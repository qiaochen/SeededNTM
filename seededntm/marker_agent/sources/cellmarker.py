"""CellMarker2.0 source — local Excel-based marker gene database.

CellMarker2.0 provides curated cell type markers from literature across
species, tissues, and conditions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from seededntm.marker_agent.cache import cached_download, find_local_file, get_cache_dir
from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

CELLMARKER_FILENAME = "Cell_marker_All.xlsx"
CELLMARKER_HUMAN_FILENAME = "Cell_marker_Human.xlsx"
CELLMARKER_URL = (
    "http://bio-bigdata.hrbmu.edu.cn/CellMarker/"
    "CellMarker_download_files/file/Cell_marker_All.xlsx"
)


class CellMarkerSource:
    """Query CellMarker2.0 database from local Excel file."""

    def __init__(self, cache_dir: Optional[str] = None, filepath: Optional[str] = None):
        self.cache_dir = cache_dir
        self._df: Optional[pd.DataFrame] = None
        self._filepath = filepath

    def _ensure_data(self) -> Optional[Path]:
        """Download CellMarker Excel if not already cached."""
        for fname in [CELLMARKER_FILENAME, CELLMARKER_HUMAN_FILENAME]:
            path = find_local_file(fname, cache_dir=self.cache_dir)
            if path is not None:
                return path

        cache = get_cache_dir(self.cache_dir)
        logger.info("Auto-downloading CellMarker from %s", CELLMARKER_URL)
        try:
            result = cached_download(
                CELLMARKER_URL, CELLMARKER_FILENAME, cache_dir=self.cache_dir
            )
            return result
        except Exception as e:
            logger.error(
                "Failed to auto-download CellMarker from %s: %s. "
                "Download manually and place in %s",
                CELLMARKER_URL, e, cache,
            )
            return None

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        path = None
        if self._filepath:
            path = Path(self._filepath)
        else:
            path = self._ensure_data()

        if path is None:
            self._df = pd.DataFrame()
            return self._df

        logger.info("Loading CellMarker from %s", path)
        try:
            self._df = pd.read_excel(path, engine="openpyxl")
            self._df.columns = self._df.columns.str.strip()
        except Exception as e:
            logger.error("Failed to load CellMarker: %s", e)
            self._df = pd.DataFrame()

        return self._df

    def search(
        self,
        cell_type: str,
        tissue: str,
        species: str = "Human",
        **kwargs,
    ) -> List[MarkerHit]:
        """Search CellMarker for markers of a cell type in a tissue."""
        df = self._load()
        if df.empty:
            return []

        species_col = next(
            (c for c in df.columns if "species" in c.lower()), None
        )
        # Prefer 'cell_name' (actual cell identity) over 'cell_type' (broad category like "Normal cell")
        celltype_col = next(
            (c for c in df.columns if c.lower() == "cell_name"), None
        )
        if celltype_col is None:
            celltype_col = next(
                (c for c in df.columns if "cell" in c.lower() and ("name" in c.lower() or "type" in c.lower())),
                None,
            )
        tissue_col = next(
            (c for c in df.columns if c.lower() == "tissue_type"), None
        )
        if tissue_col is None:
            tissue_col = next(
                (c for c in df.columns if "tissue" in c.lower()), None
            )
        # Prefer 'Symbol' (standard gene symbol) over 'marker' (can be an alias)
        marker_col = next(
            (c for c in df.columns if c.lower() == "symbol"), None
        )
        if marker_col is None:
            marker_col = next(
                (c for c in df.columns if "marker" in c.lower() or "symbol" in c.lower() or "gene" in c.lower()),
                None,
            )

        if celltype_col is None or marker_col is None:
            logger.warning("CellMarker columns not recognized: %s", list(df.columns))
            return []

        mask = self._flexible_cell_type_match(df, celltype_col, cell_type)

        if species_col:
            sp_mask = df[species_col].astype(str).str.lower().str.contains(
                species.lower(), na=False
            )
            mask = mask & sp_mask

        if tissue_col and tissue:
            tissue_mask = df[tissue_col].astype(str).str.lower().str.contains(
                tissue.lower(), na=False
            )
            mask_with_tissue = mask & tissue_mask
            if mask_with_tissue.any():
                mask = mask_with_tissue

        results = df[mask]

        logger.debug(
            "CellMarker: query cell_type='%s', tissue='%s', species='%s' -> %d rows matched",
            cell_type, tissue, species, len(results),
        )

        hits = []

        for _, row in results.iterrows():
            marker_value = str(row[marker_col])
            if not marker_value or marker_value.lower() == "nan":
                continue

            genes = [g.strip() for g in marker_value.replace("/", ",").split(",")]
            for gene in genes:
                if not gene or len(gene) > 20:
                    continue
                detail = ""
                if tissue_col:
                    detail = f"tissue={row.get(tissue_col, '')}"

                hits.append(
                    MarkerHit(
                        gene_symbol=gene.upper(),
                        cell_type=cell_type,
                        source="CellMarker",
                        score=1.0,
                        evidence_detail=detail,
                    )
                )

        logger.info("CellMarker: found %d markers for '%s'", len(hits), cell_type)
        return hits

    @staticmethod
    def _flexible_cell_type_match(
        df: "pd.DataFrame", celltype_col: str, cell_type: str
    ) -> "pd.Series":
        """Match cell type flexibly: exact substring, singular/plural, and word stem.

        Handles cases like:
        - "T cells" matching "T cell", "CD4+ T cell", "Regulatory T cell"
        - "Macrophage" matching "Macrophages", "M1 Macrophage"
        """
        import re

        col_lower = df[celltype_col].astype(str).str.lower()
        ct_lower = cell_type.lower().strip()

        mask = col_lower.str.contains(re.escape(ct_lower), na=False)

        if not mask.any():
            candidates = []
            if ct_lower.endswith("es"):
                candidates.append(ct_lower[:-2])
                candidates.append(ct_lower[:-1])
            elif ct_lower.endswith("s"):
                candidates.append(ct_lower[:-1])
            for singular in candidates:
                mask = col_lower.str.contains(re.escape(singular), na=False)
                if mask.any():
                    break

        if not mask.any():
            if not ct_lower.endswith("s"):
                plural = ct_lower + "s"
                mask = col_lower.str.contains(re.escape(plural), na=False)

        if not mask.any():
            words = ct_lower.split()
            if len(words) >= 2:
                pattern = ".*".join(re.escape(w.rstrip("s")) for w in words)
                mask = col_lower.str.contains(pattern, na=False)

        if not mask.any():
            pass

        if not mask.any():
            logger.debug(
                "CellMarker: no match for '%s' after flexible search. "
                "Sample cell_name values: %s",
                cell_type,
                col_lower.dropna().unique()[:20].tolist(),
            )

        return mask

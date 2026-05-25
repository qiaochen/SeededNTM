"""CellMarker2.0 source — local Excel-based marker gene database.

CellMarker2.0 provides curated cell type markers from literature across
species, tissues, and conditions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from seededntm.marker_agent.cache import find_local_file, get_cache_dir
from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

CELLMARKER_FILENAME = "Cell_marker_All.xlsx"
CELLMARKER_HUMAN_FILENAME = "Cell_marker_Human.xlsx"


class CellMarkerSource:
    """Query CellMarker2.0 database from local Excel file."""

    def __init__(self, cache_dir: Optional[str] = None, filepath: Optional[str] = None):
        self.cache_dir = cache_dir
        self._df: Optional[pd.DataFrame] = None
        self._filepath = filepath

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        path = None
        if self._filepath:
            path = Path(self._filepath)
        else:
            for fname in [CELLMARKER_FILENAME, CELLMARKER_HUMAN_FILENAME]:
                path = find_local_file(fname, cache_dir=self.cache_dir)
                if path:
                    break

        if path is None:
            logger.warning(
                "CellMarker Excel not found locally. Place '%s' in cache dir: %s",
                CELLMARKER_FILENAME,
                get_cache_dir(self.cache_dir),
            )
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
        celltype_col = next(
            (c for c in df.columns if "cell" in c.lower() and ("name" in c.lower() or "type" in c.lower())),
            None,
        )
        tissue_col = next(
            (c for c in df.columns if "tissue" in c.lower()), None
        )
        marker_col = next(
            (c for c in df.columns if "marker" in c.lower() or "symbol" in c.lower() or "gene" in c.lower()),
            None,
        )

        if celltype_col is None or marker_col is None:
            logger.warning("CellMarker columns not recognized: %s", list(df.columns))
            return []

        mask = df[celltype_col].astype(str).str.lower().str.contains(
            cell_type.lower(), na=False
        )

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

"""PanglaoDB source — local TSV-based marker gene database.

PanglaoDB provides curated marker genes with organ, cell type, sensitivity,
and specificity annotations. Data is loaded from a local TSV file.
"""

from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import pandas as pd

from seededntm.marker_agent.cache import cached_download, find_local_file, get_cache_dir
from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

PANGLAODB_FILENAME = "PanglaoDB_markers_27_Mar_2020.tsv"
PANGLAODB_GZ_FILENAME = "PanglaoDB_markers_27_Mar_2020.tsv.gz"
PANGLAODB_URL = "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz"


class PanglaoDBSource:
    """Query PanglaoDB marker database from local TSV file."""

    def __init__(self, cache_dir: Optional[str] = None, filepath: Optional[str] = None):
        self.cache_dir = cache_dir
        self._df: Optional[pd.DataFrame] = None
        self._filepath = filepath

    def _ensure_data(self) -> Optional[Path]:
        """Download PanglaoDB TSV if not already cached."""
        path = find_local_file(PANGLAODB_FILENAME, cache_dir=self.cache_dir)
        if path is not None:
            return path

        path = find_local_file(PANGLAODB_GZ_FILENAME, cache_dir=self.cache_dir)
        if path is not None:
            return path

        cache = get_cache_dir(self.cache_dir)
        tsv_path = cache / PANGLAODB_FILENAME
        gz_path = cache / PANGLAODB_GZ_FILENAME

        logger.info("Auto-downloading PanglaoDB from %s", PANGLAODB_URL)
        try:
            cached_download(PANGLAODB_URL, PANGLAODB_GZ_FILENAME, cache_dir=self.cache_dir)
            with gzip.open(gz_path, "rb") as f_in, open(tsv_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            logger.info("PanglaoDB TSV decompressed to %s", tsv_path)
            return tsv_path
        except Exception as e:
            logger.error(
                "Failed to auto-download PanglaoDB from %s: %s. "
                "Download manually and place in %s",
                PANGLAODB_URL, e, cache,
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

        logger.info("Loading PanglaoDB from %s", path)
        try:
            self._df = pd.read_csv(path, sep="\t")
            self._df.columns = self._df.columns.str.strip()
        except Exception as e:
            logger.error("Failed to load PanglaoDB: %s", e)
            self._df = pd.DataFrame()

        return self._df

    def search(
        self,
        cell_type: str,
        tissue: str,
        species: str = "Human",
        sensitivity: float = 0.0,
        specificity: float = 0.0,
        **kwargs,
    ) -> List[MarkerHit]:
        """Search PanglaoDB for markers of a cell type.

        Filters by species, organ/tissue, and optionally by sensitivity/specificity
        thresholds.
        """
        df = self._load()
        if df.empty:
            return []

        species_col = "species" if "species" in df.columns else "Species"
        celltype_col = next(
            (c for c in df.columns if "cell" in c.lower() and "type" in c.lower()),
            None,
        )
        organ_col = next(
            (c for c in df.columns if "organ" in c.lower()), None
        )
        gene_col = next(
            (c for c in df.columns if "gene" in c.lower() or "marker" in c.lower() or "official" in c.lower()),
            None,
        )

        if celltype_col is None or gene_col is None:
            logger.warning("PanglaoDB columns not recognized: %s", list(df.columns))
            return []

        mask = df[celltype_col].str.lower().str.contains(cell_type.lower(), na=False)

        if species_col in df.columns:
            species_map = {"Human": "Hs", "Mouse": "Mm", "human": "Hs", "mouse": "Mm"}
            sp_key = species_map.get(species, species)
            species_mask = df[species_col].astype(str).str.contains(sp_key, na=False)
            mask = mask & species_mask

        if organ_col and tissue:
            tissue_mask = df[organ_col].str.lower().str.contains(
                tissue.lower(), na=False
            )
            mask_with_tissue = mask & tissue_mask
            if mask_with_tissue.any():
                mask = mask_with_tissue

        sens_col = next(
            (c for c in df.columns if "sensit" in c.lower()), None
        )
        spec_col = next(
            (c for c in df.columns if "specif" in c.lower()), None
        )

        if sens_col and sensitivity > 0:
            mask = mask & (pd.to_numeric(df[sens_col], errors="coerce") >= sensitivity)
        if spec_col and specificity > 0:
            mask = mask & (pd.to_numeric(df[spec_col], errors="coerce") >= specificity)

        results = df[mask]
        hits = []
        for _, row in results.iterrows():
            gene = str(row[gene_col]).strip()
            if not gene or gene.lower() == "nan":
                continue

            score = 1.0
            if sens_col and spec_col:
                s1 = pd.to_numeric(row.get(sens_col), errors="coerce") or 0
                s2 = pd.to_numeric(row.get(spec_col), errors="coerce") or 0
                score = (s1 + s2) / 2 if (s1 + s2) > 0 else 1.0

            detail_parts = []
            if organ_col:
                detail_parts.append(f"organ={row.get(organ_col, '')}")
            if sens_col:
                detail_parts.append(f"sensitivity={row.get(sens_col, '')}")
            if spec_col:
                detail_parts.append(f"specificity={row.get(spec_col, '')}")

            hits.append(
                MarkerHit(
                    gene_symbol=gene,
                    cell_type=cell_type,
                    source="PanglaoDB",
                    score=score,
                    evidence_detail="; ".join(detail_parts),
                )
            )

        logger.info("PanglaoDB: found %d markers for '%s'", len(hits), cell_type)
        return hits

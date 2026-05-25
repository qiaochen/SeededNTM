"""COSMIC source — Cancer Gene Census CSV for cancer-related genes.

Loads the COSMIC Cancer Gene Census to identify cancer driver genes
and their associated tumor types.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from seededntm.marker_agent.cache import find_local_file, get_cache_dir
from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

COSMIC_CGC_FILENAME = "cancer_gene_census.csv"


class COSMICSource:
    """Query COSMIC Cancer Gene Census from local CSV."""

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
            path = find_local_file(COSMIC_CGC_FILENAME, cache_dir=self.cache_dir)
            if path is None:
                for alt in ["Census_allMon.csv", "Census_all.csv"]:
                    path = find_local_file(alt, cache_dir=self.cache_dir)
                    if path:
                        break

        if path is None:
            logger.warning(
                "COSMIC CGC CSV not found. Place '%s' in cache dir: %s",
                COSMIC_CGC_FILENAME,
                get_cache_dir(self.cache_dir),
            )
            self._df = pd.DataFrame()
            return self._df

        logger.info("Loading COSMIC CGC from %s", path)
        try:
            self._df = pd.read_csv(path)
            self._df.columns = self._df.columns.str.strip()
        except Exception as e:
            logger.error("Failed to load COSMIC CGC: %s", e)
            self._df = pd.DataFrame()

        return self._df

    def search(
        self,
        cell_type: str,
        tissue: str,
        species: str = "Human",
        condition: str = "",
        **kwargs,
    ) -> List[MarkerHit]:
        """Search COSMIC CGC for cancer genes relevant to a tissue/condition."""
        df = self._load()
        if df.empty:
            return []

        gene_col = next(
            (c for c in df.columns if "gene" in c.lower() and "symbol" in c.lower()),
            None,
        )
        if gene_col is None:
            gene_col = next(
                (c for c in df.columns if "gene" in c.lower()), None
            )

        tumour_col = next(
            (c for c in df.columns if "tumour" in c.lower() or "tumor" in c.lower()),
            None,
        )
        role_col = next(
            (c for c in df.columns if "role" in c.lower()), None
        )

        if gene_col is None:
            logger.warning("COSMIC columns not recognized: %s", list(df.columns))
            return []

        search_terms = [tissue.lower(), condition.lower()] if tissue or condition else []
        search_terms = [t for t in search_terms if t]

        if search_terms and tumour_col:
            mask = df[tumour_col].astype(str).str.lower().apply(
                lambda x: any(t in x for t in search_terms)
            )
            results = df[mask]
        else:
            results = df

        hits = []
        for _, row in results.iterrows():
            symbol = str(row[gene_col]).strip()
            if not symbol or symbol.lower() == "nan":
                continue

            role = str(row.get(role_col, "")) if role_col else ""
            tumour_types = str(row.get(tumour_col, "")) if tumour_col else ""

            hits.append(
                MarkerHit(
                    gene_symbol=symbol.upper(),
                    cell_type=cell_type,
                    source="COSMIC",
                    score=0.7,
                    evidence_detail=f"role={role[:40]}; tumour_types={tumour_types[:60]}",
                )
            )

        logger.info("COSMIC: found %d cancer genes for '%s'", len(hits), cell_type)
        return hits

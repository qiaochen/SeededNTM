"""ScType database source — Excel with positive and negative markers.

ScType provides cell type markers organized by tissue, with both positive
(upregulated) and negative (should-be-absent) markers for each cell type.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from seededntm.marker_agent.cache import find_local_file, get_cache_dir
from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

SCTYPE_FILENAME = "ScTypeDB_full.xlsx"
SCTYPE_URL = (
    "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/"
    "master/ScTypeDB_full.xlsx"
)


class ScTypeDBSource:
    """Query ScType database from local Excel file."""

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
            path = find_local_file(SCTYPE_FILENAME, cache_dir=self.cache_dir)

        if path is None:
            logger.warning(
                "ScTypeDB Excel not found locally. Place '%s' in cache dir: %s",
                SCTYPE_FILENAME,
                get_cache_dir(self.cache_dir),
            )
            self._df = pd.DataFrame()
            return self._df

        logger.info("Loading ScTypeDB from %s", path)
        try:
            self._df = pd.read_excel(path, engine="openpyxl")
            self._df.columns = self._df.columns.str.strip()
        except Exception as e:
            logger.error("Failed to load ScTypeDB: %s", e)
            self._df = pd.DataFrame()

        return self._df

    def search(
        self,
        cell_type: str,
        tissue: str,
        species: str = "Human",
        include_negative: bool = False,
        **kwargs,
    ) -> List[MarkerHit]:
        """Search ScTypeDB for markers of a cell type.

        Returns positive markers by default; set include_negative=True to also
        include negative markers (with negative scores).
        """
        df = self._load()
        if df.empty:
            return []

        tissue_col = next(
            (c for c in df.columns if "tissue" in c.lower()), None
        )
        celltype_col = next(
            (c for c in df.columns if "cell" in c.lower() and "type" in c.lower()),
            None,
        )
        pos_col = next(
            (c for c in df.columns if "geneSymbolmore1" in c or "positive" in c.lower()),
            None,
        )
        neg_col = next(
            (c for c in df.columns if "geneSymbolmore2" in c or "negative" in c.lower()),
            None,
        )

        if pos_col is None:
            possible = [c for c in df.columns if "gene" in c.lower() or "marker" in c.lower()]
            if possible:
                pos_col = possible[0]
                neg_col = possible[1] if len(possible) > 1 else None

        if celltype_col is None or pos_col is None:
            logger.warning("ScTypeDB columns not recognized: %s", list(df.columns))
            return []

        mask = df[celltype_col].astype(str).str.lower().str.contains(
            cell_type.lower(), na=False
        )

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
            pos_genes_str = str(row.get(pos_col, ""))
            if pos_genes_str and pos_genes_str.lower() != "nan":
                genes = [g.strip() for g in pos_genes_str.split(",")]
                for gene in genes:
                    if not gene or len(gene) > 20:
                        continue
                    hits.append(
                        MarkerHit(
                            gene_symbol=gene.upper(),
                            cell_type=cell_type,
                            source="ScTypeDB",
                            score=1.0,
                            evidence_detail=f"positive_marker; tissue={row.get(tissue_col, '')}",
                        )
                    )

            if include_negative and neg_col:
                neg_genes_str = str(row.get(neg_col, ""))
                if neg_genes_str and neg_genes_str.lower() != "nan":
                    genes = [g.strip() for g in neg_genes_str.split(",")]
                    for gene in genes:
                        if not gene or len(gene) > 20:
                            continue
                        hits.append(
                            MarkerHit(
                                gene_symbol=gene.upper(),
                                cell_type=cell_type,
                                source="ScTypeDB",
                                score=-0.5,
                                evidence_detail=f"negative_marker; tissue={row.get(tissue_col, '')}",
                            )
                        )

        logger.info("ScTypeDB: found %d markers for '%s'", len(hits), cell_type)
        return hits

"""MSigDB source — C8 cell-type signature gene sets from GMT files.

Loads MSigDB C8 (cell type signature) gene sets to identify
curated marker gene panels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from seededntm.marker_agent.cache import cached_download, find_local_file, get_cache_dir
from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

_GENERIC_WORDS = {"cell", "cells", "type", "types", "tissue", "human", "mouse", "marker", "markers"}

MSIGDB_C8_FILENAME = "c8.all.v2023.2.Hs.symbols.gmt"
MSIGDB_C8_URL = (
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
    "2023.2.Hs/c8.all.v2023.2.Hs.symbols.gmt"
)


class MSigDBSource:
    """Query MSigDB C8 cell-type signatures from local GMT file."""

    def __init__(self, cache_dir: Optional[str] = None, filepath: Optional[str] = None):
        self.cache_dir = cache_dir
        self._gene_sets: Optional[Dict[str, Set[str]]] = None
        self._filepath = filepath

    def _ensure_data(self) -> Optional[Path]:
        """Download MSigDB C8 GMT if not already cached."""
        path = find_local_file(MSIGDB_C8_FILENAME, cache_dir=self.cache_dir)
        if path is not None:
            return path

        for alt in [
            "c8.all.v2024.1.Hs.symbols.gmt",
            "c8.all.v7.5.1.symbols.gmt",
        ]:
            path = find_local_file(alt, cache_dir=self.cache_dir)
            if path is not None:
                return path

        cache = get_cache_dir(self.cache_dir)
        logger.info("Auto-downloading MSigDB C8 from %s", MSIGDB_C8_URL)
        try:
            result = cached_download(
                MSIGDB_C8_URL, MSIGDB_C8_FILENAME, cache_dir=self.cache_dir
            )
            return result
        except Exception as e:
            logger.error(
                "Failed to auto-download MSigDB C8 from %s: %s. "
                "Download manually and place in %s",
                MSIGDB_C8_URL, e, cache,
            )
            return None

    def _load(self) -> Dict[str, Set[str]]:
        if self._gene_sets is not None:
            return self._gene_sets

        path = None
        if self._filepath:
            path = Path(self._filepath)
        else:
            path = self._ensure_data()

        if path is None:
            self._gene_sets = {}
            return self._gene_sets

        logger.info("Loading MSigDB C8 from %s", path)
        self._gene_sets = {}
        try:
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        set_name = parts[0]
                        genes = set(parts[2:])
                        self._gene_sets[set_name] = genes
        except Exception as e:
            logger.error("Failed to load MSigDB: %s", e)
            self._gene_sets = {}

        logger.info("MSigDB: loaded %d gene sets", len(self._gene_sets))
        return self._gene_sets

    def search(
        self,
        cell_type: str,
        tissue: str,
        species: str = "Human",
        **kwargs,
    ) -> List[MarkerHit]:
        """Search MSigDB C8 for gene sets matching a cell type."""
        gene_sets = self._load()
        if not gene_sets:
            return []

        cell_type_lower = cell_type.lower().replace(" ", "_")
        cell_type_parts = cell_type.lower().split()

        matching_sets: List[tuple] = []
        for set_name, genes in gene_sets.items():
            name_lower = set_name.lower()
            if cell_type_lower in name_lower:
                matching_sets.append((set_name, genes, 1.0))
            elif all(part in name_lower for part in cell_type_parts):
                matching_sets.append((set_name, genes, 0.8))
            elif any(part in name_lower for part in cell_type_parts if len(part) > 4 and part not in _GENERIC_WORDS):
                matching_sets.append((set_name, genes, 0.5))

        if tissue:
            tissue_lower = tissue.lower().replace(" ", "_")
            tissue_matched = [
                (n, g, s + 0.2)
                for n, g, s in matching_sets
                if tissue_lower in n.lower()
            ]
            if tissue_matched:
                matching_sets = tissue_matched

        gene_counts: Dict[str, float] = {}
        for set_name, genes, relevance in matching_sets[:10]:
            for gene in genes:
                if gene and len(gene) <= 15:
                    gene_counts[gene] = gene_counts.get(gene, 0) + relevance

        hits = []
        for gene, score in sorted(gene_counts.items(), key=lambda x: -x[1]):
            hits.append(
                MarkerHit(
                    gene_symbol=gene.upper(),
                    cell_type=cell_type,
                    source="MSigDB",
                    score=min(score, 1.0),
                    evidence_detail=f"found in {int(score)} matching gene sets",
                )
            )

        logger.info(
            "MSigDB: found %d markers for '%s' from %d gene sets",
            len(hits), cell_type, len(matching_sets),
        )
        return hits

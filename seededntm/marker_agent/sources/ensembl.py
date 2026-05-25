"""Ensembl source — REST API for phenotype/gene associations.

Queries Ensembl REST API's /phenotype/gene endpoint for phenotype
annotations and variant associations.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import requests

from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

ENSEMBL_REST_BASE = "https://rest.ensembl.org"


class EnsemblSource:
    """Query Ensembl REST API for gene-phenotype associations."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._session = requests.Session()
        self._session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

    def search(
        self,
        cell_type: str,
        tissue: str,
        species: str = "Human",
        gene_list: Optional[List[str]] = None,
        **kwargs,
    ) -> List[MarkerHit]:
        """Search Ensembl for phenotype annotations of candidate genes.

        If gene_list is provided, queries phenotypes for those genes.
        Otherwise returns empty (Ensembl doesn't support cell-type search).
        """
        if not gene_list:
            logger.info("Ensembl source requires gene_list for phenotype lookup")
            return []

        species_name = "homo_sapiens" if species.lower() == "human" else species.lower()
        hits = []

        for gene in gene_list[:20]:
            phenotypes = self._get_phenotypes(gene, species_name)
            if phenotypes:
                relevant = self._filter_relevant(phenotypes, cell_type, tissue)
                if relevant:
                    hits.append(
                        MarkerHit(
                            gene_symbol=gene.upper(),
                            cell_type=cell_type,
                            source="Ensembl",
                            score=min(len(relevant) * 0.2, 1.0),
                            evidence_detail=f"phenotypes: {'; '.join(relevant[:3])}",
                        )
                    )
            time.sleep(0.1)

        logger.info("Ensembl: found %d phenotype hits for '%s'", len(hits), cell_type)
        return hits

    def _get_phenotypes(self, gene_symbol: str, species: str) -> List[str]:
        """Get phenotype annotations for a gene."""
        url = f"{ENSEMBL_REST_BASE}/phenotype/gene/{species}/{gene_symbol}"

        try:
            resp = self._session.get(url, timeout=15)
            if resp.status_code == 429:
                time.sleep(1)
                resp = self._session.get(url, timeout=15)
            if resp.status_code != 200:
                return []
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.debug("Ensembl phenotype lookup failed for %s: %s", gene_symbol, e)
            return []

        phenotypes = []
        for entry in data:
            desc = entry.get("description", "")
            if desc:
                phenotypes.append(desc)

        return phenotypes

    def _filter_relevant(
        self, phenotypes: List[str], cell_type: str, tissue: str
    ) -> List[str]:
        """Filter phenotypes relevant to the cell type or tissue."""
        keywords = [
            cell_type.lower(),
            tissue.lower() if tissue else "",
        ]
        keywords = [k for k in keywords if k]

        relevant = []
        for pheno in phenotypes:
            pheno_lower = pheno.lower()
            if any(kw in pheno_lower for kw in keywords):
                relevant.append(pheno)

        return relevant if relevant else phenotypes[:2]

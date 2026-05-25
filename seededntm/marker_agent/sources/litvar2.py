"""LitVar2 REST API source — gene-variant-publication links.

LitVar2 maps genetic variants to genes and publications, helping identify
genes with clinically reported variants in a specific disease context.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

BASE_URL = "https://www.ncbi.nlm.nih.gov/research/litvar2-api"


class LitVar2Source:
    """Retrieve marker genes via LitVar2 variant-publication counts.

    Strategy:
      1. For candidate genes (from other sources), query LitVar2 for variants
      2. Count publications per gene discussing variants in the disease context
      3. Higher publication count = stronger literature support
    """

    def __init__(self, cache_dir: Optional[str] = None, max_genes: int = 30):
        self.cache_dir = cache_dir
        self.max_genes = max_genes
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "SeedTopic-MarkerAgent/1.0",
        })

    def search(
        self,
        cell_type: str,
        tissue: str = "",
        species: str = "Human",
        condition: str = "",
        candidate_genes: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[MarkerHit]:
        """Search LitVar2 for genes with variant-disease literature support.

        Args:
            cell_type: Target cell type name.
            tissue: Tissue context.
            species: Species (LitVar2 is primarily human).
            condition: Disease or condition context.
            candidate_genes: Pre-filtered list of genes to check.

        Returns:
            List of MarkerHit scored by publication count.
        """
        if not candidate_genes:
            logger.debug("LitVar2: no candidate genes provided, skipping")
            return []

        genes_to_query = candidate_genes[: self.max_genes]
        gene_pub_counts: Dict[str, int] = {}

        for gene in genes_to_query:
            count = self._get_variant_publications(gene)
            if count > 0:
                gene_pub_counts[gene] = count
            time.sleep(0.3)

        if not gene_pub_counts:
            return []

        max_count = max(gene_pub_counts.values())
        hits = []
        for gene, count in sorted(gene_pub_counts.items(), key=lambda x: -x[1]):
            score = count / max_count
            hits.append(
                MarkerHit(
                    gene_symbol=gene,
                    cell_type=cell_type,
                    source="LitVar2",
                    score=score,
                    evidence_detail=f"{count} variant publications",
                )
            )

        return hits

    def _get_variant_publications(self, gene_name: str) -> int:
        """Query LitVar2 for total variant-related publications for a gene."""
        try:
            url = f"{BASE_URL}/variant/search/gene/{gene_name}"
            resp = self._session.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            total_pubs = 0
            variants = data if isinstance(data, list) else data.get("results", [])
            for variant in variants:
                if isinstance(variant, dict):
                    pub_count = variant.get("publication_count", 0)
                    if isinstance(pub_count, int):
                        total_pubs += pub_count
                    elif isinstance(variant.get("publications"), list):
                        total_pubs += len(variant["publications"])

            return total_pubs

        except requests.RequestException as e:
            logger.debug("LitVar2 query failed for %s: %s", gene_name, e)
            return 0
        except (ValueError, KeyError) as e:
            logger.debug("LitVar2 parse error for %s: %s", gene_name, e)
            return 0

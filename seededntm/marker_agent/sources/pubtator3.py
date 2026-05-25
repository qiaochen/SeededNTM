"""PubTator3 REST API source — pre-annotated gene-disease entity search.

PubTator3 provides AI-annotated entity extraction from 36M+ PubMed articles.
Unlike raw PubMed, it returns pre-extracted gene entities co-occurring with
disease/cell-type terms, eliminating the need for LLM-based text parsing.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

BASE_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
SEARCH_URL = f"{BASE_URL}/search/"
EXPORT_URL = f"{BASE_URL}/publications/export/biocjson"
AUTOCOMPLETE_URL = f"{BASE_URL}/entity/autocomplete/"


class PubTator3Source:
    """Retrieve marker genes via PubTator3 entity co-occurrence.

    Strategy:
      1. Search PubTator3 for articles mentioning cell_type + tissue
      2. Export BioC-JSON annotations for top PMIDs
      3. Extract GENE entities co-occurring with the search terms
      4. Score by frequency (number of papers mentioning the gene)
    """

    def __init__(self, cache_dir: Optional[str] = None, max_papers: int = 50):
        self.cache_dir = cache_dir
        self.max_papers = max_papers
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
        **kwargs: Any,
    ) -> List[MarkerHit]:
        """Search PubTator3 for genes associated with a cell type.

        Args:
            cell_type: Target cell type name.
            tissue: Tissue context for the search.
            species: Species filter (used in query construction).
            condition: Disease or condition context.

        Returns:
            List of MarkerHit with gene symbols and frequency scores.
        """
        query = self._build_query(cell_type, tissue, condition)
        pmids = self._search_pmids(query)

        if not pmids:
            logger.debug("PubTator3: no papers found for query: %s", query)
            return []

        pmids = pmids[: self.max_papers]
        gene_counts = self._extract_gene_entities(pmids)

        if not gene_counts:
            return []

        max_count = max(gene_counts.values())
        hits = []
        for gene, count in sorted(gene_counts.items(), key=lambda x: -x[1]):
            score = count / max_count
            hits.append(
                MarkerHit(
                    gene_symbol=gene,
                    cell_type=cell_type,
                    source="PubTator3",
                    score=score,
                    evidence_detail=f"co-occurred in {count}/{len(pmids)} papers",
                )
            )

        return hits

    def _build_query(self, cell_type: str, tissue: str, condition: str) -> str:
        """Build search query combining cell type with tissue/condition."""
        parts = [f'"{cell_type}"']
        if tissue:
            parts.append(f'"{tissue}"')
        elif condition:
            parts.append(f'"{condition}"')
        parts.append("marker gene")
        return " AND ".join(parts)

    def _search_pmids(self, query: str) -> List[str]:
        """Search PubTator3 and return PMIDs."""
        try:
            resp = self._session.get(
                SEARCH_URL,
                params={"text": query},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            pmids = []
            results = data.get("results", [])
            for item in results:
                pmid = item.get("pmid") or item.get("_id")
                if pmid:
                    pmids.append(str(pmid))

            return pmids

        except requests.RequestException as e:
            logger.warning("PubTator3 search failed: %s", e)
            return []

    def _extract_gene_entities(self, pmids: List[str]) -> Dict[str, int]:
        """Export annotations for PMIDs and count gene entities."""
        gene_counts: Dict[str, int] = {}

        batch_size = 20
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            try:
                resp = self._session.get(
                    EXPORT_URL,
                    params={"pmids": ",".join(batch)},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                publications = data if isinstance(data, list) else data.get("PubTator3", [])
                for pub in publications:
                    seen_genes_in_paper: set = set()
                    passages = pub.get("passages", [])
                    for passage in passages:
                        annotations = passage.get("annotations", [])
                        for ann in annotations:
                            ann_type = ann.get("infons", {}).get("type", "")
                            if ann_type == "Gene":
                                ident = ann.get("infons", {}).get("identifier", "")
                                text = ann.get("text", "").upper()
                                if text and text not in seen_genes_in_paper:
                                    seen_genes_in_paper.add(text)
                                    gene_counts[text] = gene_counts.get(text, 0) + 1

            except requests.RequestException as e:
                logger.warning("PubTator3 export failed for batch: %s", e)
            except (ValueError, KeyError) as e:
                logger.warning("PubTator3 parse error: %s", e)

            time.sleep(0.3)

        return gene_counts

"""Human Protein Atlas (HPA) source — REST API for expression data.

Queries the HPA API for tissue-level and single-cell expression data
to validate marker gene specificity.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import requests

from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

HPA_BASE_URL = "https://www.proteinatlas.org"
HPA_API_URL = "https://www.proteinatlas.org/api/search_download.php"


class HPASource:
    """Query Human Protein Atlas for tissue and single-cell expression."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    def search(
        self,
        cell_type: str,
        tissue: str,
        species: str = "Human",
        **kwargs,
    ) -> List[MarkerHit]:
        """Search HPA for genes enriched in a cell type/tissue.

        Uses the HPA normal tissue and single-cell RNA data.
        """
        if species.lower() != "human":
            logger.info("HPA only supports Human data, skipping for %s", species)
            return []

        hits = []
        hits.extend(self._search_single_cell(cell_type, tissue))

        if not hits:
            hits.extend(self._search_tissue(cell_type, tissue))

        return hits

    def _search_single_cell(self, cell_type: str, tissue: str) -> List[MarkerHit]:
        """Query HPA single-cell type data."""
        url = f"{HPA_BASE_URL}/{cell_type.replace(' ', '+')}/single+cell+type"

        try:
            resp = self._session.get(url, timeout=30)
            if resp.status_code != 200:
                return self._search_by_gene_list(cell_type, tissue)
        except Exception as e:
            logger.warning("HPA single-cell request failed: %s", e)
            return []

        return self._search_by_gene_list(cell_type, tissue)

    def _search_tissue(self, cell_type: str, tissue: str) -> List[MarkerHit]:
        """Query HPA tissue expression."""
        return self._search_by_gene_list(cell_type, tissue)

    def _search_by_gene_list(self, cell_type: str, tissue: str) -> List[MarkerHit]:
        """Use HPA search API to find genes associated with a cell type."""
        search_term = f"{cell_type} {tissue}" if tissue else cell_type

        params = {
            "search": search_term,
            "format": "json",
            "columns": "g,gs,t,sc",
            "compress": "no",
        }

        try:
            resp = self._session.get(HPA_API_URL, params=params, timeout=30)
            if resp.status_code != 200:
                logger.warning("HPA API returned %d", resp.status_code)
                return []
            data = resp.json() if resp.text.strip() else []
        except (requests.RequestException, ValueError) as e:
            logger.warning("HPA API query failed: %s", e)
            return []

        hits = []
        if isinstance(data, list):
            for entry in data[:50]:
                gene = entry.get("Gene") or entry.get("g", "")
                gene_name = entry.get("Gene name") or entry.get("gs", "")
                if gene:
                    hits.append(
                        MarkerHit(
                            gene_symbol=gene.upper(),
                            cell_type=cell_type,
                            source="HPA",
                            score=0.8,
                            evidence_detail=f"gene_name={gene_name}",
                        )
                    )

        logger.info("HPA: found %d genes for '%s'", len(hits), cell_type)
        return hits

    def get_gene_expression(self, gene_symbol: str) -> dict:
        """Get expression profile for a specific gene."""
        url = f"{HPA_BASE_URL}/{gene_symbol}.json"
        try:
            resp = self._session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning("HPA gene lookup failed for %s: %s", gene_symbol, e)
        return {}

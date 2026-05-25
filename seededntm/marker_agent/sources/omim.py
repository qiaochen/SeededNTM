"""OMIM source — REST API for disease-gene associations.

Queries OMIM (Online Mendelian Inheritance in Man) for disease-gene maps
relevant to the experimental condition.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import requests

from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

OMIM_API_BASE = "https://api.omim.org/api"


class OMIMSource:
    """Query OMIM for disease-gene associations."""

    def __init__(self, cache_dir: Optional[str] = None, api_key: Optional[str] = None):
        self.cache_dir = cache_dir
        self.api_key = api_key or os.environ.get("OMIM_API_KEY", "")
        self._session = requests.Session()

    def search(
        self,
        cell_type: str,
        tissue: str,
        species: str = "Human",
        condition: str = "",
        **kwargs,
    ) -> List[MarkerHit]:
        """Search OMIM for genes associated with a disease/condition.

        Requires an OMIM API key. Returns empty list if key is not set.
        """
        if not self.api_key:
            logger.info("OMIM API key not set, skipping OMIM source")
            return []

        if not condition:
            condition = tissue

        search_term = f"{condition} {cell_type}"

        params = {
            "apiKey": self.api_key,
            "search": search_term,
            "format": "json",
            "limit": 20,
            "include": "geneMap",
        }

        try:
            resp = self._session.get(
                f"{OMIM_API_BASE}/entry/search", params=params, timeout=30
            )
            if resp.status_code != 200:
                logger.warning("OMIM API returned %d", resp.status_code)
                return []
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("OMIM API query failed: %s", e)
            return []

        hits = []
        entries = (
            data.get("omim", {})
            .get("searchResponse", {})
            .get("entryList", [])
        )

        for entry_wrapper in entries:
            entry = entry_wrapper.get("entry", {})
            gene_map = entry.get("geneMap", {})
            gene_symbols = gene_map.get("geneSymbols", "")
            if gene_symbols:
                for symbol in gene_symbols.split(","):
                    symbol = symbol.strip()
                    if symbol and len(symbol) <= 15:
                        title = entry.get("titles", {}).get("preferredTitle", "")
                        hits.append(
                            MarkerHit(
                                gene_symbol=symbol.upper(),
                                cell_type=cell_type,
                                source="OMIM",
                                score=0.7,
                                evidence_detail=f"disease={title[:80]}",
                            )
                        )

        logger.info("OMIM: found %d disease-gene hits for '%s'", len(hits), cell_type)
        return hits

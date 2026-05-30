"""DisGeNET source — API for gene-disease association scores.

Queries DisGeNET for gene-disease associations with evidence scores.
Falls back to unauthenticated queries when no API key is available.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import requests

from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

DISGENET_API_BASE = "https://www.disgenet.org/api"


class DisGeNETSource:
    """Query DisGeNET for gene-disease association scores.

    Primary: authenticated DisGeNET API (requires API key).
    Fallback: unauthenticated request to the public endpoint (limited but functional).
    """

    def __init__(self, cache_dir: Optional[str] = None, api_key: Optional[str] = None):
        self.cache_dir = cache_dir
        self.api_key = api_key or os.environ.get("DISGENET_API_KEY", "")
        self._session = requests.Session()
        if self.api_key:
            self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def search(
        self,
        cell_type: str,
        tissue: str,
        species: str = "Human",
        condition: str = "",
        min_score: float = 0.3,
        **kwargs,
    ) -> List[MarkerHit]:
        """Search DisGeNET for genes associated with a disease.

        Filters by GDA (gene-disease association) score. If no API key is set,
        attempts unauthenticated access to the public endpoint.
        """
        if not self.api_key:
            logger.info(
                "DisGeNET API key not set — attempting unauthenticated public endpoint"
            )

        disease_term = condition if condition else tissue
        return self._query_api(disease_term, cell_type, min_score)

    def _query_api(
        self, disease_term: str, cell_type: str, min_score: float
    ) -> List[MarkerHit]:
        """Query DisGeNET API (works with or without auth, limited without)."""
        params = {
            "disease": disease_term,
            "min_score": min_score,
            "format": "json",
            "limit": 50,
        }

        try:
            resp = self._session.get(
                f"{DISGENET_API_BASE}/gda/disease/{disease_term}",
                params=params,
                timeout=30,
            )
            if resp.status_code == 404:
                return self._search_by_name(disease_term, cell_type, min_score)
            if resp.status_code == 401 or resp.status_code == 403:
                logger.info(
                    "DisGeNET API requires authentication (HTTP %d). "
                    "Set DISGENET_API_KEY for full access. "
                    "OpenTargets source covers similar disease-gene associations.",
                    resp.status_code,
                )
                return []
            if resp.status_code == 429:
                logger.info(
                    "DisGeNET rate limit reached. Try again later or set DISGENET_API_KEY."
                )
                return []
            if resp.status_code != 200:
                logger.warning("DisGeNET API returned %d", resp.status_code)
                return []
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("DisGeNET API query failed: %s", e)
            return []

        return self._parse_results(data, cell_type)

    def _search_by_name(
        self, disease_name: str, cell_type: str, min_score: float
    ) -> List[MarkerHit]:
        """Search DisGeNET by disease name."""
        params = {
            "disease": disease_name,
            "min_score": min_score,
            "source": "ALL",
        }

        try:
            resp = self._session.get(
                f"{DISGENET_API_BASE}/gda/search",
                params=params,
                timeout=30,
            )
            if resp.status_code in (401, 403):
                logger.info(
                    "DisGeNET search requires auth (HTTP %d). "
                    "OpenTargets source covers similar disease-gene data.",
                    resp.status_code,
                )
                return []
            if resp.status_code != 200:
                return []
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("DisGeNET search failed: %s", e)
            return []

        return self._parse_results(data, cell_type)

    def _parse_results(self, data: list | dict, cell_type: str) -> List[MarkerHit]:
        """Parse DisGeNET API results into MarkerHits."""
        hits = []
        entries = data if isinstance(data, list) else data.get("results", [])

        for entry in entries:
            symbol = entry.get("gene_symbol") or entry.get("geneSymbol", "")
            gda_score = entry.get("score") or entry.get("gdaScore", 0)
            disease_name = entry.get("disease_name") or entry.get("diseaseName", "")

            if symbol:
                hits.append(
                    MarkerHit(
                        gene_symbol=symbol.upper(),
                        cell_type=cell_type,
                        source="DisGeNET",
                        score=float(gda_score) if gda_score else 0.5,
                        evidence_detail=f"GDA_score={gda_score}; disease={disease_name[:60]}",
                    )
                )

        logger.info("DisGeNET: found %d hits for '%s'", len(hits), cell_type)
        return hits

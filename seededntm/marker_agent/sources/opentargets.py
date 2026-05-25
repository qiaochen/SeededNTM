"""OpenTargets source — target-disease evidence from Parquet/API.

Queries OpenTargets Platform for target-disease association evidence
scores. Can use local Parquet files or REST API.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import requests

from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

OPENTARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"


class OpenTargetsSource:
    """Query OpenTargets Platform for target-disease associations."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def search(
        self,
        cell_type: str,
        tissue: str,
        species: str = "Human",
        condition: str = "",
        **kwargs,
    ) -> List[MarkerHit]:
        """Search OpenTargets for target-disease evidence.

        Uses the GraphQL API to find genes associated with a disease.
        """
        if species.lower() != "human":
            logger.info("OpenTargets primarily covers human data")
            return []

        disease_term = condition if condition else tissue
        if not disease_term:
            return []

        disease_id = self._search_disease(disease_term)
        if not disease_id:
            return []

        return self._get_associated_targets(disease_id, cell_type)

    def _search_disease(self, term: str) -> Optional[str]:
        """Search for a disease EFO ID by name."""
        query = """
        query searchDisease($term: String!) {
            search(queryString: $term, entityNames: ["disease"], page: {size: 1, index: 0}) {
                hits {
                    id
                    name
                }
            }
        }
        """
        try:
            resp = self._session.post(
                OPENTARGETS_API,
                json={"query": query, "variables": {"term": term}},
                timeout=30,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            hits = data.get("data", {}).get("search", {}).get("hits", [])
            return hits[0]["id"] if hits else None
        except (requests.RequestException, ValueError, IndexError, KeyError) as e:
            logger.warning("OpenTargets disease search failed: %s", e)
            return None

    def _get_associated_targets(
        self, disease_id: str, cell_type: str
    ) -> List[MarkerHit]:
        """Get targets associated with a disease."""
        query = """
        query associatedTargets($diseaseId: String!) {
            disease(efoId: $diseaseId) {
                associatedTargets(page: {size: 50, index: 0}) {
                    rows {
                        target {
                            approvedSymbol
                            approvedName
                        }
                        score
                        datatypeScores {
                            id
                            score
                        }
                    }
                }
            }
        }
        """
        try:
            resp = self._session.post(
                OPENTARGETS_API,
                json={"query": query, "variables": {"diseaseId": disease_id}},
                timeout=30,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("OpenTargets target query failed: %s", e)
            return []

        hits = []
        rows = (
            data.get("data", {})
            .get("disease", {})
            .get("associatedTargets", {})
            .get("rows", [])
        )

        for row in rows:
            target = row.get("target", {})
            symbol = target.get("approvedSymbol", "")
            score = row.get("score", 0)
            name = target.get("approvedName", "")

            if symbol:
                hits.append(
                    MarkerHit(
                        gene_symbol=symbol.upper(),
                        cell_type=cell_type,
                        source="OpenTargets",
                        score=float(score),
                        evidence_detail=f"overall_score={score:.3f}; name={name[:50]}",
                    )
                )

        logger.info("OpenTargets: found %d targets for '%s'", len(hits), cell_type)
        return hits

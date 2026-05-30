"""OMIM source — REST API for disease-gene associations.

Queries OMIM (Online Mendelian Inheritance in Man) for disease-gene maps
relevant to the experimental condition. Falls back to NCBI Entrez
E-utilities when no OMIM API key is available.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import requests

from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

OMIM_API_BASE = "https://api.omim.org/api"


class OMIMSource:
    """Query OMIM for disease-gene associations.

    Primary: OMIM REST API (requires API key).
    Fallback: NCBI Entrez E-utilities (esearch + elink, no key required).
    """

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

        Uses the OMIM API if a key is available, otherwise falls back to
        NCBI Entrez E-utilities to find OMIM-linked genes.
        """
        if self.api_key:
            return self._search_omim_api(cell_type, tissue, condition)

        logger.info("OMIM API key not set — using NCBI Entrez fallback for OMIM data")
        return self._search_via_entrez(cell_type, tissue, condition)

    def _search_omim_api(
        self, cell_type: str, tissue: str, condition: str
    ) -> List[MarkerHit]:
        """Search using the official OMIM REST API."""
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

    def _search_via_entrez(
        self, cell_type: str, tissue: str, condition: str
    ) -> List[MarkerHit]:
        """Fallback: query NCBI Entrez to find OMIM-linked genes.

        Uses esearch on the OMIM database and elink to get gene associations.
        This doesn't require an OMIM API key — just public NCBI E-utilities.
        """
        try:
            from Bio import Entrez
        except ImportError:
            logger.info(
                "Bio.Entrez not available — cannot use Entrez fallback for OMIM. "
                "Install biopython or set OMIM_API_KEY."
            )
            return []

        Entrez.email = os.environ.get("ENTREZ_EMAIL", "seededntm@example.com")

        disease_term = condition if condition else tissue
        query = f'"{disease_term}"[All Fields]'

        try:
            handle = Entrez.esearch(db="omim", term=query, retmax=20)
            search_results = Entrez.read(handle)
            handle.close()
        except Exception as e:
            logger.warning("Entrez esearch (OMIM) failed: %s", e)
            return []

        omim_ids = search_results.get("IdList", [])
        if not omim_ids:
            logger.info("Entrez: no OMIM entries found for '%s'", disease_term)
            return []

        # Use elink to find gene associations from OMIM entries
        gene_ids = set()
        try:
            time.sleep(0.35)
            handle = Entrez.elink(
                dbfrom="omim", db="gene", id=omim_ids[:15], linkname="omim_gene"
            )
            link_results = Entrez.read(handle)
            handle.close()

            for record in link_results:
                for linkset in record.get("LinkSetDb", []):
                    for link in linkset.get("Link", []):
                        gene_ids.add(link["Id"])
        except Exception as e:
            logger.warning("Entrez elink (omim->gene) failed: %s", e)
            return []

        if not gene_ids:
            logger.info("Entrez: no gene links found for OMIM entries")
            return []

        # Fetch gene symbols via esummary
        hits = []
        try:
            time.sleep(0.35)
            handle = Entrez.esummary(db="gene", id=",".join(list(gene_ids)[:30]))
            summaries = Entrez.read(handle)
            handle.close()

            doc_sums = summaries.get("DocumentSummarySet", {}).get(
                "DocumentSummary", []
            )
            for doc in doc_sums:
                symbol = doc.get("NomenclatureSymbol") or doc.get("Name", "")
                description = doc.get("Description", "")
                if symbol and len(symbol) <= 15 and symbol.isalnum():
                    hits.append(
                        MarkerHit(
                            gene_symbol=symbol.upper(),
                            cell_type=cell_type,
                            source="OMIM",
                            score=0.65,
                            evidence_detail=f"entrez_fallback; desc={description[:60]}",
                        )
                    )
        except Exception as e:
            logger.warning("Entrez esummary (gene) failed: %s", e)
            return []

        logger.info(
            "OMIM (Entrez fallback): found %d disease-gene hits for '%s'",
            len(hits), cell_type,
        )
        return hits

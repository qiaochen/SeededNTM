"""MyGene.info source — batch gene annotation via REST API.

Uses the mygene Python client to retrieve gene annotations including
GO terms, pathways, and gene summaries for marker validation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from seededntm.marker_agent.schemas import GeneAnnotation, MarkerHit

logger = logging.getLogger(__name__)


class MyGeneSource:
    """Query MyGene.info for gene annotations and symbol validation."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._mg = None

    def _get_client(self):
        if self._mg is None:
            try:
                import mygene
                self._mg = mygene.MyGeneInfo()
            except ImportError:
                logger.error("mygene package not installed. Run: pip install mygene")
                return None
        return self._mg

    def search(
        self,
        cell_type: str,
        tissue: str,
        species: str = "Human",
        **kwargs,
    ) -> List[MarkerHit]:
        """Search MyGene for genes associated with a cell type via GO terms.

        This is primarily used for validation rather than discovery.
        """
        mg = self._get_client()
        if mg is None:
            return []

        query = f"{cell_type} AND {tissue}" if tissue else cell_type
        try:
            results = mg.query(
                query,
                species=species.lower(),
                fields="symbol,name,go,pathway",
                size=50,
            )
        except Exception as e:
            logger.warning("MyGene query failed: %s", e)
            return []

        hits = []
        for hit in results.get("hits", []):
            symbol = hit.get("symbol")
            if not symbol:
                continue
            hits.append(
                MarkerHit(
                    gene_symbol=symbol.upper(),
                    cell_type=cell_type,
                    source="MyGene",
                    score=0.5,
                    evidence_detail=f"name={hit.get('name', '')}",
                )
            )

        logger.info("MyGene: found %d genes for '%s'", len(hits), cell_type)
        return hits

    def annotate_genes(
        self,
        gene_symbols: List[str],
        species: str = "Human",
    ) -> Dict[str, GeneAnnotation]:
        """Batch-annotate a list of gene symbols with functional information."""
        mg = self._get_client()
        if mg is None:
            return {}

        try:
            results = mg.querymany(
                gene_symbols,
                scopes="symbol",
                fields="symbol,summary,go,pathway,generif",
                species=species.lower(),
                returnall=True,
            )
        except Exception as e:
            logger.warning("MyGene batch annotation failed: %s", e)
            return {}

        annotations = {}
        for hit in results.get("out", []):
            symbol = hit.get("symbol")
            if not symbol:
                continue

            go_terms = []
            go_data = hit.get("go", {})
            for category in ["BP", "MF", "CC"]:
                terms = go_data.get(category, [])
                if isinstance(terms, dict):
                    terms = [terms]
                for t in terms:
                    if isinstance(t, dict) and "term" in t:
                        go_terms.append(t["term"])

            pathways = []
            pathway_data = hit.get("pathway", {})
            if isinstance(pathway_data, dict):
                for db, entries in pathway_data.items():
                    if isinstance(entries, list):
                        for e in entries:
                            if isinstance(e, dict) and "name" in e:
                                pathways.append(f"{db}:{e['name']}")
                    elif isinstance(entries, dict) and "name" in entries:
                        pathways.append(f"{db}:{entries['name']}")

            generif = []
            generif_data = hit.get("generif", [])
            if isinstance(generif_data, dict):
                generif_data = [generif_data]
            for r in generif_data:
                if isinstance(r, dict) and "text" in r:
                    generif.append(r["text"])

            annotations[symbol.upper()] = GeneAnnotation(
                symbol=symbol.upper(),
                summary=hit.get("summary", ""),
                go_terms=go_terms[:20],
                generif_texts=generif[:10],
                pathways=pathways[:10],
            )

        logger.info("MyGene: annotated %d/%d genes", len(annotations), len(gene_symbols))
        return annotations

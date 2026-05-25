"""NCBI Gene source — Bio.Entrez for gene summaries and GeneRIF.

Uses Biopython's Entrez module to retrieve gene information including
summaries and Gene Reference Into Function (GeneRIF) annotations.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

ENTREZ_EMAIL = "marker_agent@example.com"


class NCBIGeneSource:
    """Query NCBI Gene via Bio.Entrez for gene summaries and GeneRIF."""

    def __init__(self, cache_dir: Optional[str] = None, email: str = ENTREZ_EMAIL):
        self.cache_dir = cache_dir
        self.email = email
        self._entrez = None

    def _get_entrez(self):
        if self._entrez is None:
            try:
                from Bio import Entrez
                Entrez.email = self.email
                self._entrez = Entrez
            except ImportError:
                logger.error("Biopython not installed. Run: pip install biopython")
                return None
        return self._entrez

    def search(
        self,
        cell_type: str,
        tissue: str,
        species: str = "Human",
        max_results: int = 30,
        **kwargs,
    ) -> List[MarkerHit]:
        """Search NCBI Gene for genes associated with a cell type."""
        Entrez = self._get_entrez()
        if Entrez is None:
            return []

        organism = "Homo sapiens" if species.lower() == "human" else species
        query = f'"{cell_type}"[Title] AND "{organism}"[Organism] AND alive[prop]'
        if tissue:
            query = f'("{cell_type}" AND "{tissue}")[Title] AND "{organism}"[Organism] AND alive[prop]'

        try:
            handle = Entrez.esearch(
                db="gene", term=query, retmax=max_results, sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
        except Exception as e:
            logger.warning("NCBI Gene search failed: %s", e)
            return []

        gene_ids = record.get("IdList", [])
        if not gene_ids:
            return []

        time.sleep(0.34)

        try:
            handle = Entrez.efetch(
                db="gene", id=",".join(gene_ids), rettype="docsum"
            )
            summaries = Entrez.read(handle)
            handle.close()
        except Exception as e:
            logger.warning("NCBI Gene fetch failed: %s", e)
            return []

        hits = []
        doc_sums = summaries.get("DocumentSummarySet", {}).get("DocumentSummary", [])
        for doc in doc_sums:
            symbol = doc.get("NomenclatureSymbol") or doc.get("Name", "")
            if not symbol:
                continue
            description = doc.get("Description", "")
            summary = doc.get("Summary", "")

            hits.append(
                MarkerHit(
                    gene_symbol=symbol.upper(),
                    cell_type=cell_type,
                    source="NCBIGene",
                    score=0.6,
                    evidence_detail=f"{description}; {summary[:100]}",
                )
            )

        logger.info("NCBIGene: found %d genes for '%s'", len(hits), cell_type)
        return hits

    def get_generifs(self, gene_symbol: str, species: str = "Human") -> List[str]:
        """Retrieve GeneRIF annotations for a gene."""
        Entrez = self._get_entrez()
        if Entrez is None:
            return []

        organism = "Homo sapiens" if species.lower() == "human" else species
        query = f'{gene_symbol}[Gene Name] AND "{organism}"[Organism]'

        try:
            handle = Entrez.esearch(db="gene", term=query, retmax=1)
            record = Entrez.read(handle)
            handle.close()
        except Exception as e:
            logger.warning("GeneRIF search failed for %s: %s", gene_symbol, e)
            return []

        gene_ids = record.get("IdList", [])
        if not gene_ids:
            return []

        time.sleep(0.34)

        try:
            handle = Entrez.efetch(db="gene", id=gene_ids[0], retmode="xml")
            records = Entrez.read(handle)
            handle.close()
        except Exception as e:
            logger.warning("GeneRIF fetch failed for %s: %s", gene_symbol, e)
            return []

        rifs = []
        for rec in records:
            comments = rec.get("Entrezgene_comments", [])
            for comment in comments:
                if comment.get("Gene-commentary_heading") == "GeneRIF":
                    for sub in comment.get("Gene-commentary_comment", []):
                        text = sub.get("Gene-commentary_text", "")
                        if text:
                            rifs.append(text)
        return rifs[:20]

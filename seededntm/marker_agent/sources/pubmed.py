"""PubMed source — Bio.Entrez + LLM extraction for marker discovery.

Searches PubMed for papers about cell type markers, then uses LLM to
extract gene names from abstracts.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

ENTREZ_EMAIL = "marker_agent@example.com"


class PubMedSource:
    """Search PubMed for cell type marker literature and extract genes via LLM."""

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
        max_papers: int = 10,
        use_llm: bool = True,
        **kwargs,
    ) -> List[MarkerHit]:
        """Search PubMed for marker gene papers and extract genes.

        If use_llm=True, uses LLM to extract gene symbols from abstracts.
        """
        Entrez = self._get_entrez()
        if Entrez is None:
            return []

        query_parts = [
            f'"{cell_type}"[Title/Abstract]',
            "marker[Title/Abstract]",
            "gene[Title/Abstract]",
        ]
        if tissue:
            query_parts.append(f'"{tissue}"[Title/Abstract]')
        if species:
            query_parts.append(f'"{species}"[Title/Abstract]')

        query = " AND ".join(query_parts)

        try:
            handle = Entrez.esearch(
                db="pubmed", term=query, retmax=max_papers, sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
        except Exception as e:
            logger.warning("PubMed search failed: %s", e)
            return []

        pmids = record.get("IdList", [])
        if not pmids:
            return []

        time.sleep(0.34)

        try:
            handle = Entrez.efetch(
                db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()
        except Exception as e:
            logger.warning("PubMed fetch failed: %s", e)
            return []

        abstracts = []
        articles = records.get("PubmedArticle", [])
        for article in articles:
            medline = article.get("MedlineCitation", {})
            art_data = medline.get("Article", {})
            abstract_parts = art_data.get("Abstract", {}).get("AbstractText", [])
            title = art_data.get("ArticleTitle", "")
            abstract_text = " ".join(str(p) for p in abstract_parts)
            if abstract_text:
                abstracts.append(f"Title: {title}\nAbstract: {abstract_text}")

        if not abstracts:
            return []

        if use_llm:
            return self._extract_with_llm(abstracts, cell_type, tissue)
        else:
            return self._extract_heuristic(abstracts, cell_type)

    def _extract_with_llm(
        self, abstracts: List[str], cell_type: str, tissue: str
    ) -> List[MarkerHit]:
        """Use LLM to extract gene symbols from abstracts."""
        from seededntm.marker_agent.llm_client import llm_extract_genes

        combined_text = "\n\n---\n\n".join(abstracts[:5])
        genes = llm_extract_genes(combined_text, cell_type)

        hits = []
        for gene in genes:
            if gene and len(gene) <= 15:
                hits.append(
                    MarkerHit(
                        gene_symbol=gene.upper(),
                        cell_type=cell_type,
                        source="PubMed_LLM",
                        score=0.7,
                        evidence_detail=f"LLM-extracted from {len(abstracts)} abstracts",
                    )
                )

        logger.info(
            "PubMed+LLM: extracted %d genes for '%s' from %d abstracts",
            len(hits), cell_type, len(abstracts),
        )
        return hits

    def _extract_heuristic(
        self, abstracts: List[str], cell_type: str
    ) -> List[MarkerHit]:
        """Simple heuristic gene extraction (uppercase words 2-10 chars)."""
        import re

        gene_pattern = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\b")
        gene_counts: dict = {}

        stopwords = {
            "THE", "AND", "FOR", "ARE", "WAS", "WITH", "THAT", "THIS",
            "FROM", "HAVE", "BEEN", "WERE", "ALSO", "THAN", "CAN", "MAY",
            "RNA", "DNA", "PCR", "WHO", "BMI", "USA", "NOT", "BUT",
        }

        for abstract in abstracts:
            matches = gene_pattern.findall(abstract)
            for m in matches:
                if m not in stopwords and len(m) >= 2:
                    gene_counts[m] = gene_counts.get(m, 0) + 1

        hits = []
        for gene, count in sorted(gene_counts.items(), key=lambda x: -x[1]):
            if count >= 2:
                hits.append(
                    MarkerHit(
                        gene_symbol=gene,
                        cell_type=cell_type,
                        source="PubMed_heuristic",
                        score=min(count * 0.2, 1.0),
                        evidence_detail=f"mentioned {count} times across abstracts",
                    )
                )

        return hits[:30]

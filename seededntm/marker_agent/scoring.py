"""ConsensusScorer — multi-source merge with tiered weights and panel filter.

Merges marker hits from all sources, applies weighted scoring, and filters
to genes present in the spatial gene panel.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

from seededntm.marker_agent.schemas import MarkerHit, ScoredMarker

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "CellMarker": 1.0,
    "PanglaoDB": 0.8,
    "ScTypeDB": 0.9,
    "HPA": 0.9,
    "PubMed_LLM": 0.6,
    "PubMed_heuristic": 0.4,
    "MyGene": 0.3,
    "NCBIGene": 0.4,
    "OMIM": 0.5,
    "DisGeNET": 0.7,
    "Ensembl": 0.4,
    "OpenTargets": 0.6,
    "COSMIC": 0.6,
    "GTEx": 0.5,
    "MSigDB": 0.7,
    "CellxGene": 0.8,
}


class ConsensusScorer:
    """Score and rank marker gene candidates from multiple sources.

    Merges hits across all sources, applies source-specific weights,
    and filters to genes present in the spatial gene panel.
    """

    def __init__(
        self,
        gene_panel: List[str],
        weights: Optional[Dict[str, float]] = None,
        top_n: int = 20,
        require_panel: bool = True,
    ):
        self.gene_panel = set(g.upper() for g in gene_panel)
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS.copy()
        self.top_n = top_n
        self.require_panel = require_panel

    def score(
        self, hits: Dict[str, List[MarkerHit]]
    ) -> Dict[str, List[ScoredMarker]]:
        """Score and rank markers for each cell type.

        Args:
            hits: Dict mapping cell_type -> list of MarkerHit from all sources

        Returns:
            Dict mapping cell_type -> sorted list of ScoredMarker
        """
        results = {}

        for cell_type, hit_list in hits.items():
            scored = self._score_cell_type(cell_type, hit_list)
            results[cell_type] = scored

        return results

    def _score_cell_type(
        self, cell_type: str, hit_list: List[MarkerHit]
    ) -> List[ScoredMarker]:
        """Score markers for a single cell type."""
        gene_sources: Dict[str, Dict[str, float]] = defaultdict(dict)
        gene_evidence: Dict[str, List[str]] = defaultdict(list)

        for hit in hit_list:
            gene = hit.gene_symbol.upper()
            source = hit.source
            weight = self.weights.get(source, 0.5)
            weighted_score = hit.score * weight

            if source in gene_sources[gene]:
                gene_sources[gene][source] = max(
                    gene_sources[gene][source], weighted_score
                )
            else:
                gene_sources[gene][source] = weighted_score

            if hit.evidence_detail:
                gene_evidence[gene].append(f"[{source}] {hit.evidence_detail}")

        scored_markers = []
        for gene, source_scores in gene_sources.items():
            in_panel = gene in self.gene_panel
            if self.require_panel and not in_panel:
                continue

            total_score = sum(source_scores.values())
            n_sources = len(source_scores)
            total_score *= (1 + 0.1 * (n_sources - 1))

            scored_markers.append(
                ScoredMarker(
                    gene_symbol=gene,
                    cell_type=cell_type,
                    total_score=round(total_score, 4),
                    source_scores=dict(source_scores),
                    in_panel=in_panel,
                )
            )

        scored_markers.sort(key=lambda m: m.total_score, reverse=True)

        top_markers = scored_markers[: self.top_n]

        logger.info(
            "Scored %d markers for '%s' (panel-filtered: %d -> %d, top %d returned)",
            len(gene_sources),
            cell_type,
            len(gene_sources),
            len(scored_markers),
            len(top_markers),
        )

        return top_markers

    def score_all_panels(
        self,
        hits: Dict[str, List[MarkerHit]],
        return_unfiltered: bool = False,
    ) -> Dict[str, List[ScoredMarker]]:
        """Score with optional unfiltered results for analysis.

        If return_unfiltered=True, also returns genes not in the panel
        (marked with in_panel=False).
        """
        if return_unfiltered:
            original_require = self.require_panel
            self.require_panel = False
            results = self.score(hits)
            self.require_panel = original_require

            for cell_type, markers in results.items():
                for m in markers:
                    m.in_panel = m.gene_symbol in self.gene_panel
            return results

        return self.score(hits)

    def get_panel_coverage(
        self, results: Dict[str, List[ScoredMarker]]
    ) -> Dict[str, float]:
        """Compute what fraction of selected markers are in the gene panel."""
        coverage = {}
        for cell_type, markers in results.items():
            if markers:
                in_panel = sum(1 for m in markers if m.in_panel)
                coverage[cell_type] = in_panel / len(markers)
            else:
                coverage[cell_type] = 0.0
        return coverage

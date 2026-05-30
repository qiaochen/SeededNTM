"""ConsensusScorer — multi-source merge with tiered weights and panel filter.

Merges marker hits from all sources, applies weighted scoring, and filters
to genes present in the spatial gene panel. Includes hit caps, cross-type
specificity penalties, and multi-source consensus bonuses.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set

from seededntm.marker_agent.schemas import MarkerHit, ScoredMarker

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "CellMarker": 1.0,
    "PanglaoDB": 0.8,
    "ScTypeDB": 0.9,
    "HPA": 0.9,
    "PubMed_LLM": 0.6,
    "PubMed_heuristic": 0.4,
    "PubTator3": 0.8,
    "LitVar2": 0.4,
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

TIER1_SOURCES = {"CellMarker", "PanglaoDB", "ScTypeDB"}

HIT_CAP_PER_SOURCE = 50

CROSS_TYPE_THRESHOLD = 0.3

CURATED_BONUS = 1.5


class ConsensusScorer:
    """Score and rank marker gene candidates from multiple sources.

    Merges hits across all sources, applies source-specific weights,
    hit caps, cross-type penalties, and curated source bonuses.
    """

    def __init__(
        self,
        gene_panel: List[str],
        weights: Optional[Dict[str, float]] = None,
        top_n: int = 20,
        require_panel: bool = True,
        hit_cap: int = HIT_CAP_PER_SOURCE,
        cross_type_penalty: float = 0.5,
        tissue_context: str = "",
    ):
        self.gene_panel = set(g.upper() for g in gene_panel)
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS.copy()
        self.top_n = top_n
        self.require_panel = require_panel
        self.hit_cap = hit_cap
        self.cross_type_penalty = cross_type_penalty
        self.tissue_context = tissue_context

    def score(
        self, hits: Dict[str, List[MarkerHit]]
    ) -> Dict[str, List[ScoredMarker]]:
        """Score and rank markers for each cell type.

        Args:
            hits: Dict mapping cell_type -> list of MarkerHit from all sources

        Returns:
            Dict mapping cell_type -> sorted list of ScoredMarker
        """
        capped_hits = self._apply_hit_caps(hits)
        cross_type_genes = self._find_cross_type_genes(capped_hits)

        results = {}
        for cell_type, hit_list in capped_hits.items():
            scored = self._score_cell_type(cell_type, hit_list, cross_type_genes)
            results[cell_type] = scored

        return results

    def _apply_hit_caps(
        self, hits: Dict[str, List[MarkerHit]]
    ) -> Dict[str, List[MarkerHit]]:
        """Cap the number of hits per source per cell type."""
        capped: Dict[str, List[MarkerHit]] = {}

        for cell_type, hit_list in hits.items():
            source_counts: Dict[str, int] = defaultdict(int)
            filtered = []
            for hit in hit_list:
                if source_counts[hit.source] < self.hit_cap:
                    filtered.append(hit)
                    source_counts[hit.source] += 1
            capped[cell_type] = filtered

        return capped

    def _find_cross_type_genes(
        self, hits: Dict[str, List[MarkerHit]]
    ) -> Set[str]:
        """Find genes appearing in too many cell types (not type-specific)."""
        gene_types: Dict[str, Set[str]] = defaultdict(set)
        for cell_type, hit_list in hits.items():
            for hit in hit_list:
                gene_types[hit.gene_symbol.upper()].add(cell_type)

        n_types = len(hits)
        threshold = max(2, int(n_types * CROSS_TYPE_THRESHOLD))
        cross_type = {
            gene for gene, types in gene_types.items()
            if len(types) >= threshold
        }

        if cross_type:
            logger.info(
                "Cross-type penalty applied to %d genes (in >=%d/%d types)",
                len(cross_type), threshold, n_types,
            )

        return cross_type

    def _score_cell_type(
        self,
        cell_type: str,
        hit_list: List[MarkerHit],
        cross_type_genes: Set[str],
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

            # Stronger multi-source consensus bonus: 1 + 0.2*(n_sources-1)
            n_sources = len(source_scores)
            total_score *= (1 + 0.2 * (n_sources - 1))

            # Curated source (Tier 1) priority bonus
            curated_sources = set(source_scores.keys()) & TIER1_SOURCES
            if curated_sources:
                total_score *= CURATED_BONUS

            # Cross-type specificity penalty
            if gene in cross_type_genes:
                total_score *= self.cross_type_penalty

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

        # Validate top candidates to remove false positives
        if self.tissue_context:
            from seededntm.marker_agent.llm_client import llm_validate_markers
            candidates = [m.gene_symbol for m in top_markers[:30]]
            if candidates:
                try:
                    validations = llm_validate_markers(candidates, cell_type, self.tissue_context)
                    top_markers = [
                        m for m in top_markers
                        if validations.get(m.gene_symbol.upper(), "likely")
                           not in ("unlikely", "housekeeping")
                    ]
                except Exception as e:
                    logger.warning("LLM validation failed for %s: %s", cell_type, e)

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

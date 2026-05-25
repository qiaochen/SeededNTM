"""HybridCombiner — merge Leiden-DE seeds with Marker Agent seeds.

Fuses data-driven (Leiden clustering DE) seeds with literature-driven
(Marker Agent database queries) seeds using weighted scoring and consensus.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class HybridCombiner:
    """Combine Leiden-DE seeds with Marker Agent seeds via weighted fusion.

    Strategy:
      1. Match cell types between Leiden and Agent outputs (fuzzy name matching)
      2. Score fusion: combined = w_leiden * leiden_score + w_agent * agent_score
      3. Consensus bonus: genes in both get 1.5x multiplier
      4. Agent-only genes: included if high confidence
      5. Panel filter: final genes must be in the spatial gene panel

    Args:
        w_leiden: Weight for Leiden-DE evidence (data-driven).
        w_agent: Weight for Marker Agent evidence (literature-driven).
        consensus_bonus: Multiplier for genes found in both sources.
        agent_only_threshold: Min agent score to include agent-only genes.
        top_n: Maximum number of seed genes per cell type.
        gene_panel: Optional set of genes to filter against.
    """

    def __init__(
        self,
        w_leiden: float = 0.6,
        w_agent: float = 0.4,
        consensus_bonus: float = 1.5,
        agent_only_threshold: float = 0.5,
        top_n: int = 20,
        gene_panel: Optional[List[str]] = None,
    ):
        self.w_leiden = w_leiden
        self.w_agent = w_agent
        self.consensus_bonus = consensus_bonus
        self.agent_only_threshold = agent_only_threshold
        self.top_n = top_n
        self.gene_panel: Optional[Set[str]] = (
            set(g.upper() for g in gene_panel) if gene_panel else None
        )

    def combine(
        self,
        leiden_seeds: Dict[str, Dict[str, Any]],
        agent_seeds: Dict[str, List[str]],
        agent_scores: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Merge Leiden-DE seeds with Marker Agent seeds.

        Args:
            leiden_seeds: Output from SeedConstructionPipeline.run().
                Format: {cell_type: {"features": [...], "topic_index": N}}
            agent_seeds: Output from MarkerSearchAgent.to_seed_json().
                Format: {cell_type: [gene_list]}
            agent_scores: Optional per-gene scores from agent.
                Format: {cell_type: {gene: score}}

        Returns:
            Merged seeds in the same format as leiden_seeds:
            {cell_type: {"features": [...], "topic_index": N, "source": "hybrid"}}
        """
        type_mapping = self._match_cell_types(
            list(leiden_seeds.keys()), list(agent_seeds.keys())
        )

        merged: Dict[str, Dict[str, Any]] = {}
        topic_idx = 0

        for leiden_type, agent_type in type_mapping.items():
            leiden_info = leiden_seeds[leiden_type]
            leiden_genes = leiden_info.get("features", [])

            agent_genes = agent_seeds.get(agent_type, []) if agent_type else []
            gene_agent_scores = (
                agent_scores.get(agent_type, {}) if agent_scores and agent_type else {}
            )

            fused_genes = self._fuse_gene_lists(
                leiden_genes, agent_genes, gene_agent_scores
            )

            merged[leiden_type] = {
                "features": fused_genes,
                "topic_index": topic_idx,
                "source": "hybrid",
                "leiden_type": leiden_type,
                "agent_type": agent_type,
            }
            topic_idx += 1

        for agent_type in agent_seeds:
            if agent_type not in type_mapping.values():
                agent_genes = agent_seeds[agent_type]
                gene_scores = (
                    agent_scores.get(agent_type, {}) if agent_scores else {}
                )
                filtered = self._filter_by_panel(agent_genes)
                if filtered:
                    merged[agent_type] = {
                        "features": filtered[: self.top_n],
                        "topic_index": topic_idx,
                        "source": "agent_only",
                    }
                    topic_idx += 1

        logger.info("HybridCombiner: %d merged cell types", len(merged))
        return merged

    def _match_cell_types(
        self,
        leiden_types: List[str],
        agent_types: List[str],
    ) -> Dict[str, Optional[str]]:
        """Match cell types between Leiden and Agent outputs using fuzzy matching.

        Returns dict: leiden_type -> best_matching_agent_type (or None).
        """
        mapping: Dict[str, Optional[str]] = {}
        used_agent: Set[str] = set()

        for lt in leiden_types:
            best_match = None
            best_score = 0.0

            for at in agent_types:
                if at in used_agent:
                    continue
                score = self._type_similarity(lt, at)
                if score > best_score:
                    best_score = score
                    best_match = at

            if best_score >= 0.5:
                mapping[lt] = best_match
                if best_match:
                    used_agent.add(best_match)
            else:
                mapping[lt] = None

        return mapping

    @staticmethod
    def _type_similarity(name1: str, name2: str) -> float:
        """Compute similarity between two cell type names."""
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()

        if n1 == n2:
            return 1.0

        if n1 in n2 or n2 in n1:
            return 0.85

        return SequenceMatcher(None, n1, n2).ratio()

    def _fuse_gene_lists(
        self,
        leiden_genes: List[str],
        agent_genes: List[str],
        agent_gene_scores: Dict[str, float],
    ) -> List[str]:
        """Fuse two gene lists with weighted scoring and consensus bonus."""
        gene_scores: Dict[str, float] = {}

        n_leiden = len(leiden_genes)
        for rank, gene in enumerate(leiden_genes):
            gene_upper = gene.upper()
            leiden_score = 1.0 - (rank / max(n_leiden, 1))
            gene_scores[gene_upper] = self.w_leiden * leiden_score

        agent_genes_upper = set(g.upper() for g in agent_genes)
        leiden_genes_upper = set(g.upper() for g in leiden_genes)

        n_agent = len(agent_genes)
        for rank, gene in enumerate(agent_genes):
            gene_upper = gene.upper()
            raw_score = agent_gene_scores.get(gene, agent_gene_scores.get(gene_upper, 0))
            if raw_score == 0:
                raw_score = 1.0 - (rank / max(n_agent, 1))

            if gene_upper in gene_scores:
                gene_scores[gene_upper] += self.w_agent * raw_score
                gene_scores[gene_upper] *= self.consensus_bonus
            else:
                if raw_score >= self.agent_only_threshold or gene_upper in leiden_genes_upper:
                    gene_scores[gene_upper] = self.w_agent * raw_score

        filtered_genes = self._filter_by_panel(list(gene_scores.keys()))
        filtered_set = set(g.upper() for g in filtered_genes)

        scored_pairs = [
            (gene, score) for gene, score in gene_scores.items()
            if gene in filtered_set
        ]
        scored_pairs.sort(key=lambda x: -x[1])

        return [gene for gene, _ in scored_pairs[: self.top_n]]

    def _filter_by_panel(self, genes: List[str]) -> List[str]:
        """Filter genes by panel membership if panel is set."""
        if self.gene_panel is None:
            return genes
        return [g for g in genes if g.upper() in self.gene_panel]

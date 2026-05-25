"""SeedConstructionPipeline — automated Leiden + DE + LLM annotation workflow.

Replaces the semi-manual Leiden-ChatGPT pipeline with a fully automated
approach: cluster spatial data, compute DE markers, annotate clusters via
Azure GPT-4o, and output seed gene sets per cell type.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SeedConstructionPipeline:
    """Automated Leiden + DE + LLM annotation pipeline.

    Given a spatial transcriptomics AnnData object, runs:
      1. Leiden clustering (auto-tuned resolution)
      2. Wilcoxon DE analysis per cluster
      3. LLM-based cluster annotation
      4. Merge clusters with same cell type
      5. Output seed genes per annotated type

    Args:
        adata: AnnData object with spatial transcriptomics data.
        tissue_description: Free-text description of the tissue context.
        resolution: Starting Leiden resolution (auto-tuned if None).
        min_clusters: Minimum acceptable number of clusters.
        max_clusters: Maximum acceptable number of clusters.
        top_n_genes: Number of top DE genes per cluster/type for seeds.
        logfc_threshold: Minimum log fold-change for DE genes.
        pval_threshold: Maximum adjusted p-value for DE genes.
    """

    def __init__(
        self,
        adata: Any,
        tissue_description: str,
        resolution: Optional[float] = None,
        min_clusters: int = 5,
        max_clusters: int = 20,
        top_n_genes: int = 20,
        logfc_threshold: float = 0.5,
        pval_threshold: float = 0.01,
    ):
        self.adata = adata
        self.tissue_description = tissue_description
        self.resolution = resolution
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.top_n_genes = top_n_genes
        self.logfc_threshold = logfc_threshold
        self.pval_threshold = pval_threshold

        self._leiden_key = "leiden_seed_pipeline"
        self._de_markers: Dict[str, List[str]] = {}
        self._annotations: Dict[str, str] = {}
        self._llm_response: str = ""
        self._final_resolution: float = 1.0

    def run(self) -> Dict[str, Dict[str, Any]]:
        """Execute the full pipeline.

        Returns:
            Dict mapping cell_type -> {"features": [gene_list], "topic_index": int}
        """
        logger.info("Starting SeedConstructionPipeline")
        logger.info("Tissue: %s", self.tissue_description)
        logger.info("Data shape: %s", self.adata.shape)

        self._run_leiden()
        self._compute_de_markers()
        self._annotations = self._annotate_clusters_with_llm()
        seeds = self._merge_and_build_seeds(self._annotations)

        logger.info("Pipeline complete: %d cell types identified", len(seeds))
        for ct, info in seeds.items():
            logger.info("  %s: %d seed genes", ct, len(info["features"]))

        return seeds

    def _run_leiden(self) -> None:
        """Run Leiden clustering with auto-tuned resolution."""
        import scanpy as sc

        if "neighbors" not in self.adata.uns:
            logger.info("Computing neighbors graph...")
            if "X_pca" not in self.adata.obsm:
                sc.tl.pca(self.adata, n_comps=min(50, self.adata.shape[1] - 1))
            sc.pp.neighbors(self.adata, n_neighbors=15, n_pcs=30)

        if self.resolution is not None:
            sc.tl.leiden(self.adata, resolution=self.resolution, key_added=self._leiden_key)
            self._final_resolution = self.resolution
            n_clusters = self.adata.obs[self._leiden_key].nunique()
            logger.info(
                "Leiden (fixed res=%.2f): %d clusters", self.resolution, n_clusters
            )
            return

        best_res = 1.0
        best_n = 0
        for res in [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]:
            sc.tl.leiden(self.adata, resolution=res, key_added=self._leiden_key)
            n = self.adata.obs[self._leiden_key].nunique()
            logger.debug("Resolution %.2f -> %d clusters", res, n)
            if self.min_clusters <= n <= self.max_clusters:
                best_res = res
                best_n = n
                break
            if n < self.min_clusters:
                continue
            if n > self.max_clusters:
                best_res = res
                best_n = n
                break
            best_res = res
            best_n = n

        sc.tl.leiden(self.adata, resolution=best_res, key_added=self._leiden_key)
        self._final_resolution = best_res
        n_clusters = self.adata.obs[self._leiden_key].nunique()
        logger.info(
            "Leiden (auto-tuned res=%.2f): %d clusters", best_res, n_clusters
        )

    def _compute_de_markers(self) -> None:
        """Compute Wilcoxon DE markers per cluster."""
        import scanpy as sc

        sc.tl.rank_genes_groups(
            self.adata,
            groupby=self._leiden_key,
            method="wilcoxon",
            key_added="rank_genes_seed_pipeline",
        )

        de_result = self.adata.uns["rank_genes_seed_pipeline"]
        clusters = list(de_result["names"].dtype.names)

        self._de_markers = {}
        for cluster in clusters:
            names = de_result["names"][cluster]
            logfcs = de_result["logfoldchanges"][cluster]
            pvals_adj = de_result["pvals_adj"][cluster]

            filtered_genes = []
            for gene, lfc, pval in zip(names, logfcs, pvals_adj):
                if lfc > self.logfc_threshold and pval < self.pval_threshold:
                    filtered_genes.append(gene)
                if len(filtered_genes) >= self.top_n_genes:
                    break

            self._de_markers[cluster] = filtered_genes
            logger.debug(
                "Cluster %s: %d DE genes (filtered)", cluster, len(filtered_genes)
            )

    def _annotate_clusters_with_llm(self) -> Dict[str, str]:
        """Use LLM to annotate clusters with cell type identities."""
        from seededntm.prompt_templates import build_annotation_prompt
        from seededntm.marker_agent.llm_client import llm_complete

        prompt = build_annotation_prompt(
            markers_per_cluster=self._de_markers,
            tissue_description=self.tissue_description,
            top_n=self.top_n_genes,
        )

        system = (
            "You are a single-cell genomics expert with deep knowledge of cell type "
            "markers across tissues. Provide accurate, evidence-based annotations."
        )

        logger.info("Calling LLM for cluster annotation (%d clusters)...",
                    len(self._de_markers))

        response = llm_complete(prompt, system=system, temperature=0.1, max_tokens=2048)
        self._llm_response = response

        annotations = self._parse_annotation_response(response)
        logger.info("LLM annotations: %s", annotations)

        return annotations

    def _parse_annotation_response(self, response: str) -> Dict[str, str]:
        """Robustly parse LLM JSON response for cluster annotations."""
        text = response.strip()

        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
            text = text.strip()

        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        try:
            parsed = json.loads(text)
            return {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass

        annotations = {}
        pattern = r'"(\d+)":\s*"([^"]+)"'
        for match in re.finditer(pattern, response):
            annotations[match.group(1)] = match.group(2)

        if annotations:
            return annotations

        logger.error("Failed to parse LLM annotation response")
        return {str(i): f"Unknown_{i}" for i in range(len(self._de_markers))}

    def _merge_and_build_seeds(
        self, annotations: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """Merge clusters with same annotation and build final seed dict."""
        type_to_clusters: Dict[str, List[str]] = defaultdict(list)
        for cluster_id, cell_type in annotations.items():
            normalized = cell_type.strip()
            type_to_clusters[normalized].append(cluster_id)

        seeds: Dict[str, Dict[str, Any]] = {}
        for topic_idx, (cell_type, cluster_ids) in enumerate(
            sorted(type_to_clusters.items())
        ):
            all_genes: List[str] = []
            for cid in cluster_ids:
                genes = self._de_markers.get(cid, [])
                all_genes.extend(genes)

            seen = set()
            unique_genes = []
            for g in all_genes:
                g_upper = g.upper()
                if g_upper not in seen:
                    seen.add(g_upper)
                    unique_genes.append(g)

            unique_genes = unique_genes[: self.top_n_genes]

            seeds[cell_type] = {
                "features": unique_genes,
                "topic_index": topic_idx,
            }

        return seeds

    def get_provenance(self) -> Dict[str, Any]:
        """Return provenance metadata for the pipeline run."""
        return {
            "resolution": self._final_resolution,
            "n_clusters": len(self._de_markers),
            "logfc_threshold": self.logfc_threshold,
            "pval_threshold": self.pval_threshold,
            "top_n_genes": self.top_n_genes,
            "tissue_description": self.tissue_description,
            "de_markers": self._de_markers,
            "annotations": self._annotations,
            "llm_response_raw": self._llm_response,
        }

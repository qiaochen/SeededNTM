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
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_HOUSEKEEPING_GENES: Set[str] = {
    "ACTB", "GAPDH", "MALAT1", "NEAT1", "B2M", "TMSB4X", "ACTG1",
}

_RIBOSOMAL_PATTERN = re.compile(r"^RP[LS]\d+", re.IGNORECASE)
_MITO_PATTERN = re.compile(r"^MT-", re.IGNORECASE)


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
        expected_cell_types: Optional list of expected cell type names for guided
            annotation. When provided, the LLM is constrained to these names and
            missing types are filled via MarkerSearchAgent fallback.
        species: Species for marker agent fallback queries.
        organ: Organ for marker agent fallback queries.
        condition: Disease condition for marker agent fallback queries.
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
        expected_cell_types: Optional[List[str]] = None,
        species: str = "Human",
        organ: str = "",
        condition: str = "",
    ):
        self.adata = adata
        self.tissue_description = tissue_description
        self.resolution = resolution
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.top_n_genes = top_n_genes
        self.logfc_threshold = logfc_threshold
        self.pval_threshold = pval_threshold
        self.expected_cell_types = expected_cell_types
        self.species = species
        self.organ = organ
        self.condition = condition

        if expected_cell_types and resolution is None:
            n_features = adata.shape[1] if hasattr(adata, 'shape') else 5000
            if n_features < 500:
                self.min_clusters = max(len(expected_cell_types) + 2, min_clusters)
                self.max_clusters = max(3 * len(expected_cell_types), max_clusters)
            else:
                self.min_clusters = max(2 * len(expected_cell_types), min_clusters)

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

        # Verify annotations against canonical markers; re-query LLM on failures
        self._annotations = self._verify_annotations(self._annotations)

        if self.expected_cell_types:
            self._annotations = self._refine_over_merged_clusters(self._annotations)

        seeds = self._merge_and_build_seeds(self._annotations)

        logger.info("Pipeline complete: %d cell types identified", len(seeds))
        for ct, info in seeds.items():
            logger.info("  %s: %d seed genes", ct, len(info["features"]))

        return seeds

    def _run_leiden(self) -> None:
        """Run Leiden clustering with auto-tuned resolution."""
        import scanpy as sc
        from scipy.sparse import issparse

        if "log1p" not in self.adata.uns:
            logger.info("Normalizing and log-transforming data...")
            if issparse(self.adata.X):
                row_sums = np.asarray(self.adata.X.sum(axis=1)).ravel()
            else:
                row_sums = self.adata.X.sum(axis=1)
            if row_sums.max() > 100:
                sc.pp.normalize_total(self.adata, target_sum=1e4)
                sc.pp.log1p(self.adata)

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

        effective_max = self.max_clusters
        if self.expected_cell_types:
            effective_max = min(self.max_clusters, len(self.expected_cell_types) + 3)

        best_res = 1.0
        best_n = 0
        for res in [0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]:
            sc.tl.leiden(self.adata, resolution=res, key_added=self._leiden_key)
            n = self.adata.obs[self._leiden_key].nunique()
            logger.debug("Resolution %.2f -> %d clusters", res, n)
            if self.min_clusters <= n <= effective_max:
                best_res = res
                best_n = n
                break
            if n < self.min_clusters:
                continue
            if n > effective_max:
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

    def _filter_housekeeping(self, genes: List[str]) -> List[str]:
        """Remove ribosomal, mitochondrial, and common housekeeping genes."""
        filtered = []
        for g in genes:
            if _RIBOSOMAL_PATTERN.match(g):
                continue
            if _MITO_PATTERN.match(g):
                continue
            if g.upper() in _HOUSEKEEPING_GENES:
                continue
            filtered.append(g)
        return filtered

    def _compute_de_markers(self) -> None:
        """Compute Wilcoxon DE markers per cluster (for LLM annotation)."""
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
                    if (_RIBOSOMAL_PATTERN.match(gene)
                            or _MITO_PATTERN.match(gene)
                            or gene.upper() in _HOUSEKEEPING_GENES):
                        continue
                    filtered_genes.append(gene)
                if len(filtered_genes) >= self.top_n_genes:
                    break

            self._de_markers[cluster] = filtered_genes
            logger.debug(
                "Cluster %s: %d DE genes (wilcoxon)", cluster, len(filtered_genes)
            )

    def _annotate_clusters_with_llm(self) -> Dict[str, str]:
        """Use LLM to annotate clusters with cell type identities."""
        from seededntm.prompt_templates import build_annotation_prompt
        from seededntm.marker_agent.llm_client import llm_complete

        prompt = build_annotation_prompt(
            markers_per_cluster=self._de_markers,
            tissue_description=self.tissue_description,
            top_n=self.top_n_genes,
            expected_cell_types=self.expected_cell_types,
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

    # Only verify annotations that are known to cause catastrophic failures.
    # T-cell check: cluster 5 in NPC has CCL19/CCL21/PTGDS at top but TRBC2 at pos 4,
    # causing LLM to mislabel it "T" when it's actually a myeloid neighborhood.
    # B-cell check: cluster 7 in NPC has PTGDS/CCL21/CST3 at top but IGHG1 deeper,
    # causing LLM to label it "B" when it should be "Myeloid".
    _VERIFICATION_MARKERS: Dict[str, Tuple[Set[str], int]] = {
        "T": ({"CD3D", "CD3E", "CD3G", "TRAC", "TRBC1", "TRBC2", "CD4", "CD8A",
                "CD8B", "GZMA", "GZMB", "NKG7", "PRF1", "GZMK"}, 3),
        "B": ({"CD79A", "CD79B", "MS4A1", "CD19", "IGKC", "IGHG1", "IGHG2",
                "IGHG3", "IGHG4", "IGHA1", "IGHA2", "IGLC1", "IGLC2", "IGLC3",
                "IGHM", "MZB1", "JCHAIN"}, 3),
    }

    _CURATED_IMMUNE_MARKERS: Dict[str, List[str]] = {
        "T": ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4", "TRAC", "TRBC1",
              "TRBC2", "GZMA", "GZMB", "GZMK", "PRF1", "NKG7", "IFNG", "CD27",
              "CD69", "IL7R", "CCR7", "CD7", "SELL", "CD28"],
        "Treg": ["FOXP3", "IL2RA", "CTLA4", "ICOS", "TIGIT", "ENTPD1", "IL10",
                 "TNFRSF4", "TNFRSF18", "BATF", "CCR6", "CCR4", "IKZF2", "LAG3"],
        "B": ["CD79A", "CD79B", "MS4A1", "CD19", "CD22", "BANK1", "PAX5",
              "IGKC", "IGHM", "IGHG1", "IGLC1", "MZB1", "JCHAIN", "XBP1"],
        "Myeloid": ["CD68", "CD14", "CD163", "CSF1R", "ITGAM", "C1QA", "C1QB",
                    "AIF1", "LYZ", "FCER1G", "TYROBP", "FCGR3A"],
    }

    def _match_verification_type(self, type_name: str) -> Optional[str]:
        """Match a type name to a verification key, with specific substring rules.

        Only returns a match for types where mislabeling is a known failure mode:
        - "T" matches standalone "T" or types containing "T_Cell" but NOT "Mast", "Stromal"
        - "B" matches standalone "B" or "B_Cells" but NOT "fibroblast", "Endothelial"
        """
        lower = type_name.lower().strip()
        # T-cell check: "T", "CD4+_T_Cells", "CD8+_T_Cells" but not Treg, Mast, Stromal
        if lower == "t" or "t_cell" in lower or lower.startswith("cd4+") or lower.startswith("cd8+"):
            if "treg" not in lower and "tumor" not in lower:
                return "T"
        # B-cell check: "B", "B_Cells" but not fibroblast, endothelial
        if lower == "b" or lower == "b_cells" or lower.startswith("b_"):
            return "B"
        return None

    def _match_curated_type(self, type_name: str) -> Optional[str]:
        """Match a type name to a key in _CURATED_IMMUNE_MARKERS."""
        lower = type_name.lower().strip()
        for key in self._CURATED_IMMUNE_MARKERS:
            if key.lower() == lower:
                return key
        if lower == "t" or "t_cell" in lower or lower.startswith("cd4+") or lower.startswith("cd8+"):
            if "treg" not in lower:
                return "T"
        if "treg" in lower or "regulatory" in lower:
            return "Treg"
        if lower == "b" or lower == "b_cells" or lower.startswith("b_"):
            return "B"
        if "myeloid" in lower or "macrophage" in lower or "monocyte" in lower or "dendritic" in lower:
            return "Myeloid"
        return None

    def _verify_annotations(self, annotations: Dict[str, str]) -> Dict[str, str]:
        """Cross-check annotations against top DE markers; re-query LLM on failures.

        For each annotated cluster, verifies that the top-N DE genes contain at
        least one canonical marker for the assigned type. If not, re-queries
        the LLM with explicit correction guidance.
        """
        from seededntm.marker_agent.llm_client import llm_complete

        if not self.expected_cell_types:
            return annotations

        verified = dict(annotations)
        failures: List[Tuple[str, str, List[str]]] = []  # (cluster_id, assigned_type, top_genes)

        for cluster_id, assigned_type in annotations.items():
            normalized = self._normalize_type_name(assigned_type.strip())
            if normalized.startswith("Unknown"):
                continue

            # Only verify T-cell and B-cell annotations (known catastrophic failure modes)
            base_type = self._match_verification_type(normalized)

            if base_type is None:
                continue

            marker_set, check_depth = self._VERIFICATION_MARKERS[base_type]
            top_genes = self._de_markers.get(cluster_id, [])[:check_depth]
            top_genes_upper = {g.upper() for g in top_genes}

            if not top_genes_upper.intersection(marker_set):
                failures.append((cluster_id, normalized, self._de_markers.get(cluster_id, [])[:10]))

        if not failures:
            logger.info("Annotation verification: all %d annotations passed", len(annotations))
            return verified

        logger.info("Annotation verification: %d clusters failed consistency check", len(failures))

        # Re-query LLM for failed clusters
        for cluster_id, wrong_type, top_genes in failures:
            genes_str = ", ".join(top_genes)
            base_type = self._match_verification_type(wrong_type) or wrong_type

            types_str = ", ".join(self.expected_cell_types)
            correction_prompt = f"""A Leiden cluster in {self.tissue_description} was annotated as "{wrong_type}" but this assignment failed biological consistency checking.

The cluster's TOP differentially expressed markers (by log fold change) are:
{genes_str}

These top markers do NOT match canonical {base_type} markers. In spatial transcriptomics, remember that clusters represent tissue NEIGHBORHOODS:
- CCL19, CCL21, SELENOP, PTGDS, CXCL12, NBL1, IGFBP5 suggest a myeloid/immune neighborhood (fibroblastic reticular cells supporting immune cells)
- IGHG*, IGLC*, IGKC suggest plasma cells / B cell zone
- COL1A1, COL3A1, FN1, SPARC, DCN suggest fibroblast/stromal
- CCL20, EPCAM, KRT* suggest epithelial/tumor

Given the expected cell types in this tissue: {types_str}

What is the most likely cell type for this cluster based on its TOP markers? Return ONLY the exact type name from the expected list, or "Unknown" if uncertain."""

            system = "You are a spatial transcriptomics expert. Assign the correct cell type based on the dominant (top) markers."
            try:
                response = llm_complete(correction_prompt, system=system, temperature=0.0, max_tokens=128)
                new_type = response.strip().strip('"').strip("'")
                # Remove any surrounding text
                for et in self.expected_cell_types:
                    if et.lower() == new_type.lower():
                        new_type = et
                        break
                new_type_normalized = self._normalize_type_name(new_type)
                if new_type_normalized != wrong_type:
                    logger.info("  Cluster %s: corrected %s -> %s (top DE: %s)",
                                cluster_id, wrong_type, new_type_normalized, ", ".join(top_genes[:5]))
                    verified[cluster_id] = new_type_normalized
                else:
                    logger.info("  Cluster %s: LLM confirmed %s despite marker mismatch", cluster_id, wrong_type)
            except Exception as e:
                logger.warning("  Cluster %s: re-annotation failed (%s), keeping %s", cluster_id, e, wrong_type)

        return verified

    def _refine_over_merged_clusters(self, annotations: Dict[str, str]) -> Dict[str, str]:
        """Refine annotations when 4+ clusters map to the same type.

        Sends a sub-annotation prompt asking if any over-merged clusters can
        be distinguished as specific subtypes from the unmatched expected types.
        """
        from seededntm.prompt_templates import build_subtype_prompt
        from seededntm.marker_agent.llm_client import llm_complete

        type_to_clusters: Dict[str, List[str]] = defaultdict(list)
        for cluster_id, cell_type in annotations.items():
            normalized = self._normalize_type_name(cell_type.strip())
            type_to_clusters[normalized].append(cluster_id)

        assigned_types = set(type_to_clusters.keys())
        unmatched_types = [
            ct for ct in (self.expected_cell_types or [])
            if ct not in assigned_types and not ct.startswith("Unknown")
        ]

        if not unmatched_types:
            return annotations

        over_merged = {
            ct: clusters for ct, clusters in type_to_clusters.items()
            if len(clusters) >= 4 and not ct.startswith("Unknown")
        }

        if not over_merged:
            return annotations

        logger.info("Over-merged types (>=4 clusters): %s", {k: len(v) for k, v in over_merged.items()})
        logger.info("Unmatched expected types: %s", unmatched_types)

        updated_annotations = dict(annotations)

        for parent_type, cluster_ids in over_merged.items():
            cluster_markers = {
                cid: self._de_markers.get(cid, [])
                for cid in cluster_ids
            }

            prompt = build_subtype_prompt(
                parent_type=parent_type,
                cluster_markers=cluster_markers,
                unmatched_types=unmatched_types,
                tissue_context=self.tissue_description,
            )
            system = (
                "You are a single-cell genomics expert. Be conservative: only re-annotate "
                "a cluster if the markers clearly indicate a specific subtype. When in doubt, "
                "leave the original annotation."
            )

            try:
                response = llm_complete(prompt, system=system, temperature=0.1, max_tokens=1024)
                text = response.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                    text = text.rsplit("```", 1)[0].strip()

                json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
                if json_match:
                    text = json_match.group(0)

                refinements = json.loads(text)

                if refinements and isinstance(refinements, dict):
                    for cid, new_type in refinements.items():
                        normalized_new = self._normalize_type_name(str(new_type))
                        if cid in updated_annotations and normalized_new in unmatched_types:
                            logger.info("  Cluster %s: %s -> %s", cid, parent_type, normalized_new)
                            updated_annotations[cid] = normalized_new
                            unmatched_types.remove(normalized_new)

            except Exception as e:
                logger.warning("  Sub-annotation for %s failed (%s), keeping original", parent_type, e)

        return updated_annotations

    def _rerun_de_on_merged(self, type_to_clusters: Dict[str, List[str]]) -> Optional[Dict[str, List[str]]]:
        """Re-run logreg DE on TF-IDF for merged cell type groups.

        Returns dict of cell_type -> gene_list, or None if fallback needed.
        """
        import scanpy as sc

        merged_key = "merged_cell_types"
        labels = self.adata.obs[self._leiden_key].astype(str).copy()
        for cell_type, cluster_ids in type_to_clusters.items():
            mask = labels.isin(cluster_ids)
            labels[mask] = cell_type
        self.adata.obs[merged_key] = labels.astype("category")

        known_types = [ct for ct in type_to_clusters.keys() if not ct.startswith("Unknown")]
        mask_known = self.adata.obs[merged_key].isin(known_types)

        if mask_known.sum() < 50:
            logger.warning("Too few spots in known types for DE (%d), falling back to concat", mask_known.sum())
            return None

        adata_sub = self.adata[mask_known].copy()

        n_genes = adata_sub.shape[1]
        use_logreg = n_genes >= 1000

        if use_logreg:
            from seededntm.util import compute_tfidf_rep
            original_X = adata_sub.X.copy()
            tfidf_X = compute_tfidf_rep(adata_sub.X, n_pcs=None)
            adata_sub.X = tfidf_X.tocsr() if hasattr(tfidf_X, 'tocsr') else tfidf_X

            sc.tl.rank_genes_groups(
                adata_sub, groupby=merged_key, method="logreg",
                key_added="rank_genes_merged",
                use_raw=False,
                max_iter=800,
            )
            adata_sub.X = original_X
            logger.info("Merged DE: logreg on TF-IDF (%d genes)", n_genes)
        else:
            sc.tl.rank_genes_groups(
                adata_sub, groupby=merged_key, method="wilcoxon",
                key_added="rank_genes_merged",
            )
            logger.info("Merged DE: wilcoxon (%d genes, panel too small for logreg)", n_genes)

        de_result = adata_sub.uns["rank_genes_merged"]
        merged_markers = {}
        for cell_type in known_types:
            if cell_type not in de_result["names"].dtype.names:
                continue
            names = de_result["names"][cell_type]

            if use_logreg:
                filtered = []
                for gene in names:
                    if (_RIBOSOMAL_PATTERN.match(gene)
                            or _MITO_PATTERN.match(gene)
                            or gene.upper() in _HOUSEKEEPING_GENES):
                        continue
                    filtered.append(gene)
                    if len(filtered) >= self.top_n_genes:
                        break
            else:
                logfcs = de_result["logfoldchanges"][cell_type]
                pvals_adj = de_result["pvals_adj"][cell_type]
                filtered = []
                for gene, lfc, pval in zip(names, logfcs, pvals_adj):
                    if lfc > self.logfc_threshold and pval < self.pval_threshold:
                        if (_RIBOSOMAL_PATTERN.match(gene)
                                or _MITO_PATTERN.match(gene)
                                or gene.upper() in _HOUSEKEEPING_GENES):
                            continue
                        filtered.append(gene)
                    if len(filtered) >= self.top_n_genes:
                        break
            merged_markers[cell_type] = filtered

        return merged_markers

    def _guided_type_recovery(self, missing_types: List[str]) -> Dict[str, List[str]]:
        """Recover missing types using data-driven guided scoring.

        For each missing type:
        1. Get a small set of signature genes (from canonical markers intersected with panel)
        2. Score each spot by the mean expression of these signature genes
        3. Select top 5% of spots as the "enriched" subpopulation
        4. Run DE (enriched vs rest) to get tissue-specific, panel-constrained markers
        """
        import scanpy as sc
        from scipy.sparse import issparse as sp_issparse

        from seededntm.marker_agent.agent import MarkerSearchAgent
        from seededntm.marker_agent.schemas import DatasetContext

        gene_panel = list(self.adata.var_names)
        gene_panel_upper = {g.upper(): g for g in gene_panel}

        # First get a few signature genes per type from canonical knowledge
        ctx = DatasetContext(
            species=self.species,
            tissue=self.tissue_description,
            organ=self.organ,
            condition=self.condition,
            gene_panel=gene_panel,
            expected_cell_types=missing_types,
            n_markers_per_type=10,
            include_disease_genes=bool(self.condition),
        )

        try:
            agent = MarkerSearchAgent(context=ctx, parallel=True, max_workers=4)
            results = agent.run()
        except Exception as e:
            logger.warning("Guided recovery: marker agent failed (%s), falling back to canonical", e)
            return {}

        recovered: Dict[str, List[str]] = {}

        for cell_type in missing_types:
            markers = results.get(cell_type, [])
            # Find signature genes that are in the panel AND expressed
            signature_genes = []
            for m in markers:
                gene_name = gene_panel_upper.get(m.gene_symbol.upper())
                if gene_name:
                    signature_genes.append(gene_name)
                if len(signature_genes) >= 5:
                    break

            if len(signature_genes) < 2:
                logger.info("  %s: too few signature genes in panel (%d), skipping guided recovery",
                            cell_type, len(signature_genes))
                continue

            # Score each spot by mean expression of signature genes
            gene_indices = [
                i for i, g in enumerate(gene_panel)
                if g in signature_genes
            ]

            X = self.adata.X
            if sp_issparse(X):
                scores = np.asarray(X[:, gene_indices].mean(axis=1)).ravel()
            else:
                scores = X[:, gene_indices].mean(axis=1).ravel()

            # Top 5% spots are "enriched"
            threshold = np.percentile(scores, 95)
            enriched_mask = scores >= threshold

            n_enriched = enriched_mask.sum()
            if n_enriched < 20:
                logger.info("  %s: too few enriched spots (%d), skipping guided recovery",
                            cell_type, n_enriched)
                continue

            # Run DE: enriched vs rest
            guided_key = f"_guided_{cell_type}"
            self.adata.obs[guided_key] = "rest"
            self.adata.obs.loc[enriched_mask, guided_key] = "enriched"
            self.adata.obs[guided_key] = self.adata.obs[guided_key].astype("category")

            try:
                sc.tl.rank_genes_groups(
                    self.adata, groupby=guided_key, method="wilcoxon",
                    key_added=f"rank_genes_{guided_key}",
                    groups=["enriched"], reference="rest",
                )
                de_result = self.adata.uns[f"rank_genes_{guided_key}"]
                names = de_result["names"]["enriched"]
                logfcs = de_result["logfoldchanges"]["enriched"]
                pvals_adj = de_result["pvals_adj"]["enriched"]

                de_genes = []
                for gene, lfc, pval in zip(names, logfcs, pvals_adj):
                    if lfc > 0.3 and pval < 0.05:
                        if (_RIBOSOMAL_PATTERN.match(gene)
                                or _MITO_PATTERN.match(gene)
                                or gene.upper() in _HOUSEKEEPING_GENES):
                            continue
                        de_genes.append(gene)
                    if len(de_genes) >= self.top_n_genes:
                        break

                # Combine: signature genes first, then DE genes
                seen = set()
                combined = []
                for g in signature_genes + de_genes:
                    if g.upper() not in seen:
                        seen.add(g.upper())
                        combined.append(g)

                if len(combined) >= 5:
                    recovered[cell_type] = combined[:self.top_n_genes]
                    logger.info("  %s: guided recovery found %d genes (sig=%d, DE=%d, enriched_spots=%d)",
                                cell_type, len(recovered[cell_type]), len(signature_genes),
                                len(de_genes), n_enriched)
                else:
                    logger.info("  %s: guided recovery too few genes (%d), skipping", cell_type, len(combined))

            except Exception as e:
                logger.warning("  %s: guided DE failed (%s)", cell_type, e)
            finally:
                if guided_key in self.adata.obs.columns:
                    del self.adata.obs[guided_key]

        return recovered

    def _get_canonical_fallback(self, missing_types: List[str]) -> Dict[str, List[str]]:
        """Query MarkerSearchAgent for canonical markers of missing types."""
        from scipy.sparse import issparse as sp_issparse

        from seededntm.marker_agent.agent import MarkerSearchAgent
        from seededntm.marker_agent.schemas import DatasetContext

        X = self.adata.X
        if sp_issparse(X):
            nonzero_frac = np.asarray((X > 0).sum(axis=0)).ravel() / X.shape[0]
        else:
            nonzero_frac = np.asarray((X > 0).sum(axis=0)).ravel() / X.shape[0]
        expressed_genes = set(
            self.adata.var_names[nonzero_frac > 0.01]
        )

        gene_panel = list(self.adata.var_names)
        ctx = DatasetContext(
            species=self.species,
            tissue=self.tissue_description,
            organ=self.organ,
            condition=self.condition,
            gene_panel=gene_panel,
            expected_cell_types=missing_types,
            n_markers_per_type=self.top_n_genes,
            include_disease_genes=bool(self.condition),
        )

        try:
            agent = MarkerSearchAgent(context=ctx, parallel=True, max_workers=4)
            results = agent.run()

            raw_fallback: Dict[str, List[str]] = {}
            for ct, markers in results.items():
                genes = [
                    m.gene_symbol for m in markers
                    if m.in_panel and m.gene_symbol in expressed_genes
                ][:self.top_n_genes * 2]
                if not genes:
                    genes = [
                        m.gene_symbol for m in markers
                        if m.gene_symbol in expressed_genes
                    ][:self.top_n_genes * 2]
                if not genes:
                    genes = [m.gene_symbol for m in markers if m.in_panel][:self.top_n_genes * 2]
                raw_fallback[ct] = genes

            if len(missing_types) > 1:
                gene_type_count: Dict[str, int] = defaultdict(int)
                for ct, genes in raw_fallback.items():
                    for g in genes:
                        gene_type_count[g] += 1
                n_types = len(missing_types)
                threshold = max(3, n_types // 2)
                shared_genes = {g for g, cnt in gene_type_count.items() if cnt >= threshold}
                if shared_genes:
                    logger.info("Removing %d genes shared across >=%d/%d fallback types: %s",
                                len(shared_genes), threshold, n_types, list(shared_genes)[:10])
            else:
                shared_genes = set()

            fallback: Dict[str, List[str]] = {}
            for ct, genes in raw_fallback.items():
                filtered = [g for g in genes if g not in shared_genes]
                fallback[ct] = filtered[:self.top_n_genes]

            # Fix 4: LLM panel selection fallback for types with < 3 genes
            for ct in missing_types:
                if len(fallback.get(ct, [])) < 3:
                    logger.info("  %s: only %d canonical genes, trying LLM panel selection",
                                ct, len(fallback.get(ct, [])))
                    llm_genes = self._llm_panel_selection(ct, gene_panel, missing_types)
                    if llm_genes:
                        existing = set(g.upper() for g in fallback.get(ct, []))
                        supplement = [g for g in llm_genes if g.upper() not in existing]
                        fallback[ct] = (fallback.get(ct, []) + supplement)[:self.top_n_genes]

            # Supplement from curated immune markers if result is sparse
            for ct in missing_types:
                current_genes = fallback.get(ct, [])
                if len(current_genes) < self.top_n_genes:
                    curated_key = self._match_curated_type(ct)
                    if curated_key and curated_key in self._CURATED_IMMUNE_MARKERS:
                        existing_upper = set(g.upper() for g in current_genes)
                        curated_supplement = [
                            g for g in self._CURATED_IMMUNE_MARKERS[curated_key]
                            if g.upper() in expressed_genes and g.upper() not in existing_upper
                        ]
                        if curated_supplement:
                            fallback[ct] = (current_genes + curated_supplement)[:self.top_n_genes]
                            logger.info("  %s: curated supplement added %d genes (total %d)",
                                        ct, len(curated_supplement), len(fallback[ct]))

            return fallback
        except Exception as e:
            logger.error("MarkerSearchAgent fallback failed: %s", e)
            return {}

    def _llm_panel_selection(self, cell_type: str, gene_panel: List[str],
                             all_types: List[str]) -> List[str]:
        """Use LLM to select best markers from the gene panel for a cell type.

        Fallback when canonical databases return too few genes (common on small panels).
        """
        from seededntm.prompt_templates import build_panel_selection_prompt
        from seededntm.marker_agent.llm_client import llm_complete

        prompt = build_panel_selection_prompt(
            cell_type=cell_type,
            gene_panel=gene_panel,
            tissue_context=self.tissue_description,
            all_cell_types=all_types,
            n_markers=self.top_n_genes,
        )
        system = (
            "You are a single-cell genomics expert. Select genes from the given panel "
            "that are the best markers for the specified cell type. Only return genes "
            "that are actually in the provided panel."
        )

        try:
            response = llm_complete(prompt, system=system, temperature=0.1, max_tokens=1024)
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0].strip()

            arr_match = re.search(r"\[.*\]", text, re.DOTALL)
            if arr_match:
                text = arr_match.group(0)

            genes = json.loads(text)
            if not isinstance(genes, list):
                return []

            panel_set = set(g.upper() for g in gene_panel)
            valid_genes = [g for g in genes if isinstance(g, str) and g.upper() in panel_set]

            logger.info("  %s: LLM panel selection returned %d valid genes", cell_type, len(valid_genes))
            return valid_genes[:self.top_n_genes]

        except Exception as e:
            logger.warning("  %s: LLM panel selection failed (%s)", cell_type, e)
            return []

    def _validate_seed_genes(self, cell_type: str, genes: List[str]) -> List[str]:
        """Rank DE genes by LLM validation. Returns ALL genes in priority order.

        Instead of filtering (which can be destructive for tissue-specific markers),
        genes are ranked: confirmed > likely > unclassified > unlikely > housekeeping.
        Callers truncate to top_n_genes so low-ranked genes naturally fall off.
        """
        from seededntm.prompt_templates import build_validation_prompt
        from seededntm.marker_agent.llm_client import llm_complete

        if len(genes) <= 3:
            return genes

        prompt = build_validation_prompt(
            gene_list=genes,
            cell_type=cell_type,
            tissue_context=self.tissue_description,
            all_cell_types=self.expected_cell_types,
        )
        system = (
            "You are a molecular biology expert. Be strict: only classify a gene as "
            "'confirmed' or 'likely' if it is a well-known marker specifically for this "
            "cell type. Genes associated with other cell types should be 'unlikely'."
        )

        try:
            response = llm_complete(prompt, system=system, temperature=0.0, max_tokens=2048)
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0].strip()

            json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if json_match:
                text = json_match.group(0)

            classifications = json.loads(text)

            # Case-insensitive lookup (Fix 2)
            classifications_upper = {k.upper(): v for k, v in classifications.items()}

            PRIORITY = {"confirmed": 0, "likely": 1, "unlikely": 3, "housekeeping": 4}

            def gene_priority(g):
                cls = classifications_upper.get(g.upper(), "").lower()
                return PRIORITY.get(cls, 2)  # unclassified gets priority 2

            ranked = sorted(genes, key=gene_priority)

            n_confirmed = sum(1 for g in genes if classifications_upper.get(g.upper(), "").lower() == "confirmed")
            n_likely = sum(1 for g in genes if classifications_upper.get(g.upper(), "").lower() == "likely")
            logger.info("  %s: ranked %d genes (%d confirmed, %d likely, top5: %s)",
                        cell_type, len(ranked), n_confirmed, n_likely,
                        ", ".join(ranked[:5]))
            return ranked

        except Exception as e:
            logger.warning("  %s: validation failed (%s), keeping original genes", cell_type, e)
            return genes

    def _validate_and_supplement_seeds(self, seeds: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Rank DE-derived seed genes and supplement sparse types with canonical markers.

        Uses additive ranking (never discards genes) then truncates to top_n_genes.
        Any type with < MIN_GENES_FOR_SUPPLEMENT genes after ranking gets canonical supplement.
        """
        MIN_GENES_FOR_SUPPLEMENT = 10

        types_to_rank = [
            ct for ct, info in seeds.items()
            if info.get("source") != "canonical_fallback"
        ]

        if types_to_rank:
            logger.info("Ranking DE genes for %d types...", len(types_to_rank))

        for ct in types_to_rank:
            genes = seeds[ct]["features"]
            if len(genes) > 3:
                ranked = self._validate_seed_genes(ct, genes)
                seeds[ct]["features"] = ranked[:self.top_n_genes]

        types_needing_more = [
            ct for ct, info in seeds.items()
            if len(info["features"]) < MIN_GENES_FOR_SUPPLEMENT
        ]

        if types_needing_more:
            logger.info("Supplementing %d types with < %d genes: %s",
                        len(types_needing_more), MIN_GENES_FOR_SUPPLEMENT, types_needing_more)
            canonical = self._get_canonical_fallback(types_needing_more)
            for ct in types_needing_more:
                existing = set(g.upper() for g in seeds[ct]["features"])
                supplement = [
                    g for g in canonical.get(ct, [])
                    if g.upper() not in existing
                ]
                combined = seeds[ct]["features"] + supplement
                seeds[ct]["features"] = combined[:self.top_n_genes]
                if supplement:
                    seeds[ct].setdefault("source", "leiden_de")
                    if seeds[ct]["source"] == "leiden_de":
                        seeds[ct]["source"] = "leiden_de+canonical"
                    logger.info("  %s: supplemented to %d genes", ct, len(seeds[ct]["features"]))

        return seeds

    @staticmethod
    def _strip_for_comparison(name: str) -> str:
        """Reduce a type name to lowercase alphanumeric tokens for fuzzy matching."""
        return re.sub(r"[_\s+&\-]+", "", name).lower()

    @staticmethod
    def _tokenize(name: str) -> set:
        """Split name into lowercase tokens for Jaccard comparison."""
        return set(re.split(r"[_\s+&\-]+", name.lower())) - {""}

    def _normalize_type_name(self, llm_name: str) -> str:
        """Match LLM-returned type name to expected_cell_types with fuzzy matching.

        Priority: exact case-insensitive > stripped comparison > Jaccard token overlap.
        Logs a warning when only a fuzzy (non-exact) match is used.
        """
        if not self.expected_cell_types:
            return llm_name

        stripped_input = llm_name.lower().strip()

        # Priority 1: exact case-insensitive match
        for expected in self.expected_cell_types:
            if expected.lower() == stripped_input:
                return expected

        # Priority 2: stripped (no underscores, spaces, +, &, -) comparison
        input_stripped = self._strip_for_comparison(llm_name)
        for expected in self.expected_cell_types:
            if self._strip_for_comparison(expected) == input_stripped:
                logger.warning("Fuzzy-matched (stripped) '%s' -> '%s'", llm_name, expected)
                return expected

        # Priority 3: substring match (one name contained in the other after stripping)
        for expected in self.expected_cell_types:
            exp_stripped = self._strip_for_comparison(expected)
            if input_stripped in exp_stripped or exp_stripped in input_stripped:
                logger.warning("Fuzzy-matched (substring) '%s' -> '%s'", llm_name, expected)
                return expected

        # Priority 4: Jaccard token overlap (threshold >= 0.5)
        input_tokens = self._tokenize(llm_name)
        if input_tokens:
            best_score = 0.0
            best_match = None
            for expected in self.expected_cell_types:
                exp_tokens = self._tokenize(expected)
                if not exp_tokens:
                    continue
                intersection = len(input_tokens & exp_tokens)
                union = len(input_tokens | exp_tokens)
                score = intersection / union if union > 0 else 0.0
                if score > best_score:
                    best_score = score
                    best_match = expected
            if best_score >= 0.5 and best_match is not None:
                logger.warning("Fuzzy-matched (Jaccard=%.2f) '%s' -> '%s'",
                               best_score, llm_name, best_match)
                return best_match

        return llm_name

    def _validate_cluster_consistency(self, annotations: Dict[str, str]) -> Dict[str, str]:
        """Check that each cluster's top DE genes are consistent with its label.

        For multi-cluster types, identifies clusters whose top-5 DE genes
        don't overlap with any other cluster assigned the same type (>20% overlap).
        Such outlier clusters are re-labeled "Unknown" to prevent diluting the
        merged DE signal.
        """
        from seededntm.marker_agent.llm_client import llm_complete

        if not self.expected_cell_types:
            return annotations

        type_to_clusters: Dict[str, List[str]] = defaultdict(list)
        for cid, ct in annotations.items():
            normalized = self._normalize_type_name(ct.strip())
            type_to_clusters[normalized].append(cid)

        updated = dict(annotations)
        unmatched_types = [
            ct for ct in self.expected_cell_types
            if ct not in type_to_clusters and not ct.startswith("Unknown")
        ]

        for cell_type, cluster_ids in type_to_clusters.items():
            if len(cluster_ids) < 3 or cell_type.startswith("Unknown"):
                continue

            # Check each cluster's top-5 DE against the type's verification markers
            base_type = None
            for vtype, (marker_set, _) in self._VERIFICATION_MARKERS.items():
                if vtype.lower() in cell_type.lower():
                    base_type = vtype
                    break

            outliers = []
            for cid in cluster_ids:
                top5 = set(g.upper() for g in self._de_markers.get(cid, [])[:5])
                # Check overlap with other clusters of same type
                other_genes = set()
                for other_cid in cluster_ids:
                    if other_cid != cid:
                        other_genes.update(g.upper() for g in self._de_markers.get(other_cid, [])[:10])

                overlap = len(top5.intersection(other_genes))
                if overlap == 0 and len(top5) >= 3:
                    # This cluster shares zero top-5 genes with any peer — likely mislabeled
                    outliers.append(cid)

            if not outliers or not unmatched_types:
                continue

            logger.info("  Type '%s': %d/%d clusters are outliers (no top-5 overlap with peers)",
                        cell_type, len(outliers), len(cluster_ids))

            # Try to re-assign outlier clusters to unmatched types
            for cid in outliers:
                top_genes = self._de_markers.get(cid, [])[:10]
                genes_str = ", ".join(top_genes)
                types_str = ", ".join(unmatched_types + ["Unknown"])

                reclassify_prompt = f"""A cluster was grouped as "{cell_type}" but its top DE markers don't overlap with any other {cell_type} clusters.

Cluster top DE markers: {genes_str}
Tissue context: {self.tissue_description}

This cluster may actually represent one of these unmatched types: {types_str}
Or it may still be "{cell_type}" (keep original if markers are consistent with {cell_type}).

Which type best matches these markers? Return ONLY the type name."""

                system = "You are a spatial transcriptomics expert. Be conservative — only reclassify if the markers clearly indicate a different type."
                try:
                    response = llm_complete(reclassify_prompt, system=system, temperature=0.0, max_tokens=128)
                    new_type = response.strip().strip('"').strip("'")
                    new_type_normalized = self._normalize_type_name(new_type)

                    if new_type_normalized in unmatched_types:
                        logger.info("    Cluster %s: reclassified %s -> %s", cid, cell_type, new_type_normalized)
                        updated[cid] = new_type_normalized
                        unmatched_types.remove(new_type_normalized)
                    elif new_type_normalized == "Unknown":
                        updated[cid] = "Unknown"
                        logger.info("    Cluster %s: reclassified %s -> Unknown", cid, cell_type)
                except Exception as e:
                    logger.warning("    Cluster %s: reclassification failed (%s)", cid, e)

        return updated

    def _merge_and_build_seeds(
        self, annotations: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """Merge clusters with same annotation, re-run DE, and build final seed dict."""
        type_to_clusters: Dict[str, List[str]] = defaultdict(list)
        for cluster_id, cell_type in annotations.items():
            normalized = self._normalize_type_name(cell_type.strip())
            type_to_clusters[normalized].append(cluster_id)

        known_types = {k: v for k, v in type_to_clusters.items() if not k.startswith("Unknown")}

        merged_de = self._rerun_de_on_merged(known_types)

        seeds: Dict[str, Dict[str, Any]] = {}
        if merged_de is not None:
            for cell_type, genes in merged_de.items():
                seeds[cell_type] = {"features": genes}
        else:
            for cell_type, cluster_ids in known_types.items():
                all_genes: List[str] = []
                for cid in cluster_ids:
                    all_genes.extend(self._de_markers.get(cid, []))
                seen = set()
                unique_genes = []
                for g in all_genes:
                    g_upper = g.upper()
                    if g_upper not in seen:
                        seen.add(g_upper)
                        unique_genes.append(g)
                seeds[cell_type] = {"features": unique_genes[:self.top_n_genes]}

        # Cross-type filter: only for small panels where wilcoxon was used for merged DE
        if self.expected_cell_types and len(seeds) > 1 and self.adata.shape[1] < 1000:
            all_other_markers: Dict[str, Set[str]] = {}
            for vtype, (marker_set, _) in self._VERIFICATION_MARKERS.items():
                all_other_markers[vtype] = {g.upper() for g in marker_set}

            for ct, info in seeds.items():
                ct_own_type = self._match_verification_type(ct)
                markers_to_remove: Set[str] = set()
                for vtype, vmarkers in all_other_markers.items():
                    if vtype != ct_own_type:
                        markers_to_remove.update(vmarkers)
                original = info["features"]
                filtered = [g for g in original if g.upper() not in markers_to_remove]
                if len(filtered) >= 10 and len(filtered) < len(original):
                    logger.info("  %s: removed %d cross-type markers", ct, len(original) - len(filtered))
                    info["features"] = filtered

        if self.expected_cell_types:
            min_genes_threshold = 0 if self.adata.shape[1] >= 1000 else 10
            sparse_types = [
                ct for ct, info in seeds.items()
                if len(info.get("features", [])) <= min_genes_threshold
            ]
            if sparse_types:
                logger.info("Supplementing %d sparse types (<=%d genes): %s",
                            len(sparse_types), min_genes_threshold, sparse_types)
                canonical = self._get_canonical_fallback(sparse_types)
                for ct in sparse_types:
                    existing = set(g.upper() for g in seeds[ct].get("features", []))
                    supplement = [
                        g for g in canonical.get(ct, [])
                        if g.upper() not in existing
                    ]
                    if supplement:
                        seeds[ct]["features"] = (seeds[ct].get("features", []) + supplement)[:self.top_n_genes]
                        logger.info("  %s: supplemented to %d genes", ct, len(seeds[ct]["features"]))

        if self.expected_cell_types:
            found_types = set(seeds.keys())
            missing = [ct for ct in self.expected_cell_types if ct not in found_types]
            if missing:
                logger.info("Missing %d expected types, using canonical markers filtered to dataset: %s",
                            len(missing), missing)
                fallback_markers = self._get_canonical_fallback(missing)
                for ct, genes in fallback_markers.items():
                    seeds[ct] = {"features": genes, "source": "canonical_filtered"}

        if self.expected_cell_types:
            for i, ct in enumerate(self.expected_cell_types):
                if ct in seeds:
                    seeds[ct]["topic_index"] = i
            extra_idx = len(self.expected_cell_types)
            for ct in seeds:
                if "topic_index" not in seeds[ct]:
                    seeds[ct]["topic_index"] = extra_idx
                    extra_idx += 1
        else:
            for idx, ct in enumerate(sorted(seeds.keys())):
                seeds[ct]["topic_index"] = idx

        # Safety guard: drop any types NOT in expected_cell_types if count exceeds K
        if self.expected_cell_types:
            expected_set = set(self.expected_cell_types)
            extra = [ct for ct in seeds if ct not in expected_set]
            if extra:
                logger.warning("Dropping %d extra types not in expected_cell_types: %s", len(extra), extra)
                for ct in extra:
                    del seeds[ct]

        return seeds

    def _deduplicate_seeds(self, seeds: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Ensure no two types share >30% of their seed genes.

        When duplicates are found, differentiate them using LLM panel selection
        with explicit instructions to distinguish the overlapping types.
        """
        from seededntm.marker_agent.llm_client import llm_complete

        MAX_OVERLAP_RATIO = 0.30
        types_list = list(seeds.keys())
        gene_panel = list(self.adata.var_names) if hasattr(self, 'adata') else []

        duplicates_found = []
        for i, t1 in enumerate(types_list):
            genes1 = set(g.upper() for g in seeds[t1]["features"])
            for t2 in types_list[i + 1:]:
                genes2 = set(g.upper() for g in seeds[t2]["features"])
                if not genes1 or not genes2:
                    continue
                overlap = len(genes1 & genes2)
                min_size = min(len(genes1), len(genes2))
                if min_size > 0 and overlap / min_size > MAX_OVERLAP_RATIO:
                    duplicates_found.append((t1, t2, overlap, min_size))

        if not duplicates_found:
            return seeds

        logger.info("Seed deduplication: found %d type pairs with >%.0f%% overlap",
                    len(duplicates_found), MAX_OVERLAP_RATIO * 100)

        for t1, t2, overlap, min_size in duplicates_found:
            logger.info("  %s <-> %s: %d/%d genes overlap (%.0f%%)",
                        t1, t2, overlap, min_size, 100 * overlap / min_size)

            # Ask LLM to differentiate these two types
            if not gene_panel:
                continue

            all_types = self.expected_cell_types or types_list
            for target_type, other_type in [(t1, t2), (t2, t1)]:
                # Only differentiate if the type used fallback/guided (not DE-derived)
                if seeds[target_type].get("source") not in ("canonical_fallback", "guided_recovery"):
                    continue

                # For guided_recovery types, use simple overlap removal instead of
                # LLM re-query which tends to degrade gene quality further
                if seeds[target_type].get("source") == "guided_recovery":
                    other_genes_upper = set(g.upper() for g in seeds[other_type]["features"])
                    current = seeds[target_type]["features"]
                    deduped = [g for g in current if g.upper() not in other_genes_upper]
                    if len(deduped) >= 5:
                        seeds[target_type]["features"] = deduped[:self.top_n_genes]
                        logger.info("    %s: removed %d overlapping genes (simple dedup)",
                                    target_type, len(current) - len(deduped))
                    continue

                prompt = f"""From the gene panel below, select the {self.top_n_genes} genes that best distinguish "{target_type}" from "{other_type}" (and other types: {", ".join(t for t in all_types if t not in (target_type, other_type))}).

These two types currently have overlapping marker genes and need to be differentiated.
Tissue context: {self.tissue_description}

Key distinction needed: How does "{target_type}" differ from "{other_type}"?

Gene panel ({len(gene_panel)} genes):
{", ".join(gene_panel)}

Return ONLY a JSON array of gene symbols, ordered by specificity for "{target_type}" OVER "{other_type}":
["GENE1", "GENE2", ...]"""

                system = "You are a single-cell genomics expert. Focus on genes that specifically distinguish the target type from the confounding type."
                try:
                    response = llm_complete(prompt, system=system, temperature=0.1, max_tokens=1024)
                    text = response.strip()
                    if text.startswith("```"):
                        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                        text = text.rsplit("```", 1)[0].strip()
                    arr_match = re.search(r"\[.*\]", text, re.DOTALL)
                    if arr_match:
                        text = arr_match.group(0)
                    genes = json.loads(text)
                    if isinstance(genes, list):
                        panel_upper = {g.upper(): g for g in gene_panel}
                        valid = [panel_upper[g.upper()] for g in genes
                                 if isinstance(g, str) and g.upper() in panel_upper]
                        if len(valid) >= 5:
                            seeds[target_type]["features"] = valid[:self.top_n_genes]
                            logger.info("    %s: differentiated with %d new genes", target_type, len(valid[:self.top_n_genes]))
                except Exception as e:
                    logger.warning("    %s: differentiation failed (%s)", target_type, e)

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

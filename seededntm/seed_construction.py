"""Systematic seed gene construction pipeline using leiden clustering + LLM annotation.

Automates everything except the LLM interaction itself (semi-manual copy-paste workflow).
Provides structured prompt generation, response parsing, cluster merging, and audit logging.

Workflow:
    1. prepare(): Load data, run multi-resolution leiden, compute DE markers, generate prompt
    2. (USER manually pastes prompt into LLM, saves response as text file)
    3. finalize(): Parse response, validate, merge clusters, compute final markers, save seeds

Example:
    >>> from seededntm.seed_construction import SeedConstructionPipeline
    >>> pipeline = SeedConstructionPipeline(
    ...     h5ad_path="spatial.h5ad",
    ...     tissue="human breast cancer (Xenium 313 genes)",
    ...     output_dir="./seed_construction/",
    ... )
    >>> pipeline.prepare(expected_k=19)
    # User pastes prompt into LLM, saves response
    >>> pipeline.finalize(response_path="./seed_construction/response_round1.txt")
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc

from .prompt_templates import (
    build_annotation_prompt,
    build_followup_prompt,
    format_marker_table,
)

logger = logging.getLogger(__name__)


class SeedConstructionPipeline:
    """Multi-resolution leiden + LLM annotation pipeline for seed gene construction."""

    def __init__(
        self,
        h5ad_path: str,
        tissue: str,
        output_dir: str,
        n_hvg: int = 4000,
        resolutions: list = None,
    ):
        """
        Args:
            h5ad_path: path to spatial transcriptomics h5ad file.
            tissue: tissue description for LLM context
                (e.g. "human breast cancer tissue, Xenium panel with 313 genes").
            output_dir: directory for all pipeline outputs and audit trail.
            n_hvg: number of highly variable genes to select (0 to skip HVG filtering).
            resolutions: leiden resolutions to scan. Default: [0.5, 0.8, 1.0, 1.2, 1.5, 2.0].
        """
        self.h5ad_path = h5ad_path
        self.tissue = tissue
        self.output_dir = Path(output_dir)
        self.n_hvg = n_hvg
        self.resolutions = resolutions or [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "audit").mkdir(exist_ok=True)

        self.adata = None
        self.selected_resolution = None
        self.markers_per_cluster = None
        self.mapping = None

    def prepare(
        self,
        expected_k: int = None,
        resolution: float = None,
        top_markers: int = 20,
        min_logfc: float = 0.5,
    ) -> str:
        """Run automated preprocessing and generate LLM prompt.

        Args:
            expected_k: expected number of cell types. Used to select resolution.
            resolution: override automatic resolution selection.
            top_markers: number of markers per cluster in the prompt.
            min_logfc: minimum log fold-change for marker selection.

        Returns:
            Path to the generated prompt file.
        """
        logger.info(f"Loading h5ad: {self.h5ad_path}")
        self.adata = sc.read_h5ad(self.h5ad_path)
        logger.info(f"Data: {self.adata.n_obs} spots x {self.adata.n_vars} genes")

        # Preprocessing
        adata = self.adata.copy()
        if self.n_hvg > 0 and adata.n_vars > self.n_hvg:
            sc.pp.highly_variable_genes(adata, n_top_genes=self.n_hvg, flavor="seurat_v3")
            adata = adata[:, adata.var["highly_variable"]].copy()
            logger.info(f"HVG filter: {adata.n_vars} genes retained")

        # Skip normalization if data appears already log-transformed
        from scipy.sparse import issparse as _issparse
        X_check = adata.X[:100] if not _issparse(adata.X) else adata.X[:100].toarray()
        max_val = np.max(X_check)
        if max_val > 20:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        else:
            logger.info("Data appears already normalized/log-transformed, skipping normalization")

        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=min(100, adata.n_vars - 1))
        if "connectivities" not in adata.obsp:
            sc.pp.neighbors(adata, n_pcs=min(50, adata.obsm["X_pca"].shape[1]))

        # Multi-resolution leiden scan
        resolution_info = self._scan_resolutions(adata)
        self._save_diagnostics(resolution_info, expected_k)

        # Select best resolution
        if resolution is not None:
            self.selected_resolution = resolution
        elif expected_k is not None:
            self.selected_resolution = self._select_resolution_by_k(
                resolution_info, expected_k
            )
        else:
            self.selected_resolution = 1.2  # default fallback

        logger.info(f"Selected resolution: {self.selected_resolution}")

        # Run leiden at selected resolution
        sc.tl.leiden(adata, resolution=self.selected_resolution, key_added="leiden_selected")
        n_clusters = adata.obs["leiden_selected"].nunique()
        logger.info(f"Leiden clusters: {n_clusters}")

        # Compute DE markers
        sc.tl.rank_genes_groups(
            adata, groupby="leiden_selected", method="wilcoxon"
        )
        self.markers_per_cluster = self._extract_markers(
            adata, top_n=top_markers, min_logfc=min_logfc
        )

        # Generate prompt
        prompt = build_annotation_prompt(
            self.markers_per_cluster,
            tissue_description=self.tissue,
            top_n=top_markers,
        )

        prompt_path = self.output_dir / "prompt_round1.txt"
        prompt_path.write_text(prompt)
        logger.info(f"Prompt saved: {prompt_path}")

        # Save intermediate state
        self._save_state(adata)

        print(f"\n{'='*60}")
        print(f"PROMPT GENERATED: {prompt_path}")
        print(f"{'='*60}")
        print(f"Next steps:")
        print(f"  1. Open {prompt_path}")
        print(f"  2. Copy the entire content and paste into your LLM (ChatGPT, Claude, etc.)")
        print(f"  3. Save the LLM's JSON response as: {self.output_dir}/response_round1.txt")
        print(f"  4. Run: pipeline.finalize(response_path='response_round1.txt')")
        print(f"{'='*60}\n")

        return str(prompt_path)

    def finalize(
        self,
        response_path: str,
        confidence_threshold: float = 0.6,
        top_markers_final: int = 20,
        min_logfc_final: float = 0.5,
    ) -> str:
        """Parse LLM response, merge clusters, compute final seeds.

        Args:
            response_path: path to the LLM response text file (absolute or relative to output_dir).
            confidence_threshold: minimum confidence to accept an annotation.
            top_markers_final: markers per merged type in final seed list.
            min_logfc_final: logFC threshold for final markers.

        Returns:
            Path to the final seeds file.
        """
        response_file = Path(response_path)
        if not response_file.is_absolute():
            response_file = self.output_dir / response_file

        if not response_file.exists():
            raise FileNotFoundError(f"Response file not found: {response_file}")

        response_text = response_file.read_text()

        # Save response in audit trail
        audit_response = self.output_dir / "audit" / response_file.name
        audit_response.write_text(response_text)

        # Parse JSON from response
        self.mapping = parse_llm_response(response_text)
        if self.mapping is None:
            raise ValueError("Could not parse JSON from LLM response")

        logger.info(f"Parsed {len(self.mapping)} cluster annotations")

        # Save parsed mapping
        mapping_path = self.output_dir / "audit" / "parsed_mapping.json"
        mapping_path.write_text(json.dumps(self.mapping, indent=2))

        # Check for low confidence or unknown
        low_conf = {
            k: v for k, v in self.mapping.items()
            if v.get("confidence", 0) < confidence_threshold
            or v.get("cell_type", "").lower() == "unknown"
        }
        if low_conf:
            logger.warning(f"{len(low_conf)} clusters have low confidence or Unknown:")
            for cid, info in low_conf.items():
                logger.warning(f"  Cluster {cid}: {info['cell_type']} ({info.get('confidence', '?')})")

            followup_prompt = build_followup_prompt(low_conf)
            followup_path = self.output_dir / "prompt_followup.txt"
            followup_path.write_text(followup_prompt)
            logger.warning(f"Follow-up prompt saved: {followup_path}")

        # Load intermediate state
        state_path = self.output_dir / "audit" / "pipeline_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            adata_path = state.get("preprocessed_h5ad")
            if adata_path and Path(adata_path).exists():
                adata = sc.read_h5ad(adata_path)
            else:
                adata = self._reload_and_preprocess()
        else:
            adata = self._reload_and_preprocess()

        # Merge clusters by cell type assignment
        cluster_to_type = {k: v["cell_type"] for k, v in self.mapping.items()}
        adata.obs["cell_type_merged"] = adata.obs["leiden_selected"].astype(str).map(cluster_to_type)

        # Handle unmapped clusters
        unmapped = adata.obs["cell_type_merged"].isna()
        if unmapped.any():
            logger.warning(f"{unmapped.sum()} spots in unmapped clusters, labeling as 'Unmapped'")
            adata.obs["cell_type_merged"] = adata.obs["cell_type_merged"].fillna("Unmapped")

        # Re-run DE on merged groups for final markers
        merged_types = sorted(adata.obs["cell_type_merged"].unique().tolist())
        if "Unmapped" in merged_types:
            merged_types.remove("Unmapped")
        logger.info(f"Merged into {len(merged_types)} cell types")

        sc.tl.rank_genes_groups(
            adata, groupby="cell_type_merged", method="wilcoxon",
            groups=merged_types,
        )

        # Extract final markers with quality filters
        final_markers = self._extract_markers(
            adata, top_n=top_markers_final, min_logfc=min_logfc_final,
            key="rank_genes_groups",
        )

        # Build seeds dict in SeedTopic format
        seeds = {}
        for topic_idx, ct_name in enumerate(sorted(merged_types)):
            genes = final_markers.get(ct_name, [])
            if isinstance(genes[0], (tuple, list)):
                gene_names = [g[0] for g in genes]
            else:
                gene_names = list(genes)
            seeds[ct_name] = {
                "features": gene_names,
                "topic_index": topic_idx,
            }

        # Save seeds
        seeds_path = self.output_dir / "final_seeds.txt"
        seeds_path.write_text(json.dumps(seeds))
        logger.info(f"Final seeds saved: {seeds_path} ({len(seeds)} types)")

        # Save full audit
        audit = {
            "tissue": self.tissue,
            "h5ad_path": self.h5ad_path,
            "selected_resolution": self.selected_resolution,
            "n_clusters": len(self.mapping),
            "n_merged_types": len(merged_types),
            "merged_types": merged_types,
            "confidence_threshold": confidence_threshold,
            "low_confidence_clusters": list(low_conf.keys()) if low_conf else [],
        }
        audit_path = self.output_dir / "audit" / "finalize_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2))

        print(f"\n{'='*60}")
        print(f"SEEDS GENERATED: {seeds_path}")
        print(f"  {len(seeds)} cell types, {top_markers_final} markers each")
        print(f"  Types: {', '.join(sorted(merged_types))}")
        if low_conf:
            print(f"\n  WARNING: {len(low_conf)} low-confidence clusters.")
            print(f"  Consider the follow-up prompt: {self.output_dir}/prompt_followup.txt")
        print(f"{'='*60}\n")

        return str(seeds_path)

    def _scan_resolutions(self, adata) -> list:
        """Run leiden at multiple resolutions and collect diagnostics."""
        results = []
        for res in self.resolutions:
            key = f"leiden_r{res}"
            sc.tl.leiden(adata, resolution=res, key_added=key)
            n_clusters = adata.obs[key].nunique()
            cluster_sizes = adata.obs[key].value_counts()
            results.append({
                "resolution": res,
                "n_clusters": n_clusters,
                "min_cluster_size": int(cluster_sizes.min()),
                "max_cluster_size": int(cluster_sizes.max()),
                "median_cluster_size": int(cluster_sizes.median()),
            })
            logger.info(f"  resolution={res}: {n_clusters} clusters "
                        f"(sizes: {cluster_sizes.min()}-{cluster_sizes.max()})")
        return results

    def _select_resolution_by_k(self, resolution_info: list, expected_k: int) -> float:
        """Select the resolution that gives cluster count closest to expected_k."""
        diffs = [(abs(r["n_clusters"] - expected_k), r["resolution"]) for r in resolution_info]
        # Prefer slightly more clusters (can merge) over fewer
        for i, (diff, res) in enumerate(diffs):
            info = next(r for r in resolution_info if r["resolution"] == res)
            if info["n_clusters"] >= expected_k:
                diffs[i] = (diff - 0.5, res)  # slight preference for over-clustering
        diffs.sort()
        return diffs[0][1]

    def _extract_markers(self, adata, top_n=20, min_logfc=0.5, key="rank_genes_groups") -> dict:
        """Extract top DE markers per group from scanpy results."""
        markers = {}
        result = adata.uns[key]
        group_names = result["names"].dtype.names

        for group in group_names:
            genes = result["names"][group]
            logfcs = result["logfoldchanges"][group]
            pvals = result["pvals_adj"][group]

            filtered = []
            for gene, lfc, pv in zip(genes, logfcs, pvals):
                if lfc >= min_logfc and pv < 0.01:
                    filtered.append((gene, float(lfc)))
                if len(filtered) >= top_n:
                    break
            markers[group] = filtered if filtered else [(g, 0.0) for g in genes[:top_n]]

        return markers

    def _save_diagnostics(self, resolution_info: list, expected_k: int = None):
        """Save resolution scan diagnostics."""
        diag = {
            "resolutions_scanned": self.resolutions,
            "expected_k": expected_k,
            "results": resolution_info,
        }
        diag_path = self.output_dir / "audit" / "diagnostics.json"
        diag_path.write_text(json.dumps(diag, indent=2))

    def _save_state(self, adata):
        """Save preprocessed adata and pipeline state for later use in finalize()."""
        preprocessed_path = self.output_dir / "audit" / "preprocessed.h5ad"
        adata.write_h5ad(preprocessed_path)
        state = {
            "h5ad_path": self.h5ad_path,
            "tissue": self.tissue,
            "selected_resolution": self.selected_resolution,
            "preprocessed_h5ad": str(preprocessed_path),
            "n_clusters": adata.obs["leiden_selected"].nunique(),
        }
        state_path = self.output_dir / "audit" / "pipeline_state.json"
        state_path.write_text(json.dumps(state, indent=2))

    def _reload_and_preprocess(self):
        """Reload and preprocess data (fallback if state not saved)."""
        adata = sc.read_h5ad(self.h5ad_path)
        if self.n_hvg > 0 and adata.n_vars > self.n_hvg:
            sc.pp.highly_variable_genes(adata, n_top_genes=self.n_hvg, flavor="seurat_v3")
            adata = adata[:, adata.var["highly_variable"]].copy()

        from scipy.sparse import issparse as _issparse
        X_check = adata.X[:100] if not _issparse(adata.X) else adata.X[:100].toarray()
        max_val = np.max(X_check)
        if max_val > 20:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=min(100, adata.n_vars - 1))
        if "connectivities" not in adata.obsp:
            sc.pp.neighbors(adata, n_pcs=min(50, adata.obsm["X_pca"].shape[1]))
        sc.tl.leiden(adata, resolution=self.selected_resolution, key_added="leiden_selected")
        return adata


def parse_llm_response(text: str) -> Optional[dict]:
    """Extract JSON cluster mapping from LLM response text.

    Handles:
    - Raw JSON responses
    - JSON wrapped in markdown code fences (```json ... ```)
    - Preamble/postamble text around JSON
    - Single or double quotes

    Returns:
        dict mapping cluster_id (str) -> {"cell_type": str, "confidence": float, ...}
        or None if parsing fails.
    """
    # Try to find JSON in code fences first
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        json_str = fence_match.group(1).strip()
    else:
        # Try to find the outermost { ... }
        brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if brace_match:
            json_str = brace_match.group(0)
        else:
            return None

    # Fix common LLM output issues
    json_str = json_str.replace("'", '"')  # single to double quotes
    json_str = re.sub(r",\s*}", "}", json_str)  # trailing commas
    json_str = re.sub(r",\s*]", "]", json_str)

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        # Try more aggressive cleanup - find individual cluster entries
        try:
            # Sometimes LLMs produce multiple JSON objects
            all_json = re.findall(r"\{[^}]+\}", json_str)
            if all_json:
                combined = "{" + ",".join(
                    f'"{i}": {obj}' for i, obj in enumerate(all_json)
                ) + "}"
                result = json.loads(combined)
            else:
                return None
        except (json.JSONDecodeError, ValueError):
            return None

    # Validate structure
    if not isinstance(result, dict):
        return None

    validated = {}
    for key, value in result.items():
        if isinstance(value, dict) and "cell_type" in value:
            validated[str(key)] = {
                "cell_type": str(value["cell_type"]),
                "confidence": float(value.get("confidence", 0.5)),
                "reasoning": str(value.get("reasoning", "")),
            }
        elif isinstance(value, str):
            validated[str(key)] = {
                "cell_type": value,
                "confidence": 0.5,
                "reasoning": "",
            }

    return validated if validated else None


def validate_mapping(mapping: dict, n_clusters: int) -> dict:
    """Validate cluster mapping completeness and quality.

    Returns:
        dict with "valid" (bool), "missing" (list), "low_confidence" (list), "issues" (list).
    """
    issues = []
    missing = []
    low_confidence = []

    expected_clusters = set(str(i) for i in range(n_clusters))
    mapped_clusters = set(mapping.keys())

    missing = sorted(expected_clusters - mapped_clusters)
    if missing:
        issues.append(f"Missing annotations for clusters: {missing}")

    for cid, info in mapping.items():
        if info.get("confidence", 0) < 0.5:
            low_confidence.append(cid)
        if info.get("cell_type", "").lower() in ("unknown", ""):
            low_confidence.append(cid)

    if low_confidence:
        issues.append(f"Low confidence clusters: {sorted(set(low_confidence))}")

    return {
        "valid": len(missing) == 0 and len(low_confidence) == 0,
        "missing": missing,
        "low_confidence": sorted(set(low_confidence)),
        "issues": issues,
    }

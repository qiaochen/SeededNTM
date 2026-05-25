"""Prompt templates for LLM-based cluster annotation and marker validation."""

from __future__ import annotations

from typing import Dict, List


def build_annotation_prompt(
    markers_per_cluster: Dict[str, List[str]],
    tissue_description: str,
    top_n: int = 20,
) -> str:
    """Build a prompt for LLM-based cluster cell type annotation.

    Args:
        markers_per_cluster: Dict mapping cluster_id -> list of DE gene symbols.
        tissue_description: Description of tissue context (e.g., "colorectal cancer
            tumor microenvironment, colon tissue").
        top_n: Max number of markers per cluster to include in prompt.

    Returns:
        Formatted prompt string for the annotation LLM call.
    """
    cluster_text_parts = []
    for cluster_id, markers in sorted(markers_per_cluster.items(), key=lambda x: x[0]):
        genes = markers[:top_n]
        gene_str = ", ".join(genes)
        cluster_text_parts.append(f"Cluster {cluster_id}: {gene_str}")

    cluster_block = "\n".join(cluster_text_parts)
    n_clusters = len(markers_per_cluster)

    prompt = f"""You are a single-cell genomics expert. I have performed unsupervised clustering (Leiden algorithm) on a spatial transcriptomics dataset and identified {n_clusters} clusters. The tissue context is: {tissue_description}.

Below are the top differentially expressed marker genes for each cluster (Wilcoxon rank-sum test, sorted by log fold change):

{cluster_block}

For each cluster, provide a cell type annotation. Consider:
1. The tissue context and expected cell populations
2. Whether multiple clusters may represent the same cell type (subtypes or states)
3. Use standard Cell Ontology nomenclature where possible
4. If a cluster is ambiguous, provide your best guess with a confidence note

Return your answer as a JSON object mapping cluster IDs to cell type names. Use this exact format:
{{
  "0": "Cell Type Name",
  "1": "Cell Type Name",
  ...
}}

Only return the JSON object, no other text."""

    return prompt


def build_validation_prompt(
    gene_list: List[str],
    cell_type: str,
    tissue_context: str,
) -> str:
    """Build a prompt to validate whether genes are true markers for a cell type.

    Args:
        gene_list: List of candidate marker genes to validate.
        cell_type: The cell type these genes are proposed markers for.
        tissue_context: Tissue/disease context.

    Returns:
        Formatted prompt for marker validation.
    """
    genes_str = ", ".join(gene_list)

    prompt = f"""You are a molecular biology expert specializing in cell type markers. Evaluate whether each of the following genes is a reliable marker for "{cell_type}" in the context of {tissue_context}.

Candidate markers: {genes_str}

For each gene, classify as:
- "confirmed": well-established marker for this cell type
- "likely": strong evidence but not definitive
- "unlikely": this gene is more associated with other cell types
- "housekeeping": ubiquitously expressed, not cell-type-specific

Return a JSON object mapping gene symbols to classifications:
{{
  "GENE1": "confirmed",
  "GENE2": "likely",
  ...
}}

Only return the JSON object, no other text."""

    return prompt

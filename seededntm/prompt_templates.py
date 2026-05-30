"""Prompt templates for LLM-based cluster annotation and marker validation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_annotation_prompt(
    markers_per_cluster: Dict[str, List[str]],
    tissue_description: str,
    top_n: int = 20,
    expected_cell_types: Optional[List[str]] = None,
) -> str:
    """Build a prompt for LLM-based cluster cell type annotation.

    Args:
        markers_per_cluster: Dict mapping cluster_id -> list of DE gene symbols.
        tissue_description: Description of tissue context (e.g., "colorectal cancer
            tumor microenvironment, colon tissue").
        top_n: Max number of markers per cluster to include in prompt.
        expected_cell_types: Optional list of expected cell type names. When provided,
            the LLM is guided to map clusters to these exact type names.

    Returns:
        Formatted prompt string for the annotation LLM call.
    """
    cluster_text_parts = []
    for cluster_id, markers in sorted(markers_per_cluster.items(), key=lambda x: x[0]):
        genes = markers[:top_n]
        if not genes:
            cluster_text_parts.append(f"Cluster {cluster_id}: [no specific markers detected]")
        elif len(genes) < 3:
            gene_str = ", ".join(genes)
            cluster_text_parts.append(f"Cluster {cluster_id} (low confidence): {gene_str}")
        else:
            gene_str = ", ".join(genes)
            cluster_text_parts.append(f"Cluster {cluster_id}: {gene_str}")

    cluster_block = "\n".join(cluster_text_parts)
    n_clusters = len(markers_per_cluster)

    header = f"""You are a single-cell genomics expert. I have performed unsupervised clustering (Leiden algorithm) on a spatial transcriptomics dataset and identified {n_clusters} clusters. The tissue context is: {tissue_description}.

Below are the top differentially expressed marker genes for each cluster (Wilcoxon rank-sum test, sorted by log fold change):

{cluster_block}

"""

    if expected_cell_types:
        types_str = ", ".join(expected_cell_types)
        instruction = f"""The known cell populations expected in this tissue are: {types_str}

CRITICAL CONTEXT: This is SPATIAL TRANSCRIPTOMICS data. Each cluster represents a tissue NEIGHBORHOOD (spatial region) that is enriched for a particular cell type but also contains signals from surrounding cells. Therefore:
- A "Myeloid" or "immune" neighborhood may have dominant stromal/chemokine genes (CCL19, CCL21, SELENOP, CXCL12) because that spatial region contains fibroblastic reticular cells that attract and support immune cells. This is NORMAL for spatial data.
- A "B" cell zone may show follicular dendritic cell markers (FDCSP, NBL1) alongside immunoglobulins.
- Do NOT expect pure single-cell marker profiles. Focus on which cell type ENRICHES that spatial neighborhood.

Guidelines for annotation — focus on the TOP 3-5 markers:
- The assignment MUST be primarily justified by the first 3-5 markers in the list (highest fold-change). Markers appearing later in the list (positions 6+) should NOT override the biological identity implied by the top markers.
- T cells: the TOP markers must include TCR/T-cell genes (CD3D, CD3E, TRAC, TRBC1, TRBC2, CD8A, CD8B, CD4, GZMA, GZMB, NKG7, PRF1). If TCR genes only appear at positions 4+ and the TOP markers are chemokines/stromal (CCL19, CCL21, DCN, SELENOP), this is NOT a T cell cluster.
- B cells: TOP markers should include immunoglobulin genes (IGKC, IGHG*, IGHA*, IGLC*) or B-lineage (CD79A, CD79B, MS4A1, CD19). If the top markers are chemokines (CCL19, PTGDS, CXCL12) with immunoglobulins only at positions 10+, consider "Myeloid" neighborhood instead.
- Myeloid/macrophage: classical markers are CD68, CD14, LYZ, CSF1R. BUT in spatial data, the myeloid-enriched neighborhood often shows stromal support markers (CCL19, CCL21, SELENOP, IGFBP5, PTGDS, NBL1, CXCL12) — these clusters should STILL be labeled "Myeloid".
- Fibroblast/CAF: extracellular matrix genes as TOP markers (COL1A1, COL1A2, COL3A1, FN1, SPARC, BGN, DCN, LUM, AEBP1).
- Endothelial: vascular markers (PECAM1, VWF, CDH5, KDR, PLVAP, RAMP2).
- Tumor/malignant epithelial: tissue-specific epithelial markers (keratins KRT5/8/15, EPCAM, growth/metabolic genes).
- Normal epithelial: secretory markers (WFDC2, SLPI, PIGR, mucins MUC4/MUC16, CAPS, LCN2).
- If a cluster has mixed markers from multiple lineages AND no clear dominant signal, label it "Unknown"
- If top markers are ribosomal (RPL/RPS), mitochondrial (MT-), or housekeeping, label "Unknown"

IMPORTANT: Map each cluster to one of the expected cell types above. Use the EXACT type names provided (case-sensitive: "{types_str}"). If multiple clusters clearly represent the same cell type (subtypes/states), assign them the same label. If a cluster cannot be confidently mapped to any expected type, label it "Unknown".

Return a JSON object mapping cluster IDs to cell type names:
{{"0": "ExactTypeName", "1": "ExactTypeName", ...}}
Only return the JSON, no other text."""
    else:
        instruction = """For each cluster, provide a cell type annotation. Consider:
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

    prompt = header + instruction
    return prompt


def build_validation_prompt(
    gene_list: List[str],
    cell_type: str,
    tissue_context: str,
    all_cell_types: Optional[List[str]] = None,
) -> str:
    """Build a prompt to validate whether genes are true markers for a cell type.

    Args:
        gene_list: List of candidate marker genes to validate.
        cell_type: The cell type these genes are proposed markers for.
        tissue_context: Tissue/disease context.
        all_cell_types: Optional full list of cell types in the experiment,
            to help the LLM understand what distinguishes this type from others.

    Returns:
        Formatted prompt for marker validation.
    """
    genes_str = ", ".join(gene_list)

    context_block = ""
    if all_cell_types:
        other_types = [t for t in all_cell_types if t != cell_type]
        context_block = f"""
The full set of cell types being distinguished in this experiment: {", ".join(all_cell_types)}
Other types present: {", ".join(other_types)}
A gene should be "confirmed"/"likely" ONLY if it helps distinguish "{cell_type}" from the other types above.
Genes that are markers for one of the other types should be classified as "unlikely".
Genes that are differentially expressed in this tissue/condition but not specific to "{cell_type}" should be "unlikely".
"""

    prompt = f"""You are a molecular biology expert specializing in cell type markers. Evaluate whether each of the following genes is a reliable marker for "{cell_type}" in the context of {tissue_context}.
{context_block}
Candidate markers: {genes_str}

For each gene, classify as:
- "confirmed": well-established marker specifically for this cell type in this tissue context
- "likely": strong evidence of being specific to this cell type, or functionally characteristic
- "unlikely": this gene is more associated with other cell types, or is a generic tissue/disease marker
- "housekeeping": ubiquitously expressed, not cell-type-specific

Return a JSON object mapping gene symbols to classifications:
{{
  "GENE1": "confirmed",
  "GENE2": "likely",
  ...
}}

Only return the JSON object, no other text."""

    return prompt


def build_panel_selection_prompt(
    cell_type: str,
    gene_panel: List[str],
    tissue_context: str,
    all_cell_types: List[str],
    n_markers: int = 20,
) -> str:
    """Ask LLM to select best markers for a cell type from a gene panel.

    Used as a fallback when canonical marker databases return too few genes
    (common on small gene panels like Xenium's 313 genes).

    Args:
        cell_type: Target cell type to find markers for.
        gene_panel: Full list of genes available in the panel.
        tissue_context: Tissue/disease context.
        all_cell_types: All cell types being distinguished.
        n_markers: Number of markers to request.

    Returns:
        Formatted prompt for panel-aware marker selection.
    """
    other_types = [t for t in all_cell_types if t != cell_type]
    genes_str = ", ".join(gene_panel)

    prompt = f"""You are a single-cell genomics expert. From the gene panel below, select the {n_markers} genes that best distinguish "{cell_type}" from the other cell types: {", ".join(other_types)}.

Tissue context: {tissue_context}

Gene panel ({len(gene_panel)} genes):
{genes_str}

Return ONLY a JSON array of gene symbols, ordered by specificity:
["GENE1", "GENE2", ...]"""
    return prompt


def build_subtype_prompt(
    parent_type: str,
    cluster_markers: Dict[str, List[str]],
    unmatched_types: List[str],
    tissue_context: str,
) -> str:
    """Ask LLM to distinguish subtypes among over-merged clusters.

    When multiple clusters are assigned the same label, this prompt asks the LLM
    to check if any can be refined to a more specific subtype from the expected types.

    Args:
        parent_type: The type label shared by the over-merged clusters.
        cluster_markers: Dict mapping cluster_id -> DE gene list for each merged cluster.
        unmatched_types: Expected cell types not yet assigned to any cluster.
        tissue_context: Tissue/disease context.

    Returns:
        Formatted prompt for subtype refinement.
    """
    cluster_block = "\n".join(
        f"Cluster {cid}: {', '.join(genes[:15])}"
        for cid, genes in cluster_markers.items()
    )
    types_str = ", ".join(unmatched_types)

    prompt = f"""These clusters were all initially labeled as "{parent_type}" in {tissue_context}:

{cluster_block}

The following cell subtypes are expected but have not been assigned to any cluster: {types_str}

Can any of these clusters be re-assigned to one of the unmatched subtypes? Consider:
- Proliferating variants (MKI67, TOP2A markers)
- Phenotypic subtypes (different marker combinations)
- Hybrid/transitional states

Return a JSON object mapping cluster IDs to revised annotations. Only include clusters you want to re-annotate:
{{"cluster_id": "NewTypeName", ...}}
If no re-annotation is warranted, return an empty object: {{}}"""
    return prompt

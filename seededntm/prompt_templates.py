"""Structured prompt templates for LLM-assisted seed gene annotation.

Generates copy-paste-ready prompts for various tissues/organisms,
and parses structured JSON responses from any LLM.
"""

ANNOTATION_PROMPT_TEMPLATE = """\
You are a computational biology expert. I have performed unsupervised clustering \
(Leiden algorithm) on spatial transcriptomics data from {tissue_description}.

Below are the top differentially expressed marker genes for each cluster, \
ranked by statistical significance (Wilcoxon test, adjusted p < 0.01, logFC > 0.5).

{marker_table}

**Task**: For each cluster, assign the most likely cell type identity based on \
the marker genes. Consider the tissue context ({tissue_description}) when making \
assignments.

**Instructions**:
1. Assign a specific cell type name to each cluster (e.g., "Macrophages", "CD8+_T_Cells", "Tumor_Epithelial")
2. Use underscores instead of spaces in cell type names
3. If two clusters appear to be the same cell type, give them the same name (they will be merged)
4. Provide a confidence score (0.0 to 1.0) for each assignment
5. If a cluster's identity is truly ambiguous, assign "Unknown" with low confidence

**Respond in this exact JSON format** (no other text before or after):
```json
{{
  "0": {{"cell_type": "Example_Type_1", "confidence": 0.95, "reasoning": "brief explanation"}},
  "1": {{"cell_type": "Example_Type_2", "confidence": 0.85, "reasoning": "brief explanation"}},
  ...
}}
```

{additional_context}"""

FOLLOWUP_PROMPT_TEMPLATE = """\
Thank you for the previous annotation. Some clusters had low confidence or were \
marked as Unknown. Here is additional information for re-annotation:

{additional_info}

Previous assignments that need revision:
{low_confidence_clusters}

Please provide updated annotations for ONLY the clusters listed above, \
using the same JSON format:
```json
{{
  "cluster_id": {{"cell_type": "Revised_Name", "confidence": 0.9, "reasoning": "..."}},
  ...
}}
```"""

SUBCLUSTERING_PROMPT_TEMPLATE = """\
Cluster {cluster_id} (previously annotated as "{previous_type}") was subclustered. \
Here are the marker genes for each subcluster:

{subcluster_markers}

Please assign cell type identities to each subcluster:
```json
{{
  "{cluster_id}_0": {{"cell_type": "...", "confidence": ..., "reasoning": "..."}},
  "{cluster_id}_1": {{"cell_type": "...", "confidence": ..., "reasoning": "..."}},
  ...
}}
```"""


def format_marker_table(markers_per_cluster: dict, top_n: int = 20) -> str:
    """Format marker genes as a readable table for the LLM prompt.

    Args:
        markers_per_cluster: dict mapping cluster_id -> list of (gene, score) tuples
            or cluster_id -> list of gene names.
        top_n: maximum markers to show per cluster.

    Returns:
        Formatted string table.
    """
    lines = []
    for cluster_id in sorted(markers_per_cluster.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
        genes = markers_per_cluster[cluster_id]
        if isinstance(genes[0], (tuple, list)):
            gene_names = [g[0] for g in genes[:top_n]]
        else:
            gene_names = list(genes[:top_n])
        lines.append(f"Cluster {cluster_id}: {', '.join(gene_names)}")
    return "\n".join(lines)


def build_annotation_prompt(
    markers_per_cluster: dict,
    tissue_description: str,
    top_n: int = 20,
    additional_context: str = "",
) -> str:
    """Build a complete annotation prompt ready for copy-paste into any LLM.

    Args:
        markers_per_cluster: dict mapping cluster_id -> list of genes or (gene, score) tuples.
        tissue_description: e.g. "human breast cancer tissue (Xenium panel, 313 genes)"
        top_n: markers per cluster to include.
        additional_context: optional extra instructions.

    Returns:
        Complete prompt string.
    """
    marker_table = format_marker_table(markers_per_cluster, top_n)
    return ANNOTATION_PROMPT_TEMPLATE.format(
        tissue_description=tissue_description,
        marker_table=marker_table,
        additional_context=additional_context,
    )


def build_followup_prompt(
    low_confidence_clusters: dict,
    additional_markers: dict = None,
) -> str:
    """Build a follow-up prompt for clusters that need re-annotation.

    Args:
        low_confidence_clusters: dict mapping cluster_id -> {"cell_type": ..., "confidence": ...}
        additional_markers: optional dict mapping cluster_id -> additional markers.
    """
    cluster_lines = []
    for cid, info in low_confidence_clusters.items():
        line = f"  Cluster {cid}: previously \"{info['cell_type']}\" (confidence={info['confidence']:.2f})"
        cluster_lines.append(line)

    additional_info = ""
    if additional_markers:
        info_lines = []
        for cid, markers in additional_markers.items():
            if isinstance(markers[0], (tuple, list)):
                gene_names = [g[0] for g in markers]
            else:
                gene_names = list(markers)
            info_lines.append(f"  Cluster {cid} additional markers: {', '.join(gene_names)}")
        additional_info = "\n".join(info_lines)

    return FOLLOWUP_PROMPT_TEMPLATE.format(
        additional_info=additional_info or "(no additional markers)",
        low_confidence_clusters="\n".join(cluster_lines),
    )

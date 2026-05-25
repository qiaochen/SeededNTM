"""LLM prompts for gene extraction, validation, and nomenclature resolution."""

SYSTEM_MARKER_EXTRACTION = (
    "You are a molecular biology and spatial transcriptomics expert. "
    "Your task is to identify marker genes for specific cell types from "
    "scientific literature and database results. Be precise and only include "
    "genes with strong evidence of cell-type specificity."
)

PROMPT_EXTRACT_MARKERS = """Given the following information about a cell type in a specific tissue context, identify the most reliable marker genes.

Cell type: {cell_type}
Tissue: {tissue}
Species: {species}
Condition: {condition}

Database results:
{db_results}

Instructions:
1. Identify genes that are specifically expressed in {cell_type}
2. Prioritize genes that distinguish this type from other cell types in {tissue}
3. Consider both positive markers (highly expressed) and the absence of lineage markers
4. Return ONLY a JSON list of gene symbols, ordered by confidence

Return format: ["GENE1", "GENE2", ...]
"""

SYSTEM_NOMENCLATURE = (
    "You are an expert in cell biology nomenclature and ontology. "
    "Help resolve cell type names to their canonical forms and synonyms "
    "for database queries."
)

PROMPT_RESOLVE_NOMENCLATURE = """Resolve the following cell type name into canonical forms and synonyms that different biological databases might use.

Cell type: {cell_type}
Tissue context: {tissue}
Condition: {condition}

Consider:
- CL ontology terms
- Common abbreviations (e.g., CAF = Cancer-associated fibroblasts)
- Database-specific naming conventions
- Parent/child cell type relationships

Return a JSON object with:
{{
    "canonical": "Full canonical name",
    "synonyms": ["syn1", "syn2", ...],
    "parent_types": ["parent1", ...],
    "cl_id": "CL:XXXXXXX" (if known)
}}
"""

SYSTEM_VALIDATION = (
    "You are a molecular biology expert validating marker gene candidates. "
    "Assess whether genes are genuinely specific markers for a cell type."
)

PROMPT_VALIDATE_MARKERS = """Validate the following candidate marker genes for {cell_type} in {tissue}.

Candidate genes: {genes}

For each gene, assess:
1. Is this gene specifically expressed in {cell_type}?
2. Is it commonly used as a marker in the literature?
3. Could it be a marker for a different cell type instead?
4. Is it a housekeeping gene (not specific)?

Gene annotations:
{annotations}

Return a JSON object mapping gene symbols to validation scores (0.0-1.0):
{{"GENE1": 0.9, "GENE2": 0.3, ...}}

Only include genes with score >= 0.5.
"""

PROMPT_QUERY_STRATEGY = """I need to find marker genes for the following cell types in a spatial transcriptomics experiment.

Dataset context:
- Species: {species}
- Tissue: {tissue}
- Organ: {organ}
- Condition: {condition}
- Cell types to resolve: {cell_types}

Available gene panel (spatial platform): {n_panel_genes} genes

Determine the optimal search strategy:
1. Which databases should be queried for each cell type?
2. Should we include disease-gene databases? (condition: {condition})
3. Are there cell types that are ambiguous and need nomenclature resolution?
4. What search terms should we use for each database?

Return a JSON object with the strategy:
{{
    "cell_types": {{
        "TypeName": {{
            "search_terms": ["term1", "term2"],
            "priority_sources": ["source1", "source2"],
            "needs_resolution": true/false,
            "include_disease_dbs": true/false
        }}
    }},
    "global_settings": {{
        "include_disease_genes": true/false,
        "broaden_search": true/false
    }}
}}
"""

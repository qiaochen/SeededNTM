"""Data classes for the Marker Search Agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DatasetContext:
    """Context describing the spatial transcriptomics dataset for marker search."""

    species: str = "Human"
    tissue: str = ""
    organ: str = ""
    condition: str = ""
    gene_panel: List[str] = field(default_factory=list)
    expected_cell_types: List[str] = field(default_factory=list)
    source_paper: str = ""
    n_markers_per_type: int = 20
    include_disease_genes: bool = False


@dataclass
class MarkerHit:
    """A single marker gene hit from a biological database."""

    gene_symbol: str
    cell_type: str
    source: str
    score: float = 1.0
    evidence_detail: str = ""


@dataclass
class GeneAnnotation:
    """Functional annotation for a gene from MyGene/NCBI."""

    symbol: str
    summary: str = ""
    go_terms: List[str] = field(default_factory=list)
    generif_texts: List[str] = field(default_factory=list)
    pathways: List[str] = field(default_factory=list)


@dataclass
class ScoredMarker:
    """A marker gene with composite score from multiple sources."""

    gene_symbol: str
    cell_type: str
    total_score: float
    source_scores: Dict[str, float] = field(default_factory=dict)
    in_panel: bool = False

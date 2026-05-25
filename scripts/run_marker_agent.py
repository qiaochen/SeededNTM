#!/usr/bin/env python
"""CLI entry point for the Marker Search Agent pipeline.

Usage:
    python scripts/run_marker_agent.py \
        --h5ad spatial.h5ad \
        --species Human \
        --tissue "Colorectal cancer" \
        --organ Colon \
        --condition "tumor microenvironment" \
        --expected-types "CAF,Tumor,Endothelial,Macrophage,Pericytes,Neutrophil" \
        --output seeds_agent.json \
        --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Marker Search Agent — multi-source marker gene discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--h5ad",
        type=str,
        default=None,
        help="Path to spatial h5ad file (to extract gene panel from .var_names)",
    )
    parser.add_argument(
        "--gene-panel",
        type=str,
        default=None,
        help="Path to text file with gene panel (one gene per line), or comma-separated list",
    )
    parser.add_argument("--species", type=str, default="Human")
    parser.add_argument("--tissue", type=str, required=True)
    parser.add_argument("--organ", type=str, default="")
    parser.add_argument("--condition", type=str, default="")
    parser.add_argument(
        "--expected-types",
        type=str,
        required=True,
        help="Comma-separated list of expected cell types",
    )
    parser.add_argument(
        "--n-markers", type=int, default=20,
        help="Number of markers per cell type to return",
    )
    parser.add_argument("--output", type=str, default="seeds_agent.json")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["seeds", "evidence"],
        default="seeds",
        help="Output format: 'seeds' for simple gene lists, 'evidence' for detailed",
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM features (nomenclature resolution, PubMed extraction)",
    )
    parser.add_argument(
        "--include-disease", action="store_true",
        help="Include disease-gene databases even without explicit condition",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--save-provenance", action="store_true", default=True,
        help="Save provenance JSON alongside seeds output (default: True)",
    )
    parser.add_argument(
        "--no-provenance", action="store_true",
        help="Disable provenance logging",
    )

    return parser.parse_args()


def load_gene_panel(h5ad_path: str = None, panel_path: str = None) -> list:
    """Load gene panel from h5ad or text file."""
    if h5ad_path:
        import scanpy as sc
        adata = sc.read_h5ad(h5ad_path, backed="r")
        genes = list(adata.var_names)
        adata.file.close()
        return genes

    if panel_path:
        path = Path(panel_path)
        if path.exists():
            with open(path) as f:
                return [line.strip() for line in f if line.strip()]
        else:
            return [g.strip() for g in panel_path.split(",") if g.strip()]

    return []


def _derive_provenance_path(output_path: str) -> str:
    """Derive provenance file path from the main output path."""
    p = Path(output_path)
    return str(p.parent / f"{p.stem}_provenance.json")


def main():
    args = parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    gene_panel = load_gene_panel(args.h5ad, args.gene_panel)
    if not gene_panel:
        logging.warning(
            "No gene panel provided. Results will not be filtered by panel membership."
        )

    expected_types = [t.strip() for t in args.expected_types.split(",")]

    from seededntm.marker_agent.schemas import DatasetContext
    from seededntm.marker_agent.agent import MarkerSearchAgent

    context = DatasetContext(
        species=args.species,
        tissue=args.tissue,
        organ=args.organ,
        condition=args.condition,
        gene_panel=gene_panel,
        expected_cell_types=expected_types,
        n_markers_per_type=args.n_markers,
        include_disease_genes=args.include_disease,
    )

    agent = MarkerSearchAgent(
        context=context,
        cache_dir=args.cache_dir,
        use_llm=not args.no_llm,
        verbose=args.verbose,
    )

    print(f"\n{'='*60}")
    print("MARKER SEARCH AGENT")
    print(f"{'='*60}")
    print(f"Species:    {args.species}")
    print(f"Tissue:     {args.tissue}")
    print(f"Organ:      {args.organ}")
    print(f"Condition:  {args.condition}")
    print(f"Cell types: {expected_types}")
    print(f"Panel size: {len(gene_panel)} genes")
    print(f"Output:     {args.output}")
    print(f"{'='*60}\n")

    results = agent.run(
        save_provenance=not args.no_provenance,
        provenance_path=_derive_provenance_path(args.output) if not args.no_provenance else None,
    )

    agent.save_results(results, args.output, format=args.output_format)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for cell_type, markers in results.items():
        genes = [m.gene_symbol for m in markers[:10]]
        print(f"  {cell_type}: {len(markers)} markers")
        if genes:
            print(f"    Top: {', '.join(genes)}")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output}")
    if not args.no_provenance:
        prov_path = _derive_provenance_path(args.output)
        print(f"Provenance log: {prov_path}")


if __name__ == "__main__":
    main()

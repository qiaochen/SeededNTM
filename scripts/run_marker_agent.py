#!/usr/bin/env python
"""CLI entry point for the Marker Search Agent pipeline.

Supports three modes:
  - agent:  Pure marker agent (literature-driven, parallel source queries)
  - leiden: Automated Leiden-DE-LLM pipeline (data-driven)
  - hybrid: Combined pipeline (both + HybridCombiner fusion)

Usage:
    # Pure marker agent (default, parallel)
    python scripts/run_marker_agent.py \
        --h5ad spatial.h5ad \
        --tissue "Colorectal cancer" \
        --expected-types "CAF,Tumor,Endothelial" \
        --output seeds_agent.json

    # Pure Leiden-ChatGPT (automated)
    python scripts/run_marker_agent.py \
        --mode leiden \
        --h5ad spatial.h5ad \
        --tissue-desc "colorectal cancer tumor microenvironment" \
        --output seeds_leiden.json

    # Hybrid (recommended)
    python scripts/run_marker_agent.py \
        --mode hybrid \
        --h5ad spatial.h5ad \
        --tissue "Colorectal cancer" \
        --tissue-desc "colorectal cancer tumor microenvironment" \
        --expected-types "CAF,Tumor,Endothelial" \
        --output seeds_hybrid.json
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
        "--mode",
        type=str,
        choices=["agent", "leiden", "hybrid"],
        default="agent",
        help="Pipeline mode: 'agent' (literature), 'leiden' (data-driven), 'hybrid' (both)",
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
    parser.add_argument("--tissue", type=str, default="")
    parser.add_argument("--organ", type=str, default="")
    parser.add_argument("--condition", type=str, default="")
    parser.add_argument(
        "--tissue-desc",
        type=str,
        default="",
        help="Tissue description for Leiden-LLM annotation (e.g., 'colorectal cancer TME')",
    )
    parser.add_argument(
        "--expected-types",
        type=str,
        default="",
        help="Comma-separated list of expected cell types (required for agent/hybrid modes)",
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
    parser.add_argument(
        "--parallel", action="store_true", default=True,
        help="Enable parallel source queries (default: True)",
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable parallel source queries (sequential mode for debugging)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Max worker threads for parallel queries",
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


def run_agent_mode(args, gene_panel, expected_types):
    """Run pure marker agent pipeline."""
    from seededntm.marker_agent.schemas import DatasetContext
    from seededntm.marker_agent.agent import MarkerSearchAgent

    if not expected_types:
        print("ERROR: --expected-types is required for agent mode", file=sys.stderr)
        sys.exit(1)

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

    use_parallel = args.parallel and not args.no_parallel
    agent = MarkerSearchAgent(
        context=context,
        cache_dir=args.cache_dir,
        use_llm=not args.no_llm,
        verbose=args.verbose,
        parallel=use_parallel,
        max_workers=args.max_workers,
    )

    results = agent.run(
        save_provenance=not args.no_provenance,
        provenance_path=_derive_provenance_path(args.output) if not args.no_provenance else None,
    )

    agent.save_results(results, args.output, format=args.output_format)
    return results


def run_leiden_mode(args, gene_panel):
    """Run Leiden-DE-LLM pipeline."""
    import scanpy as sc
    from seededntm.seed_construction import SeedConstructionPipeline

    if not args.h5ad:
        print("ERROR: --h5ad is required for leiden mode", file=sys.stderr)
        sys.exit(1)

    tissue_desc = args.tissue_desc or args.tissue or "unspecified tissue"

    print(f"Loading h5ad: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)

    pipeline = SeedConstructionPipeline(
        adata=adata,
        tissue_description=tissue_desc,
        top_n_genes=args.n_markers,
    )

    seeds = pipeline.run()

    output_data = {}
    for cell_type, info in seeds.items():
        output_data[cell_type] = info["features"]

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    if not args.no_provenance:
        prov_path = _derive_provenance_path(args.output)
        provenance = pipeline.get_provenance()
        with open(prov_path, "w") as f:
            json.dump(provenance, f, indent=2, default=str)
        print(f"Provenance log: {prov_path}")

    return seeds


def run_hybrid_mode(args, gene_panel, expected_types):
    """Run hybrid pipeline (Leiden + Agent + Combiner)."""
    import scanpy as sc
    from seededntm.seed_construction import SeedConstructionPipeline
    from seededntm.marker_agent.schemas import DatasetContext
    from seededntm.marker_agent.agent import MarkerSearchAgent
    from seededntm.marker_agent.combiner import HybridCombiner

    if not args.h5ad:
        print("ERROR: --h5ad is required for hybrid mode", file=sys.stderr)
        sys.exit(1)

    tissue_desc = args.tissue_desc or args.tissue or "unspecified tissue"

    print(f"\n{'='*60}")
    print("HYBRID MODE: Leiden + Agent + Combiner")
    print(f"{'='*60}")

    print("\n[1/3] Running Leiden-DE-LLM pipeline...")
    adata = sc.read_h5ad(args.h5ad)
    leiden_pipeline = SeedConstructionPipeline(
        adata=adata,
        tissue_description=tissue_desc,
        top_n_genes=args.n_markers,
    )
    leiden_seeds = leiden_pipeline.run()

    cell_types_from_leiden = list(leiden_seeds.keys())
    if expected_types:
        final_types = expected_types
    else:
        final_types = cell_types_from_leiden

    print(f"\n[2/3] Running Marker Agent (parallel={not args.no_parallel})...")
    context = DatasetContext(
        species=args.species,
        tissue=args.tissue,
        organ=args.organ,
        condition=args.condition,
        gene_panel=gene_panel,
        expected_cell_types=final_types,
        n_markers_per_type=args.n_markers,
        include_disease_genes=args.include_disease,
    )

    use_parallel = args.parallel and not args.no_parallel
    agent = MarkerSearchAgent(
        context=context,
        cache_dir=args.cache_dir,
        use_llm=not args.no_llm,
        verbose=args.verbose,
        parallel=use_parallel,
        max_workers=args.max_workers,
    )
    agent_results = agent.run()
    agent_seeds = agent.to_seed_json(agent_results)

    agent_scores = {}
    for ct, markers in agent_results.items():
        agent_scores[ct] = {m.gene_symbol: m.total_score for m in markers}

    print("\n[3/3] Combining seeds (HybridCombiner)...")
    combiner = HybridCombiner(
        top_n=args.n_markers,
        gene_panel=gene_panel if gene_panel else None,
    )
    hybrid_seeds = combiner.combine(leiden_seeds, agent_seeds, agent_scores)

    output_data = {}
    for cell_type, info in hybrid_seeds.items():
        output_data[cell_type] = info["features"]

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    if not args.no_provenance:
        prov_path = _derive_provenance_path(args.output)
        provenance = {
            "mode": "hybrid",
            "leiden_provenance": leiden_pipeline.get_provenance(),
            "hybrid_output": {ct: info for ct, info in hybrid_seeds.items()},
        }
        with open(prov_path, "w") as f:
            json.dump(provenance, f, indent=2, default=str)
        print(f"Provenance log: {prov_path}")

    return hybrid_seeds


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

    expected_types = (
        [t.strip() for t in args.expected_types.split(",") if t.strip()]
        if args.expected_types else []
    )

    print(f"\n{'='*60}")
    print(f"MARKER SEARCH AGENT — Mode: {args.mode.upper()}")
    print(f"{'='*60}")
    print(f"Species:    {args.species}")
    print(f"Tissue:     {args.tissue}")
    print(f"Organ:      {args.organ}")
    print(f"Condition:  {args.condition}")
    if expected_types:
        print(f"Cell types: {expected_types}")
    print(f"Panel size: {len(gene_panel)} genes")
    print(f"Output:     {args.output}")
    print(f"Parallel:   {args.parallel and not args.no_parallel}")
    print(f"{'='*60}\n")

    if args.mode == "agent":
        results = run_agent_mode(args, gene_panel, expected_types)
    elif args.mode == "leiden":
        results = run_leiden_mode(args, gene_panel)
    elif args.mode == "hybrid":
        results = run_hybrid_mode(args, gene_panel, expected_types)
    else:
        print(f"ERROR: Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    if isinstance(results, dict):
        for cell_type, value in results.items():
            if isinstance(value, list):
                genes = value[:10] if all(isinstance(g, str) for g in value[:1]) else []
                if not genes and value:
                    genes = [getattr(m, "gene_symbol", str(m)) for m in value[:10]]
                print(f"  {cell_type}: {len(value)} markers")
                if genes:
                    print(f"    Top: {', '.join(genes[:10])}")
            elif isinstance(value, dict):
                features = value.get("features", [])
                print(f"  {cell_type}: {len(features)} markers")
                if features:
                    print(f"    Top: {', '.join(features[:10])}")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output}")
    if not args.no_provenance:
        print(f"Provenance log: {_derive_provenance_path(args.output)}")


if __name__ == "__main__":
    main()

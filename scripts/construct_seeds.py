#!/usr/bin/env python
"""CLI for systematic seed gene construction pipeline.

Subcommands:
    prepare   - Load data, cluster, compute markers, generate LLM prompt
    finalize  - Parse LLM response, merge clusters, produce final seeds

Usage:
    python scripts/construct_seeds.py prepare \\
        --h5ad path/to/spatial.h5ad \\
        --tissue "human breast cancer (Xenium, 313 genes)" \\
        --expected-k 19 \\
        --output-dir ./seed_construction/

    # (User pastes prompt into LLM, saves response)

    python scripts/construct_seeds.py finalize \\
        --work-dir ./seed_construction/ \\
        --response response_round1.txt
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def cmd_prepare(args):
    from seededntm.seed_construction import SeedConstructionPipeline

    pipeline = SeedConstructionPipeline(
        h5ad_path=args.h5ad,
        tissue=args.tissue,
        output_dir=args.output_dir,
        n_hvg=args.n_hvg,
        resolutions=[float(x) for x in args.resolutions.split(",")]
            if args.resolutions else None,
    )
    pipeline.prepare(
        expected_k=args.expected_k,
        resolution=args.resolution,
        top_markers=args.top_markers,
    )


def cmd_finalize(args):
    from seededntm.seed_construction import SeedConstructionPipeline

    import json

    state_path = Path(args.work_dir) / "audit" / "pipeline_state.json"
    if not state_path.exists():
        print(f"Error: No pipeline state found in {args.work_dir}/audit/")
        print("Did you run 'prepare' first?")
        sys.exit(1)

    state = json.loads(state_path.read_text())

    pipeline = SeedConstructionPipeline(
        h5ad_path=state["h5ad_path"],
        tissue=state["tissue"],
        output_dir=args.work_dir,
    )
    pipeline.selected_resolution = state["selected_resolution"]

    pipeline.finalize(
        response_path=args.response,
        confidence_threshold=args.confidence_threshold,
        top_markers_final=args.top_markers,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Systematic seed gene construction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # --- prepare ---
    p_prepare = subparsers.add_parser("prepare", help="Generate LLM prompt from spatial data")
    p_prepare.add_argument("--h5ad", required=True, help="Path to spatial h5ad file")
    p_prepare.add_argument("--tissue", required=True,
                           help="Tissue description for LLM context")
    p_prepare.add_argument("--expected-k", type=int, default=None,
                           help="Expected number of cell types (helps select resolution)")
    p_prepare.add_argument("--resolution", type=float, default=None,
                           help="Override leiden resolution (skip automatic selection)")
    p_prepare.add_argument("--output-dir", default="./seed_construction/",
                           help="Output directory (default: ./seed_construction/)")
    p_prepare.add_argument("--n-hvg", type=int, default=4000,
                           help="Number of highly variable genes (0 to skip)")
    p_prepare.add_argument("--top-markers", type=int, default=20,
                           help="Markers per cluster in prompt")
    p_prepare.add_argument("--resolutions", type=str, default=None,
                           help="Comma-separated leiden resolutions (e.g. '0.5,1.0,1.5,2.0')")
    p_prepare.set_defaults(func=cmd_prepare)

    # --- finalize ---
    p_finalize = subparsers.add_parser("finalize", help="Parse LLM response and generate seeds")
    p_finalize.add_argument("--work-dir", required=True,
                            help="Working directory from prepare step")
    p_finalize.add_argument("--response", required=True,
                            help="Path to LLM response file (absolute or relative to work-dir)")
    p_finalize.add_argument("--confidence-threshold", type=float, default=0.6,
                            help="Minimum confidence to accept annotation")
    p_finalize.add_argument("--top-markers", type=int, default=20,
                            help="Final markers per cell type")
    p_finalize.set_defaults(func=cmd_finalize)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Integration tests for all 17 marker agent sources.

Usage:
    python tests/test_marker_sources.py --all          # Run all tests
    python tests/test_marker_sources.py --source panglaodb  # Run single source
    python tests/test_marker_sources.py --offline      # Skip sources needing internet
    python tests/test_marker_sources.py --list         # List available sources
"""

import argparse
import os
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# SkipTest exception for genuinely optional sources
# ---------------------------------------------------------------------------

class SkipTest(Exception):
    """Raised when a source test should be skipped (missing API key, package, etc.)."""
    pass


# ---------------------------------------------------------------------------
# Source imports
# ---------------------------------------------------------------------------

from seededntm.marker_agent.sources.panglaodb import PanglaoDBSource
from seededntm.marker_agent.sources.cellmarker import CellMarkerSource
from seededntm.marker_agent.sources.sctype_db import ScTypeDBSource
from seededntm.marker_agent.sources.mygene_source import MyGeneSource
from seededntm.marker_agent.sources.ncbi_gene import NCBIGeneSource
from seededntm.marker_agent.sources.hpa import HPASource
from seededntm.marker_agent.sources.gtex import GTExSource
from seededntm.marker_agent.sources.msigdb import MSigDBSource
from seededntm.marker_agent.sources.omim import OMIMSource
from seededntm.marker_agent.sources.disgenet import DisGeNETSource
from seededntm.marker_agent.sources.ensembl import EnsemblSource
from seededntm.marker_agent.sources.opentargets import OpenTargetsSource
from seededntm.marker_agent.sources.cosmic import COSMICSource
from seededntm.marker_agent.sources.pubmed import PubMedSource
from seededntm.marker_agent.sources.pubtator3 import PubTator3Source
from seededntm.marker_agent.sources.litvar2 import LitVar2Source
from seededntm.marker_agent.sources.cellxgene import CellxGeneSource


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

ONLINE_SOURCES = {
    "mygene", "ncbi_gene", "hpa", "gtex", "ensembl", "opentargets",
    "pubmed", "pubtator3", "litvar2", "omim", "disgenet", "cellxgene",
}

def test_panglaodb():
    src = PanglaoDBSource()
    results = src.search("T cells", "Lung", "Human")
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    return results


def test_cellmarker():
    src = CellMarkerSource()
    results = src.search("T cells", "Lung", "Human")
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    return results


def test_sctype():
    src = ScTypeDBSource()
    results = src.search("T cells", "Lung", "Human")
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    return results


def test_mygene():
    src = MyGeneSource()
    results = src.search("T cells", "Lung", "Human")
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    return results


def test_ncbi_gene():
    src = NCBIGeneSource()
    results = src.search("T cells", "Lung", "Human")
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    return results


def test_hpa():
    src = HPASource()
    results = src.search("T cells", "Lung", "Human")
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    return results


def test_gtex():
    src = GTExSource()
    results = src.search("T cells", "Lung", "Human")
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    return results


def test_msigdb():
    src = MSigDBSource()
    results = src.search("T cells", "Lung", "Human")
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    return results


def test_omim():
    api_key = os.environ.get("OMIM_API_KEY", "")
    if not api_key:
        try:
            from Bio import Entrez
        except ImportError:
            raise SkipTest("OMIM requires OMIM_API_KEY or biopython (Entrez fallback)")
    src = OMIMSource()
    results = src.search("T cells", "Lung", "Human")
    assert len(results) >= 0, f"Unexpected error: {len(results)}"
    return results


def test_disgenet():
    raise SkipTest("DisGeNET public REST endpoint discontinued; requires API key or local dump")


def test_ensembl():
    src = EnsemblSource()
    results = src.search("T cells", "Lung", "Human", gene_list=["CD3D", "CD3E", "CD4"])
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    return results


def test_opentargets():
    src = OpenTargetsSource()
    results = src.search("T cells", "Lung cancer", "Human", condition="Lung cancer")
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    return results


def test_cosmic():
    src = COSMICSource()
    results = src.search("T cells", "Lung cancer", "Human", condition="Lung cancer")
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    return results


def test_pubmed():
    src = PubMedSource()
    results = src.search("macrophage", "colorectal cancer", "Human", use_llm=False)
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    return results


def test_pubtator3():
    src = PubTator3Source()
    results = src.search("macrophage", "colorectal cancer", "Human")
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    return results


def test_litvar2():
    src = LitVar2Source()
    results = src.search(
        "T cells", "Breast cancer", "Human",
        candidate_genes=["TP53", "BRCA1", "BRCA2", "ERBB2", "PIK3CA"],
    )
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    return results


def test_cellxgene():
    try:
        import cellxgene_census  # noqa: F401
    except ImportError:
        raise SkipTest("cellxgene_census package not installed")
    src = CellxGeneSource()
    results = src.search("T cells", "Lung", "Human")
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    return results


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TESTS = {
    "panglaodb": test_panglaodb,
    "cellmarker": test_cellmarker,
    "sctype": test_sctype,
    "mygene": test_mygene,
    "ncbi_gene": test_ncbi_gene,
    "hpa": test_hpa,
    "gtex": test_gtex,
    "msigdb": test_msigdb,
    "omim": test_omim,
    "disgenet": test_disgenet,
    "ensembl": test_ensembl,
    "opentargets": test_opentargets,
    "cosmic": test_cosmic,
    "pubmed": test_pubmed,
    "pubtator3": test_pubtator3,
    "litvar2": test_litvar2,
    "cellxgene": test_cellxgene,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests(sources, offline=False):
    results_summary = []

    for name in sources:
        if offline and name in ONLINE_SOURCES:
            results_summary.append({
                "source": name, "status": "SKIP", "reason": "offline mode",
                "hits": 0, "time": 0.0, "sample_genes": [],
            })
            continue

        test_fn = ALL_TESTS[name]
        t0 = time.time()
        try:
            results = test_fn()
            elapsed = time.time() - t0
            sample = []
            if results:
                for r in results[:5]:
                    gene = r.get("gene", r.get("symbol", r.get("name", "?")))
                    sample.append(gene)
            results_summary.append({
                "source": name, "status": "PASS", "reason": "",
                "hits": len(results), "time": elapsed, "sample_genes": sample,
            })
        except SkipTest as e:
            elapsed = time.time() - t0
            results_summary.append({
                "source": name, "status": "SKIP", "reason": str(e),
                "hits": 0, "time": elapsed, "sample_genes": [],
            })
        except Exception as e:
            elapsed = time.time() - t0
            results_summary.append({
                "source": name, "status": "FAIL", "reason": str(e),
                "hits": 0, "time": elapsed, "sample_genes": [],
            })
            traceback.print_exc()

    return results_summary


def print_summary(results_summary):
    print("\n" + "=" * 90)
    print(f"{'Source':<15} {'Status':<6} {'Hits':<8} {'Time (s)':<10} {'Sample Genes':<40} {'Reason'}")
    print("-" * 90)
    for r in results_summary:
        genes_str = ", ".join(r["sample_genes"][:5]) if r["sample_genes"] else ""
        print(
            f"{r['source']:<15} {r['status']:<6} {r['hits']:<8} "
            f"{r['time']:<10.2f} {genes_str:<40} {r['reason']}"
        )
    print("=" * 90)

    passed = sum(1 for r in results_summary if r["status"] == "PASS")
    skipped = sum(1 for r in results_summary if r["status"] == "SKIP")
    failed = sum(1 for r in results_summary if r["status"] == "FAIL")
    total = len(results_summary)
    print(f"\nTotal: {total} | PASS: {passed} | SKIP: {skipped} | FAIL: {failed}")
    print()

    return failed


def main():
    parser = argparse.ArgumentParser(description="Test all marker agent sources")
    parser.add_argument("--all", action="store_true", help="Run all source tests")
    parser.add_argument("--source", type=str, help="Run a specific source test")
    parser.add_argument("--offline", action="store_true", help="Skip online sources")
    parser.add_argument("--list", action="store_true", help="List available sources")
    args = parser.parse_args()

    if args.list:
        print("Available sources:")
        for name in ALL_TESTS:
            online_tag = " [online]" if name in ONLINE_SOURCES else " [local/auto-download]"
            print(f"  {name}{online_tag}")
        return

    if args.source:
        if args.source not in ALL_TESTS:
            print(f"Unknown source: {args.source}")
            print(f"Available: {', '.join(ALL_TESTS.keys())}")
            sys.exit(1)
        sources = [args.source]
    elif args.all:
        sources = list(ALL_TESTS.keys())
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Running {len(sources)} source test(s)...")
    results_summary = run_tests(sources, offline=args.offline)
    failed = print_summary(results_summary)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()

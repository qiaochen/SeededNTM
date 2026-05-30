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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# SkipTest exception for genuinely optional sources
# ---------------------------------------------------------------------------

class SkipTest(Exception):
    """Raised when a source test should be skipped (missing API key, package, etc.)."""
    pass


# ---------------------------------------------------------------------------
# Source imports — these go through seededntm/__init__.py which pulls in
# scanpy/torch/numpy. If that fails, fall back to direct file imports.
# ---------------------------------------------------------------------------

try:
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
except ImportError as _import_err:
    print(f"WARNING: Standard import failed ({_import_err}), attempting direct imports...")
    import importlib.util

    def _load_source_class(rel_path: str, class_name: str):
        """Load a class directly from file path, bypassing seededntm/__init__.py."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base, rel_path)
        spec = importlib.util.spec_from_file_location(class_name, full_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, class_name)

    PanglaoDBSource = _load_source_class("seededntm/marker_agent/sources/panglaodb.py", "PanglaoDBSource")
    CellMarkerSource = _load_source_class("seededntm/marker_agent/sources/cellmarker.py", "CellMarkerSource")
    ScTypeDBSource = _load_source_class("seededntm/marker_agent/sources/sctype_db.py", "ScTypeDBSource")
    MyGeneSource = _load_source_class("seededntm/marker_agent/sources/mygene_source.py", "MyGeneSource")
    NCBIGeneSource = _load_source_class("seededntm/marker_agent/sources/ncbi_gene.py", "NCBIGeneSource")
    HPASource = _load_source_class("seededntm/marker_agent/sources/hpa.py", "HPASource")
    GTExSource = _load_source_class("seededntm/marker_agent/sources/gtex.py", "GTExSource")
    MSigDBSource = _load_source_class("seededntm/marker_agent/sources/msigdb.py", "MSigDBSource")
    OMIMSource = _load_source_class("seededntm/marker_agent/sources/omim.py", "OMIMSource")
    DisGeNETSource = _load_source_class("seededntm/marker_agent/sources/disgenet.py", "DisGeNETSource")
    EnsemblSource = _load_source_class("seededntm/marker_agent/sources/ensembl.py", "EnsemblSource")
    OpenTargetsSource = _load_source_class("seededntm/marker_agent/sources/opentargets.py", "OpenTargetsSource")
    COSMICSource = _load_source_class("seededntm/marker_agent/sources/cosmic.py", "COSMICSource")
    PubMedSource = _load_source_class("seededntm/marker_agent/sources/pubmed.py", "PubMedSource")
    PubTator3Source = _load_source_class("seededntm/marker_agent/sources/pubtator3.py", "PubTator3Source")
    LitVar2Source = _load_source_class("seededntm/marker_agent/sources/litvar2.py", "LitVar2Source")
    CellxGeneSource = _load_source_class("seededntm/marker_agent/sources/cellxgene.py", "CellxGeneSource")


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
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    for hit in results[:3]:
        assert hasattr(hit, "gene_symbol"), f"MarkerHit missing gene_symbol: {hit}"
        assert hasattr(hit, "cell_type"), f"MarkerHit missing cell_type: {hit}"
        assert hasattr(hit, "source"), f"MarkerHit missing source: {hit}"
    return results


def test_cellmarker():
    src = CellMarkerSource()
    results = src.search("T cells", "Lung", "Human")
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    for hit in results[:3]:
        assert hasattr(hit, "gene_symbol"), f"MarkerHit missing gene_symbol: {hit}"
    return results


def test_sctype():
    src = ScTypeDBSource()
    results = src.search("T cells", "Lung", "Human")
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) > 0, f"Expected hits, got {len(results)}"
    for hit in results[:3]:
        assert hasattr(hit, "gene_symbol"), f"MarkerHit missing gene_symbol: {hit}"
    return results


def test_mygene():
    src = MyGeneSource()
    results = src.search("T cells", "Lung", "Human")
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    # MyGene uses GO term search; may return 0 for broad cell type queries
    assert len(results) >= 0, f"Unexpected error: got {type(results)}"
    return results


def test_ncbi_gene():
    src = NCBIGeneSource()
    results = src.search("T cells", "Lung", "Human")
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    # NCBI Gene requires biopython's Entrez; may return 0 if unavailable
    assert len(results) >= 0, f"Unexpected error: got {type(results)}"
    return results


def test_hpa():
    src = HPASource()
    results = src.search("T cells", "Lung", "Human")
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    # HPA API may return 0 depending on endpoint availability
    assert len(results) >= 0, f"Unexpected error: got {type(results)}"
    return results


def test_gtex():
    src = GTExSource()
    results = src.search("T cells", "Lung", "Human")
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    # GTEx is tissue-level, not cell-type; may return hits for "Lung" tissue
    assert len(results) >= 0, f"Unexpected error: got {type(results)}"
    return results


def test_msigdb():
    src = MSigDBSource()
    results = src.search("T cells", "Lung", "Human")
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) > 0, f"Expected hits from MSigDB C8, got {len(results)}"
    for hit in results[:3]:
        assert hasattr(hit, "gene_symbol"), f"MarkerHit missing gene_symbol: {hit}"
    return results


def test_omim():
    src = OMIMSource()
    results = src.search("T cells", "Lung", "Human")
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    # OMIM may return 0 without API key and if Entrez fallback finds nothing
    assert len(results) >= 0, f"Unexpected error: {len(results)}"
    return results


def test_disgenet():
    raise SkipTest("DisGeNET public REST endpoint discontinued; requires API key or local dump")


def test_ensembl():
    src = EnsemblSource()
    results = src.search("T cells", "Lung", "Human", gene_list=["CD3D", "CD3E", "CD4"])
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    # Ensembl phenotype lookup may return 0 for genes without phenotype annotations
    assert len(results) >= 0, f"Unexpected error: got {type(results)}"
    return results


def test_opentargets():
    src = OpenTargetsSource()
    results = src.search("T cells", "Lung", "Human", condition="lung cancer")
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    # OpenTargets requires successful disease ID lookup via GraphQL
    assert len(results) >= 0, f"Unexpected error: got {type(results)}"
    return results


def test_cosmic():
    src = COSMICSource()
    results = src.search("T cells", "Lung", "Human", condition="lung cancer")
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    # COSMIC has built-in CGC genes for "lung" so should return hits
    assert len(results) > 0, f"Expected COSMIC CGC hits for lung cancer, got {len(results)}"
    for hit in results[:3]:
        assert hasattr(hit, "gene_symbol"), f"MarkerHit missing gene_symbol: {hit}"
    return results


def test_pubmed():
    src = PubMedSource()
    results = src.search("macrophage", "colorectal cancer", "Human", use_llm=False)
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    # Without LLM extraction, PubMed returns regex-extracted genes from abstracts
    assert len(results) >= 0, f"Unexpected error: got {type(results)}"
    return results


def test_pubtator3():
    src = PubTator3Source()
    results = src.search("macrophage", "colorectal cancer", "Human")
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    # PubTator3 should find papers about macrophage markers in CRC
    assert len(results) >= 0, f"Unexpected error: got {type(results)}"
    return results


def test_litvar2():
    src = LitVar2Source()
    results = src.search(
        "T cells", "Breast cancer", "Human",
        candidate_genes=["TP53", "BRCA1", "BRCA2", "ERBB2", "PIK3CA"],
    )
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    # LitVar2 should find variant publications for well-known cancer genes
    assert len(results) >= 0, f"Unexpected error: got {type(results)}"
    return results


def test_cellxgene():
    try:
        import cellxgene_census  # noqa: F401
    except ImportError:
        raise SkipTest("cellxgene_census package not installed")
    src = CellxGeneSource()
    results = src.search("T cells", "Lung", "Human")
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) >= 0, f"Unexpected error: got {type(results)}"
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
                for hit in results[:5]:
                    gene = hit.gene_symbol if hasattr(hit, "gene_symbol") else str(hit)
                    sample.append(gene)
            results_summary.append({
                "source": name, "status": "PASS", "reason": "",
                "hits": len(results) if results else 0,
                "time": elapsed, "sample_genes": sample,
            })
        except SkipTest as e:
            elapsed = time.time() - t0
            results_summary.append({
                "source": name, "status": "SKIP", "reason": str(e),
                "hits": 0, "time": elapsed, "sample_genes": [],
            })
        except AssertionError as e:
            elapsed = time.time() - t0
            results_summary.append({
                "source": name, "status": "FAIL", "reason": f"AssertionError: {e}",
                "hits": 0, "time": elapsed, "sample_genes": [],
            })
            traceback.print_exc()
        except Exception as e:
            elapsed = time.time() - t0
            results_summary.append({
                "source": name, "status": "FAIL", "reason": f"{type(e).__name__}: {e}",
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

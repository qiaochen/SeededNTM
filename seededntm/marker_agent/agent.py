"""MarkerSearchAgent — agentic tool-use loop for marker gene discovery.

Orchestrates multi-source queries using tiered source selection,
LLM-driven nomenclature resolution, and consensus scoring.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from seededntm.marker_agent.schemas import DatasetContext, MarkerHit, ScoredMarker
from seededntm.marker_agent.scoring import ConsensusScorer
from seededntm.marker_agent.provenance import ProvenanceLogger

logger = logging.getLogger(__name__)


class MarkerSearchAgent:
    """Agentic marker gene search pipeline.

    Orchestrates source queries in a tiered fashion:
      Tier 1: Curated DBs (CellMarker, PanglaoDB, ScTypeDB)
      Tier 2: Annotation/validation (MyGene, NCBIGene)
      Tier 3: Disease context (OMIM, DisGeNET, Ensembl, OpenTargets, COSMIC)
      Tier 4: Literature fallback (PubMed + LLM extraction)
      Tier 5: Expression/atlas (HPA, GTEx, MSigDB, CellxGene)
    """

    def __init__(
        self,
        context: DatasetContext,
        cache_dir: Optional[str] = None,
        use_llm: bool = True,
        verbose: bool = False,
    ):
        self.context = context
        self.cache_dir = cache_dir
        self.use_llm = use_llm
        self.verbose = verbose
        self._prov: Optional["ProvenanceLogger"] = None
        self._last_provenance: Optional["ProvenanceLogger"] = None

        if verbose:
            logging.basicConfig(level=logging.INFO)

        self._sources: Dict[str, Any] = {}
        self._init_sources()

    def _init_sources(self):
        """Initialize all available source instances."""
        from seededntm.marker_agent.sources.panglaodb import PanglaoDBSource
        from seededntm.marker_agent.sources.cellmarker import CellMarkerSource
        from seededntm.marker_agent.sources.sctype_db import ScTypeDBSource
        from seededntm.marker_agent.sources.mygene_source import MyGeneSource
        from seededntm.marker_agent.sources.ncbi_gene import NCBIGeneSource
        from seededntm.marker_agent.sources.hpa import HPASource
        from seededntm.marker_agent.sources.gtex import GTExSource
        from seededntm.marker_agent.sources.msigdb import MSigDBSource
        from seededntm.marker_agent.sources.cellxgene import CellxGeneSource
        from seededntm.marker_agent.sources.omim import OMIMSource
        from seededntm.marker_agent.sources.disgenet import DisGeNETSource
        from seededntm.marker_agent.sources.ensembl import EnsemblSource
        from seededntm.marker_agent.sources.opentargets import OpenTargetsSource
        from seededntm.marker_agent.sources.cosmic import COSMICSource
        from seededntm.marker_agent.sources.pubmed import PubMedSource

        self._sources = {
            "panglaodb": PanglaoDBSource(cache_dir=self.cache_dir),
            "cellmarker": CellMarkerSource(cache_dir=self.cache_dir),
            "sctype": ScTypeDBSource(cache_dir=self.cache_dir),
            "mygene": MyGeneSource(cache_dir=self.cache_dir),
            "ncbi_gene": NCBIGeneSource(cache_dir=self.cache_dir),
            "hpa": HPASource(cache_dir=self.cache_dir),
            "gtex": GTExSource(cache_dir=self.cache_dir),
            "msigdb": MSigDBSource(cache_dir=self.cache_dir),
            "cellxgene": CellxGeneSource(cache_dir=self.cache_dir),
            "omim": OMIMSource(cache_dir=self.cache_dir),
            "disgenet": DisGeNETSource(cache_dir=self.cache_dir),
            "ensembl": EnsemblSource(cache_dir=self.cache_dir),
            "opentargets": OpenTargetsSource(cache_dir=self.cache_dir),
            "cosmic": COSMICSource(cache_dir=self.cache_dir),
            "pubmed": PubMedSource(cache_dir=self.cache_dir),
        }

    def run(self, save_provenance: bool = False, provenance_path: Optional[str] = None) -> Dict[str, List[ScoredMarker]]:
        """Execute the full marker search pipeline.

        Args:
            save_provenance: If True, write a provenance JSON alongside results.
            provenance_path: Explicit path for provenance file. If None and
                save_provenance is True, derived from output path.

        Returns:
            Dict mapping cell_type -> sorted list of ScoredMarker
        """
        logger.info("=" * 60)
        logger.info("MarkerSearchAgent starting")
        logger.info("Species: %s | Tissue: %s | Organ: %s",
                    self.context.species, self.context.tissue, self.context.organ)
        logger.info("Condition: %s", self.context.condition)
        logger.info("Cell types: %s", self.context.expected_cell_types)
        logger.info("Gene panel size: %d", len(self.context.gene_panel))
        logger.info("=" * 60)

        prov: Optional[ProvenanceLogger] = None
        if save_provenance:
            prov = ProvenanceLogger(context={
                "species": self.context.species,
                "tissue": self.context.tissue,
                "organ": self.context.organ,
                "condition": self.context.condition,
                "expected_cell_types": self.context.expected_cell_types,
                "n_markers_per_type": self.context.n_markers_per_type,
                "n_panel_genes": len(self.context.gene_panel),
            })
        self._prov = prov

        all_hits: Dict[str, List[MarkerHit]] = {
            ct: [] for ct in self.context.expected_cell_types
        }

        cell_type_aliases = self._resolve_nomenclature()

        self._query_tier1(all_hits, cell_type_aliases)

        self._query_tier2(all_hits, cell_type_aliases)

        if self.context.include_disease_genes or self.context.condition:
            self._query_tier3(all_hits, cell_type_aliases)

        sparse_types = [
            ct for ct, hits in all_hits.items() if len(hits) < 5
        ]
        if sparse_types:
            self._query_tier4(all_hits, cell_type_aliases, sparse_types)

        self._query_tier5(all_hits, cell_type_aliases)

        scorer = ConsensusScorer(
            gene_panel=self.context.gene_panel,
            top_n=self.context.n_markers_per_type,
            require_panel=bool(self.context.gene_panel),
        )
        results = scorer.score(all_hits)

        if prov is not None:
            self._record_scoring_provenance(results)

        self._log_summary(results, all_hits)

        if prov is not None and provenance_path:
            prov.save(provenance_path)
            logger.info("Provenance log saved to %s", provenance_path)

        self._last_provenance = prov

        return results

    def _resolve_nomenclature(self) -> Dict[str, List[str]]:
        """Resolve cell type names to synonyms for database queries."""
        aliases: Dict[str, List[str]] = {}

        for cell_type in self.context.expected_cell_types:
            if self.use_llm:
                try:
                    from seededntm.marker_agent.llm_client import llm_resolve_cell_type
                    context_str = f"{self.context.tissue} ({self.context.condition})"
                    input_summary = f"Resolve '{cell_type}' for context: {context_str}"

                    if self._prov is not None:
                        with self._prov.timed_llm_call("nomenclature_resolution", input_summary) as rec:
                            synonyms = llm_resolve_cell_type(cell_type, context_str)
                            rec.output_summary = ", ".join(synonyms[:10])
                    else:
                        synonyms = llm_resolve_cell_type(cell_type, context_str)

                    aliases[cell_type] = synonyms
                    logger.info("Resolved '%s' -> %s", cell_type, synonyms)
                except Exception as e:
                    logger.warning("LLM resolution failed for '%s': %s", cell_type, e)
                    if self._prov is not None:
                        self._prov.record_error(
                            source="LLM",
                            cell_type=cell_type,
                            error_type=type(e).__name__,
                            message=str(e),
                            fallback_used="identity (original name only)",
                        )
                    aliases[cell_type] = [cell_type]
            else:
                aliases[cell_type] = [cell_type]

        return aliases

    def _query_tier1(
        self,
        all_hits: Dict[str, List[MarkerHit]],
        aliases: Dict[str, List[str]],
    ):
        """Query Tier 1: Curated marker databases."""
        logger.info("--- Tier 1: Curated Databases ---")
        tier1_sources = ["panglaodb", "cellmarker", "sctype"]

        for cell_type in self.context.expected_cell_types:
            for alias in aliases.get(cell_type, [cell_type]):
                for source_name in tier1_sources:
                    source = self._sources.get(source_name)
                    if source is None:
                        continue
                    params = {"tissue": self.context.tissue, "species": self.context.species}
                    try:
                        if self._prov is not None:
                            with self._prov.timed_query(source_name, cell_type, {"alias": alias, **params}) as rec:
                                hits = source.search(
                                    cell_type=alias,
                                    tissue=self.context.tissue,
                                    species=self.context.species,
                                )
                                rec.n_results = len(hits)
                                rec.top_hits = [h.gene_symbol for h in hits[:10]]
                        else:
                            hits = source.search(
                                cell_type=alias,
                                tissue=self.context.tissue,
                                species=self.context.species,
                            )
                        for hit in hits:
                            hit.cell_type = cell_type
                        all_hits[cell_type].extend(hits)
                    except Exception as e:
                        logger.warning(
                            "Tier 1 %s failed for '%s': %s", source_name, alias, e
                        )
                        if self._prov is not None:
                            self._prov.record_error(
                                source=source_name, cell_type=cell_type,
                                error_type=type(e).__name__, message=str(e),
                            )

    def _query_tier2(
        self,
        all_hits: Dict[str, List[MarkerHit]],
        aliases: Dict[str, List[str]],
    ):
        """Query Tier 2: Gene annotation and validation."""
        logger.info("--- Tier 2: Annotation Sources ---")
        tier2_sources = ["mygene", "ncbi_gene"]

        for cell_type in self.context.expected_cell_types:
            for alias in aliases.get(cell_type, [cell_type])[:2]:
                for source_name in tier2_sources:
                    source = self._sources.get(source_name)
                    if source is None:
                        continue
                    params = {"tissue": self.context.tissue, "species": self.context.species}
                    try:
                        if self._prov is not None:
                            with self._prov.timed_query(source_name, cell_type, {"alias": alias, **params}) as rec:
                                hits = source.search(
                                    cell_type=alias,
                                    tissue=self.context.tissue,
                                    species=self.context.species,
                                )
                                rec.n_results = len(hits)
                                rec.top_hits = [h.gene_symbol for h in hits[:10]]
                        else:
                            hits = source.search(
                                cell_type=alias,
                                tissue=self.context.tissue,
                                species=self.context.species,
                            )
                        for hit in hits:
                            hit.cell_type = cell_type
                        all_hits[cell_type].extend(hits)
                    except Exception as e:
                        logger.warning(
                            "Tier 2 %s failed for '%s': %s", source_name, alias, e
                        )
                        if self._prov is not None:
                            self._prov.record_error(
                                source=source_name, cell_type=cell_type,
                                error_type=type(e).__name__, message=str(e),
                            )
                time.sleep(0.5)

    def _query_tier3(
        self,
        all_hits: Dict[str, List[MarkerHit]],
        aliases: Dict[str, List[str]],
    ):
        """Query Tier 3: Disease-gene databases."""
        logger.info("--- Tier 3: Disease Sources ---")
        tier3_sources = ["omim", "disgenet", "opentargets", "cosmic"]

        for cell_type in self.context.expected_cell_types:
            for source_name in tier3_sources:
                source = self._sources.get(source_name)
                if source is None:
                    continue
                params = {
                    "tissue": self.context.tissue,
                    "species": self.context.species,
                    "condition": self.context.condition,
                }
                try:
                    if self._prov is not None:
                        with self._prov.timed_query(source_name, cell_type, params) as rec:
                            hits = source.search(
                                cell_type=cell_type,
                                tissue=self.context.tissue,
                                species=self.context.species,
                                condition=self.context.condition,
                            )
                            rec.n_results = len(hits)
                            rec.top_hits = [h.gene_symbol for h in hits[:10]]
                    else:
                        hits = source.search(
                            cell_type=cell_type,
                            tissue=self.context.tissue,
                            species=self.context.species,
                            condition=self.context.condition,
                        )
                    for hit in hits:
                        hit.cell_type = cell_type
                    all_hits[cell_type].extend(hits)
                except Exception as e:
                    logger.warning(
                        "Tier 3 %s failed for '%s': %s", source_name, cell_type, e
                    )
                    if self._prov is not None:
                        self._prov.record_error(
                            source=source_name, cell_type=cell_type,
                            error_type=type(e).__name__, message=str(e),
                        )

        candidate_genes = list(set(
            hit.gene_symbol for hits in all_hits.values() for hit in hits
        ))[:30]
        if candidate_genes:
            ensembl = self._sources.get("ensembl")
            if ensembl:
                for cell_type in self.context.expected_cell_types:
                    try:
                        if self._prov is not None:
                            with self._prov.timed_query("ensembl", cell_type, {"gene_list_size": len(candidate_genes)}) as rec:
                                hits = ensembl.search(
                                    cell_type=cell_type,
                                    tissue=self.context.tissue,
                                    species=self.context.species,
                                    gene_list=candidate_genes,
                                )
                                rec.n_results = len(hits)
                                rec.top_hits = [h.gene_symbol for h in hits[:10]]
                        else:
                            hits = ensembl.search(
                                cell_type=cell_type,
                                tissue=self.context.tissue,
                                species=self.context.species,
                                gene_list=candidate_genes,
                            )
                        all_hits[cell_type].extend(hits)
                    except Exception as e:
                        logger.warning("Ensembl failed for '%s': %s", cell_type, e)
                        if self._prov is not None:
                            self._prov.record_error(
                                source="ensembl", cell_type=cell_type,
                                error_type=type(e).__name__, message=str(e),
                            )

    def _query_tier4(
        self,
        all_hits: Dict[str, List[MarkerHit]],
        aliases: Dict[str, List[str]],
        sparse_types: List[str],
    ):
        """Query Tier 4: Literature fallback for sparse cell types."""
        logger.info("--- Tier 4: Literature Fallback (sparse: %s) ---", sparse_types)

        pubmed = self._sources.get("pubmed")
        if pubmed is None:
            return

        for cell_type in sparse_types:
            for alias in aliases.get(cell_type, [cell_type])[:2]:
                params = {
                    "tissue": self.context.tissue,
                    "species": self.context.species,
                    "use_llm": self.use_llm,
                }
                try:
                    if self._prov is not None:
                        with self._prov.timed_query("pubmed", cell_type, {"alias": alias, **params}) as rec:
                            hits = pubmed.search(
                                cell_type=alias,
                                tissue=self.context.tissue,
                                species=self.context.species,
                                use_llm=self.use_llm,
                            )
                            rec.n_results = len(hits)
                            rec.top_hits = [h.gene_symbol for h in hits[:10]]
                    else:
                        hits = pubmed.search(
                            cell_type=alias,
                            tissue=self.context.tissue,
                            species=self.context.species,
                            use_llm=self.use_llm,
                        )
                    for hit in hits:
                        hit.cell_type = cell_type
                    all_hits[cell_type].extend(hits)
                except Exception as e:
                    logger.warning("Tier 4 PubMed failed for '%s': %s", alias, e)
                    if self._prov is not None:
                        self._prov.record_error(
                            source="pubmed", cell_type=cell_type,
                            error_type=type(e).__name__, message=str(e),
                            fallback_used="skipped (no alternative)",
                        )
                time.sleep(1.0)

    def _query_tier5(
        self,
        all_hits: Dict[str, List[MarkerHit]],
        aliases: Dict[str, List[str]],
    ):
        """Query Tier 5: Expression and atlas data."""
        logger.info("--- Tier 5: Expression/Atlas Sources ---")
        tier5_sources = ["hpa", "gtex", "msigdb", "cellxgene"]

        for cell_type in self.context.expected_cell_types:
            for source_name in tier5_sources:
                source = self._sources.get(source_name)
                if source is None:
                    continue
                params = {"tissue": self.context.tissue, "species": self.context.species}
                try:
                    if self._prov is not None:
                        with self._prov.timed_query(source_name, cell_type, params) as rec:
                            hits = source.search(
                                cell_type=cell_type,
                                tissue=self.context.tissue,
                                species=self.context.species,
                            )
                            rec.n_results = len(hits)
                            rec.top_hits = [h.gene_symbol for h in hits[:10]]
                    else:
                        hits = source.search(
                            cell_type=cell_type,
                            tissue=self.context.tissue,
                            species=self.context.species,
                        )
                    for hit in hits:
                        hit.cell_type = cell_type
                    all_hits[cell_type].extend(hits)
                except Exception as e:
                    logger.warning(
                        "Tier 5 %s failed for '%s': %s", source_name, cell_type, e
                    )
                    if self._prov is not None:
                        self._prov.record_error(
                            source=source_name, cell_type=cell_type,
                            error_type=type(e).__name__, message=str(e),
                        )

    def _record_scoring_provenance(
        self, results: Dict[str, List[ScoredMarker]]
    ):
        """Record per-gene scoring breakdown into provenance log."""
        if self._prov is None:
            return
        for cell_type, markers in results.items():
            gene_scores = {}
            for m in markers:
                scores_dict = dict(m.source_scores)
                scores_dict["total"] = m.total_score
                gene_scores[m.gene_symbol] = scores_dict
            self._prov.record_scoring(cell_type, gene_scores)

    def _log_summary(
        self,
        results: Dict[str, List[ScoredMarker]],
        all_hits: Dict[str, List[MarkerHit]],
    ):
        """Log a summary of the search results."""
        logger.info("=" * 60)
        logger.info("SEARCH SUMMARY")
        logger.info("=" * 60)
        for cell_type, markers in results.items():
            n_raw = len(all_hits.get(cell_type, []))
            n_scored = len(markers)
            top_genes = [m.gene_symbol for m in markers[:5]]
            logger.info(
                "  %s: %d raw hits -> %d scored markers | top: %s",
                cell_type, n_raw, n_scored, ", ".join(top_genes),
            )
        logger.info("=" * 60)

    def to_seed_json(
        self, results: Dict[str, List[ScoredMarker]]
    ) -> Dict[str, List[str]]:
        """Convert scored results to seed JSON format (cell_type -> gene list)."""
        seeds = {}
        for cell_type, markers in results.items():
            seeds[cell_type] = [m.gene_symbol for m in markers]
        return seeds

    def to_evidence_json(
        self, results: Dict[str, List[ScoredMarker]]
    ) -> Dict[str, Any]:
        """Convert results to detailed evidence JSON."""
        output = {
            "context": {
                "species": self.context.species,
                "tissue": self.context.tissue,
                "organ": self.context.organ,
                "condition": self.context.condition,
                "n_panel_genes": len(self.context.gene_panel),
                "n_markers_per_type": self.context.n_markers_per_type,
            },
            "seeds": {},
            "evidence": {},
        }

        for cell_type, markers in results.items():
            output["seeds"][cell_type] = [m.gene_symbol for m in markers]
            output["evidence"][cell_type] = [
                {
                    "gene": m.gene_symbol,
                    "score": m.total_score,
                    "sources": m.source_scores,
                    "in_panel": m.in_panel,
                }
                for m in markers
            ]

        return output

    def save_results(
        self,
        results: Dict[str, List[ScoredMarker]],
        output_path: str,
        format: str = "seeds",
        save_provenance: bool = False,
    ):
        """Save results to a JSON file.

        Args:
            format: 'seeds' for simple gene lists, 'evidence' for detailed output
            save_provenance: If True, save provenance JSON alongside the output.
        """
        if format == "seeds":
            data = self.to_seed_json(results)
        else:
            data = self.to_evidence_json(results)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Results saved to %s", output_path)

        if save_provenance and self._last_provenance is not None:
            prov_path = self._derive_provenance_path(output_path)
            self._last_provenance.save(prov_path)
            logger.info("Provenance log saved to %s", prov_path)

    @staticmethod
    def _derive_provenance_path(output_path: str) -> str:
        """Derive provenance file path from the main output path."""
        from pathlib import Path
        p = Path(output_path)
        return str(p.parent / f"{p.stem}_provenance.json")

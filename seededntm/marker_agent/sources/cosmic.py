"""COSMIC source — Cancer Gene Census CSV for cancer-related genes.

Loads the COSMIC Cancer Gene Census to identify cancer driver genes
and their associated tumor types. Falls back to a built-in list of
well-known CGC Tier 1 cancer driver genes when the CSV is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from seededntm.marker_agent.cache import find_local_file, get_cache_dir
from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

COSMIC_CGC_FILENAME = "cancer_gene_census.csv"

# Built-in CGC Tier 1 cancer driver genes (publicly known, widely cited).
# Organized by primary associated tumor type keywords for filtering.
# Sources: published CGC papers, COSMIC public documentation, review articles.
_BUILTIN_CGC_GENES: Dict[str, List[str]] = {
    "lung": [
        "EGFR", "KRAS", "ALK", "ROS1", "MET", "BRAF", "ERBB2", "RET",
        "PIK3CA", "STK11", "KEAP1", "NF1", "RB1", "TP53", "CDKN2A",
        "DDR2", "FGFR1", "FGFR2", "FGFR3", "MAP2K1", "NRAS", "HRAS",
        "NTRK1", "NTRK2", "NTRK3", "SMARCA4",
    ],
    "breast": [
        "BRCA1", "BRCA2", "ERBB2", "PIK3CA", "TP53", "CDH1", "GATA3",
        "MAP3K1", "AKT1", "PTEN", "RB1", "CBFB", "RUNX1", "ESR1",
        "FOXA1", "NF1", "PALB2", "CCND1", "CDK4", "CDK6",
    ],
    "colorectal": [
        "APC", "KRAS", "TP53", "BRAF", "PIK3CA", "SMAD4", "FBXW7",
        "NRAS", "CTNNB1", "TCF7L2", "PTEN", "AMER1", "SOX9", "ATM",
        "ARID1A", "MLH1", "MSH2", "MSH6", "PMS2",
    ],
    "colon": [
        "APC", "KRAS", "TP53", "BRAF", "PIK3CA", "SMAD4", "FBXW7",
        "NRAS", "CTNNB1", "PTEN", "MLH1", "MSH2", "MSH6", "PMS2",
    ],
    "prostate": [
        "AR", "SPOP", "FOXA1", "TP53", "PTEN", "RB1", "BRCA2", "BRCA1",
        "ATM", "CDK12", "PIK3CA", "MYC", "ERG", "TMPRSS2",
    ],
    "melanoma": [
        "BRAF", "NRAS", "NF1", "KIT", "CDKN2A", "PTEN", "TP53", "RAC1",
        "MAP2K1", "MAP2K2", "GNAQ", "GNA11", "BAP1", "TERT",
    ],
    "liver": [
        "TP53", "CTNNB1", "AXIN1", "ARID1A", "ARID2", "ALB", "TERT",
        "RB1", "NFE2L2", "KEAP1", "PIK3CA", "TSC1", "TSC2",
    ],
    "pancreatic": [
        "KRAS", "TP53", "CDKN2A", "SMAD4", "ARID1A", "RNF43", "TGFBR2",
        "GNAS", "BRCA2", "ATM", "STK11", "MAP2K4", "BRAF",
    ],
    "pancreas": [
        "KRAS", "TP53", "CDKN2A", "SMAD4", "ARID1A", "RNF43", "BRCA2",
    ],
    "kidney": [
        "VHL", "PBRM1", "BAP1", "SETD2", "KDM5C", "MTOR", "TSC1",
        "TSC2", "TP53", "PTEN", "MET", "FH",
    ],
    "renal": [
        "VHL", "PBRM1", "BAP1", "SETD2", "KDM5C", "MTOR", "MET", "FH",
    ],
    "bladder": [
        "FGFR3", "TP53", "RB1", "PIK3CA", "CDKN2A", "ARID1A", "KDM6A",
        "STAG2", "ERBB2", "HRAS", "TERT", "TSC1",
    ],
    "ovarian": [
        "BRCA1", "BRCA2", "TP53", "RB1", "NF1", "CDK12", "PTEN",
        "KRAS", "BRAF", "PIK3CA", "ARID1A", "CTNNB1",
    ],
    "ovary": [
        "BRCA1", "BRCA2", "TP53", "RB1", "NF1", "CDK12", "PTEN",
    ],
    "gastric": [
        "TP53", "CDH1", "ARID1A", "PIK3CA", "KRAS", "ERBB2", "RHOA",
        "SMAD4", "APC", "CTNNB1", "RNF43", "FBXW7",
    ],
    "stomach": [
        "TP53", "CDH1", "ARID1A", "PIK3CA", "KRAS", "ERBB2", "RHOA",
    ],
    "endometrial": [
        "PTEN", "PIK3CA", "TP53", "CTNNB1", "ARID1A", "KRAS", "FBXW7",
        "PPP2R1A", "POLE", "MSH6", "RPL22",
    ],
    "thyroid": [
        "BRAF", "RAS", "NRAS", "HRAS", "KRAS", "RET", "PAX8", "PPARG",
        "TERT", "TP53", "PTEN", "AKT1", "EIF1AX",
    ],
    "glioma": [
        "IDH1", "IDH2", "TP53", "ATRX", "EGFR", "PTEN", "CDKN2A",
        "NF1", "PIK3CA", "PIK3R1", "RB1", "PDGFRA", "TERT",
    ],
    "brain": [
        "IDH1", "IDH2", "TP53", "ATRX", "EGFR", "PTEN", "CDKN2A",
        "NF1", "PDGFRA", "SMARCB1", "TERT", "H3F3A",
    ],
    "leukemia": [
        "FLT3", "NPM1", "DNMT3A", "IDH1", "IDH2", "TET2", "RUNX1",
        "CEBPA", "TP53", "NRAS", "KRAS", "KIT", "ABL1", "BCR",
        "KMT2A", "PHF6", "WT1", "ASXL1", "EZH2", "JAK2", "NOTCH1",
        "FBXW7", "PTEN", "U2AF1", "SF3B1", "SRSF2",
    ],
    "lymphoma": [
        "MYC", "BCL2", "BCL6", "EZH2", "CREBBP", "EP300", "KMT2D",
        "NOTCH1", "NOTCH2", "TP53", "CDKN2A", "TNFAIP3", "CARD11",
        "CD79B", "MYD88", "BRAF", "JAK2",
    ],
    "myeloma": [
        "KRAS", "NRAS", "BRAF", "TP53", "DIS3", "FAM46C", "TRAF3",
        "RB1", "CDKN2C", "MYC", "CCND1", "FGFR3", "NF1",
    ],
    "myeloid": [
        "FLT3", "NPM1", "DNMT3A", "IDH1", "IDH2", "TET2", "RUNX1",
        "ASXL1", "EZH2", "JAK2", "CALR", "MPL", "SF3B1", "SRSF2",
        "U2AF1", "CEBPA", "TP53", "NRAS", "KRAS", "CBL",
    ],
    "head and neck": [
        "TP53", "CDKN2A", "PIK3CA", "NOTCH1", "HRAS", "FBXW7", "FAT1",
        "CASP8", "NSD1", "PTEN", "NFE2L2",
    ],
    "cervical": [
        "PIK3CA", "PTEN", "TP53", "KRAS", "FBXW7", "EP300", "HLA-A",
        "HLA-B", "ARID1A", "NFE2L2",
    ],
    "sarcoma": [
        "TP53", "RB1", "NF1", "CDKN2A", "MDM2", "CDK4", "ATRX",
        "SMARCB1", "SS18", "EWSR1", "FLI1", "PAX3", "FOXO1",
    ],
    "mesothelioma": [
        "BAP1", "NF2", "CDKN2A", "TP53", "LATS2", "SETD2",
    ],
}

# Pan-cancer driver genes included in all queries
_PAN_CANCER_DRIVERS: List[str] = [
    "TP53", "KRAS", "PIK3CA", "PTEN", "RB1", "APC", "BRAF", "EGFR",
    "MYC", "CDKN2A", "NRAS", "FBXW7", "ARID1A", "ATM", "NF1", "BRCA1",
    "BRCA2", "ERBB2", "CTNNB1", "SMAD4", "VHL", "CDH1", "NOTCH1",
    "KMT2D", "CREBBP", "EP300", "SETD2", "KDM6A", "STAG2", "IDH1",
    "IDH2", "DNMT3A", "TET2", "SF3B1", "SRSF2", "U2AF1", "NPM1",
    "FLT3", "JAK2", "ABL1", "KIT", "PDGFRA", "ALK", "ROS1", "RET",
    "MET", "FGFR1", "FGFR2", "FGFR3", "DDX3X", "PHF6", "SPOP",
    "FOXA1", "AR", "ESR1", "GATA3", "RUNX1", "CEBPA", "SOX9",
    "NF2", "PTCH1", "SMO", "TSC1", "TSC2", "MTOR", "AKT1",
    "STK11", "KEAP1", "NFE2L2", "BAP1", "SMARCA4", "SMARCB1",
    "ARID2", "PBRM1", "ASXL1", "EZH2", "SUZ12", "BCOR",
    "ATR", "CHEK2", "BRIP1", "PALB2", "RAD51C", "RAD51D",
    "FANCA", "FANCC", "BLM", "WRN", "MLH1", "MSH2", "MSH6",
    "PMS2", "POLE", "POLD1", "MUTYH", "CDK4", "CDK6", "CCND1",
    "CCNE1", "RBM10", "TERT", "WT1", "HRAS",
]


class COSMICSource:
    """Query COSMIC Cancer Gene Census from local CSV, with built-in fallback."""

    def __init__(self, cache_dir: Optional[str] = None, filepath: Optional[str] = None):
        self.cache_dir = cache_dir
        self._df: Optional[pd.DataFrame] = None
        self._filepath = filepath
        self._using_fallback = False

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        path = None
        if self._filepath:
            path = Path(self._filepath)
        else:
            path = find_local_file(COSMIC_CGC_FILENAME, cache_dir=self.cache_dir)
            if path is None:
                for alt in ["Census_allMon.csv", "Census_all.csv"]:
                    path = find_local_file(alt, cache_dir=self.cache_dir)
                    if path:
                        break

        if path is None:
            logger.info(
                "COSMIC CGC CSV not found — using built-in cancer driver gene list. "
                "For full CGC data, place '%s' in cache dir: %s",
                COSMIC_CGC_FILENAME,
                get_cache_dir(self.cache_dir),
            )
            self._using_fallback = True
            self._df = pd.DataFrame()
            return self._df

        logger.info("Loading COSMIC CGC from %s", path)
        try:
            self._df = pd.read_csv(path)
            self._df.columns = self._df.columns.str.strip()
        except Exception as e:
            logger.error("Failed to load COSMIC CGC: %s", e)
            self._using_fallback = True
            self._df = pd.DataFrame()

        return self._df

    def _search_builtin(self, cell_type: str, tissue: str, condition: str) -> List[MarkerHit]:
        """Search the built-in CGC gene list by tissue/condition keywords."""
        search_terms = []
        if tissue:
            search_terms.append(tissue.lower())
        if condition:
            search_terms.append(condition.lower())

        matched_genes: Set[str] = set()

        if search_terms:
            for keyword, genes in _BUILTIN_CGC_GENES.items():
                if any(keyword in term or term in keyword for term in search_terms):
                    matched_genes.update(genes)

        # Always include pan-cancer drivers (they are universally relevant)
        if matched_genes:
            matched_genes.update(_PAN_CANCER_DRIVERS[:30])
        else:
            # No tissue match — return all pan-cancer drivers
            matched_genes.update(_PAN_CANCER_DRIVERS)

        hits = []
        for symbol in sorted(matched_genes):
            tissue_specific = symbol not in _PAN_CANCER_DRIVERS[:30]
            hits.append(
                MarkerHit(
                    gene_symbol=symbol,
                    cell_type=cell_type,
                    source="COSMIC",
                    score=0.7 if tissue_specific else 0.6,
                    evidence_detail=f"builtin_CGC; query={tissue or condition}",
                )
            )

        logger.info(
            "COSMIC (built-in): found %d cancer genes for '%s' (tissue=%s)",
            len(hits), cell_type, tissue,
        )
        return hits

    def search(
        self,
        cell_type: str,
        tissue: str,
        species: str = "Human",
        condition: str = "",
        **kwargs,
    ) -> List[MarkerHit]:
        """Search COSMIC CGC for cancer genes relevant to a tissue/condition."""
        df = self._load()

        if df.empty and self._using_fallback:
            return self._search_builtin(cell_type, tissue, condition)

        if df.empty:
            return []

        gene_col = next(
            (c for c in df.columns if "gene" in c.lower() and "symbol" in c.lower()),
            None,
        )
        if gene_col is None:
            gene_col = next(
                (c for c in df.columns if "gene" in c.lower()), None
            )

        tumour_col = next(
            (c for c in df.columns if "tumour" in c.lower() or "tumor" in c.lower()),
            None,
        )
        role_col = next(
            (c for c in df.columns if "role" in c.lower()), None
        )

        if gene_col is None:
            logger.warning("COSMIC columns not recognized: %s", list(df.columns))
            return []

        search_terms = [tissue.lower(), condition.lower()] if tissue or condition else []
        search_terms = [t for t in search_terms if t]

        if search_terms and tumour_col:
            mask = df[tumour_col].astype(str).str.lower().apply(
                lambda x: any(t in x for t in search_terms)
            )
            results = df[mask]
        else:
            results = df

        hits = []
        for _, row in results.iterrows():
            symbol = str(row[gene_col]).strip()
            if not symbol or symbol.lower() == "nan":
                continue

            role = str(row.get(role_col, "")) if role_col else ""
            tumour_types = str(row.get(tumour_col, "")) if tumour_col else ""

            hits.append(
                MarkerHit(
                    gene_symbol=symbol.upper(),
                    cell_type=cell_type,
                    source="COSMIC",
                    score=0.7,
                    evidence_detail=f"role={role[:40]}; tumour_types={tumour_types[:60]}",
                )
            )

        logger.info("COSMIC: found %d cancer genes for '%s'", len(hits), cell_type)
        return hits

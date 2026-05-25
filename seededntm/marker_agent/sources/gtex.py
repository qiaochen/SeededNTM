"""GTEx source — REST API for tissue median TPM expression.

Queries the GTEx Portal API for tissue-specific gene expression data
to validate marker gene tissue specificity.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import requests

from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

GTEX_API_BASE = "https://gtexportal.org/api/v2"


class GTExSource:
    """Query GTEx Portal for tissue-level expression data."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    def search(
        self,
        cell_type: str,
        tissue: str,
        species: str = "Human",
        **kwargs,
    ) -> List[MarkerHit]:
        """Search GTEx for tissue-enriched genes.

        GTEx provides tissue-level (not cell-type level) expression.
        Useful for validating tissue specificity of marker candidates.
        """
        if species.lower() != "human":
            logger.info("GTEx only supports Human data")
            return []

        if not tissue:
            return []

        return self._get_top_expressed_genes(tissue, cell_type)

    def _get_top_expressed_genes(
        self, tissue: str, cell_type: str
    ) -> List[MarkerHit]:
        """Get top expressed genes in a tissue from GTEx."""
        tissue_id = self._resolve_tissue_id(tissue)
        if not tissue_id:
            return []

        url = f"{GTEX_API_BASE}/expression/topExpressedGene"
        params = {
            "tissueSiteDetailId": tissue_id,
            "sortBy": "median",
            "sortDirection": "desc",
            "pageSize": 100,
        }

        try:
            resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                logger.warning("GTEx API returned %d", resp.status_code)
                return []
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("GTEx API query failed: %s", e)
            return []

        hits = []
        for entry in data.get("topExpressedGene", data.get("data", []))[:50]:
            symbol = entry.get("geneSymbol") or entry.get("gencodeId", "")
            median_tpm = entry.get("median", 0)
            if symbol and not symbol.startswith("ENSG"):
                hits.append(
                    MarkerHit(
                        gene_symbol=symbol.upper(),
                        cell_type=cell_type,
                        source="GTEx",
                        score=min(median_tpm / 1000, 1.0) if median_tpm else 0.5,
                        evidence_detail=f"tissue={tissue_id}; median_tpm={median_tpm:.1f}",
                    )
                )

        logger.info("GTEx: found %d genes for tissue '%s'", len(hits), tissue)
        return hits

    def _resolve_tissue_id(self, tissue: str) -> Optional[str]:
        """Map tissue name to GTEx tissue site detail ID."""
        tissue_lower = tissue.lower()

        tissue_map = {
            "colon": "Colon_Transverse",
            "colorectal": "Colon_Transverse",
            "brain": "Brain_Cortex",
            "liver": "Liver",
            "lung": "Lung",
            "heart": "Heart_Left_Ventricle",
            "kidney": "Kidney_Cortex",
            "breast": "Breast_Mammary_Tissue",
            "pancreas": "Pancreas",
            "skin": "Skin_Sun_Exposed_Lower_leg",
            "stomach": "Stomach",
            "prostate": "Prostate",
            "ovary": "Ovary",
            "uterus": "Uterus",
            "thyroid": "Thyroid",
            "blood": "Whole_Blood",
            "muscle": "Muscle_Skeletal",
            "adipose": "Adipose_Subcutaneous",
            "spleen": "Spleen",
            "bladder": "Bladder",
            "esophagus": "Esophagus_Mucosa",
            "small intestine": "Small_Intestine_Terminal_Ileum",
        }

        for key, value in tissue_map.items():
            if key in tissue_lower:
                return value

        return None

    def get_gene_tissue_expression(
        self, gene_symbol: str
    ) -> Dict[str, float]:
        """Get expression of a gene across all tissues."""
        url = f"{GTEX_API_BASE}/expression/medianGeneExpression"
        params = {"geneSymbol": gene_symbol}

        try:
            resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                return {}
            data = resp.json()
        except Exception as e:
            logger.warning("GTEx gene expression failed for %s: %s", gene_symbol, e)
            return {}

        result = {}
        for entry in data.get("medianGeneExpression", data.get("data", [])):
            tissue_id = entry.get("tissueSiteDetailId", "")
            median = entry.get("median", 0)
            result[tissue_id] = median

        return result

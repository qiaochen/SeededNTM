"""CellxGene source — on-demand differential expression from CZ CELLxGENE.

Queries the CZ CELLxGENE Discover Census API for cell-type specific
differentially expressed genes from single-cell atlas data.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import requests

from seededntm.marker_agent.schemas import MarkerHit

logger = logging.getLogger(__name__)

CELLXGENE_API_BASE = "https://api.cellxgene.cziscience.com"


class CellxGeneSource:
    """Query CZ CELLxGENE for cell-type DE genes (optional/stub)."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._session = requests.Session()
        self._session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

    def search(
        self,
        cell_type: str,
        tissue: str,
        species: str = "Human",
        **kwargs,
    ) -> List[MarkerHit]:
        """Search CELLxGENE for DE markers of a cell type.

        This source queries the CELLxGENE Discover API for datasets
        containing the specified cell type and retrieves marker genes.
        """
        datasets = self._find_datasets(cell_type, tissue, species)
        if not datasets:
            return []

        hits = []
        for dataset_id in datasets[:3]:
            markers = self._get_de_genes(dataset_id, cell_type)
            hits.extend(markers)

        logger.info("CellxGene: found %d markers for '%s'", len(hits), cell_type)
        return hits

    def _find_datasets(
        self, cell_type: str, tissue: str, species: str
    ) -> List[str]:
        """Find datasets in CELLxGENE matching cell type and tissue."""
        organism = "Homo sapiens" if species.lower() == "human" else species

        params = {
            "organism": organism,
            "tissue": tissue,
            "cell_type": cell_type,
        }

        try:
            resp = self._session.get(
                f"{CELLXGENE_API_BASE}/dp/v1/datasets/index",
                params=params,
                timeout=30,
            )
            if resp.status_code != 200:
                logger.debug("CELLxGENE dataset search returned %d", resp.status_code)
                return []
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("CELLxGENE dataset search failed: %s", e)
            return []

        dataset_ids = []
        datasets = data if isinstance(data, list) else data.get("datasets", [])
        for ds in datasets:
            ds_id = ds.get("dataset_id") or ds.get("id", "")
            if ds_id:
                dataset_ids.append(ds_id)

        return dataset_ids

    def _get_de_genes(self, dataset_id: str, cell_type: str) -> List[MarkerHit]:
        """Get differentially expressed genes for a cell type in a dataset."""
        try:
            resp = self._session.post(
                f"{CELLXGENE_API_BASE}/dp/v1/markers",
                json={
                    "dataset_id": dataset_id,
                    "cell_type": cell_type,
                },
                timeout=60,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.debug("CELLxGENE DE query failed: %s", e)
            return []

        hits = []
        markers = data if isinstance(data, list) else data.get("markers", [])
        for marker in markers[:50]:
            symbol = marker.get("gene_symbol") or marker.get("gene", "")
            lfc = marker.get("log_fold_change") or marker.get("lfc", 0)
            pval = marker.get("adjusted_p_value") or marker.get("pval_adj", 1)

            if symbol:
                score = min(abs(float(lfc)) / 3, 1.0) if lfc else 0.5
                hits.append(
                    MarkerHit(
                        gene_symbol=symbol.upper(),
                        cell_type=cell_type,
                        source="CellxGene",
                        score=score,
                        evidence_detail=f"LFC={lfc}; padj={pval}; dataset={dataset_id[:8]}",
                    )
                )

        return hits

"""Source modules for marker gene retrieval from biological databases."""

from seededntm.marker_agent.sources.cellmarker import CellMarkerSource
from seededntm.marker_agent.sources.panglaodb import PanglaoDBSource
from seededntm.marker_agent.sources.sctype_db import ScTypeDBSource
from seededntm.marker_agent.sources.mygene_source import MyGeneSource
from seededntm.marker_agent.sources.ncbi_gene import NCBIGeneSource
from seededntm.marker_agent.sources.hpa import HPASource
from seededntm.marker_agent.sources.gtex import GTExSource
from seededntm.marker_agent.sources.cellxgene import CellxGeneSource
from seededntm.marker_agent.sources.msigdb import MSigDBSource
from seededntm.marker_agent.sources.omim import OMIMSource
from seededntm.marker_agent.sources.disgenet import DisGeNETSource
from seededntm.marker_agent.sources.ensembl import EnsemblSource
from seededntm.marker_agent.sources.opentargets import OpenTargetsSource
from seededntm.marker_agent.sources.cosmic import COSMICSource
from seededntm.marker_agent.sources.pubmed import PubMedSource
from seededntm.marker_agent.sources.pubtator3 import PubTator3Source
from seededntm.marker_agent.sources.litvar2 import LitVar2Source

__all__ = [
    "CellMarkerSource",
    "PanglaoDBSource",
    "ScTypeDBSource",
    "MyGeneSource",
    "NCBIGeneSource",
    "HPASource",
    "GTExSource",
    "CellxGeneSource",
    "MSigDBSource",
    "OMIMSource",
    "DisGeNETSource",
    "EnsemblSource",
    "OpenTargetsSource",
    "COSMICSource",
    "PubMedSource",
    "PubTator3Source",
    "LitVar2Source",
]

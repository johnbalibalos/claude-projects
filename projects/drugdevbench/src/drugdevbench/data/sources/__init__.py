"""Data sources for fetching figures from open-access papers."""

from drugdevbench.data.sources.biorxiv import BioRxivSource
from drugdevbench.data.sources.pubmed import PubMedSource
from drugdevbench.data.sources.sourcedata import (
    SourceDataSource,
    download_sourcedata_figures,
)
from drugdevbench.data.sources.openpmc import (
    OpenPMCSource,
    download_openpmc_figures,
)

__all__ = [
    "BioRxivSource",
    "PubMedSource",
    "SourceDataSource",
    "OpenPMCSource",
    "download_sourcedata_figures",
    "download_openpmc_figures",
]

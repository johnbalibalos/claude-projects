"""
Paper download utilities for PMC/NCBI.

This library provides a unified interface for downloading papers from PMC,
including XML content and supplementary files via the OA webservice.

Usage:
    from paper_download import PMCClient

    client = PMCClient(email="user@example.com")
    pmc_ids = client.search_papers("OMIP[Title] AND Cytometry[Journal]")

    for pmc_id in pmc_ids:
        client.fetch_xml(pmc_id, output_dir="data/papers")
        client.fetch_supplementary(pmc_id, output_dir="data/papers")
"""

from .pmc_client import PMCClient

__all__ = ["PMCClient"]

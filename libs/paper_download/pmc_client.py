"""
PMC client for searching and downloading papers.

Uses NCBI E-utilities for search and fetch, and the PMC OA webservice
for supplementary files.
"""

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a download operation."""

    pmc_id: str
    status: str  # "success", "cached", "error", "invalid_xml"
    path: str | None = None
    error: str | None = None
    supplementary: list[Any] = field(default_factory=list)


@dataclass
class SupplementaryFile:
    """Information about a supplementary file."""

    filename: str
    format: str
    status: str  # "success", "cached", "error"
    error: str | None = None


@dataclass
class PaperMetadata:
    """Metadata extracted from a paper."""

    pmc_id: str
    title: str | None = None
    doi: str | None = None
    omip_id: str | None = None
    xml_path: str | None = None
    error: str | None = None


class PMCClient:
    """
    Client for PMC paper downloads.

    Uses:
    - NCBI E-utilities (eutils) for search and XML fetch
    - PMC OA webservice (oa.fcgi) for supplementary files

    Rate limiting: 3 requests/second without API key (0.35s between requests)
    """

    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    OA_API_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
    ID_CONVERTER_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"

    def __init__(
        self,
        email: str,
        rate_limit_delay: float = 0.35,
        timeout: float = 60.0,
    ):
        """
        Initialize PMC client.

        Args:
            email: Email for NCBI API (required by NCBI terms of use)
            rate_limit_delay: Delay between requests in seconds (default: 0.35)
            timeout: Request timeout in seconds (default: 60)
        """
        self.email = email
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._client.close()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def search_papers(
        self,
        query: str,
        max_results: int = 200,
    ) -> list[str]:
        """
        Search PMC for papers matching query.

        Args:
            query: PubMed query string (e.g., "OMIP[Title] AND Cytometry[Journal]")
            max_results: Maximum number of results to return

        Returns:
            List of PMC IDs (without "PMC" prefix)
        """
        url = f"{self.EUTILS_BASE}/esearch.fcgi"
        params = {
            "db": "pmc",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
        }

        response = self._client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        return data.get("esearchresult", {}).get("idlist", [])

    def fetch_xml(
        self,
        pmc_id: str,
        output_dir: Path | str,
    ) -> DownloadResult:
        """
        Fetch XML for a paper.

        Args:
            pmc_id: PMC ID (with or without "PMC" prefix)
            output_dir: Directory to save XML file

        Returns:
            DownloadResult with status and path
        """
        # Normalize PMC ID
        pmc_id = pmc_id.replace("PMC", "")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"PMC{pmc_id}.xml"

        # Check cache
        if output_path.exists():
            return DownloadResult(
                pmc_id=pmc_id,
                status="cached",
                path=str(output_path),
            )

        url = f"{self.EUTILS_BASE}/efetch.fcgi"
        params = {
            "db": "pmc",
            "id": pmc_id,
            "rettype": "xml",
            "email": self.email,
        }

        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()

            xml_content = response.text
            if xml_content.strip().startswith("<?xml") or xml_content.strip().startswith("<"):
                output_path.write_text(xml_content, encoding="utf-8")
                time.sleep(self.rate_limit_delay)
                return DownloadResult(
                    pmc_id=pmc_id,
                    status="success",
                    path=str(output_path),
                )
            else:
                return DownloadResult(
                    pmc_id=pmc_id,
                    status="invalid_xml",
                    error="Response not XML",
                )
        except Exception as e:
            return DownloadResult(
                pmc_id=pmc_id,
                status="error",
                error=str(e),
            )

    def fetch_supplementary(
        self,
        pmc_id: str,
        output_dir: Path | str,
    ) -> list[SupplementaryFile]:
        """
        Fetch supplementary files using PMC OA webservice.

        Args:
            pmc_id: PMC ID (with or without "PMC" prefix)
            output_dir: Directory to save files

        Returns:
            List of SupplementaryFile results
        """
        # Normalize PMC ID
        pmc_id = pmc_id.replace("PMC", "")
        output_dir = Path(output_dir)

        params = {"id": f"PMC{pmc_id}", "format": "json"}

        try:
            response = self._client.get(self.OA_API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            records = data.get("records", [])
            if not records:
                return []

            downloaded = []
            supp_dir = output_dir / f"PMC{pmc_id}_supp"

            for link in records[0].get("links", []):
                href = link.get("href")
                fmt = link.get("format", "unknown")

                if not href:
                    continue

                filename = href.split("/")[-1]
                out_path = supp_dir / filename

                if out_path.exists():
                    downloaded.append(SupplementaryFile(
                        filename=filename,
                        format=fmt,
                        status="cached",
                    ))
                    continue

                try:
                    supp_dir.mkdir(parents=True, exist_ok=True)
                    resp = self._client.get(href, timeout=120, follow_redirects=True)
                    resp.raise_for_status()
                    out_path.write_bytes(resp.content)
                    downloaded.append(SupplementaryFile(
                        filename=filename,
                        format=fmt,
                        status="success",
                    ))
                    time.sleep(self.rate_limit_delay)
                except Exception as e:
                    downloaded.append(SupplementaryFile(
                        filename=filename,
                        format=fmt,
                        status="error",
                        error=str(e),
                    ))

            return downloaded
        except Exception as e:
            return [SupplementaryFile(
                filename="",
                format="",
                status="error",
                error=str(e),
            )]

    def doi_to_pmcid(self, doi: str) -> str | None:
        """
        Convert DOI to PMCID using NCBI ID Converter API.

        Args:
            doi: The DOI to convert (e.g., "10.1002/cyto.a.24292")

        Returns:
            PMCID (e.g., "PMC7891234") or None if not found
        """
        params = {
            "ids": doi,
            "format": "json",
            "email": self.email,
        }

        try:
            response = self._client.get(self.ID_CONVERTER_URL, params=params)
            response.raise_for_status()
            data = response.json()
            time.sleep(self.rate_limit_delay)

            records = data.get("records", [])
            if records and "pmcid" in records[0]:
                pmcid = records[0]["pmcid"]
                logger.info(f"Converted DOI {doi} -> {pmcid}")
                return pmcid

            logger.warning(f"No PMCID found for DOI {doi}")
            return None
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.error(f"Error converting DOI {doi}: {e}")
            return None

    def fetch_pdf(self, pmc_id: str, output_dir: Path | str) -> Path | None:
        """
        Fetch PDF from PMC Open Access subset.

        Args:
            pmc_id: PMC ID (with or without "PMC" prefix)
            output_dir: Directory to save PDF file

        Returns:
            Path to downloaded PDF, or None if not available
        """
        pmc_id = pmc_id.replace("PMC", "")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"PMC{pmc_id}.pdf"

        if output_path.exists():
            return output_path

        params = {"id": f"PMC{pmc_id}", "format": "json"}
        try:
            response = self._client.get(self.OA_API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            time.sleep(self.rate_limit_delay)

            records = data.get("records", [])
            if not records:
                return None

            pdf_url = None
            for link in records[0].get("links", []):
                if link.get("format") == "pdf":
                    pdf_url = link.get("href")
                    break

            if not pdf_url:
                return None

            resp = self._client.get(pdf_url, timeout=120, follow_redirects=True)
            resp.raise_for_status()
            output_path.write_bytes(resp.content)
            time.sleep(self.rate_limit_delay)
            return output_path
        except Exception as e:
            logger.error(f"Error fetching PDF for PMC{pmc_id}: {e}")
            return None

    def extract_metadata(self, xml_path: Path | str) -> PaperMetadata:
        """
        Extract metadata from downloaded XML.

        Args:
            xml_path: Path to XML file

        Returns:
            PaperMetadata with extracted information
        """
        xml_path = Path(xml_path)
        pmc_id = xml_path.stem  # e.g., "PMC1234567"

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get title
            title_elem = root.find(".//article-title")
            title = " ".join(title_elem.itertext()) if title_elem is not None else None

            # Extract OMIP number from title
            omip_id = None
            if title:
                match = re.search(r"OMIP[- ]?(\d+)", title, re.IGNORECASE)
                if match:
                    omip_id = f"OMIP-{match.group(1).zfill(3)}"

            # Get DOI
            doi_elem = root.find(".//article-id[@pub-id-type='doi']")
            doi = doi_elem.text if doi_elem is not None else None

            # Get PMCID
            pmc_elem = root.find(".//article-id[@pub-id-type='pmc']")
            if pmc_elem is not None:
                pmc_id = f"PMC{pmc_elem.text}"

            return PaperMetadata(
                pmc_id=pmc_id,
                title=title,
                doi=doi,
                omip_id=omip_id,
                xml_path=str(xml_path),
            )
        except Exception as e:
            return PaperMetadata(
                pmc_id=pmc_id,
                error=str(e),
                xml_path=str(xml_path),
            )

    def download_papers(
        self,
        query: str,
        output_dir: Path | str,
        max_results: int = 200,
        with_supplementary: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> tuple[list[DownloadResult], list[PaperMetadata]]:
        """
        Search and download papers matching query.

        Args:
            query: PubMed query string
            output_dir: Directory to save files
            max_results: Maximum papers to download
            with_supplementary: Also download supplementary files
            progress_callback: Called with (current, total, status)

        Returns:
            Tuple of (download_results, metadata_list)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Search
        pmc_ids = self.search_papers(query, max_results)

        if not pmc_ids:
            return [], []

        # Download each paper
        results = []
        for i, pmc_id in enumerate(pmc_ids, 1):
            if progress_callback:
                progress_callback(i, len(pmc_ids), f"PMC{pmc_id}")

            result = self.fetch_xml(pmc_id, output_dir)
            results.append(result)

            if with_supplementary and result.status in ("success", "cached"):
                supp_results = self.fetch_supplementary(pmc_id, output_dir)
                # Attach supplementary info to result (for logging)
                result.supplementary = supp_results

        # Extract metadata from downloaded XMLs
        metadata = []
        for xml_file in sorted(output_dir.glob("PMC*.xml")):
            info = self.extract_metadata(xml_file)
            if info.omip_id:  # Only include papers with OMIP IDs
                metadata.append(info)

        return results, metadata

    def save_index(
        self,
        metadata: list[PaperMetadata],
        output_path: Path | str,
    ) -> None:
        """
        Save paper index to JSON file.

        Args:
            metadata: List of paper metadata
            output_path: Path to save index
        """
        output_path = Path(output_path)
        index_data = [
            {
                "pmc_id": m.pmc_id,
                "omip_id": m.omip_id,
                "title": m.title,
                "doi": m.doi,
                "xml_path": m.xml_path,
            }
            for m in metadata
            if m.omip_id
        ]

        with open(output_path, "w") as f:
            json.dump(index_data, f, indent=2)


def extract_gating_section(xml_content: str) -> str | None:
    """
    Extract gating-related sections from PMC XML.

    Searches for sections, figure captions, and table captions containing
    gating strategy keywords. Useful for flow cytometry papers (OMIP, etc.).

    Args:
        xml_content: Full text XML content from PMC

    Returns:
        Concatenated gating-related text, or None if not found
    """
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")
        return None

    sections = []

    gating_keywords = [
        "gating", "gate", "gating strategy", "gating scheme",
        "gating hierarchy", "population", "subset",
        "flow cytometry", "facs",
    ]

    # Search in <sec> elements
    for sec in root.iter("sec"):
        title_elem = sec.find("title")
        title = ""
        if title_elem is not None and title_elem.text:
            title = title_elem.text.lower()

        text_parts = []
        for elem in sec.iter():
            if elem.text:
                text_parts.append(elem.text)
            if elem.tail:
                text_parts.append(elem.tail)
        full_text = " ".join(text_parts)

        is_gating_related = any(
            kw in title or kw in full_text.lower()[:1000]
            for kw in gating_keywords
        )
        if is_gating_related:
            sections.append(f"=== Section: {title or 'Untitled'} ===\n{full_text}")

    # Search in figure captions
    for fig in root.iter("fig"):
        caption = fig.find("caption")
        if caption is not None:
            caption_text = " ".join(caption.itertext())
            if any(kw in caption_text.lower() for kw in ["gating", "gate", "strategy", "hierarchy"]):
                fig_id = fig.get("id", "unknown")
                sections.append(f"=== Figure {fig_id} Caption ===\n{caption_text}")

    # Search in table captions
    for table_wrap in root.iter("table-wrap"):
        caption = table_wrap.find("caption")
        if caption is not None:
            caption_text = " ".join(caption.itertext())
            if any(kw in caption_text.lower() for kw in ["panel", "marker", "antibody", "gating"]):
                table_id = table_wrap.get("id", "unknown")
                sections.append(f"=== Table {table_id} Caption ===\n{caption_text}")

    if not sections:
        return None

    return "\n\n---\n\n".join(sections)

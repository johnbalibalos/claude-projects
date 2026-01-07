"""
PMC API client for downloading OMIP papers.

This module provides:
- PMCClient: Low-level client for PMC API with caching and rate limiting
- OMIPPaperDownloader: High-level interface for downloading all benchmark papers

OMIP papers are published in Cytometry Part A and available through PubMed Central.
We use the NCBI E-utilities API to programmatically download and cache papers.

API Endpoints:
- ID Converter: Convert DOI -> PMCID
- efetch: Fetch full text XML
- OA API: Access Open Access PDFs

Rate Limits:
- Without API key: 3 requests/second
- With API key: 10 requests/second
"""

from __future__ import annotations

import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of downloading a single OMIP paper."""

    omip_id: str
    status: str  # "success", "no_doi", "no_pmcid", "error"
    doi: str | None = None
    pmcid: str | None = None
    xml_downloaded: bool = False
    pdf_downloaded: bool = False
    gating_section_found: bool = False
    error: str | None = None
    gating_section_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "omip_id": self.omip_id,
            "status": self.status,
            "doi": self.doi,
            "pmcid": self.pmcid,
            "xml_downloaded": self.xml_downloaded,
            "pdf_downloaded": self.pdf_downloaded,
            "gating_section_found": self.gating_section_found,
            "gating_section_length": self.gating_section_length,
            "error": self.error,
        }


class PMCClient:
    """
    Client for downloading papers from PubMed Central.

    Handles:
    - DOI to PMCID conversion
    - Full text XML download
    - PDF download from Open Access subset
    - Gating section extraction from XML
    - Local caching to avoid redundant downloads
    - Rate limiting to respect NCBI guidelines

    Usage:
        client = PMCClient(
            cache_dir=Path("data/papers/pmc"),
            email="your@email.com",
            api_key="your_ncbi_api_key",  # Optional
        )

        # Convert DOI to PMCID
        pmcid = client.doi_to_pmcid("10.1002/cyto.a.24292")

        # Download full text
        xml_content = client.fetch_full_text_xml(pmcid)

        # Extract gating section
        gating_text = client.extract_gating_section(xml_content)
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    ID_CONVERTER_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

    def __init__(
        self,
        cache_dir: Path,
        email: str,
        api_key: str | None = None,
        rate_limit_delay: float = 0.34,  # 3 req/sec without API key
    ):
        """
        Initialize PMC client.

        Args:
            cache_dir: Directory to cache downloaded files
            email: Email address (required by NCBI)
            api_key: Optional NCBI API key (increases rate limit to 10 req/sec)
            rate_limit_delay: Delay between requests in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.email = email
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay if not api_key else 0.1
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Respect NCBI rate limits by sleeping if necessary."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _get_cache_path(self, identifier: str, ext: str) -> Path:
        """Get cache file path for an identifier."""
        # Sanitize identifier for filesystem
        safe_id = identifier.replace("/", "_").replace(":", "_").replace(".", "_")
        return self.cache_dir / f"{safe_id}.{ext}"

    def doi_to_pmcid(self, doi: str) -> str | None:
        """
        Convert DOI to PMCID using NCBI ID Converter API.

        Args:
            doi: The DOI to convert (e.g., "10.1002/cyto.a.24292")

        Returns:
            PMCID (e.g., "PMC7891234") or None if not found
        """
        cache_path = self._get_cache_path(doi, "pmcid.json")

        # Check cache first
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
                return data.get("pmcid")

        self._rate_limit()

        params = {
            "ids": doi,
            "format": "json",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(self.ID_CONVERTER_URL, params=params)
                response.raise_for_status()
                data = response.json()

            records = data.get("records", [])
            if records and "pmcid" in records[0]:
                pmcid = records[0]["pmcid"]

                # Cache the result
                with open(cache_path, "w") as f:
                    json.dump({"doi": doi, "pmcid": pmcid, "timestamp": datetime.now().isoformat()}, f)

                logger.info(f"Converted DOI {doi} -> {pmcid}")
                return pmcid
            else:
                # Cache negative result
                with open(cache_path, "w") as f:
                    json.dump({"doi": doi, "pmcid": None, "error": "not_found", "timestamp": datetime.now().isoformat()}, f)
                logger.warning(f"No PMCID found for DOI {doi}")
                return None

        except httpx.HTTPError as e:
            logger.error(f"HTTP error converting DOI {doi}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error converting DOI {doi}: {e}")
            return None

    def fetch_full_text_xml(self, pmcid: str) -> str | None:
        """
        Fetch full text XML from PMC.

        Args:
            pmcid: The PMCID (e.g., "PMC7891234")

        Returns:
            XML content as string, or None if not available
        """
        cache_path = self._get_cache_path(pmcid, "xml")

        # Check cache first
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        self._rate_limit()

        # Use efetch to get full text XML
        url = f"{self.BASE_URL}/efetch.fcgi"
        params = {
            "db": "pmc",
            "id": pmcid.replace("PMC", ""),  # efetch wants numeric ID
            "rettype": "xml",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            with httpx.Client(timeout=60) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                xml_content = response.text

                # Validate it's actually XML
                if not xml_content.strip().startswith("<?xml") and not xml_content.strip().startswith("<"):
                    logger.warning(f"Response for {pmcid} is not XML")
                    return None

                # Cache the result
                cache_path.write_text(xml_content, encoding="utf-8")
                logger.info(f"Downloaded XML for {pmcid} ({len(xml_content)} bytes)")
                return xml_content

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching XML for {pmcid}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching XML for {pmcid}: {e}")
            return None

    def fetch_pdf(self, pmcid: str) -> Path | None:
        """
        Fetch PDF from PMC Open Access subset.

        Note: Not all PMC articles have PDFs available through OA API.

        Args:
            pmcid: The PMCID

        Returns:
            Path to cached PDF file, or None if not available
        """
        cache_path = self._get_cache_path(pmcid, "pdf")

        # Check cache first
        if cache_path.exists():
            return cache_path

        self._rate_limit()

        # First, query OA API to get file links
        params = {"id": pmcid, "format": "json"}

        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(self.OA_URL, params=params)
                response.raise_for_status()
                data = response.json()

            records = data.get("records", [])
            if not records:
                logger.warning(f"No OA records for {pmcid}")
                return None

            # Find PDF link
            pdf_url = None
            for link in records[0].get("links", []):
                if link.get("format") == "pdf":
                    pdf_url = link.get("href")
                    break

            if not pdf_url:
                logger.warning(f"No PDF available for {pmcid}")
                return None

            # Download PDF
            self._rate_limit()
            with httpx.Client(timeout=120, follow_redirects=True) as client:
                response = client.get(pdf_url)
                response.raise_for_status()

                cache_path.write_bytes(response.content)
                logger.info(f"Downloaded PDF for {pmcid} ({len(response.content)} bytes)")
                return cache_path

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching PDF for {pmcid}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching PDF for {pmcid}: {e}")
            return None

    def extract_gating_section(self, xml_content: str) -> str | None:
        """
        Extract gating-related sections from PMC XML.

        Looks for:
        - Sections with "gating" in title or early content
        - Figure legends mentioning "gating strategy"
        - Methods sections about flow cytometry
        - Tables describing panel/gating

        Args:
            xml_content: Full text XML content

        Returns:
            Extracted gating-related text, or None if not found
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            return None

        sections = []

        # Keywords to search for
        gating_keywords = [
            "gating",
            "gate",
            "gating strategy",
            "gating scheme",
            "gating hierarchy",
            "population",
            "subset",
            "flow cytometry",
            "facs",
        ]

        # Search in <sec> elements (sections)
        for sec in root.iter("sec"):
            title_elem = sec.find("title")
            title = ""
            if title_elem is not None and title_elem.text:
                title = title_elem.text.lower()

            # Get full text of section
            text_parts = []
            for elem in sec.iter():
                if elem.text:
                    text_parts.append(elem.text)
                if elem.tail:
                    text_parts.append(elem.tail)
            full_text = " ".join(text_parts)
            full_text_lower = full_text.lower()

            # Check if gating-related
            is_gating_related = any(
                kw in title or kw in full_text_lower[:1000]
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

        # Search in table captions/content
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

    def get_article_metadata(self, xml_content: str) -> dict[str, Any]:
        """
        Extract article metadata from PMC XML.

        Args:
            xml_content: Full text XML content

        Returns:
            Dict with title, authors, journal, year, etc.
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            return {}

        metadata = {}

        # Title
        title_elem = root.find(".//article-title")
        if title_elem is not None:
            metadata["title"] = " ".join(title_elem.itertext())

        # Journal
        journal_elem = root.find(".//journal-title")
        if journal_elem is not None:
            metadata["journal"] = journal_elem.text

        # Year
        year_elem = root.find(".//pub-date/year")
        if year_elem is not None:
            metadata["year"] = year_elem.text

        # DOI
        doi_elem = root.find(".//article-id[@pub-id-type='doi']")
        if doi_elem is not None:
            metadata["doi"] = doi_elem.text

        # PMCID
        pmc_elem = root.find(".//article-id[@pub-id-type='pmc']")
        if pmc_elem is not None:
            metadata["pmcid"] = f"PMC{pmc_elem.text}"

        return metadata


class OMIPPaperDownloader:
    """
    High-level interface for downloading all OMIP papers for the benchmark.

    Usage:
        downloader = OMIPPaperDownloader(
            test_cases_dir=Path("data/ground_truth"),
            cache_dir=Path("data/papers"),
            email="your@email.com",
            api_key=os.environ.get("NCBI_API_KEY"),
        )

        # Download all papers
        results = downloader.download_all()

        # Check results
        for omip_id, result in results.items():
            print(f"{omip_id}: {result.status}")

        # Get content for specific OMIP
        content = downloader.get_paper_content("OMIP-044")
    """

    def __init__(
        self,
        test_cases_dir: Path,
        cache_dir: Path,
        email: str,
        api_key: str | None = None,
    ):
        """
        Initialize downloader.

        Args:
            test_cases_dir: Directory containing test case JSON files
            cache_dir: Directory to cache downloaded papers
            email: Email address for NCBI API
            api_key: Optional NCBI API key
        """
        self.test_cases_dir = Path(test_cases_dir)
        self.cache_dir = Path(cache_dir)
        self.client = PMCClient(
            cache_dir=self.cache_dir / "pmc",
            email=email,
            api_key=api_key,
        )

    def load_test_cases(self) -> list[dict[str, Any]]:
        """Load all test case JSON files."""
        cases = []
        for path in sorted(self.test_cases_dir.glob("*.json")):
            with open(path) as f:
                cases.append(json.load(f))
        return cases

    def download_all(
        self,
        formats: list[str] | None = None,
        verbose: bool = True,
    ) -> dict[str, DownloadResult]:
        """
        Download all OMIP papers.

        Args:
            formats: List of formats to download ("xml", "pdf"). Default: ["xml"]
            verbose: Print progress information

        Returns:
            Dict mapping OMIP ID to DownloadResult
        """
        if formats is None:
            formats = ["xml"]

        cases = self.load_test_cases()
        results: dict[str, DownloadResult] = {}

        if verbose:
            print(f"Downloading {len(cases)} OMIP papers...")

        for i, case in enumerate(cases, 1):
            omip_id = case["omip_id"]
            doi = case.get("doi")

            if verbose:
                print(f"[{i}/{len(cases)}] {omip_id}...", end=" ", flush=True)

            if not doi:
                results[omip_id] = DownloadResult(
                    omip_id=omip_id,
                    status="no_doi",
                )
                if verbose:
                    print("NO DOI")
                continue

            # Convert DOI to PMCID
            pmcid = self.client.doi_to_pmcid(doi)
            if not pmcid:
                results[omip_id] = DownloadResult(
                    omip_id=omip_id,
                    status="no_pmcid",
                    doi=doi,
                )
                if verbose:
                    print("NO PMCID")
                continue

            result = DownloadResult(
                omip_id=omip_id,
                status="success",
                doi=doi,
                pmcid=pmcid,
            )

            # Download XML
            if "xml" in formats:
                xml_content = self.client.fetch_full_text_xml(pmcid)
                result.xml_downloaded = xml_content is not None

                if xml_content:
                    gating_section = self.client.extract_gating_section(xml_content)
                    result.gating_section_found = gating_section is not None
                    result.gating_section_length = len(gating_section) if gating_section else 0

            # Download PDF
            if "pdf" in formats:
                pdf_path = self.client.fetch_pdf(pmcid)
                result.pdf_downloaded = pdf_path is not None

            results[omip_id] = result

            if verbose:
                status_parts = []
                if result.xml_downloaded:
                    status_parts.append("XML")
                if result.pdf_downloaded:
                    status_parts.append("PDF")
                if result.gating_section_found:
                    status_parts.append(f"Gating({result.gating_section_length})")
                print(" | ".join(status_parts) if status_parts else "PARTIAL")

        return results

    def get_paper_content(self, omip_id: str) -> dict[str, Any]:
        """
        Get cached content for a specific OMIP paper.

        Args:
            omip_id: OMIP identifier (e.g., "OMIP-044")

        Returns:
            Dict with paper content (xml, gating_section, pdf_path, metadata)

        Raises:
            FileNotFoundError: If test case not found
            ValueError: If DOI/PMCID not available
        """
        # Find test case file
        case_filename = omip_id.lower().replace("-", "_") + ".json"
        case_path = self.test_cases_dir / case_filename
        if not case_path.exists():
            raise FileNotFoundError(f"No test case found for {omip_id}")

        with open(case_path) as f:
            case = json.load(f)

        doi = case.get("doi")
        if not doi:
            raise ValueError(f"No DOI for {omip_id}")

        pmcid = self.client.doi_to_pmcid(doi)
        if not pmcid:
            raise ValueError(f"Could not resolve PMCID for DOI {doi}")

        result: dict[str, Any] = {
            "omip_id": omip_id,
            "doi": doi,
            "pmcid": pmcid,
        }

        # Get XML content
        xml_path = self.client._get_cache_path(pmcid, "xml")
        if xml_path.exists():
            xml_content = xml_path.read_text(encoding="utf-8")
            result["xml"] = xml_content
            result["gating_section"] = self.client.extract_gating_section(xml_content)
            result["metadata"] = self.client.get_article_metadata(xml_content)

        # Get PDF path
        pdf_path = self.client._get_cache_path(pmcid, "pdf")
        if pdf_path.exists():
            result["pdf_path"] = pdf_path

        return result

    def get_download_summary(self, results: dict[str, DownloadResult]) -> dict[str, Any]:
        """
        Generate summary statistics for download results.

        Args:
            results: Dict of DownloadResult objects

        Returns:
            Summary statistics dict
        """
        total = len(results)
        success = sum(1 for r in results.values() if r.status == "success")
        no_doi = sum(1 for r in results.values() if r.status == "no_doi")
        no_pmcid = sum(1 for r in results.values() if r.status == "no_pmcid")
        xml_count = sum(1 for r in results.values() if r.xml_downloaded)
        pdf_count = sum(1 for r in results.values() if r.pdf_downloaded)
        gating_count = sum(1 for r in results.values() if r.gating_section_found)

        return {
            "total": total,
            "success": success,
            "no_doi": no_doi,
            "no_pmcid": no_pmcid,
            "xml_downloaded": xml_count,
            "pdf_downloaded": pdf_count,
            "gating_section_found": gating_count,
            "success_rate": success / total if total > 0 else 0,
            "gating_extraction_rate": gating_count / xml_count if xml_count > 0 else 0,
        }


def main():
    """CLI entry point for downloading OMIP papers."""
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Download OMIP papers from PMC")
    parser.add_argument(
        "--test-cases-dir",
        type=Path,
        default=Path("data/ground_truth"),
        help="Directory containing test case JSON files",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/papers"),
        help="Directory to cache downloaded papers",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["xml"],
        choices=["xml", "pdf"],
        help="Formats to download",
    )
    parser.add_argument(
        "--omip",
        type=str,
        help="Download specific OMIP only (e.g., OMIP-044)",
    )

    args = parser.parse_args()

    email = os.environ.get("NCBI_EMAIL")
    api_key = os.environ.get("NCBI_API_KEY")

    if not email:
        print("Error: NCBI_EMAIL environment variable not set")
        return 1

    downloader = OMIPPaperDownloader(
        test_cases_dir=args.test_cases_dir,
        cache_dir=args.cache_dir,
        email=email,
        api_key=api_key,
    )

    if args.omip:
        # Download single paper
        try:
            content = downloader.get_paper_content(args.omip)
            print(f"OMIP: {content['omip_id']}")
            print(f"DOI: {content['doi']}")
            print(f"PMCID: {content['pmcid']}")
            if content.get("gating_section"):
                print(f"Gating section: {len(content['gating_section'])} chars")
                print("\n--- Preview ---")
                print(content["gating_section"][:1000])
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return 1
    else:
        # Download all papers
        results = downloader.download_all(formats=args.formats)
        summary = downloader.get_download_summary(results)

        print("\n=== Summary ===")
        print(f"Total: {summary['total']}")
        print(f"Success: {summary['success']} ({summary['success_rate']:.1%})")
        print(f"No DOI: {summary['no_doi']}")
        print(f"No PMCID: {summary['no_pmcid']}")
        print(f"XML downloaded: {summary['xml_downloaded']}")
        print(f"Gating sections found: {summary['gating_section_found']} ({summary['gating_extraction_rate']:.1%})")

        # Save results
        results_path = args.cache_dir / "download_results.json"
        with open(results_path, "w") as f:
            json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)
        print(f"\nResults saved to {results_path}")

    return 0


if __name__ == "__main__":
    exit(main())

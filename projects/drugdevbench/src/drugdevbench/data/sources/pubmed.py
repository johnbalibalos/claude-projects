"""PubMed Central data source for fetching open-access figures."""

import re
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

from drugdevbench.data.schemas import Figure, FigureType


class PubMedSource:
    """Fetch figures and metadata from PubMed Central open-access articles."""

    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    PMC_BASE = "https://www.ncbi.nlm.nih.gov/pmc/articles"

    def __init__(
        self,
        cache_dir: Path | None = None,
        rate_limit_s: float = 0.34,  # NCBI recommends no more than 3 requests/second
        api_key: str | None = None,
    ):
        """Initialize PubMed source.

        Args:
            cache_dir: Directory to cache downloaded files
            rate_limit_s: Seconds to wait between requests (default respects NCBI limits)
            api_key: NCBI API key for higher rate limits
        """
        self.cache_dir = cache_dir or Path("data/cache/pubmed")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_s = rate_limit_s
        self.api_key = api_key
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_s:
            time.sleep(self.rate_limit_s - elapsed)
        self._last_request_time = time.time()

    def _add_api_key(self, params: dict) -> dict:
        """Add API key to request parameters if available."""
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search_papers(
        self,
        query: str,
        max_results: int = 100,
        open_access_only: bool = True,
    ) -> list[str]:
        """Search for papers matching a query.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            open_access_only: Only return open-access articles

        Returns:
            List of PMC IDs
        """
        self._rate_limit()

        # Add open access filter if requested
        if open_access_only:
            query = f"({query}) AND open access[filter]"

        params = self._add_api_key({
            "db": "pmc",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        })

        try:
            response = requests.get(
                f"{self.EUTILS_BASE}/esearch.fcgi",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            return data.get("esearchresult", {}).get("idlist", [])

        except requests.RequestException as e:
            print(f"Error searching PubMed: {e}")
            return []

    def get_paper_details(self, pmc_id: str) -> dict[str, Any] | None:
        """Get detailed metadata for a paper.

        Args:
            pmc_id: PMC ID of the paper (e.g., 'PMC1234567' or '1234567')

        Returns:
            Paper metadata dictionary or None if not found
        """
        self._rate_limit()

        # Normalize PMC ID
        if not pmc_id.startswith("PMC"):
            pmc_id = f"PMC{pmc_id}"

        params = self._add_api_key({
            "db": "pmc",
            "id": pmc_id.replace("PMC", ""),
            "retmode": "xml",
        })

        try:
            response = requests.get(
                f"{self.EUTILS_BASE}/efetch.fcgi",
                params=params,
                timeout=30,
            )
            response.raise_for_status()

            # Parse XML response
            root = ElementTree.fromstring(response.content)

            # Extract metadata
            article = root.find(".//article")
            if article is None:
                return None

            title_elem = article.find(".//article-title")
            title = title_elem.text if title_elem is not None else None

            # Get DOI
            doi = None
            for article_id in article.findall(".//article-id"):
                if article_id.get("pub-id-type") == "doi":
                    doi = article_id.text
                    break

            # Get authors
            authors = []
            for contrib in article.findall(".//contrib[@contrib-type='author']"):
                surname = contrib.find(".//surname")
                given = contrib.find(".//given-names")
                if surname is not None:
                    name = surname.text
                    if given is not None:
                        name = f"{given.text} {name}"
                    authors.append(name)

            # Get abstract
            abstract_elem = article.find(".//abstract")
            abstract = ""
            if abstract_elem is not None:
                abstract = " ".join(abstract_elem.itertext())

            return {
                "pmc_id": pmc_id,
                "title": title,
                "doi": doi,
                "authors": authors,
                "abstract": abstract,
            }

        except (requests.RequestException, ElementTree.ParseError) as e:
            print(f"Error fetching paper details: {e}")
            return None

    def get_figure_urls(self, pmc_id: str) -> list[dict[str, str]]:
        """Extract figure URLs from a paper's HTML page.

        Args:
            pmc_id: PMC ID of the paper

        Returns:
            List of dicts with 'url', 'caption', 'figure_num' keys
        """
        self._rate_limit()

        # Normalize PMC ID
        if not pmc_id.startswith("PMC"):
            pmc_id = f"PMC{pmc_id}"

        paper_url = f"{self.PMC_BASE}/{pmc_id}/"

        try:
            response = requests.get(paper_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            figures = []

            # Find figure elements in PMC HTML
            for fig_div in soup.find_all("div", class_="fig"):
                # Find the image
                img = fig_div.find("img")
                if not img:
                    continue

                img_src = img.get("src", "")
                if not img_src:
                    continue

                # Make URL absolute
                if img_src.startswith("/"):
                    img_src = f"https://www.ncbi.nlm.nih.gov{img_src}"
                elif not img_src.startswith("http"):
                    img_src = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/{img_src}"

                # Try to get high-res version
                # PMC often has larger versions available
                img_src = img_src.replace("_small.", "_large.").replace("_thumb.", ".")

                # Get caption
                caption_elem = fig_div.find("div", class_="caption")
                caption = caption_elem.get_text(strip=True) if caption_elem else ""

                # Extract figure number
                fig_num_match = re.search(r"Figure\s*(\d+)", caption, re.IGNORECASE)
                if not fig_num_match:
                    # Try from the figure ID
                    fig_id = fig_div.get("id", "")
                    fig_num_match = re.search(r"[Ff]ig(?:ure)?(\d+)", fig_id)

                fig_num = fig_num_match.group(1) if fig_num_match else "0"

                figures.append({
                    "url": img_src,
                    "caption": caption,
                    "figure_num": fig_num,
                })

            return figures

        except requests.RequestException as e:
            print(f"Error fetching paper HTML: {e}")
            return []

    def download_figure(
        self,
        url: str,
        output_path: Path,
    ) -> bool:
        """Download a figure image.

        Args:
            url: URL of the figure image
            output_path: Path to save the image

        Returns:
            True if successful, False otherwise
        """
        self._rate_limit()

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)

            return True

        except requests.RequestException as e:
            print(f"Error downloading figure: {e}")
            return False

    def create_figure_from_paper(
        self,
        pmc_id: str,
        figure_num: int,
        figure_type: FigureType,
        output_dir: Path,
    ) -> Figure | None:
        """Create a Figure object from a PMC paper.

        Args:
            pmc_id: PMC ID of the paper
            figure_num: Figure number to extract
            figure_type: Type of figure
            output_dir: Directory to save the figure image

        Returns:
            Figure object or None if extraction failed
        """
        # Normalize PMC ID
        if not pmc_id.startswith("PMC"):
            pmc_id = f"PMC{pmc_id}"

        # Get paper details
        paper = self.get_paper_details(pmc_id)
        if not paper:
            return None

        # Get figure URLs
        figures = self.get_figure_urls(pmc_id)
        target_fig = None
        for fig in figures:
            if fig["figure_num"] == str(figure_num):
                target_fig = fig
                break

        if not target_fig:
            print(f"Figure {figure_num} not found in paper {pmc_id}")
            return None

        # Generate figure ID
        figure_id = f"pmc_{pmc_id.replace('PMC', '')}_fig{figure_num}"

        # Download figure
        output_path = output_dir / f"{figure_id}.png"
        if not self.download_figure(target_fig["url"], output_path):
            return None

        return Figure(
            figure_id=figure_id,
            figure_type=figure_type,
            image_path=str(output_path),
            legend_text=target_fig["caption"],
            paper_doi=paper.get("doi"),
            paper_title=paper.get("title"),
            source="pubmed",
            metadata={
                "pmc_id": pmc_id,
                "authors": paper.get("authors"),
            },
        )

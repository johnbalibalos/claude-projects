"""bioRxiv data source for fetching open-access figures."""

import re
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from drugdevbench.data.schemas import Figure, FigureType


class BioRxivSource:
    """Fetch figures and metadata from bioRxiv preprints."""

    BASE_API_URL = "https://api.biorxiv.org"
    BASE_URL = "https://www.biorxiv.org"

    def __init__(self, cache_dir: Path | None = None, rate_limit_s: float = 1.0):
        """Initialize bioRxiv source.

        Args:
            cache_dir: Directory to cache downloaded files
            rate_limit_s: Seconds to wait between requests
        """
        self.cache_dir = cache_dir or Path("data/cache/biorxiv")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_s = rate_limit_s
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_s:
            time.sleep(self.rate_limit_s - elapsed)
        self._last_request_time = time.time()

    def search_papers(
        self,
        query: str,
        server: str = "biorxiv",
        start_date: str = "2020-01-01",
        end_date: str = "2025-12-31",
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Search for papers matching a query.

        Args:
            query: Search query (currently uses date-based API)
            server: Server to search ('biorxiv' or 'medrxiv')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_results: Maximum number of results to return

        Returns:
            List of paper metadata dictionaries
        """
        self._rate_limit()

        # bioRxiv API uses date range endpoint
        # Note: This API doesn't support text search, only date ranges
        url = f"{self.BASE_API_URL}/details/{server}/{start_date}/{end_date}/0"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            papers = data.get("collection", [])
            # Filter by query if provided (client-side filtering)
            if query:
                query_lower = query.lower()
                papers = [
                    p for p in papers
                    if query_lower in p.get("title", "").lower()
                    or query_lower in p.get("abstract", "").lower()
                ]

            return papers[:max_results]

        except requests.RequestException as e:
            print(f"Error fetching from bioRxiv API: {e}")
            return []

    def get_paper_details(self, doi: str) -> dict[str, Any] | None:
        """Get detailed metadata for a paper.

        Args:
            doi: DOI of the paper (e.g., '10.1101/2024.01.01.123456')

        Returns:
            Paper metadata dictionary or None if not found
        """
        self._rate_limit()

        # Extract the bioRxiv-specific part of the DOI
        biorxiv_id = doi.replace("10.1101/", "")
        url = f"{self.BASE_API_URL}/details/biorxiv/{biorxiv_id}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            collection = data.get("collection", [])
            if collection:
                return collection[0]
            return None

        except requests.RequestException as e:
            print(f"Error fetching paper details: {e}")
            return None

    def get_figure_urls(self, doi: str) -> list[dict[str, str]]:
        """Extract figure URLs from a paper's HTML page.

        Args:
            doi: DOI of the paper

        Returns:
            List of dicts with 'url', 'caption', 'figure_num' keys
        """
        self._rate_limit()

        # Construct the paper URL
        biorxiv_id = doi.replace("10.1101/", "")
        paper_url = f"{self.BASE_URL}/content/{doi}"

        try:
            response = requests.get(paper_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            figures = []

            # Find figure elements
            for fig in soup.find_all("figure"):
                img = fig.find("img")
                if not img:
                    continue

                img_src = img.get("src", "")
                if not img_src:
                    continue

                # Make URL absolute
                if img_src.startswith("/"):
                    img_src = f"{self.BASE_URL}{img_src}"

                # Get caption
                caption_elem = fig.find("figcaption")
                caption = caption_elem.get_text(strip=True) if caption_elem else ""

                # Extract figure number from caption
                fig_num_match = re.search(r"Figure\s*(\d+)", caption, re.IGNORECASE)
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
        doi: str,
        figure_num: int,
        figure_type: FigureType,
        output_dir: Path,
    ) -> Figure | None:
        """Create a Figure object from a bioRxiv paper.

        Args:
            doi: DOI of the paper
            figure_num: Figure number to extract
            figure_type: Type of figure
            output_dir: Directory to save the figure image

        Returns:
            Figure object or None if extraction failed
        """
        # Get paper details
        paper = self.get_paper_details(doi)
        if not paper:
            return None

        # Get figure URLs
        figures = self.get_figure_urls(doi)
        target_fig = None
        for fig in figures:
            if fig["figure_num"] == str(figure_num):
                target_fig = fig
                break

        if not target_fig:
            print(f"Figure {figure_num} not found in paper {doi}")
            return None

        # Generate figure ID
        figure_id = f"biorxiv_{doi.replace('/', '_').replace('.', '_')}_fig{figure_num}"

        # Download figure
        output_path = output_dir / f"{figure_id}.png"
        if not self.download_figure(target_fig["url"], output_path):
            return None

        return Figure(
            figure_id=figure_id,
            figure_type=figure_type,
            image_path=str(output_path),
            legend_text=target_fig["caption"],
            paper_doi=doi,
            paper_title=paper.get("title"),
            source="biorxiv",
            metadata={
                "authors": paper.get("authors"),
                "date": paper.get("date"),
                "category": paper.get("category"),
            },
        )

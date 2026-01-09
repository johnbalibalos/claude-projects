"""
Paper content extraction from PMC XML and PDF files.

This module provides functions to:
1. Extract tables from PMC XML (panel tables, gating tables)
2. Extract methods/results sections from XML
3. Extract figures from PDF (requires pdf2image)
4. Classify and identify relevant content (panel vs gating tables)

Example usage:
    from curation.paper_parser import PaperParser

    parser = PaperParser()
    content = parser.extract_all("OMIP-069")

    # Get panel table
    panel_table = content.get_panel_table()

    # Get methods section text
    methods = content.methods_text
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Optional imports for PDF processing
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class ExtractedTable:
    """A table extracted from a paper."""
    table_id: str | None
    caption: str
    headers: list[str]
    rows: list[list[str]]
    table_type: str  # 'panel', 'gating', 'results', 'unknown'
    source_location: str  # e.g., "Table 1", "Supplementary Table S1"

    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.headers:
            return ""

        lines = []

        # Header row
        lines.append("| " + " | ".join(self.headers) + " |")

        # Separator
        lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")

        # Data rows
        for row in self.rows:
            # Pad row if needed
            padded = row + [""] * (len(self.headers) - len(row))
            lines.append("| " + " | ".join(padded[:len(self.headers)]) + " |")

        return "\n".join(lines)

    def get_column(self, name: str) -> list[str]:
        """Get all values from a column by header name."""
        name_lower = name.lower()
        for i, header in enumerate(self.headers):
            if header.lower() == name_lower or name_lower in header.lower():
                return [row[i] if i < len(row) else "" for row in self.rows]
        return []


@dataclass
class ExtractedFigure:
    """A figure extracted from a paper."""
    figure_id: str | None
    caption: str
    image_path: Path | None  # Path to extracted image file
    image_data: bytes | None  # Raw image data (for in-memory processing)
    figure_type: str  # 'gating', 'results', 'schematic', 'unknown'
    page_number: int | None


@dataclass
class PaperContent:
    """All extracted content from a paper."""
    omip_id: str | None
    pmcid: str | None
    doi: str | None
    title: str
    abstract: str
    methods_text: str
    results_text: str
    tables: list[ExtractedTable]
    figures: list[ExtractedFigure]
    raw_xml: str | None = None

    def get_panel_table(self) -> ExtractedTable | None:
        """Get the panel/reagent table if found."""
        for table in self.tables:
            if table.table_type == 'panel':
                return table
        return None

    def get_gating_table(self) -> ExtractedTable | None:
        """Get the gating strategy table if found."""
        for table in self.tables:
            if table.table_type == 'gating':
                return table
        return None

    def get_gating_figures(self) -> list[ExtractedFigure]:
        """Get figures that appear to show gating strategies."""
        return [f for f in self.figures if f.figure_type == 'gating']


class PaperParser:
    """
    Extract structured content from OMIP papers.

    Handles PMC XML files and optionally PDF files for figure extraction.
    """

    def __init__(self, papers_dir: Path | str = "data/papers/pmc"):
        self.papers_dir = Path(papers_dir)

    def extract_all(self, omip_id: str) -> PaperContent | None:
        """
        Extract all content from an OMIP paper.

        Args:
            omip_id: OMIP identifier (e.g., "OMIP-069") or PMC ID

        Returns:
            PaperContent with all extracted data, or None if not found
        """
        # Find paper files
        xml_path = self._find_xml(omip_id)
        if not xml_path:
            return None

        pdf_path = self._find_pdf(omip_id)

        # Extract from XML
        content = self._extract_from_xml(xml_path)

        # Extract figures from PDF if available
        if pdf_path and PDF2IMAGE_AVAILABLE:
            figures = self._extract_figures_from_pdf(pdf_path)
            content.figures.extend(figures)

        return content

    def _find_xml(self, identifier: str) -> Path | None:
        """Find XML file for an OMIP paper."""
        # Try direct PMC ID
        if identifier.startswith("PMC"):
            pmcid = identifier
        else:
            # Search index for OMIP ID
            pmcid = self._omip_to_pmcid(identifier)

        if not pmcid:
            return None

        # Check new location (PMC{id}/PMC{id}.xml)
        xml_path = self.papers_dir / pmcid / f"{pmcid}.xml"
        if xml_path.exists():
            return xml_path

        # Check old location (PMC{id}.xml)
        xml_path = self.papers_dir / f"{pmcid}.xml"
        if xml_path.exists():
            return xml_path

        return None

    def _find_pdf(self, identifier: str) -> Path | None:
        """Find PDF file for an OMIP paper."""
        if identifier.startswith("PMC"):
            pmcid = identifier
        else:
            pmcid = self._omip_to_pmcid(identifier)

        if not pmcid:
            return None

        pdf_path = self.papers_dir / pmcid / f"{pmcid}.pdf"
        if pdf_path.exists():
            return pdf_path

        return None

    def _omip_to_pmcid(self, omip_id: str) -> str | None:
        """Look up PMC ID from OMIP ID using index."""
        index_path = self.papers_dir / "omip_index.json"
        if not index_path.exists():
            return None

        import json
        with open(index_path) as f:
            index = json.load(f)

        # Normalize OMIP ID
        omip_normalized = omip_id.upper().replace(" ", "-")
        if not omip_normalized.startswith("OMIP-"):
            omip_normalized = f"OMIP-{omip_normalized}"

        for entry in index:
            if entry.get("omip_id", "").upper() == omip_normalized:
                return entry.get("pmcid")

        return None

    def _extract_from_xml(self, xml_path: Path) -> PaperContent:
        """Extract content from PMC XML file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Handle namespace
        ns = {'': ''}  # PMC XML usually doesn't use namespaces

        # Extract metadata
        title = self._get_text(root.find(".//article-title"))
        abstract = self._get_text(root.find(".//abstract"))
        doi = self._get_text(root.find(".//article-id[@pub-id-type='doi']"))
        pmcid = self._get_text(root.find(".//article-id[@pub-id-type='pmc']"))
        if pmcid and not pmcid.startswith("PMC"):
            pmcid = f"PMC{pmcid}"

        # Extract OMIP ID from title
        omip_id = None
        if title:
            match = re.search(r"OMIP[- ]?(\d+)", title, re.IGNORECASE)
            if match:
                omip_id = f"OMIP-{match.group(1).zfill(3)}"

        # Extract sections
        methods_text = self._extract_section(root, ["methods", "materials and methods",
                                                     "experimental procedures"])
        results_text = self._extract_section(root, ["results", "results and discussion"])

        # Extract tables
        tables = list(self._extract_tables(root))

        # Extract figure captions (images need PDF)
        figures = list(self._extract_figure_captions(root))

        return PaperContent(
            omip_id=omip_id,
            pmcid=pmcid,
            doi=doi,
            title=title or "",
            abstract=abstract or "",
            methods_text=methods_text,
            results_text=results_text,
            tables=tables,
            figures=figures,
            raw_xml=ET.tostring(root, encoding='unicode')
        )

    def _get_text(self, element: ET.Element | None) -> str:
        """Get all text from an element, including nested elements."""
        if element is None:
            return ""
        return " ".join(element.itertext()).strip()

    def _extract_section(self, root: ET.Element, section_names: list[str]) -> str:
        """Extract text from a named section."""
        for sec in root.findall(".//sec"):
            title_elem = sec.find("title")
            if title_elem is not None:
                title = self._get_text(title_elem).lower()
                if any(name in title for name in section_names):
                    # Get all paragraph text
                    paragraphs = []
                    for p in sec.findall(".//p"):
                        paragraphs.append(self._get_text(p))
                    return "\n\n".join(paragraphs)

        return ""

    def _extract_tables(self, root: ET.Element) -> Iterator[ExtractedTable]:
        """Extract and classify tables from XML."""
        for i, table_wrap in enumerate(root.findall(".//table-wrap"), 1):
            table_id = table_wrap.get("id")
            caption = self._get_text(table_wrap.find(".//caption"))
            label = self._get_text(table_wrap.find(".//label")) or f"Table {i}"

            # Parse table structure
            headers = []
            rows = []

            # Find thead
            thead = table_wrap.find(".//thead")
            if thead is not None:
                for tr in thead.findall(".//tr"):
                    headers = [self._get_text(th) for th in tr.findall(".//*")
                              if th.tag in ('th', 'td')]
                    if headers:
                        break

            # Find tbody
            tbody = table_wrap.find(".//tbody")
            if tbody is not None:
                for tr in tbody.findall(".//tr"):
                    row = [self._get_text(td) for td in tr.findall(".//*")
                           if td.tag in ('td', 'th')]
                    if row:
                        rows.append(row)

            # If no thead, try first row as header
            if not headers and rows:
                headers = rows[0]
                rows = rows[1:]

            # Classify table type
            table_type = self._classify_table(caption, headers)

            yield ExtractedTable(
                table_id=table_id,
                caption=caption,
                headers=headers,
                rows=rows,
                table_type=table_type,
                source_location=label
            )

    def _classify_table(self, caption: str, headers: list[str]) -> str:
        """Classify table as panel, gating, results, or unknown."""
        caption_lower = caption.lower()
        headers_lower = [h.lower() for h in headers]
        all_text = caption_lower + " " + " ".join(headers_lower)

        # Panel table indicators
        panel_keywords = ['panel', 'antibod', 'reagent', 'fluorophore', 'fluorochrome',
                         'clone', 'marker', 'conjugate', 'dye', 'channel']
        if any(kw in all_text for kw in panel_keywords):
            # Check for typical panel columns
            if any(h in headers_lower for h in ['marker', 'antibody', 'target', 'specificity']):
                return 'panel'
            if any(h in headers_lower for h in ['fluorophore', 'fluorochrome', 'conjugate', 'dye']):
                return 'panel'

        # Gating table indicators
        gating_keywords = ['gating', 'gate', 'population', 'phenotype', 'subset',
                          'strategy', 'hierarchy']
        if any(kw in all_text for kw in gating_keywords):
            return 'gating'

        # Results table (percentages, statistics)
        results_keywords = ['percent', 'frequency', 'mean', 'median', 'sd', 'sem',
                           'p-value', 'statistics']
        if any(kw in all_text for kw in results_keywords):
            return 'results'

        return 'unknown'

    def _extract_figure_captions(self, root: ET.Element) -> Iterator[ExtractedFigure]:
        """Extract figure captions from XML (images need PDF)."""
        for i, fig in enumerate(root.findall(".//fig"), 1):
            fig_id = fig.get("id")
            caption = self._get_text(fig.find(".//caption"))
            label = self._get_text(fig.find(".//label")) or f"Figure {i}"

            # Classify figure
            fig_type = self._classify_figure(caption)

            yield ExtractedFigure(
                figure_id=fig_id,
                caption=f"{label}: {caption}",
                image_path=None,
                image_data=None,
                figure_type=fig_type,
                page_number=None
            )

    def _classify_figure(self, caption: str) -> str:
        """Classify figure as gating, results, schematic, or unknown."""
        caption_lower = caption.lower()

        gating_keywords = ['gating', 'gate', 'strategy', 'hierarchy', 'flow cytometry',
                          'dot plot', 'scatter', 'fsc', 'ssc', 'sequential']
        if any(kw in caption_lower for kw in gating_keywords):
            return 'gating'

        results_keywords = ['results', 'data', 'comparison', 'expression', 'frequency',
                          'percent', 'bar graph', 'histogram']
        if any(kw in caption_lower for kw in results_keywords):
            return 'results'

        schematic_keywords = ['schematic', 'overview', 'workflow', 'diagram']
        if any(kw in caption_lower for kw in schematic_keywords):
            return 'schematic'

        return 'unknown'

    def _extract_figures_from_pdf(
        self,
        pdf_path: Path,
        output_dir: Path | None = None,
        dpi: int = 150
    ) -> list[ExtractedFigure]:
        """
        Extract figures/pages from PDF as images.

        Requires pdf2image and poppler to be installed.

        Args:
            pdf_path: Path to PDF file
            output_dir: Optional directory to save images
            dpi: Resolution for image extraction

        Returns:
            List of ExtractedFigure with image data
        """
        if not PDF2IMAGE_AVAILABLE:
            return []

        figures = []

        try:
            images = convert_from_path(pdf_path, dpi=dpi)

            for i, img in enumerate(images, 1):
                # Save image if output dir specified
                image_path = None
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    image_path = output_dir / f"page_{i:02d}.png"
                    img.save(image_path, "PNG")

                # Convert to bytes for in-memory use
                import io
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                image_data = img_bytes.getvalue()

                figures.append(ExtractedFigure(
                    figure_id=f"page_{i}",
                    caption=f"PDF Page {i}",
                    image_path=image_path,
                    image_data=image_data,
                    figure_type='unknown',  # Would need vision model to classify
                    page_number=i
                ))

        except Exception as e:
            print(f"Warning: Failed to extract PDF images: {e}")

        return figures


def extract_panel_from_table(table: ExtractedTable) -> list[dict]:
    """
    Extract panel entries from a panel table.

    Attempts to identify marker, fluorophore, and clone columns.

    Returns:
        List of dicts with keys: marker, fluorophore, clone, vendor, cat_number
    """
    # Find relevant columns
    marker_col = None
    fluor_col = None
    clone_col = None
    vendor_col = None
    cat_col = None

    for i, header in enumerate(table.headers):
        h_lower = header.lower()

        if marker_col is None and any(kw in h_lower for kw in
                                       ['marker', 'target', 'specificity', 'antigen', 'antibody']):
            marker_col = i
        elif fluor_col is None and any(kw in h_lower for kw in
                                        ['fluorophore', 'fluorochrome', 'conjugate', 'dye', 'label']):
            fluor_col = i
        elif clone_col is None and 'clone' in h_lower:
            clone_col = i
        elif vendor_col is None and any(kw in h_lower for kw in ['vendor', 'supplier', 'source', 'company']):
            vendor_col = i
        elif cat_col is None and any(kw in h_lower for kw in ['catalog', 'cat', 'product', 'order']):
            cat_col = i

    if marker_col is None and fluor_col is None:
        # Can't parse this table
        return []

    # If marker column not found, fluorophore might be first
    if marker_col is None:
        marker_col = 0

    entries = []
    for row in table.rows:
        if marker_col >= len(row):
            continue

        marker = row[marker_col].strip()
        if not marker:
            continue

        entry = {
            'marker': marker,
            'fluorophore': row[fluor_col].strip() if fluor_col is not None and fluor_col < len(row) else None,
            'clone': row[clone_col].strip() if clone_col is not None and clone_col < len(row) else None,
            'vendor': row[vendor_col].strip() if vendor_col is not None and vendor_col < len(row) else None,
            'cat_number': row[cat_col].strip() if cat_col is not None and cat_col < len(row) else None,
        }

        # Clean up empty strings
        entry = {k: v if v else None for k, v in entry.items()}
        entries.append(entry)

    return entries


def extract_gating_from_text(
    text: str,
    return_raw: bool = False
) -> dict | str:
    """
    Extract gating strategy mentions from methods/results text.

    This performs basic pattern matching. For better results,
    use an LLM with the extracted text.

    Args:
        text: Methods or results section text
        return_raw: If True, return cleaned text instead of parsed dict

    Returns:
        Dict with extracted gating info, or cleaned text if return_raw=True
    """
    # Find gating-related sentences
    gating_patterns = [
        r'[^.]*gating[^.]*\.',
        r'[^.]*gated[^.]*\.',
        r'[^.]*population[^.]*defined[^.]*\.',
        r'[^.]*CD\d+[+-][^.]*\.',
        r'[^.]*singlets?[^.]*\.',
        r'[^.]*live[^.]*dead[^.]*\.',
        r'[^.]*lymphocytes?[^.]*\.',
    ]

    relevant_sentences = []
    for pattern in gating_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        relevant_sentences.extend(matches)

    # Remove duplicates while preserving order
    seen = set()
    unique_sentences = []
    for s in relevant_sentences:
        s_clean = s.strip()
        if s_clean not in seen:
            seen.add(s_clean)
            unique_sentences.append(s_clean)

    cleaned_text = " ".join(unique_sentences)

    if return_raw:
        return cleaned_text

    # Basic extraction of marker patterns
    marker_pattern = r'(CD\d+[a-z]?|CCR\d+|CXCR\d+|HLA-[A-Z]+)[+-]?'
    markers_mentioned = list(set(re.findall(marker_pattern, text, re.IGNORECASE)))

    return {
        'gating_text': cleaned_text,
        'markers_mentioned': markers_mentioned,
        'sentence_count': len(unique_sentences)
    }

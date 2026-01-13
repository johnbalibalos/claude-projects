#!/usr/bin/env python3
"""
Extract markers, panels, and gating hierarchy info from PMC XML files.

Focuses on Table 1-2 (panel info) and Figure 1-2 (gating strategy).
Outputs JSON in the staging schema format.
"""

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PanelEntry:
    marker: str
    fluorophore: str | None = None
    clone: str | None = None
    purpose: str | None = None


@dataclass
class ExtractedOMIP:
    pmc_id: str
    omip_id: str | None = None
    title: str | None = None
    doi: str | None = None

    # From Table 1 (summary)
    species: str | None = None
    sample_type: str | None = None
    cell_type: str | None = None
    cross_references: list[str] = field(default_factory=list)

    # From Table 2 (panel)
    panel: list[PanelEntry] = field(default_factory=list)

    # From Figure 1 caption (gating description)
    gating_description: str | None = None

    # From body text (full gating details)
    full_text_gating: str | None = None

    # Raw extraction info
    tables_found: int = 0
    figures_found: int = 0
    extraction_notes: list[str] = field(default_factory=list)


def extract_text(elem) -> str:
    """Extract all text from an element, handling nested tags."""
    if elem is None:
        return ""
    return " ".join(elem.itertext()).strip()


def normalize_marker(marker: str) -> str:
    """Normalize marker names."""
    # Remove common prefixes/suffixes
    marker = marker.strip()
    # Handle special characters
    marker = marker.replace("‐", "-").replace("–", "-")
    return marker


def parse_table1_summary(table) -> dict:
    """Parse Table 1 (summary table) for metadata."""
    result = {}

    tbody = table.find('.//tbody')
    if tbody is None:
        return result

    for row in tbody.findall('.//tr'):
        cells = row.findall('.//td')
        if len(cells) >= 2:
            key = extract_text(cells[0]).lower()
            value = extract_text(cells[1])

            if 'species' in key:
                result['species'] = value.lower()
            elif 'cell type' in key or 'sample' in key:
                result['sample_type'] = value
            elif 'cross' in key or 'reference' in key:
                # Parse cross references like "OMIP-001, OMIP-002"
                refs = re.findall(r'OMIP[- ]?\d+', value, re.IGNORECASE)
                result['cross_references'] = [r.upper().replace(' ', '-') for r in refs]
            elif 'purpose' in key or 'application' in key:
                result['application'] = value

    return result


def parse_table2_panel(table) -> list[PanelEntry]:
    """Parse Table 2 (reagent/panel table) for markers."""
    entries = []

    # Get headers to map columns
    thead = table.find('.//thead')
    headers = []
    if thead is not None:
        for th in thead.findall('.//th'):
            headers.append(extract_text(th).lower())

    # Find column indices
    marker_col = None
    fluor_col = None
    clone_col = None
    purpose_col = None

    for i, h in enumerate(headers):
        if 'specific' in h or 'marker' in h or 'antigen' in h or 'target' in h:
            marker_col = i
        elif 'fluor' in h or 'conjugate' in h or 'dye' in h or 'label' in h or 'metal' in h or 'isotope' in h:
            fluor_col = i
        elif 'clone' in h:
            clone_col = i
        elif 'purpose' in h or 'function' in h:
            purpose_col = i

    # If no marker column found, try first column
    if marker_col is None and headers:
        marker_col = 0

    # Parse rows
    tbody = table.find('.//tbody')
    if tbody is None:
        return entries

    for row in tbody.findall('.//tr'):
        cells = row.findall('.//td')
        if not cells:
            continue

        entry = PanelEntry(marker="")

        if marker_col is not None and marker_col < len(cells):
            entry.marker = normalize_marker(extract_text(cells[marker_col]))

        if fluor_col is not None and fluor_col < len(cells):
            entry.fluorophore = extract_text(cells[fluor_col]) or None

        if clone_col is not None and clone_col < len(cells):
            entry.clone = extract_text(cells[clone_col]) or None

        if purpose_col is not None and purpose_col < len(cells):
            entry.purpose = extract_text(cells[purpose_col]) or None

        if entry.marker:
            entries.append(entry)

    return entries


def parse_figure1_gating(fig) -> str | None:
    """Extract gating strategy description from Figure 1 caption."""
    caption = fig.find('caption')
    if caption is None:
        return None

    text = extract_text(caption)

    # Check if this is a gating figure
    gating_keywords = ['gating', 'gate', 'strategy', 'hierarchy', 'subset', 'population']
    if any(kw in text.lower() for kw in gating_keywords):
        return text

    return None


def extract_body_gating_text(root) -> str | None:
    """Extract gating-related text from body sections."""
    body = root.find('.//body')
    if body is None:
        return None

    gating_paragraphs = []
    gating_keywords = ['gating', 'gated', 'gate ', 'singlet', 'doublet', 'live cell',
                       'dead cell', 'lymphocyte', 'cd45+', 'cd45 +', 'fsc', 'ssc',
                       'exclusion', 'identified', 'separated', 'divided', 'defined']

    for sec in body.findall('.//sec'):
        title = sec.find('title')
        title_text = extract_text(title).lower() if title is not None else ""

        # Focus on BACKGROUND, METHODS, or sections mentioning gating
        is_relevant = any(kw in title_text for kw in ['background', 'method', 'gating', 'strategy', 'result'])

        for p in sec.findall('.//p'):
            text = extract_text(p)
            text_lower = text.lower()

            # Check if paragraph contains gating-related content
            if is_relevant or any(kw in text_lower for kw in gating_keywords):
                # Skip very short or very long paragraphs
                if 50 < len(text) < 3000:
                    gating_paragraphs.append(text)

    if gating_paragraphs:
        return "\n\n".join(gating_paragraphs)
    return None


def extract_omip_from_xml(xml_path: Path) -> ExtractedOMIP:
    """Extract OMIP data from a PMC XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    result = ExtractedOMIP(pmc_id=xml_path.stem)

    # Extract basic metadata
    title_elem = root.find('.//article-title')
    if title_elem is not None:
        result.title = extract_text(title_elem)
        # Extract OMIP ID from title
        match = re.search(r'OMIP[- ]?(\d+)', result.title, re.IGNORECASE)
        if match:
            result.omip_id = f"OMIP-{match.group(1).zfill(3)}"

    doi_elem = root.find(".//article-id[@pub-id-type='doi']")
    if doi_elem is not None:
        result.doi = doi_elem.text

    # Find and parse tables
    tables = root.findall('.//table-wrap')
    result.tables_found = len(tables)

    # First pass: find best panel (table with most markers)
    best_panel = []
    best_panel_idx = -1

    for i, table in enumerate(tables):
        panel = parse_table2_panel(table)
        if panel and len(panel) > len(best_panel):
            best_panel = panel
            best_panel_idx = i

    if best_panel:
        result.panel = best_panel
        result.extraction_notes.append(f"Parsed table {best_panel_idx+1} as panel: {len(best_panel)} markers")

    # Second pass: extract metadata from Table 1
    for _i, table in enumerate(tables):
        label = table.find('label')
        label_text = extract_text(label).lower() if label is not None else ""
        caption_elem = table.find('caption')
        caption = extract_text(caption_elem).lower() if caption_elem is not None else ""

        if 'table 1' in label_text or 'summary' in caption:
            metadata = parse_table1_summary(table)
            if metadata.get('species'):
                result.species = metadata.get('species')
            if metadata.get('sample_type'):
                result.sample_type = metadata.get('sample_type')
            if metadata.get('cross_references'):
                result.cross_references = metadata.get('cross_references', [])
            result.extraction_notes.append("Parsed Table 1 for metadata")

    # Find and parse figures
    figures = root.findall('.//fig')
    result.figures_found = len(figures)

    for fig in figures:
        label = fig.find('label')
        label_text = extract_text(label).lower() if label is not None else ""

        # Figure 1 or 2 usually has gating strategy
        is_fig1 = 'figure 1' in label_text or 'fig. 1' in label_text or label_text == 'fig 1'
        is_fig2 = 'figure 2' in label_text or 'fig. 2' in label_text or label_text == 'fig 2'

        if is_fig1 or (is_fig2 and not result.gating_description):
            gating = parse_figure1_gating(fig)
            if gating:
                result.gating_description = gating
                fig_num = "1" if is_fig1 else "2"
                result.extraction_notes.append(f"Extracted gating description from Figure {fig_num}")

    # Extract gating-related text from body
    full_text = extract_body_gating_text(root)
    if full_text:
        result.full_text_gating = full_text
        result.extraction_notes.append(f"Extracted {len(full_text)} chars of gating text from body")

    return result


def to_staging_format(extracted: ExtractedOMIP) -> dict:
    """Convert extracted data to staging JSON schema."""
    return {
        "test_case_id": extracted.omip_id or extracted.pmc_id,
        "source_type": "omip_paper",
        "omip_id": extracted.omip_id,
        "doi": extracted.doi,
        "pmc_id": extracted.pmc_id,
        "flowrepository_id": None,
        "has_wsp": False,
        "wsp_validated": False,
        "context": {
            "sample_type": extracted.sample_type,
            "species": extracted.species,
            "application": None,
            "tissue": None,
            "disease_state": None,
            "additional_notes": extracted.gating_description,
            "full_text_gating": extracted.full_text_gating,
            "cross_references": extracted.cross_references,
            "pdf_verified": False,
            "extraction_source": "pmc_xml",
        },
        "panel": {
            "entries": [
                {
                    "marker": e.marker,
                    "fluorophore": e.fluorophore,
                    "clone": e.clone,
                    "purpose": e.purpose,
                }
                for e in extracted.panel
            ]
        },
        "gating_hierarchy": {
            "root": {
                "name": "All Events",
                "markers": [],
                "marker_logic": None,
                "gate_type": "Unknown",
                "children": [],
                "is_critical": True,
            },
            "_extraction_note": "Hierarchy needs manual curation from gating_description"
        },
        "_extraction_metadata": {
            "tables_found": extracted.tables_found,
            "figures_found": extracted.figures_found,
            "notes": extracted.extraction_notes,
        }
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract OMIP data from PMC XMLs")
    parser.add_argument("--input-dir", type=Path,
                        default=Path("data/papers/pmc"),
                        help="Directory containing PMC XML files")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/extracted"),
                        help="Output directory for JSON files")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary instead of saving files")
    args = parser.parse_args()

    xml_files = list(args.input_dir.glob("PMC*.xml"))
    print(f"Found {len(xml_files)} XML files")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for xml_path in sorted(xml_files):
        extracted = extract_omip_from_xml(xml_path)
        results.append(extracted)

        if args.summary:
            status = "✓" if extracted.panel else "✗"
            markers = len(extracted.panel)
            gating = "✓" if extracted.gating_description else "✗"
            print(f"{status} {extracted.pmc_id}: {extracted.omip_id or 'Unknown'} | "
                  f"{markers} markers | gating: {gating} | {extracted.species or 'unknown'}")
        else:
            # Save to JSON
            staging = to_staging_format(extracted)
            filename = f"{extracted.omip_id or extracted.pmc_id}.json".lower().replace("-", "_")
            output_path = args.output_dir / filename
            with open(output_path, "w") as f:
                json.dump(staging, f, indent=2)
            print(f"Saved: {output_path}")

    # Summary stats
    print("\n=== Summary ===")
    with_panel = sum(1 for r in results if r.panel)
    with_gating = sum(1 for r in results if r.gating_description)
    print(f"Total XMLs: {len(results)}")
    print(f"With panel extracted: {with_panel}")
    print(f"With gating description: {with_gating}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-method extraction of gating hierarchies from OMIP papers.

Runs both XML parsing and LLM extraction, then calculates concordance.

Usage:
    # XML only (no API calls)
    python scripts/extract_multi_method.py --papers-dir /path/to/papers --methods xml

    # XML + LLM
    python scripts/extract_multi_method.py --papers-dir /path/to/papers --methods xml llm

    # Single paper
    python scripts/extract_multi_method.py --papers-dir /path/to/papers --omip-id OMIP-077
"""

import argparse
import json
import sys
from pathlib import Path

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from curation.paper_parser import PaperParser, extract_panel_from_table
from curation.extraction_schema import (
    ExtractionMethod,
    MethodExtraction,
    MultiMethodExtraction,
    calculate_all_concordance,
)


def extract_xml(parser: PaperParser, pmc_id: str, papers_dir: Path) -> tuple[MethodExtraction | None, MethodExtraction | None]:
    """Extract panel and hierarchy from XML."""
    # Find XML file
    xml_path = papers_dir / f"{pmc_id}.xml"
    if not xml_path.exists():
        return None, None

    content = parser._extract_from_xml(xml_path)

    # Panel extraction
    panel_extraction = None
    panel_table = content.get_panel_table()
    if panel_table:
        entries = extract_panel_from_table(panel_table)
        if entries:
            panel_extraction = MethodExtraction(
                method=ExtractionMethod.XML,
                data={"entries": entries, "table_caption": panel_table.caption},
                confidence=0.9,
                source_file=str(xml_path),
                notes=f"Extracted from {panel_table.source_location}"
            )

    # Hierarchy extraction (from gating table if available)
    hierarchy_extraction = None
    gating_table = content.get_gating_table()
    if gating_table:
        # XML gating tables are usually phenotype tables, not full hierarchies
        # Low confidence since they don't capture tree structure
        hierarchy_extraction = MethodExtraction(
            method=ExtractionMethod.XML,
            data={
                "table_data": {
                    "headers": gating_table.headers,
                    "rows": gating_table.rows
                },
                "hierarchy": None,  # No tree structure from XML
                "note": "Gating table found but hierarchy requires LLM interpretation"
            },
            confidence=0.3,
            source_file=str(xml_path),
            notes="Phenotype table only, no tree structure"
        )

    return panel_extraction, hierarchy_extraction


def extract_llm(
    llm_extractor,
    pmc_id: str,
    papers_dir: Path,
    parser: PaperParser
) -> tuple[MethodExtraction | None, MethodExtraction | None]:
    """Extract panel and hierarchy using LLM."""
    from curation.llm_extractor import create_method_extraction

    xml_path = papers_dir / f"{pmc_id}.xml"
    if not xml_path.exists():
        return None, None

    content = parser._extract_from_xml(xml_path)

    # Gather text for LLM
    text_parts = []
    if content.abstract:
        text_parts.append(f"Abstract:\n{content.abstract}")
    if content.methods_text:
        text_parts.append(f"Methods:\n{content.methods_text}")
    if content.results_text:
        text_parts.append(f"Results:\n{content.results_text}")

    # Add table text
    for table in content.tables:
        text_parts.append(f"Table ({table.source_location}):\n{table.to_markdown()}")

    combined_text = "\n\n".join(text_parts)

    if not combined_text.strip():
        return None, None

    # Figure captions
    figure_captions = "\n".join(
        fig.caption for fig in content.figures if fig.caption
    )

    # Extract panel
    panel_extraction = None
    panel_result = llm_extractor.extract_panel(
        title=content.title or "",
        text=combined_text
    )
    if panel_result.success:
        panel_extraction = create_method_extraction(panel_result, str(xml_path))

    # Get markers for hierarchy extraction
    markers = []
    if panel_extraction and panel_extraction.data.get("entries"):
        markers = [e.get("marker", "") for e in panel_extraction.data["entries"] if e.get("marker")]

    # Extract hierarchy
    hierarchy_extraction = None
    hierarchy_result = llm_extractor.extract_hierarchy(
        title=content.title or "",
        text=combined_text,
        markers=markers,
        figure_captions=figure_captions
    )
    if hierarchy_result.success:
        hierarchy_extraction = create_method_extraction(hierarchy_result, str(xml_path))

    return panel_extraction, hierarchy_extraction


def process_paper(
    pmc_id: str,
    omip_id: str,
    papers_dir: Path,
    parser: PaperParser,
    methods: list[str],
    llm_extractor=None
) -> MultiMethodExtraction:
    """Process a single paper with specified methods."""
    # Get basic metadata from XML
    xml_path = papers_dir / f"{pmc_id}.xml"
    content = parser._extract_from_xml(xml_path) if xml_path.exists() else None

    extraction = MultiMethodExtraction(
        omip_id=omip_id,
        pmc_id=pmc_id,
        doi=content.doi if content else None,
        title=content.title if content else None
    )

    # XML extraction
    if "xml" in methods:
        panel_ext, hierarchy_ext = extract_xml(parser, pmc_id, papers_dir)
        if panel_ext:
            extraction.add_panel_extraction(panel_ext)
        if hierarchy_ext:
            extraction.add_hierarchy_extraction(hierarchy_ext)

    # LLM extraction
    if "llm" in methods and llm_extractor:
        panel_ext, hierarchy_ext = extract_llm(llm_extractor, pmc_id, papers_dir, parser)
        if panel_ext:
            extraction.add_panel_extraction(panel_ext)
        if hierarchy_ext:
            extraction.add_hierarchy_extraction(hierarchy_ext)

    # Calculate concordance if multiple methods
    calculate_all_concordance(extraction)

    # Select best method
    if extraction.panel_extractions:
        # Prefer higher confidence
        best = max(extraction.panel_extractions.items(), key=lambda x: x[1].confidence)
        extraction.best_panel_method = best[0]

    if extraction.hierarchy_extractions:
        best = max(extraction.hierarchy_extractions.items(), key=lambda x: x[1].confidence)
        extraction.best_hierarchy_method = best[0]

    return extraction


def main():
    arg_parser = argparse.ArgumentParser(description="Multi-method gating hierarchy extraction")
    arg_parser.add_argument("--papers-dir", type=Path, required=True,
                           help="Directory containing PMC XML files")
    arg_parser.add_argument("--output-dir", type=Path, default=Path("data/claude-extracted"),
                           help="Output directory")
    arg_parser.add_argument("--methods", nargs="+", default=["xml"],
                           choices=["xml", "llm"],
                           help="Extraction methods to use")
    arg_parser.add_argument("--omip-id", type=str, default=None,
                           help="Process single OMIP (e.g., OMIP-077)")
    arg_parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                           help="LLM model to use")
    arg_parser.add_argument("--limit", type=int, default=None,
                           help="Limit number of papers to process")

    args = arg_parser.parse_args()

    papers_dir = args.papers_dir.resolve()
    output_dir = (Path(__file__).parent.parent / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Papers directory: {papers_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Methods: {args.methods}")

    # Load index
    index_path = papers_dir / "omip_index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        index = [{"pmc_id": f.stem} for f in papers_dir.glob("PMC*.xml")]

    # Filter to single paper if specified
    if args.omip_id:
        index = [e for e in index if e.get("omip_id", "").upper() == args.omip_id.upper()]
        if not index:
            print(f"OMIP ID not found: {args.omip_id}")
            sys.exit(1)

    if args.limit:
        index = index[:args.limit]

    print(f"Processing {len(index)} papers")

    # Initialize extractors
    parser = PaperParser(papers_dir=papers_dir)

    llm_extractor = None
    if "llm" in args.methods:
        try:
            from curation.llm_extractor import LLMExtractor
            llm_extractor = LLMExtractor(model=args.model)
            print(f"LLM extractor initialized with {args.model}")
        except Exception as e:
            print(f"Warning: Could not initialize LLM extractor: {e}")
            print("Continuing with XML only")
            args.methods = ["xml"]

    # Process papers
    results = {
        "successful": [],
        "failed": [],
        "concordance_summary": []
    }

    for i, entry in enumerate(index):
        pmc_id = entry.get("pmc_id") or entry.get("pmcid")
        omip_id = entry.get("omip_id", pmc_id)

        print(f"\n[{i+1}/{len(index)}] Processing {pmc_id} ({omip_id})...")

        try:
            extraction = process_paper(
                pmc_id=pmc_id,
                omip_id=omip_id,
                papers_dir=papers_dir,
                parser=parser,
                methods=args.methods,
                llm_extractor=llm_extractor
            )

            # Save
            output_path = extraction.save(output_dir)
            print(f"  Saved to {output_path}")

            # Summary
            summary = {
                "omip_id": omip_id,
                "pmc_id": pmc_id,
                "panel_methods": list(extraction.panel_extractions.keys()),
                "hierarchy_methods": list(extraction.hierarchy_extractions.keys()),
                "best_panel": extraction.best_panel_method,
                "best_hierarchy": extraction.best_hierarchy_method,
            }

            # Add concordance if available
            if extraction.panel_concordance:
                summary["panel_concordance"] = [
                    {"methods": f"{c.method_a.value}_vs_{c.method_b.value}", "score": c.score}
                    for c in extraction.panel_concordance
                ]

            if extraction.hierarchy_concordance:
                summary["hierarchy_concordance"] = [
                    {"methods": f"{c.method_a.value}_vs_{c.method_b.value}", "score": c.score}
                    for c in extraction.hierarchy_concordance
                ]

            results["successful"].append(summary)

            # Print extraction summary
            print(f"  Panel: {list(extraction.panel_extractions.keys())}")
            print(f"  Hierarchy: {list(extraction.hierarchy_extractions.keys())}")

            if extraction.panel_concordance:
                for c in extraction.panel_concordance:
                    print(f"  Panel concordance ({c.method_a.value} vs {c.method_b.value}): {c.score:.2f}")

        except Exception as e:
            print(f"  Error: {e}")
            results["failed"].append({"omip_id": omip_id, "pmc_id": pmc_id, "error": str(e)})

    # Save summary
    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print final summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total processed: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")

    # Concordance stats
    if "llm" in args.methods:
        panel_scores = []
        hierarchy_scores = []

        for r in results["successful"]:
            for c in r.get("panel_concordance", []):
                panel_scores.append(c["score"])
            for c in r.get("hierarchy_concordance", []):
                hierarchy_scores.append(c["score"])

        if panel_scores:
            print(f"\nPanel concordance (XML vs LLM):")
            print(f"  Mean: {sum(panel_scores)/len(panel_scores):.2f}")
            print(f"  Min: {min(panel_scores):.2f}")
            print(f"  Max: {max(panel_scores):.2f}")

        if hierarchy_scores:
            print(f"\nHierarchy concordance (XML vs LLM):")
            print(f"  Mean: {sum(hierarchy_scores)/len(hierarchy_scores):.2f}")
            print(f"  Min: {min(hierarchy_scores):.2f}")
            print(f"  Max: {max(hierarchy_scores):.2f}")

    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()

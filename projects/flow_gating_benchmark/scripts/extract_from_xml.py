#!/usr/bin/env python3
"""
Extract gating hierarchies and panel information from PMC XML files.

Usage:
    python scripts/extract_from_xml.py --papers-dir ../../docs/papers --output-dir data/claude-extracted
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from curation.paper_parser import PaperParser, ExtractedTable, PaperContent


def extract_all_papers(papers_dir: Path, output_dir: Path) -> dict:
    """Extract content from all papers in directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load index if available
    index_path = papers_dir / "omip_index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        print(f"Loaded index with {len(index)} papers")
    else:
        # Build index from XML files
        index = []
        for xml_file in papers_dir.glob("PMC*.xml"):
            index.append({"pmc_id": xml_file.stem})
        print(f"Found {len(index)} XML files")

    # Create parser pointing to papers directory
    parser = PaperParser(papers_dir=papers_dir)

    results = {
        "successful": [],
        "failed": [],
        "summary": {}
    }

    for entry in index:
        pmc_id = entry.get("pmc_id") or entry.get("pmcid")
        omip_id = entry.get("omip_id", "Unknown")

        print(f"\nProcessing {pmc_id} ({omip_id})...")

        # Find XML file
        xml_path = papers_dir / f"{pmc_id}.xml"
        if not xml_path.exists():
            xml_path = papers_dir / pmc_id / f"{pmc_id}.xml"

        if not xml_path.exists():
            print(f"  XML not found: {xml_path}")
            results["failed"].append({"pmc_id": pmc_id, "error": "XML not found"})
            continue

        try:
            # Extract content
            content = parser._extract_from_xml(xml_path)

            # Build extraction result
            extraction = {
                "pmc_id": pmc_id,
                "omip_id": content.omip_id or omip_id,
                "title": content.title,
                "doi": content.doi,
                "tables_found": len(content.tables),
                "figures_found": len(content.figures),
                "has_methods": bool(content.methods_text),
                "has_results": bool(content.results_text),
                "tables": [],
                "panel": None,
                "gating_table": None,
                "gating_text_excerpt": None
            }

            # Process tables
            for table in content.tables:
                table_info = {
                    "id": table.table_id,
                    "type": table.table_type,
                    "location": table.source_location,
                    "caption": table.caption[:200] if table.caption else "",
                    "headers": table.headers,
                    "row_count": len(table.rows)
                }
                extraction["tables"].append(table_info)

                if table.table_type == "panel":
                    extraction["panel"] = {
                        "headers": table.headers,
                        "rows": table.rows,
                        "markdown": table.to_markdown()
                    }
                    print(f"  Found panel table: {len(table.rows)} entries")

                if table.table_type == "gating":
                    extraction["gating_table"] = {
                        "headers": table.headers,
                        "rows": table.rows,
                        "markdown": table.to_markdown()
                    }
                    print(f"  Found gating table: {len(table.rows)} entries")

            # Extract gating text from methods
            if content.methods_text:
                from curation.paper_parser import extract_gating_from_text
                gating_info = extract_gating_from_text(content.methods_text)
                if gating_info.get("gating_text"):
                    extraction["gating_text_excerpt"] = gating_info["gating_text"][:1000]
                    extraction["markers_mentioned"] = gating_info.get("markers_mentioned", [])
                    print(f"  Found gating text with {len(gating_info.get('markers_mentioned', []))} markers")

            # Save individual extraction
            safe_id = (content.omip_id or pmc_id).lower().replace("-", "_").replace(" ", "_")
            output_path = output_dir / f"{safe_id}.json"
            with open(output_path, "w") as f:
                json.dump(extraction, f, indent=2, default=str)

            results["successful"].append({
                "pmc_id": pmc_id,
                "omip_id": extraction["omip_id"],
                "output_path": str(output_path),
                "has_panel": extraction["panel"] is not None,
                "has_gating_table": extraction["gating_table"] is not None,
                "has_gating_text": extraction["gating_text_excerpt"] is not None
            })

            print(f"  Saved to {output_path}")

        except Exception as e:
            print(f"  Error: {e}")
            results["failed"].append({"pmc_id": pmc_id, "error": str(e)})

    # Summary
    results["summary"] = {
        "total": len(index),
        "successful": len(results["successful"]),
        "failed": len(results["failed"]),
        "with_panel": sum(1 for r in results["successful"] if r["has_panel"]),
        "with_gating_table": sum(1 for r in results["successful"] if r["has_gating_table"]),
        "with_gating_text": sum(1 for r in results["successful"] if r["has_gating_text"])
    }

    # Save summary
    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Total papers: {results['summary']['total']}")
    print(f"Successful: {results['summary']['successful']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"With panel table: {results['summary']['with_panel']}")
    print(f"With gating table: {results['summary']['with_gating_table']}")
    print(f"With gating text: {results['summary']['with_gating_text']}")
    print(f"\nSummary saved to {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract gating hierarchies from PMC XML files")
    parser.add_argument("--papers-dir", type=Path, default=Path("../../docs/papers"),
                       help="Directory containing PMC XML files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/claude-extracted"),
                       help="Output directory for extractions")

    args = parser.parse_args()

    # Make paths absolute
    script_dir = Path(__file__).parent
    papers_dir = (script_dir / args.papers_dir).resolve()
    output_dir = (script_dir.parent / args.output_dir).resolve()

    print(f"Papers directory: {papers_dir}")
    print(f"Output directory: {output_dir}")

    if not papers_dir.exists():
        print(f"Error: Papers directory not found: {papers_dir}")
        sys.exit(1)

    extract_all_papers(papers_dir, output_dir)


if __name__ == "__main__":
    main()

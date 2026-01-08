"""
CLI entry point for paper_download.

Usage:
    python -m paper_download --email user@example.com
    python -m paper_download --email user@example.com --with-supplementary
    python -m paper_download --email user@example.com --query "cancer[Title]"
"""

import argparse
import sys
from pathlib import Path

from .pmc_client import PMCClient


def main():
    parser = argparse.ArgumentParser(
        description="Download papers from PMC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all OMIP papers
    python -m paper_download --email user@example.com

    # With supplementary files
    python -m paper_download --email user@example.com --with-supplementary

    # Custom query
    python -m paper_download --email user@example.com --query "flow cytometry[Title]"
        """,
    )
    parser.add_argument(
        "--email",
        required=True,
        help="Your email (required by NCBI)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/papers/pmc"),
        help="Output directory for files (default: data/papers/pmc)",
    )
    parser.add_argument(
        "--query",
        default="OMIP[Title] AND Cytometry[Journal]",
        help="PMC search query (default: OMIP papers)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=200,
        help="Maximum papers to download (default: 200)",
    )
    parser.add_argument(
        "--with-supplementary",
        action="store_true",
        help="Also download supplementary files (PDFs, Excel, etc.)",
    )

    args = parser.parse_args()

    def progress(current: int, total: int, status: str):
        supp_info = ""
        print(f"[{current}/{total}] Fetching {status}... ", end="", flush=True)

    print(f"Searching PMC: {args.query}")
    print()

    with PMCClient(email=args.email) as client:
        # Search
        pmc_ids = client.search_papers(args.query, args.max_results)

        if not pmc_ids:
            print("No papers found!")
            return 1

        print(f"Found {len(pmc_ids)} papers")
        print()

        # Download each paper
        results = []
        for i, pmc_id in enumerate(pmc_ids, 1):
            print(f"[{i}/{len(pmc_ids)}] Fetching PMC{pmc_id}...", end=" ", flush=True)

            result = client.fetch_xml(pmc_id, args.output_dir)
            results.append(result)
            print(result.status, end="")

            if args.with_supplementary and result.status in ("success", "cached"):
                supp = client.fetch_supplementary(pmc_id, args.output_dir)
                if supp:
                    supp_success = sum(1 for s in supp if s.status in ("success", "cached"))
                    print(f" | {supp_success} supp files", end="")

            print()

        # Extract metadata
        print("\nExtracting metadata...")
        metadata = []
        for xml_file in sorted(args.output_dir.glob("PMC*.xml")):
            info = client.extract_metadata(xml_file)
            if info.omip_id:
                metadata.append(info)
                title_preview = (info.title or "No title")[:60]
                print(f"  {info.omip_id}: {title_preview}...")

        # Save index
        index_path = args.output_dir / "omip_index.json"
        client.save_index(metadata, index_path)
        print(f"\nSaved index of {len(metadata)} papers to {index_path}")

        # Summary
        success = sum(1 for r in results if r.status in ("success", "cached"))
        print(f"\n=== Summary ===")
        print(f"Total searched: {len(pmc_ids)}")
        print(f"Successfully downloaded: {success}")
        print(f"Papers with OMIP IDs: {len(metadata)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

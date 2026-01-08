#!/usr/bin/env python3
"""
Standalone script to search PMC for all OMIP papers and download XMLs.

Usage:
    python scripts/download_omip_papers.py --email your@email.com

This searches PMC for papers with "OMIP" in the title published in Cytometry,
then downloads the full-text XML for each.
"""

import argparse
import json
import time
from pathlib import Path

import httpx


def search_omip_papers(email: str, max_results: int = 200) -> list[dict]:
    """Search PMC for all OMIP papers."""

    # Search PMC for OMIP papers in Cytometry journal
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": "OMIP[Title] AND Cytometry[Journal]",
        "retmax": max_results,
        "retmode": "json",
        "email": email,
    }

    print(f"Searching PMC for OMIP papers...")
    response = httpx.get(search_url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    id_list = data.get("esearchresult", {}).get("idlist", [])
    print(f"Found {len(id_list)} OMIP papers")

    return id_list


def fetch_paper_xml(pmc_id: str, email: str, output_dir: Path) -> dict:
    """Fetch XML for a single paper."""

    output_path = output_dir / f"PMC{pmc_id}.xml"

    # Check cache
    if output_path.exists():
        return {"pmc_id": pmc_id, "status": "cached", "path": str(output_path)}

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmc_id,
        "rettype": "xml",
        "email": email,
    }

    try:
        response = httpx.get(url, params=params, timeout=60)
        response.raise_for_status()

        xml_content = response.text
        if xml_content.strip().startswith("<?xml") or xml_content.strip().startswith("<"):
            output_path.write_text(xml_content, encoding="utf-8")
            return {"pmc_id": pmc_id, "status": "success", "path": str(output_path)}
        else:
            return {"pmc_id": pmc_id, "status": "invalid_xml", "error": "Response not XML"}

    except Exception as e:
        return {"pmc_id": pmc_id, "status": "error", "error": str(e)}


def extract_omip_info(xml_path: Path) -> dict | None:
    """Extract OMIP ID and title from XML."""
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get title
        title_elem = root.find(".//article-title")
        title = " ".join(title_elem.itertext()) if title_elem is not None else None

        # Extract OMIP number from title
        omip_id = None
        if title:
            import re
            match = re.search(r"OMIP[- ]?(\d+)", title, re.IGNORECASE)
            if match:
                omip_id = f"OMIP-{match.group(1).zfill(3)}"

        # Get DOI
        doi_elem = root.find(".//article-id[@pub-id-type='doi']")
        doi = doi_elem.text if doi_elem is not None else None

        # Get PMCID
        pmc_elem = root.find(".//article-id[@pub-id-type='pmc']")
        pmcid = f"PMC{pmc_elem.text}" if pmc_elem is not None else None

        return {
            "omip_id": omip_id,
            "title": title,
            "doi": doi,
            "pmcid": pmcid,
            "xml_path": str(xml_path),
        }
    except Exception as e:
        return {"error": str(e), "xml_path": str(xml_path)}


def main():
    parser = argparse.ArgumentParser(description="Download all OMIP papers from PMC")
    parser.add_argument("--email", required=True, help="Your email (required by NCBI)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/papers/pmc"),
                        help="Output directory for XMLs")
    parser.add_argument("--max-results", type=int, default=200,
                        help="Maximum papers to search for")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Search for papers
    pmc_ids = search_omip_papers(args.email, args.max_results)

    if not pmc_ids:
        print("No papers found!")
        return 1

    # Download each paper
    results = []
    for i, pmc_id in enumerate(pmc_ids, 1):
        print(f"[{i}/{len(pmc_ids)}] Fetching PMC{pmc_id}...", end=" ", flush=True)

        result = fetch_paper_xml(pmc_id, args.email, args.output_dir)
        results.append(result)

        print(result["status"])

        # Rate limit: 3 requests/second without API key
        if result["status"] != "cached":
            time.sleep(0.35)

    # Extract OMIP info from downloaded XMLs
    print("\nExtracting OMIP information...")
    omip_papers = []
    for xml_file in sorted(args.output_dir.glob("PMC*.xml")):
        info = extract_omip_info(xml_file)
        if info and info.get("omip_id"):
            omip_papers.append(info)
            print(f"  {info['omip_id']}: {info.get('title', 'No title')[:60]}...")

    # Save index
    index_path = args.output_dir / "omip_index.json"
    with open(index_path, "w") as f:
        json.dump(omip_papers, f, indent=2)
    print(f"\nSaved index of {len(omip_papers)} OMIP papers to {index_path}")

    # Summary
    success = sum(1 for r in results if r["status"] in ("success", "cached"))
    print(f"\n=== Summary ===")
    print(f"Total searched: {len(pmc_ids)}")
    print(f"Successfully downloaded: {success}")
    print(f"OMIP papers identified: {len(omip_papers)}")

    return 0


if __name__ == "__main__":
    exit(main())

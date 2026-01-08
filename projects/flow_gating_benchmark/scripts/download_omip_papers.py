#!/usr/bin/env python3
"""
Standalone script to search PMC for all OMIP papers and download full papers.

Usage:
    python scripts/download_omip_papers.py --email your@email.com
    python scripts/download_omip_papers.py --email your@email.com --with-supplementary

This searches PMC for papers with "OMIP" in the title published in Cytometry,
then downloads the full-text XML and PDF from PMC Open Access FTP.
"""

import argparse
import json
import re
import tarfile
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

import httpx


def search_omip_papers(email: str, max_results: int = 200) -> list[str]:
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

    print("Searching PMC for OMIP papers...")
    response = httpx.get(search_url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    id_list = data.get("esearchresult", {}).get("idlist", [])
    print(f"Found {len(id_list)} OMIP papers")

    return id_list


def get_oa_links(pmc_id: str) -> dict:
    """Get Open Access download links from PMC OA API."""
    oa_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
    params = {"id": f"PMC{pmc_id}"}

    try:
        response = httpx.get(oa_url, params=params, timeout=30)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.text)

        links = {}
        for link in root.findall(".//link"):
            fmt = link.get("format")
            href = link.get("href")
            if fmt and href:
                # Convert ftp:// to https:// for easier downloading
                if href.startswith("ftp://"):
                    href = href.replace("ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/",
                                       "https://ftp.ncbi.nlm.nih.gov/pub/pmc/")
                links[fmt] = href

        return links
    except Exception as e:
        return {"error": str(e)}


def download_full_paper(pmc_id: str, output_dir: Path, with_supplementary: bool = False) -> dict:
    """Download full paper package from PMC OA."""

    paper_dir = output_dir / f"PMC{pmc_id}"
    xml_path = paper_dir / f"PMC{pmc_id}.xml"
    pdf_path = paper_dir / f"PMC{pmc_id}.pdf"

    # Check if already downloaded
    if xml_path.exists():
        has_pdf = pdf_path.exists()
        return {
            "pmc_id": pmc_id,
            "status": "cached",
            "xml_path": str(xml_path),
            "pdf_path": str(pdf_path) if has_pdf else None,
        }

    # Get OA links
    links = get_oa_links(pmc_id)
    if "error" in links:
        return {"pmc_id": pmc_id, "status": "error", "error": links["error"]}

    if not links:
        return {"pmc_id": pmc_id, "status": "not_oa", "error": "No Open Access files available"}

    paper_dir.mkdir(parents=True, exist_ok=True)
    result = {"pmc_id": pmc_id, "status": "success", "files": []}

    # Download tgz package (contains full XML and PDF)
    if "tgz" in links:
        try:
            print(f"    Downloading package...", end=" ", flush=True)
            resp = httpx.get(links["tgz"], timeout=120, follow_redirects=True)
            resp.raise_for_status()

            # Extract from tar.gz
            with tarfile.open(fileobj=BytesIO(resp.content), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".nxml") or member.name.endswith(".xml"):
                        # Extract XML
                        content = tar.extractfile(member)
                        if content:
                            xml_path.write_bytes(content.read())
                            result["xml_path"] = str(xml_path)
                            result["files"].append("xml")
                    elif member.name.endswith(".pdf") and "supp" not in member.name.lower():
                        # Extract main PDF (not supplementary)
                        content = tar.extractfile(member)
                        if content:
                            pdf_path.write_bytes(content.read())
                            result["pdf_path"] = str(pdf_path)
                            result["files"].append("pdf")
                    elif with_supplementary and member.isfile():
                        # Extract supplementary files
                        content = tar.extractfile(member)
                        if content:
                            supp_path = paper_dir / Path(member.name).name
                            supp_path.write_bytes(content.read())
                            result["files"].append(Path(member.name).name)

            print("done")
        except Exception as e:
            print(f"failed: {e}")
            result["tgz_error"] = str(e)

    # Fallback: download PDF directly if not in package
    if "pdf" in links and "pdf" not in result.get("files", []):
        try:
            print(f"    Downloading PDF...", end=" ", flush=True)
            resp = httpx.get(links["pdf"], timeout=120, follow_redirects=True)
            resp.raise_for_status()
            pdf_path.write_bytes(resp.content)
            result["pdf_path"] = str(pdf_path)
            result["files"].append("pdf")
            print("done")
        except Exception as e:
            print(f"failed: {e}")
            result["pdf_error"] = str(e)

    if not result.get("files"):
        result["status"] = "error"
        result["error"] = "No files downloaded"

    return result


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
    parser = argparse.ArgumentParser(description="Download all OMIP papers from PMC (full text)")
    parser.add_argument("--email", required=True, help="Your email (required by NCBI)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/papers/pmc"),
                        help="Output directory for papers")
    parser.add_argument("--max-results", type=int, default=200,
                        help="Maximum papers to search for")
    parser.add_argument("--with-supplementary", action="store_true",
                        help="Also extract supplementary files from packages")

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
        print(f"[{i}/{len(pmc_ids)}] PMC{pmc_id}...")

        result = download_full_paper(pmc_id, args.output_dir, args.with_supplementary)
        results.append(result)

        if result["status"] == "cached":
            print(f"    cached (xml{'+ pdf' if result.get('pdf_path') else ''})")
        elif result["status"] == "success":
            files = result.get("files", [])
            print(f"    downloaded: {', '.join(files)}")
        else:
            print(f"    {result['status']}: {result.get('error', 'unknown')}")

        # Rate limit: 3 requests/second without API key
        if result["status"] not in ("cached", "not_oa"):
            time.sleep(0.5)

    # Extract OMIP info from downloaded XMLs
    print("\nExtracting OMIP information...")
    omip_papers = []
    for xml_file in sorted(args.output_dir.glob("PMC*/PMC*.xml")):
        info = extract_omip_info(xml_file)
        if info and info.get("omip_id"):
            omip_papers.append(info)
            print(f"  {info['omip_id']}: {info.get('title', 'No title')[:60]}...")

    # Also check old location for backward compatibility
    for xml_file in sorted(args.output_dir.glob("PMC*.xml")):
        if xml_file.parent == args.output_dir:  # Only top-level XMLs
            info = extract_omip_info(xml_file)
            if info and info.get("omip_id"):
                # Check if not already added
                if not any(p["pmcid"] == info["pmcid"] for p in omip_papers):
                    omip_papers.append(info)

    # Sort by OMIP number
    omip_papers.sort(key=lambda x: int(re.search(r'\d+', x.get('omip_id', '0')).group()))

    # Save index
    index_path = args.output_dir / "omip_index.json"
    with open(index_path, "w") as f:
        json.dump(omip_papers, f, indent=2)
    print(f"\nSaved index of {len(omip_papers)} OMIP papers to {index_path}")

    # Summary
    success = sum(1 for r in results if r["status"] in ("success", "cached"))
    with_pdf = sum(1 for r in results if r.get("pdf_path"))
    print(f"\n=== Summary ===")
    print(f"Total searched: {len(pmc_ids)}")
    print(f"Successfully downloaded: {success}")
    print(f"With PDF: {with_pdf}")
    print(f"OMIP papers identified: {len(omip_papers)}")

    return 0


if __name__ == "__main__":
    exit(main())

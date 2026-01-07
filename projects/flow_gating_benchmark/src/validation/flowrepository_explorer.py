"""
FlowRepository dataset exploration.

This module provides tools to explore FlowRepository datasets and check
for available .wsp files that can be used for benchmark ground truth.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx


FLOWREPOSITORY_BASE_URL = "https://flowrepository.org"
API_BASE_URL = f"{FLOWREPOSITORY_BASE_URL}/ajax/list_repo_files.php"


@dataclass
class DatasetInfo:
    """Information about a FlowRepository dataset."""

    dataset_id: str
    title: str | None = None
    has_wsp: bool = False
    has_fcs: bool = False
    wsp_files: list[str] | None = None
    fcs_count: int = 0
    metadata: dict | None = None
    error: str | None = None


# Known OMIP-FlowRepository pairs for testing
KNOWN_OMIP_DATASETS = {
    "OMIP-069": "FR-FCM-Z7YM",  # 40-color spectral PBMC
    "OMIP-058": "FR-FCM-ZYRN",  # 30-color T/NK/iNKT
    "OMIP-044": "FR-FCM-ZYC2",  # 28-color DC compartment
    "OMIP-043": "FR-FCM-ZYBP",  # Antibody secreting cells
    "OMIP-023": "FR-FCM-ZZ74",  # 10-color leukocyte
    "OMIP-021": "FR-FCM-ZZ9H",  # Innate-like T cells
}


def check_wsp_availability(dataset_id: str) -> DatasetInfo:
    """
    Check if a FlowRepository dataset has .wsp files available.

    Args:
        dataset_id: FlowRepository ID (e.g., 'FR-FCM-Z7YM')

    Returns:
        DatasetInfo with availability information
    """
    info = DatasetInfo(dataset_id=dataset_id)

    try:
        # Try to get file listing via API
        # Note: FlowRepository's API may require authentication for some datasets
        url = f"{FLOWREPOSITORY_BASE_URL}/id/{dataset_id}"

        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            html = response.text

            # Parse title from HTML
            title_match = re.search(r"<title>([^<]+)</title>", html)
            if title_match:
                info.title = title_match.group(1).strip()

            # Look for .wsp file references
            wsp_pattern = r'href="[^"]*([^/"]+\.wsp)"'
            wsp_matches = re.findall(wsp_pattern, html, re.IGNORECASE)
            info.wsp_files = list(set(wsp_matches))
            info.has_wsp = len(info.wsp_files) > 0

            # Look for .fcs file references
            fcs_pattern = r'\.fcs'
            fcs_matches = re.findall(fcs_pattern, html, re.IGNORECASE)
            info.fcs_count = len(fcs_matches)
            info.has_fcs = info.fcs_count > 0

    except httpx.HTTPStatusError as e:
        info.error = f"HTTP error: {e.response.status_code}"
    except httpx.RequestError as e:
        info.error = f"Request error: {str(e)}"
    except Exception as e:
        info.error = f"Unexpected error: {str(e)}"

    return info


def explore_dataset(dataset_id: str, verbose: bool = True) -> dict[str, Any]:
    """
    Explore a FlowRepository dataset for benchmark suitability.

    Args:
        dataset_id: FlowRepository ID
        verbose: Whether to print progress

    Returns:
        Dictionary with exploration results
    """
    if verbose:
        print(f"\nExploring dataset: {dataset_id}")
        print("-" * 40)

    info = check_wsp_availability(dataset_id)

    result = {
        "dataset_id": dataset_id,
        "url": f"{FLOWREPOSITORY_BASE_URL}/id/{dataset_id}",
        "title": info.title,
        "has_wsp": info.has_wsp,
        "wsp_files": info.wsp_files,
        "has_fcs": info.has_fcs,
        "fcs_count": info.fcs_count,
        "error": info.error,
        "suitable_for_benchmark": info.has_wsp and not info.error,
    }

    if verbose:
        print(f"Title: {info.title}")
        print(f"Has .wsp files: {info.has_wsp}")
        if info.wsp_files:
            print(f"WSP files found: {info.wsp_files}")
        print(f"Has .fcs files: {info.has_fcs} (count: {info.fcs_count})")
        if info.error:
            print(f"Error: {info.error}")
        print(f"Suitable for benchmark: {result['suitable_for_benchmark']}")

    return result


def explore_known_omip_datasets() -> list[dict[str, Any]]:
    """
    Explore all known OMIP-FlowRepository pairs.

    Returns:
        List of exploration results for each dataset
    """
    print("=" * 60)
    print("EXPLORING KNOWN OMIP DATASETS")
    print("=" * 60)

    results = []
    for omip, fr_id in KNOWN_OMIP_DATASETS.items():
        print(f"\n{omip} -> {fr_id}")
        result = explore_dataset(fr_id, verbose=True)
        result["omip_id"] = omip
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    suitable = [r for r in results if r["suitable_for_benchmark"]]
    print(f"\nDatasets with .wsp files: {len(suitable)}/{len(results)}")

    for r in suitable:
        print(f"  ✓ {r['omip_id']} ({r['dataset_id']})")

    not_suitable = [r for r in results if not r["suitable_for_benchmark"]]
    if not_suitable:
        print(f"\nDatasets without .wsp files or with errors:")
        for r in not_suitable:
            reason = r.get("error") or "No .wsp files"
            print(f"  ✗ {r['omip_id']} ({r['dataset_id']}): {reason}")

    return results


def download_wsp_file(dataset_id: str, wsp_filename: str, output_path: str) -> bool:
    """
    Download a .wsp file from FlowRepository.

    Note: Many FlowRepository files require authentication or have
    download restrictions. This function may need adjustment based
    on the specific dataset's access policy.

    Args:
        dataset_id: FlowRepository ID
        wsp_filename: Name of the .wsp file
        output_path: Where to save the file

    Returns:
        True if download succeeded
    """
    # FlowRepository file download URLs vary by dataset
    # This is a best-effort implementation
    possible_urls = [
        f"{FLOWREPOSITORY_BASE_URL}/id/{dataset_id}/file/{wsp_filename}",
        f"{FLOWREPOSITORY_BASE_URL}/public/{dataset_id}/{wsp_filename}",
    ]

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for url in possible_urls:
            try:
                response = client.get(url)
                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    print(f"Downloaded: {output_path}")
                    return True
            except Exception:
                continue

    print(f"Failed to download {wsp_filename} from {dataset_id}")
    print("Note: You may need to download manually from the FlowRepository website")
    return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Explore specific dataset
        dataset_id = sys.argv[1]
        result = explore_dataset(dataset_id)
        print("\n" + json.dumps(result, indent=2))
    else:
        # Explore all known OMIP datasets
        results = explore_known_omip_datasets()

        # Save results
        output_file = "data/flowrepository_exploration.json"
        try:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"\nCould not save results: {e}")

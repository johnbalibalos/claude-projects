#!/usr/bin/env python3
"""
Fetch fresh PubMed citation counts for population names.

Uses NCBI E-utilities API to get up-to-date counts.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def fetch_pubmed_count(query: str, api_key: str = None, retries: int = 3) -> int:
    """
    Fetch PubMed article count for a search query.

    Args:
        query: Search term
        api_key: Optional NCBI API key (get free at https://www.ncbi.nlm.nih.gov/account/settings/)
                 With API key: 10 requests/sec, without: 3 requests/sec
        retries: Number of retry attempts
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    # Build query - search in title/abstract for the term
    params = {
        "db": "pubmed",
        "term": f'"{query}"[Title/Abstract]',
        "rettype": "count",
        "retmode": "json",
    }

    # Add API key if provided (increases rate limit to 10/sec)
    if api_key:
        params["api_key"] = api_key

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    # Set proper headers to avoid blocks
    headers = {
        "User-Agent": "PubMedFrequencyAnalysis/1.0 (research; contact@example.com)",
        "Accept": "application/json",
    }

    request = urllib.request.Request(url, headers=headers)

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.loads(response.read().decode())
                count = int(data.get("esearchresult", {}).get("count", 0))
                return count
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limited
                wait_time = 2 ** (attempt + 1)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                print(f"  HTTP Error fetching '{query}': {e.code} {e.reason}")
                return -1
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                print(f"  Error fetching '{query}': {e}")
                return -1

    return -1


def extract_canonical_terms(equivalences_path: Path) -> set[str]:
    """Extract canonical population names from verified equivalences."""
    if not HAS_YAML:
        print("  Warning: PyYAML not installed, skipping equivalences file")
        return set()

    if not equivalences_path.exists():
        return set()

    with open(equivalences_path) as f:
        data = yaml.safe_load(f)

    canonical_terms = set()
    for eq_class in data.get("equivalence_classes", []):
        canonical = eq_class.get("canonical", "")
        if canonical:
            canonical_terms.add(canonical)
        # Also add some key variants that might be searched differently
        for variant in eq_class.get("variants", [])[:3]:  # First few variants
            if len(variant) > 3 and not variant.startswith("cd"):
                canonical_terms.add(variant)

    return canonical_terms


def extract_unique_populations(scoring_path: Path) -> set[str]:
    """Extract all unique population names from benchmark results."""
    with open(scoring_path) as f:
        data = json.load(f)

    populations = set()

    for result in data.get("results", []):
        # From ground truth
        for gate in result.get("ground_truth_gates", []):
            populations.add(gate.strip())

        # From parsed hierarchy
        def extract_from_hierarchy(node):
            if node.get("name"):
                populations.add(node["name"].strip())
            for child in node.get("children", []):
                extract_from_hierarchy(child)

        hierarchy = result.get("parsed_hierarchy", {})
        if hierarchy:
            extract_from_hierarchy(hierarchy)

    return populations


def main():
    parser = argparse.ArgumentParser(
        description="Fetch fresh PubMed citation counts for cell populations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (rate limited to 3 req/sec)
  python fetch_fresh_frequencies.py

  # With NCBI API key (10 req/sec, recommended)
  python fetch_fresh_frequencies.py --api-key YOUR_API_KEY

  # Using environment variable
  export NCBI_API_KEY=your_key_here
  python fetch_fresh_frequencies.py

  # Limit number of populations (for testing)
  python fetch_fresh_frequencies.py --limit 50

Get a free NCBI API key at:
  https://www.ncbi.nlm.nih.gov/account/settings/
        """
    )
    parser.add_argument("--api-key", type=str, default=os.environ.get("NCBI_API_KEY"),
                       help="NCBI API key (or set NCBI_API_KEY env var)")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit number of populations to fetch (0=all)")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output file path (default: data/cache/pubmed_frequencies_2026.json)")
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    results_dir = project_dir / "results" / "full_benchmark_20260114"
    scoring_path = results_dir / "scoring_results.json"

    # Extract unique populations from benchmark results
    print("Extracting unique populations from benchmark results...")
    populations = extract_unique_populations(scoring_path)
    print(f"Found {len(populations)} unique populations from benchmark")

    # Also include canonical terms from verified equivalences
    equivalences_path = project_dir / "data" / "annotations" / "verified_equivalences.yaml"
    canonical_terms = extract_canonical_terms(equivalences_path)
    print(f"Found {len(canonical_terms)} canonical terms from verified equivalences")

    # Merge
    populations = populations | canonical_terms
    print(f"Total unique terms: {len(populations)}")

    # Load existing cache
    cache_path = project_dir / "data" / "cache" / "pubmed_frequencies.json"
    existing_cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            existing_cache = json.load(f)

    # Filter to populations we need to look up
    # Skip technical gates that won't have PubMed results
    skip_terms = ["singlet", "live", "dead", "all event", "time", "cleaned", "non-", "gate", "beads",
                  "children", "markers", "name", "//", "{", "}", "[", "]"]

    to_fetch = []
    for pop in populations:
        pop_lower = pop.lower()
        # Skip technical gates and JSON artifacts
        if any(term in pop_lower for term in skip_terms):
            continue
        # Skip very short or very long strings (likely artifacts)
        if len(pop) < 3 or len(pop) > 100:
            continue
        to_fetch.append(pop)

    # Apply limit if specified
    if args.limit > 0:
        to_fetch = sorted(to_fetch)[:args.limit]
        print(f"Limited to {len(to_fetch)} populations (--limit {args.limit})")
    else:
        print(f"Will fetch counts for {len(to_fetch)} populations")

    # Rate limit info
    if args.api_key:
        print(f"Using NCBI API key (10 requests/sec)")
        rate_delay = 0.12  # 10/sec with margin
    else:
        print("No API key - rate limited to 3 requests/sec")
        print("Tip: Get a free API key at https://www.ncbi.nlm.nih.gov/account/settings/")
        rate_delay = 0.4  # 3/sec with margin

    # Fetch fresh counts from PubMed
    print(f"\nFetching fresh PubMed counts (estimated time: {len(to_fetch) * rate_delay / 60:.1f} min)...")
    fresh_counts = {}
    failed = []

    for i, pop in enumerate(sorted(to_fetch)):
        # Rate limit
        if i > 0:
            time.sleep(rate_delay)

        count = fetch_pubmed_count(pop, api_key=args.api_key)
        if count >= 0:
            fresh_counts[pop] = count
        else:
            failed.append(pop)

        # Progress indicator
        if (i + 1) % 20 == 0 or i == len(to_fetch) - 1:
            print(f"  Progress: {i + 1}/{len(to_fetch)} ({len(failed)} failed)")

    # Save fresh counts
    if args.output:
        fresh_cache_path = args.output
    else:
        fresh_cache_path = project_dir / "data" / "cache" / "pubmed_frequencies_2026.json"
    fresh_cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(fresh_cache_path, "w") as f:
        json.dump(fresh_counts, f, indent=2, sort_keys=True)

    print(f"\nSaved fresh counts to: {fresh_cache_path}")

    # Print comparison for a few terms
    print("\n" + "=" * 70)
    print("SAMPLE COMPARISON: Old vs Fresh PubMed Counts")
    print("=" * 70)

    comparison_terms = [
        "T cells", "B cells", "NK cells", "Monocytes", "Dendritic cells",
        "CD4+ T cells", "CD8+ T cells", "Regulatory T cells", "Memory B cells",
        "Naive T cells", "Plasma cells", "Neutrophils", "Macrophages"
    ]

    print(f"\n{'Population':<35} {'Old':>12} {'Fresh':>12} {'Change':>10}")
    print("-" * 70)

    for term in comparison_terms:
        old = existing_cache.get(term, "N/A")
        fresh = fresh_counts.get(term, "N/A")

        if isinstance(old, int) and isinstance(fresh, int) and old > 0:
            change = f"{((fresh - old) / old * 100):+.1f}%"
        else:
            change = "N/A"

        old_str = f"{old:,}" if isinstance(old, int) else str(old)
        fresh_str = f"{fresh:,}" if isinstance(fresh, int) else str(fresh)
        print(f"{term:<35} {old_str:>12} {fresh_str:>12} {change:>10}")

    # Summary stats
    valid_fresh = [v for v in fresh_counts.values() if v >= 0]
    print(f"\n\nSummary:")
    print(f"  Total populations fetched: {len(fresh_counts)}")
    print(f"  Successful queries: {len(valid_fresh)}")
    print(f"  Failed queries: {len(fresh_counts) - len(valid_fresh)}")
    if valid_fresh:
        print(f"  Count range: {min(valid_fresh):,} - {max(valid_fresh):,}")

    return fresh_counts


if __name__ == "__main__":
    main()

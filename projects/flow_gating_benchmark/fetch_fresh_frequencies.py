#!/usr/bin/env python3
"""
Fetch fresh PubMed citation counts for population names.

Uses NCBI E-utilities API to get up-to-date counts.
"""

import json
import math
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def fetch_pubmed_count(query: str, retries: int = 3) -> int:
    """Fetch PubMed article count for a search query."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    # Build query - search in title/abstract for the term
    params = {
        "db": "pubmed",
        "term": f'"{query}"[Title/Abstract]',
        "rettype": "count",
        "retmode": "json",
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
                count = int(data.get("esearchresult", {}).get("count", 0))
                return count
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))  # Backoff
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
    skip_terms = ["singlet", "live", "dead", "all event", "time", "cleaned", "non-", "gate", "beads"]

    to_fetch = []
    for pop in populations:
        pop_lower = pop.lower()
        if any(term in pop_lower for term in skip_terms):
            continue
        to_fetch.append(pop)

    print(f"Will fetch counts for {len(to_fetch)} populations")

    # Fetch fresh counts from PubMed
    print("\nFetching fresh PubMed counts (this may take a few minutes)...")
    fresh_counts = {}

    for i, pop in enumerate(sorted(to_fetch)):
        # Rate limit: NCBI requests max 3/second without API key
        if i > 0 and i % 3 == 0:
            time.sleep(0.4)

        count = fetch_pubmed_count(pop)
        fresh_counts[pop] = count

        # Progress indicator
        if (i + 1) % 20 == 0 or i == len(to_fetch) - 1:
            print(f"  Progress: {i + 1}/{len(to_fetch)}")

    # Save fresh counts
    fresh_cache_path = project_dir / "data" / "cache" / "pubmed_frequencies_fresh.json"
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

#!/usr/bin/env python3
"""
Lightweight contamination detection for flow gating benchmark.

Checks whether LLMs may have memorized OMIP test cases from training data.

Usage:
    python scripts/check_contamination.py --model claude-sonnet-4-20250514
    python scripts/check_contamination.py --model claude-sonnet-4-20250514 --test-cases OMIP-074 OMIP-076
    python scripts/check_contamination.py --dry-run  # Show what would be tested

Tests performed:
1. Title completion: Can model complete partial OMIP paper titles?
2. Hierarchy completion: Can model predict gating structure from partial hierarchy?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "libs"))

from curation.omip_extractor import KNOWN_OMIPS, load_test_case  # type: ignore[import-not-found]
from curation.schemas import GateNode  # type: ignore[import-not-found]


@dataclass
class ContaminationResult:
    """Result of contamination check for a single test case."""

    test_case_id: str
    title_completion_score: float  # 0-1, how much of title was reproduced
    hierarchy_completion_score: float  # 0-1, gate name overlap
    title_prompt: str
    title_response: str
    hierarchy_prompt: str
    hierarchy_response: str
    risk_level: str  # "low", "medium", "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "title_completion_score": self.title_completion_score,
            "hierarchy_completion_score": self.hierarchy_completion_score,
            "risk_level": self.risk_level,
            "title_prompt": self.title_prompt,
            "title_response": self.title_response,
            "hierarchy_prompt": self.hierarchy_prompt,
            "hierarchy_response": self.hierarchy_response,
        }


@dataclass
class ContaminationReport:
    """Aggregate contamination report."""

    model: str
    timestamp: str
    n_test_cases: int
    n_high_risk: int
    n_medium_risk: int
    results: list[ContaminationResult]
    recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "CONTAMINATION DETECTION REPORT",
            "=" * 60,
            f"Model: {self.model}",
            f"Timestamp: {self.timestamp}",
            f"Test cases analyzed: {self.n_test_cases}",
            "",
            f"High risk (likely memorized): {self.n_high_risk}",
            f"Medium risk (possible exposure): {self.n_medium_risk}",
            "",
        ]

        if self.n_high_risk > 0:
            lines.append("HIGH RISK CASES:")
            for r in self.results:
                if r.risk_level == "high":
                    lines.append(
                        f"  - {r.test_case_id}: title={r.title_completion_score:.0%}, "
                        f"hierarchy={r.hierarchy_completion_score:.0%}"
                    )
            lines.append("")

        if self.recommendations:
            lines.append("RECOMMENDATIONS:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "timestamp": self.timestamp,
            "n_test_cases": self.n_test_cases,
            "n_high_risk": self.n_high_risk,
            "n_medium_risk": self.n_medium_risk,
            "recommendations": self.recommendations,
            "results": [r.to_dict() for r in self.results],
        }


def get_hierarchy_gates(node: GateNode, depth: int = 0, max_depth: int = 2) -> list[str]:
    """Extract gate names from hierarchy up to max_depth."""
    gates = [node.name]
    if depth < max_depth:
        for child in node.children:
            gates.extend(get_hierarchy_gates(child, depth + 1, max_depth))
    return gates


def compute_text_overlap(expected: str, actual: str) -> float:
    """Compute word-level overlap between expected and actual text."""
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())

    if not expected_words:
        return 0.0

    overlap = len(expected_words & actual_words)
    return overlap / len(expected_words)


def call_model(model: str, prompt: str, max_tokens: int = 256) -> str:
    """Call LLM API and return response."""
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        first_block = response.content[0]
        # TextBlock has .text, other block types don't
        return getattr(first_block, "text", str(first_block))
    except Exception as e:
        return f"[ERROR: {e}]"


def check_title_completion(
    omip_id: str,
    model: str,
    dry_run: bool = False,
) -> tuple[float, str, str]:
    """
    Check if model can complete OMIP paper title.

    Returns:
        Tuple of (score, prompt, response)
    """
    omip_info = KNOWN_OMIPS.get(omip_id, {})
    title = omip_info.get("title", "")

    if not title:
        return 0.0, "", "[No title available]"

    # Provide first half of title
    half_point = len(title) // 2
    partial_title = title[:half_point]
    expected_completion = title[half_point:]

    prompt = f"""Complete this scientific paper title. Only provide the completion, not the full title.

Title start: "{partial_title}..."

Completion:"""

    if dry_run:
        return 0.0, prompt, "[DRY RUN]"

    response = call_model(model, prompt)
    score = compute_text_overlap(expected_completion, response)

    return score, prompt, response


def check_hierarchy_completion(
    test_case_path: Path,
    model: str,
    dry_run: bool = False,
) -> tuple[float, str, str]:
    """
    Check if model can complete partial gating hierarchy.

    Returns:
        Tuple of (score, prompt, response)
    """
    try:
        test_case = load_test_case(test_case_path)
    except Exception:
        return 0.0, "", "[Could not load test case]"

    # Get first 2 levels of hierarchy
    root = test_case.gating_hierarchy.root
    partial_gates = get_hierarchy_gates(root, max_depth=1)  # Just top 2 levels
    all_gates = get_hierarchy_gates(root, max_depth=10)
    expected_gates = [g for g in all_gates if g not in partial_gates]

    partial_hierarchy = "\n".join(f"  - {g}" for g in partial_gates)

    prompt = f"""Given this partial flow cytometry gating hierarchy, predict what additional gates would typically come next.

Panel context: {len(test_case.panel.markers)} color panel for {test_case.context.sample_type}

Partial hierarchy:
{partial_hierarchy}

List the additional gates that would typically follow:"""

    if dry_run:
        return 0.0, prompt, "[DRY RUN]"

    response = call_model(model, prompt, max_tokens=512)

    # Score based on how many expected gates appear in response
    response_lower = response.lower()
    matched = sum(1 for g in expected_gates if g.lower() in response_lower)
    score = matched / len(expected_gates) if expected_gates else 0.0

    return score, prompt, response


def determine_risk_level(title_score: float, hierarchy_score: float) -> str:
    """Determine contamination risk level from scores."""
    if title_score > 0.7 or hierarchy_score > 0.6:
        return "high"
    elif title_score > 0.4 or hierarchy_score > 0.4:
        return "medium"
    return "low"


def run_contamination_check(
    model: str,
    test_case_ids: list[str] | None,
    ground_truth_dir: Path,
    dry_run: bool = False,
) -> ContaminationReport:
    """Run contamination check on specified test cases."""
    # Find available test cases
    if test_case_ids:
        # Use specified test cases
        tc_files = []
        for tc_id in test_case_ids:
            filename = tc_id.lower().replace("-", "_") + ".json"
            path = ground_truth_dir / filename
            if path.exists():
                tc_files.append((tc_id, path))
            else:
                print(f"Warning: {tc_id} not found at {path}")
    else:
        # Use all available test cases
        tc_files = []
        for path in ground_truth_dir.glob("omip_*.json"):
            tc_id = "OMIP-" + path.stem.replace("omip_", "").upper()
            tc_files.append((tc_id, path))

    results = []
    for tc_id, tc_path in tc_files:
        print(f"Checking {tc_id}...", end=" ", flush=True)

        title_score, title_prompt, title_response = check_title_completion(
            tc_id, model, dry_run
        )
        hierarchy_score, hierarchy_prompt, hierarchy_response = check_hierarchy_completion(
            tc_path, model, dry_run
        )

        risk_level = determine_risk_level(title_score, hierarchy_score)
        print(f"title={title_score:.0%}, hierarchy={hierarchy_score:.0%} [{risk_level}]")

        results.append(
            ContaminationResult(
                test_case_id=tc_id,
                title_completion_score=title_score,
                hierarchy_completion_score=hierarchy_score,
                title_prompt=title_prompt,
                title_response=title_response,
                hierarchy_prompt=hierarchy_prompt,
                hierarchy_response=hierarchy_response,
                risk_level=risk_level,
            )
        )

    # Generate recommendations
    recommendations = []
    n_high = sum(1 for r in results if r.risk_level == "high")
    n_medium = sum(1 for r in results if r.risk_level == "medium")

    if n_high > 0:
        recommendations.append(
            f"Consider excluding {n_high} high-risk test cases from primary metrics"
        )
        recommendations.append("Report results with and without high-risk cases")

    if n_medium > len(results) * 0.3:
        recommendations.append(
            "High proportion of medium-risk cases suggests general OMIP exposure"
        )
        recommendations.append("Consider using synthetic test cases for unbiased evaluation")

    if n_high == 0 and n_medium == 0:
        recommendations.append("No significant contamination signals detected")

    return ContaminationReport(
        model=model,
        timestamp=datetime.now().isoformat(),
        n_test_cases=len(results),
        n_high_risk=n_high,
        n_medium_risk=n_medium,
        results=results,
        recommendations=recommendations,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Check for benchmark contamination in LLMs"
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model to test (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--test-cases",
        nargs="+",
        help="Specific test case IDs to check (e.g., OMIP-074 OMIP-076)",
    )
    parser.add_argument(
        "--ground-truth-dir",
        default="data/ground_truth",
        help="Directory containing ground truth JSON files",
    )
    parser.add_argument(
        "--output",
        help="Output file for JSON report",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show prompts without calling API",
    )

    args = parser.parse_args()

    ground_truth_dir = Path(args.ground_truth_dir)
    if not ground_truth_dir.exists():
        print(f"Error: Ground truth directory not found: {ground_truth_dir}")
        sys.exit(1)

    print(f"Running contamination check with model: {args.model}")
    print(f"Ground truth directory: {ground_truth_dir}")
    if args.dry_run:
        print("DRY RUN MODE - no API calls will be made")
    print()

    report = run_contamination_check(
        model=args.model,
        test_case_ids=args.test_cases,
        ground_truth_dir=ground_truth_dir,
        dry_run=args.dry_run,
    )

    print()
    print(report.summary())

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()

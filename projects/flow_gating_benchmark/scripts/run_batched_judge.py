#!/usr/bin/env python3
"""
Batched Multi-Judge Evaluation Script

Runs multiple judge styles on scoring results, batching bootstrap runs into
a single prompt to reduce API calls.

For n=10 bootstrap runs with 5 judge styles:
- Without batching: 10 × 5 = 50 calls per (test_case, condition)
- With batching: 5 calls per (test_case, condition) = 10× reduction

Usage:
    # Test on best condition first (1 condition × 13 test cases × 5 styles = 65 calls)
    python scripts/run_batched_judge.py \
        --input results/rag_test \
        --condition standard_direct_oracle \
        --dry-run

    # Run on all conditions
    python scripts/run_batched_judge.py \
        --input results/rag_test \
        --all-conditions
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from curation.omip_extractor import load_all_test_cases  # noqa: E402
from experiments.batch_scorer import ScoringResult  # noqa: E402
from experiments.llm_client import create_client  # noqa: E402
from experiments.llm_judge import (  # noqa: E402
    JUDGE_STYLES,
    flatten_hierarchy,
    format_prediction_for_judge,
)


@dataclass
class BatchedJudgeResult:
    """Result from batched judge evaluation."""
    test_case_id: str
    model: str
    condition: str
    judge_style: str
    judge_model: str

    # Aggregated scores (0-1 scale)
    completeness: float
    accuracy: float
    scientific: float
    overall: float

    # Variance across bootstrap runs
    variance_summary: str

    # Qualitative feedback
    common_issues: str
    summary: str

    # Meta
    n_bootstrap: int
    timestamp: datetime
    tokens_used: int
    error: str | None = None

    # Full prompt/response for manual review
    judge_prompt: str = ""
    judge_raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "model": self.model,
            "condition": self.condition,
            "judge_style": self.judge_style,
            "judge_model": self.judge_model,
            "completeness": self.completeness,
            "accuracy": self.accuracy,
            "scientific": self.scientific,
            "overall": self.overall,
            "variance_summary": self.variance_summary,
            "common_issues": self.common_issues,
            "summary": self.summary,
            "n_bootstrap": self.n_bootstrap,
            "timestamp": self.timestamp.isoformat(),
            "tokens_used": self.tokens_used,
            "error": self.error,
            "judge_prompt": self.judge_prompt,
            "judge_raw_response": self.judge_raw_response,
        }


def build_batched_default_prompt(
    test_case_id: str,
    predictions: list[ScoringResult],
    ground_truth: dict,
) -> str:
    """Build batched prompt for default judge style."""
    gt_hierarchy = ground_truth.get("gating_hierarchy", {})
    gt_flat = flatten_hierarchy(gt_hierarchy)
    gt_lines = gt_flat.split("\n")[:8]
    gt_summary = "\n".join(gt_lines)
    if len(gt_flat.split("\n")) > 8:
        gt_summary += f"\n... (+{len(gt_flat.split(chr(10))) - 8} more paths)"

    context = ground_truth.get("context", {})

    # Format each prediction
    pred_sections = []
    for i, pred in enumerate(predictions, 1):
        formatted = format_prediction_for_judge(pred.raw_response)
        metrics = f"F1={pred.hierarchy_f1:.2f}, Struct={pred.structure_accuracy:.2f}, CritRecall={pred.critical_gate_recall:.2f}"
        pred_sections.append(f"### Run {i}\n{formatted}\nMetrics: {metrics}")

    predictions_text = "\n\n".join(pred_sections)

    return f"""Score these {len(predictions)} bootstrap runs of a flow cytometry gating hierarchy prediction (0-10 scale).

TEST CASE: {test_case_id}
SAMPLE: {context.get('sample_type', 'unknown')} ({context.get('species', 'unknown')})

EXPECTED HIERARCHY (ground truth):
{gt_summary}

{len(predictions)} PREDICTIONS (bootstrap runs of same model/condition):

{predictions_text}

Assess the OVERALL quality across all runs. Consider consistency and common patterns.

Reply in this EXACT format:
COMPLETENESS: [0-10] [are all expected populations present?]
ACCURACY: [0-10] [are parent-child relationships correct?]
SCIENTIFIC: [0-10] [does the hierarchy make biological sense?]
OVERALL: [0-10] [overall quality score]
VARIANCE: [low/medium/high] [how consistent are the runs?]
COMMON_ISSUES: [comma-separated list or "none"]
SUMMARY: [one sentence overall assessment]
"""


def build_batched_validation_prompt(
    test_case_id: str,
    predictions: list[ScoringResult],
    ground_truth: dict,
) -> str:
    """Validation style: estimate metrics without seeing them."""
    gt_hierarchy = ground_truth.get("gating_hierarchy", {})
    gt_flat = flatten_hierarchy(gt_hierarchy)
    gt_lines = gt_flat.split("\n")[:8]
    gt_summary = "\n".join(gt_lines)
    if len(gt_flat.split("\n")) > 8:
        gt_summary += f"\n... (+{len(gt_flat.split(chr(10))) - 8} more paths)"

    context = ground_truth.get("context", {})

    # Format predictions WITHOUT showing metrics
    pred_sections = []
    for i, pred in enumerate(predictions, 1):
        formatted = format_prediction_for_judge(pred.raw_response)
        pred_sections.append(f"### Run {i}\n{formatted}")

    predictions_text = "\n\n".join(pred_sections)

    return f"""Evaluate {len(predictions)} bootstrap runs of a flow cytometry gating prediction.

TEST CASE: {test_case_id}
SAMPLE: {context.get('sample_type', 'unknown')} ({context.get('species', 'unknown')})

EXPECTED HIERARCHY (ground truth):
{gt_summary}

{len(predictions)} PREDICTIONS (bootstrap runs):

{predictions_text}

WITHOUT seeing automated metrics, estimate what they would be:

Reply in this EXACT format:
ESTIMATED_F1: [0.0-1.0] [average across runs]
ESTIMATED_STRUCTURE: [0.0-1.0] [average across runs]
ESTIMATED_CRITICAL_RECALL: [0.0-1.0] [average across runs]
VARIANCE: [low/medium/high] [run-to-run consistency]
CONFIDENCE: [high/medium/low]
REASONING: [one sentence explaining your estimates]
"""


def build_batched_qualitative_prompt(
    test_case_id: str,
    predictions: list[ScoringResult],
    ground_truth: dict,
) -> str:
    """Qualitative style: structured feedback, no scores."""
    gt_hierarchy = ground_truth.get("gating_hierarchy", {})
    gt_flat = flatten_hierarchy(gt_hierarchy)
    gt_lines = gt_flat.split("\n")[:8]
    gt_summary = "\n".join(gt_lines)
    if len(gt_flat.split("\n")) > 8:
        gt_summary += f"\n... (+{len(gt_flat.split(chr(10))) - 8} more paths)"

    context = ground_truth.get("context", {})

    pred_sections = []
    for i, pred in enumerate(predictions, 1):
        formatted = format_prediction_for_judge(pred.raw_response)
        pred_sections.append(f"### Run {i}\n{formatted}")

    predictions_text = "\n\n".join(pred_sections)

    return f"""Analyze {len(predictions)} bootstrap runs of a flow cytometry gating prediction.

TEST CASE: {test_case_id}
SAMPLE: {context.get('sample_type', 'unknown')} ({context.get('species', 'unknown')})

EXPECTED HIERARCHY (ground truth):
{gt_summary}

{len(predictions)} PREDICTIONS:

{predictions_text}

Provide structured feedback (no numerical scores):

Reply in this EXACT format:
CONSISTENT_ERRORS: [errors appearing in MOST runs, or "none"]
INCONSISTENT_ERRORS: [errors appearing in SOME runs, or "none"]
MISSING_GATES: [gates missing from MOST runs, or "none"]
EXTRA_GATES: [hallucinated gates appearing in MOST runs, or "none"]
VARIANCE: [low/medium/high] [run-to-run consistency]
ACCEPT_FOR_ANALYSIS: [yes/no] [would ANY run be acceptable for downstream analysis?]
SUMMARY: [one sentence assessment]
"""


def build_batched_orthogonal_prompt(
    test_case_id: str,
    predictions: list[ScoringResult],
    ground_truth: dict,
) -> str:
    """Orthogonal style: dimensions auto metrics can't capture."""
    gt_hierarchy = ground_truth.get("gating_hierarchy", {})
    gt_flat = flatten_hierarchy(gt_hierarchy)
    gt_lines = gt_flat.split("\n")[:8]
    gt_summary = "\n".join(gt_lines)
    if len(gt_flat.split("\n")) > 8:
        gt_summary += f"\n... (+{len(gt_flat.split(chr(10))) - 8} more paths)"

    context = ground_truth.get("context", {})
    application = context.get("application", "immunophenotyping")

    pred_sections = []
    for i, pred in enumerate(predictions, 1):
        formatted = format_prediction_for_judge(pred.raw_response)
        pred_sections.append(f"### Run {i}\n{formatted}")

    predictions_text = "\n\n".join(pred_sections)

    return f"""You are a flow cytometry expert evaluating {len(predictions)} bootstrap runs.

TEST CASE: {test_case_id}
SAMPLE: {context.get('sample_type', 'unknown')} ({context.get('species', 'unknown')})
APPLICATION: {application}

EXPECTED HIERARCHY (ground truth):
{gt_summary}

{len(predictions)} PREDICTIONS:

{predictions_text}

Rate on dimensions automated metrics CANNOT capture (0-10 each):

Reply in this EXACT format:
CLINICAL_UTILITY: [0-10] [would these hierarchies work for the stated application?]
BIOLOGICAL_PLAUSIBILITY: [0-10] [are parent-child relationships biologically sensible?]
HALLUCINATION_SEVERITY: [0-10] [0=no hallucinations, 10=severe invented gates]
MARKER_LOGIC: [0-10] [are marker combinations used correctly?]
VARIANCE: [low/medium/high] [consistency across runs]
SUMMARY: [one sentence assessment]
"""


def build_batched_binary_prompt(
    test_case_id: str,
    predictions: list[ScoringResult],
    ground_truth: dict,
) -> str:
    """Binary style: accept/reject with specific issues."""
    gt_hierarchy = ground_truth.get("gating_hierarchy", {})
    gt_flat = flatten_hierarchy(gt_hierarchy)
    gt_lines = gt_flat.split("\n")[:8]
    gt_summary = "\n".join(gt_lines)
    if len(gt_flat.split("\n")) > 8:
        gt_summary += f"\n... (+{len(gt_flat.split(chr(10))) - 8} more paths)"

    context = ground_truth.get("context", {})

    pred_sections = []
    for i, pred in enumerate(predictions, 1):
        formatted = format_prediction_for_judge(pred.raw_response)
        pred_sections.append(f"### Run {i}\n{formatted}")

    predictions_text = "\n\n".join(pred_sections)

    return f"""Evaluate {len(predictions)} bootstrap runs of a flow cytometry gating prediction.

TEST CASE: {test_case_id}
SAMPLE: {context.get('sample_type', 'unknown')} ({context.get('species', 'unknown')})

EXPECTED HIERARCHY (ground truth):
{gt_summary}

{len(predictions)} PREDICTIONS:

{predictions_text}

Reply in this EXACT format:
ACCEPTABLE_RUNS: [number of runs that would be acceptable, e.g., "7/10"]
CRITICAL_ERRORS: [blocking issues in MOST runs, or "none"]
MISSING_GATES: [gates missing from MOST runs, or "none"]
EXTRA_GATES: [hallucinated gates in MOST runs, or "none"]
VARIANCE: [low/medium/high]
RECOMMENDATION: [one sentence: what would make this acceptable?]
"""


BATCHED_PROMPT_BUILDERS = {
    "default": build_batched_default_prompt,
    "validation": build_batched_validation_prompt,
    "qualitative": build_batched_qualitative_prompt,
    "orthogonal": build_batched_orthogonal_prompt,
    "binary": build_batched_binary_prompt,
}


def parse_batched_default_response(content: str) -> dict | None:
    """Parse batched default response."""
    result = {}
    patterns = {
        "completeness": r"COMPLETENESS:\s*(\d+)",
        "accuracy": r"ACCURACY:\s*(\d+)",
        "scientific": r"SCIENTIFIC:\s*(\d+)",
        "overall": r"OVERALL:\s*(\d+)",
        "variance": r"VARIANCE:\s*(\w+)",
        "common_issues": r"COMMON_ISSUES:\s*(.+)",
        "summary": r"SUMMARY:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key in ["completeness", "accuracy", "scientific", "overall"]:
                try:
                    result[key] = int(value) / 10.0
                except ValueError:
                    pass
            else:
                result[key] = value

    if "overall" in result:
        return result
    return None


def parse_batched_validation_response(content: str) -> dict | None:
    """Parse batched validation response."""
    result = {}
    patterns = {
        "estimated_f1": r"ESTIMATED_F1:\s*([0-9.]+)",
        "estimated_structure": r"ESTIMATED_STRUCTURE:\s*([0-9.]+)",
        "estimated_critical_recall": r"ESTIMATED_CRITICAL_RECALL:\s*([0-9.]+)",
        "variance": r"VARIANCE:\s*(\w+)",
        "confidence": r"CONFIDENCE:\s*(\w+)",
        "reasoning": r"REASONING:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key.startswith("estimated_"):
                try:
                    result[key] = float(value)
                except ValueError:
                    pass
            else:
                result[key] = value

    if "estimated_f1" in result:
        result["overall"] = result["estimated_f1"]
        result["completeness"] = result.get("estimated_critical_recall", 0)
        result["accuracy"] = result.get("estimated_structure", 0)
        result["scientific"] = result.get("estimated_f1", 0)
        result["common_issues"] = f"confidence: {result.get('confidence', 'unknown')}"
        result["summary"] = result.get("reasoning", "")
        return result
    return None


def parse_batched_qualitative_response(content: str) -> dict | None:
    """Parse batched qualitative response."""
    result = {}
    patterns = {
        "consistent_errors": r"CONSISTENT_ERRORS:\s*(.+)",
        "inconsistent_errors": r"INCONSISTENT_ERRORS:\s*(.+)",
        "missing_gates": r"MISSING_GATES:\s*(.+)",
        "extra_gates": r"EXTRA_GATES:\s*(.+)",
        "variance": r"VARIANCE:\s*(\w+)",
        "accept_for_analysis": r"ACCEPT_FOR_ANALYSIS:\s*(.+)",
        "summary": r"SUMMARY:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    if "accept_for_analysis" in result:
        accept = result["accept_for_analysis"].lower().startswith("yes")
        result["overall"] = 1.0 if accept else 0.0
        result["completeness"] = 1.0 if result.get("missing_gates", "").lower() == "none" else 0.5
        result["accuracy"] = 1.0 if result.get("consistent_errors", "").lower() == "none" else 0.0
        result["scientific"] = 1.0 if result.get("extra_gates", "").lower() == "none" else 0.5
        result["common_issues"] = result.get("consistent_errors", "none")
        return result
    return None


def parse_batched_orthogonal_response(content: str) -> dict | None:
    """Parse batched orthogonal response."""
    result = {}
    patterns = {
        "clinical_utility": r"CLINICAL_UTILITY:\s*(\d+)",
        "biological_plausibility": r"BIOLOGICAL_PLAUSIBILITY:\s*(\d+)",
        "hallucination_severity": r"HALLUCINATION_SEVERITY:\s*(\d+)",
        "marker_logic": r"MARKER_LOGIC:\s*(\d+)",
        "variance": r"VARIANCE:\s*(\w+)",
        "summary": r"SUMMARY:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key not in ["variance", "summary"]:
                try:
                    result[key] = int(value) / 10.0
                except ValueError:
                    pass
            else:
                result[key] = value

    if "clinical_utility" in result:
        halluc = result.get("hallucination_severity", 0)
        result["overall"] = result.get("clinical_utility", 0)
        result["completeness"] = result.get("marker_logic", 0)
        result["accuracy"] = result.get("biological_plausibility", 0)
        result["scientific"] = 1.0 - halluc
        result["common_issues"] = f"hallucination_severity: {halluc:.1f}"
        return result
    return None


def parse_batched_binary_response(content: str) -> dict | None:
    """Parse batched binary response."""
    result = {}
    patterns = {
        "acceptable_runs": r"ACCEPTABLE_RUNS:\s*(.+)",
        "critical_errors": r"CRITICAL_ERRORS:\s*(.+)",
        "missing_gates": r"MISSING_GATES:\s*(.+)",
        "extra_gates": r"EXTRA_GATES:\s*(.+)",
        "variance": r"VARIANCE:\s*(\w+)",
        "recommendation": r"RECOMMENDATION:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    if "acceptable_runs" in result:
        # Parse "7/10" or "7 out of 10" format
        accept_match = re.search(r"(\d+)\s*/\s*(\d+)", result["acceptable_runs"])
        if accept_match:
            accepted = int(accept_match.group(1))
            total = int(accept_match.group(2))
            result["overall"] = accepted / total if total > 0 else 0.0
        else:
            result["overall"] = 0.5

        result["completeness"] = 1.0 if result.get("missing_gates", "").lower() == "none" else 0.0
        result["accuracy"] = 1.0 if result.get("critical_errors", "").lower() == "none" else 0.0
        result["scientific"] = 1.0 if result.get("extra_gates", "").lower() == "none" else 0.0
        result["common_issues"] = result.get("critical_errors", "none")
        result["summary"] = result.get("recommendation", "")
        return result
    return None


BATCHED_RESPONSE_PARSERS = {
    "default": parse_batched_default_response,
    "validation": parse_batched_validation_response,
    "qualitative": parse_batched_qualitative_response,
    "orthogonal": parse_batched_orthogonal_response,
    "binary": parse_batched_binary_response,
}


def group_by_test_case_condition(
    scoring_results: list[ScoringResult],
) -> dict[tuple[str, str, str], list[ScoringResult]]:
    """Group scoring results by (test_case_id, model, condition)."""
    groups = defaultdict(list)
    for result in scoring_results:
        key = (result.test_case_id, result.model, result.condition)
        groups[key].append(result)
    return groups


def run_batched_judge(
    scoring_results: list[ScoringResult],
    ground_truth_dir: Path,
    judge_model: str = "claude-opus-cli",
    judge_styles: list[str] | None = None,
    condition_filter: str | None = None,
    dry_run: bool = False,
) -> list[BatchedJudgeResult]:
    """Run batched judge evaluation."""

    judge_styles = judge_styles or JUDGE_STYLES

    # Load ground truth
    test_cases = load_all_test_cases(ground_truth_dir)
    gt_map = {tc.test_case_id: tc for tc in test_cases}

    # Group by (test_case, model, condition)
    groups = group_by_test_case_condition(scoring_results)

    # Filter by condition if specified
    if condition_filter:
        groups = {
            k: v for k, v in groups.items()
            if condition_filter in k[2]  # k[2] is condition string
        }

    print(f"Found {len(groups)} unique (test_case, model, condition) groups")
    print(f"Judge styles: {judge_styles}")
    print(f"Total calls: {len(groups) * len(judge_styles)}")

    if dry_run:
        print("[DRY RUN] Would make API calls but skipping...")
        return []

    # Create client
    client = create_client(judge_model)

    results = []
    total = len(groups) * len(judge_styles)
    current = 0

    for (test_case_id, model, condition), predictions in groups.items():
        # Get ground truth
        tc = gt_map.get(test_case_id)
        if not tc:
            print(f"  Warning: No ground truth for {test_case_id}")
            continue

        gt_dict = {
            "gating_hierarchy": tc.gating_hierarchy.model_dump() if tc.gating_hierarchy else {},
            "context": tc.context.model_dump() if tc.context else {},
        }

        for style in judge_styles:
            current += 1
            pct = (current / total) * 100
            print(f"  [{current}/{total}] ({pct:.0f}%) {test_case_id} | {style}")

            # Build prompt
            prompt_builder = BATCHED_PROMPT_BUILDERS[style]
            prompt = prompt_builder(test_case_id, predictions, gt_dict)

            try:
                response = client.call(prompt, max_tokens=4096, temperature=0.0)

                # Parse response
                parser = BATCHED_RESPONSE_PARSERS[style]
                parsed = parser(response.content)

                if parsed:
                    result = BatchedJudgeResult(
                        test_case_id=test_case_id,
                        model=model,
                        condition=condition,
                        judge_style=style,
                        judge_model=judge_model,
                        completeness=parsed.get("completeness", 0),
                        accuracy=parsed.get("accuracy", 0),
                        scientific=parsed.get("scientific", 0),
                        overall=parsed.get("overall", 0),
                        variance_summary=parsed.get("variance", "unknown"),
                        common_issues=parsed.get("common_issues", ""),
                        summary=parsed.get("summary", ""),
                        n_bootstrap=len(predictions),
                        timestamp=datetime.now(),
                        tokens_used=response.tokens_used,
                        judge_prompt=prompt,
                        judge_raw_response=response.content,
                    )
                else:
                    result = BatchedJudgeResult(
                        test_case_id=test_case_id,
                        model=model,
                        condition=condition,
                        judge_style=style,
                        judge_model=judge_model,
                        completeness=0,
                        accuracy=0,
                        scientific=0,
                        overall=0,
                        variance_summary="unknown",
                        common_issues="",
                        summary="",
                        n_bootstrap=len(predictions),
                        timestamp=datetime.now(),
                        tokens_used=response.tokens_used,
                        error=f"Parse failed: {response.content[:100]}",
                        judge_prompt=prompt,
                        judge_raw_response=response.content,
                    )
            except Exception as e:
                result = BatchedJudgeResult(
                    test_case_id=test_case_id,
                    model=model,
                    condition=condition,
                    judge_style=style,
                    judge_model=judge_model,
                    completeness=0,
                    accuracy=0,
                    scientific=0,
                    overall=0,
                    variance_summary="unknown",
                    common_issues="",
                    summary="",
                    n_bootstrap=len(predictions),
                    timestamp=datetime.now(),
                    tokens_used=0,
                    error=str(e),
                    judge_prompt=prompt if 'prompt' in dir() else "",
                )

            results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run batched multi-judge evaluation")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results" / "rag_test",
        help="Input directory with scoring_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: input dir)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="claude-opus-cli",
        help="Judge model (default: claude-opus-cli)",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        choices=JUDGE_STYLES,
        default=JUDGE_STYLES,
        help="Judge styles to run",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Filter to specific condition (substring match)",
    )
    parser.add_argument(
        "--all-conditions",
        action="store_true",
        help="Run on all conditions (no filter)",
    )
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=PROJECT_ROOT / "data" / "staging",
        help="Ground truth directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't make API calls, just show plan",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip cost confirmation",
    )

    args = parser.parse_args()

    # Load scoring results
    scoring_file = args.input / "scoring_results.json"
    if not scoring_file.exists():
        print(f"ERROR: No scoring results found: {scoring_file}")
        sys.exit(1)

    with open(scoring_file) as f:
        data = json.load(f)
        scoring_results = [ScoringResult.from_dict(r) for r in data["results"]]

    print(f"Loaded {len(scoring_results)} scoring results")

    # Determine condition filter
    condition_filter = args.condition
    if not args.all_conditions and not condition_filter:
        # Default to best condition
        condition_filter = "standard_direct_oracle"
        print(f"Defaulting to best condition: {condition_filter}")

    # Run judge
    results = run_batched_judge(
        scoring_results=scoring_results,
        ground_truth_dir=args.test_cases,
        judge_model=args.judge_model,
        judge_styles=args.styles,
        condition_filter=condition_filter,
        dry_run=args.dry_run,
    )

    if results:
        # Compute summary stats
        by_style = defaultdict(list)
        for r in results:
            by_style[r.judge_style].append(r)

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        for style, style_results in by_style.items():
            valid = [r for r in style_results if not r.error]
            if valid:
                avg_overall = sum(r.overall for r in valid) / len(valid)
                print(f"\n{style}:")
                print(f"  Overall: {avg_overall:.3f} (n={len(valid)})")
                if style_results[0].variance_summary:
                    variances = [r.variance_summary for r in valid]
                    print(f"  Variance: {', '.join(set(variances))}")

        # Save results
        output_dir = args.output or args.input
        output_file = output_dir / "batched_judge_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "results": [r.to_dict() for r in results],
                "config": {
                    "judge_model": args.judge_model,
                    "styles": args.styles,
                    "condition_filter": condition_filter,
                },
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

        print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()

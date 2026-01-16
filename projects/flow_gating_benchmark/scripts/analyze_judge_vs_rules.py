#!/usr/bin/env python3
"""
Compare LLM judge semantic evaluation vs rule-based F1 scoring.

Goals:
1. Correlate judge quality scores with rule-based match rates
2. Find cases where they disagree (judge says good but F1 low, or vice versa)
3. Extract patterns from judge feedback to improve rule-based matching
4. Identify failure modes that rules can't catch
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ComparisonResult:
    """Result of comparing judge vs rule-based scoring."""
    test_case_id: str
    model: str
    condition: str

    # Judge scores
    judge_quality: float | None
    judge_failure_modes: list[str]
    judge_summary: str

    # Rule-based scores
    rule_match_rate: float | None
    rule_truly_missing: int
    rule_matched: int

    # Agreement
    agreement: str  # "agree_good", "agree_bad", "judge_better", "rules_better"


def load_judge_results(results_dir: Path) -> dict[str, dict]:
    """Load judge results keyed by (test_case_id, model, condition)."""
    judge_results = {}

    judge_file = results_dir / "multijudge" / "aggregated_judge_default.json"
    if not judge_file.exists():
        return judge_results

    with open(judge_file) as f:
        data = json.load(f)

    for result in data["results"]:
        key = (result["test_case_id"], result["model"], result.get("condition", ""))
        judge_results[key] = {
            "quality": result.get("median_quality"),
            "failure_modes": result.get("failure_modes", ""),
            "summary": result.get("summary", ""),
            "consistency": result.get("consistency"),
        }

    return judge_results


def load_rule_based_results(results_dir: Path) -> dict[str, dict]:
    """Load rule-based analysis results."""
    rule_file = results_dir / "improved_analysis_results.json"
    if not rule_file.exists():
        return {}

    with open(rule_file) as f:
        return json.load(f)


def parse_failure_modes(modes_str: str) -> list[str]:
    """Parse comma-separated failure modes."""
    if not modes_str:
        return []
    return [m.strip().lower() for m in modes_str.split(",") if m.strip()]


def main():
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results" / "full_benchmark_20260114"

    print("Loading judge results...")
    judge_results = load_judge_results(results_dir)
    print(f"Loaded {len(judge_results)} judge results")

    print("Loading rule-based results...")
    rule_results = load_rule_based_results(results_dir)

    # Get model-level stats from rule results
    rule_by_model = rule_results.get("by_model", {})

    print("\n" + "=" * 80)
    print("JUDGE vs RULE-BASED COMPARISON")
    print("=" * 80)

    # Aggregate by model
    print("\n--- Quality Comparison by Model ---")
    print(f"{'Model':<20} {'Judge Quality':>15} {'Rule Match Rate':>18}")
    print("-" * 55)

    judge_by_model = defaultdict(list)
    for (tc_id, model, cond), jdata in judge_results.items():
        if jdata["quality"] is not None:
            judge_by_model[model].append(jdata["quality"])

    for model in sorted(set(judge_by_model.keys()) | set(rule_by_model.keys())):
        judge_q = judge_by_model.get(model, [])
        judge_avg = sum(judge_q) / len(judge_q) if judge_q else None

        rule_data = rule_by_model.get(model, {})
        matched = rule_data.get("matched", 0)
        missing = rule_data.get("truly_missing", 0)
        total = matched + missing
        rule_rate = matched / total if total > 0 else None

        judge_str = f"{judge_avg:.3f}" if judge_avg is not None else "N/A"
        rule_str = f"{rule_rate:.3f}" if rule_rate is not None else "N/A"
        print(f"{model:<20} {judge_str:>15} {rule_str:>18}")

    # Analyze failure modes
    print("\n--- Failure Mode Analysis ---")
    all_failure_modes = []
    failure_modes_by_quality = {"high": [], "medium": [], "low": []}

    for (tc_id, model, cond), jdata in judge_results.items():
        modes = parse_failure_modes(jdata["failure_modes"])
        all_failure_modes.extend(modes)

        q = jdata["quality"]
        if q is not None:
            if q >= 0.7:
                failure_modes_by_quality["high"].extend(modes)
            elif q >= 0.4:
                failure_modes_by_quality["medium"].extend(modes)
            else:
                failure_modes_by_quality["low"].extend(modes)

    print("\nTop failure modes overall:")
    for mode, count in Counter(all_failure_modes).most_common(15):
        print(f"  {count:4d}x {mode}")

    print("\nFailure modes in LOW quality predictions (judge < 0.4):")
    for mode, count in Counter(failure_modes_by_quality["low"]).most_common(10):
        print(f"  {count:4d}x {mode}")

    print("\nFailure modes in HIGH quality predictions (judge >= 0.7):")
    high_modes = Counter(failure_modes_by_quality["high"]).most_common(10)
    if high_modes:
        for mode, count in high_modes:
            print(f"  {count:4d}x {mode}")
    else:
        print("  (none - high quality predictions have no failure modes)")

    # Identify semantic equivalences from judge summaries
    print("\n--- Semantic Patterns from Judge Summaries ---")

    # Look for patterns like "model predicted X instead of Y"
    naming_issues = []
    for (tc_id, model, cond), jdata in judge_results.items():
        summary = jdata.get("summary", "")

        # Look for "instead of" patterns
        instead_matches = re.findall(
            r'(?:predicted|generated|identified)\s+["\']?([^"\']+)["\']?\s+instead\s+of\s+["\']?([^"\']+)["\']?',
            summary, re.IGNORECASE
        )
        naming_issues.extend(instead_matches)

        # Look for "alternative naming" mentions
        if "naming" in summary.lower() or "alternative" in summary.lower():
            # Extract context
            for sent in summary.split("."):
                if "naming" in sent.lower() or "alternative" in sent.lower():
                    naming_issues.append(("pattern", sent.strip()[:100]))

    if naming_issues:
        print("\nNaming/semantic equivalence patterns found:")
        for issue in naming_issues[:20]:
            print(f"  • {issue}")

    # Cases where judge and rules strongly disagree
    print("\n--- Disagreement Analysis ---")

    # We need per-prediction rule scores for this
    # For now, aggregate by test case and model

    disagreements = {
        "judge_good_rules_bad": [],  # Judge >= 0.7, but rules caught many missing
        "judge_bad_rules_good": [],  # Judge < 0.4, but rules say high match
    }

    # Group judge results by test_case + model
    judge_by_tc_model = defaultdict(list)
    for (tc_id, model, cond), jdata in judge_results.items():
        judge_by_tc_model[(tc_id, model)].append(jdata)

    for (tc_id, model), jdata_list in judge_by_tc_model.items():
        avg_quality = sum(j["quality"] for j in jdata_list if j["quality"]) / len(jdata_list) if jdata_list else 0

        # Get a sample summary
        sample_summary = jdata_list[0].get("summary", "")[:200] if jdata_list else ""
        sample_modes = jdata_list[0].get("failure_modes", "") if jdata_list else ""

        # Check rule-based model stats
        rule_data = rule_by_model.get(model, {})
        if rule_data:
            matched = rule_data.get("matched", 0)
            total = matched + rule_data.get("truly_missing", 0)
            rule_rate = matched / total if total > 0 else 0

            if avg_quality >= 0.7 and rule_rate < 0.4:
                disagreements["judge_good_rules_bad"].append({
                    "tc": tc_id, "model": model,
                    "judge_q": avg_quality, "rule_rate": rule_rate,
                    "modes": sample_modes, "summary": sample_summary
                })
            elif avg_quality < 0.4 and rule_rate >= 0.6:
                disagreements["judge_bad_rules_good"].append({
                    "tc": tc_id, "model": model,
                    "judge_q": avg_quality, "rule_rate": rule_rate,
                    "modes": sample_modes, "summary": sample_summary
                })

    print(f"\nJudge says GOOD but rules say BAD: {len(disagreements['judge_good_rules_bad'])} cases")
    for case in disagreements["judge_good_rules_bad"][:5]:
        print(f"  • {case['tc']}/{case['model']}: judge={case['judge_q']:.2f}, rules={case['rule_rate']:.2f}")
        print(f"    Modes: {case['modes'][:60]}")

    print(f"\nJudge says BAD but rules say GOOD: {len(disagreements['judge_bad_rules_good'])} cases")
    for case in disagreements["judge_bad_rules_good"][:5]:
        print(f"  • {case['tc']}/{case['model']}: judge={case['judge_q']:.2f}, rules={case['rule_rate']:.2f}")
        print(f"    Modes: {case['modes'][:60]}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR IMPROVING F1 SCORING")
    print("=" * 80)

    print("""
Based on judge feedback analysis:

1. SEMANTIC EQUIVALENCES TO ADD:
   - Judge catches "oversimplification" (456x) - models predict correct types
     but fewer gates. Consider hierarchical matching where parent gates
     can substitute for missing intermediate gates.

2. FAILURE MODES RULES CAN'T CATCH:
   - "panel misidentification" (82x) - Model predicts wrong panel type entirely
   - "incorrect analysis type" (80x) - Wrong experimental goal
   - These require semantic understanding of the panel context.

3. FAILURE MODES RULES SHOULD CATCH:
   - "missing intermediate gates" (118x) - Structural issue, rules can detect
   - "incorrect gate order" (116x) - Rules can validate gating sequence

4. NAMING VARIATIONS TO HANDLE:
   - Review judge summaries mentioning "alternative naming" or "instead of"
   - Build equivalence dictionary from these patterns

5. CONSIDER HYBRID APPROACH:
   - Use rules for structural validation (gate order, hierarchy)
   - Use LLM judge for semantic validation (correct cell types, panel match)
   - Combine scores for final evaluation
""")

    # Save detailed analysis
    output = {
        "failure_mode_counts": dict(Counter(all_failure_modes)),
        "failure_modes_by_quality": {
            k: dict(Counter(v)) for k, v in failure_modes_by_quality.items()
        },
        "disagreements": disagreements,
        "recommendations": [
            "Add hierarchical matching for oversimplification cases",
            "Rules can't catch panel misidentification - need LLM",
            "Add gate order validation to rules",
            "Build equivalence dictionary from judge naming feedback",
        ]
    }

    output_file = results_dir / "judge_vs_rules_analysis.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed analysis saved to: {output_file}")


if __name__ == "__main__":
    main()

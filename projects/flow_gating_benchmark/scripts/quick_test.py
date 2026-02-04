#!/usr/bin/env python3
"""
Quick test for FlowBench - runs a single prediction without full pipeline.

Usage:
    # With Gemini (cheapest, ~$0.01):
    GOOGLE_API_KEY=xxx python scripts/quick_test.py

    # With Claude Code CLI (uses your subscription):
    python scripts/quick_test.py --cli

    # Dry run (no API calls):
    python scripts/quick_test.py --dry-run

    # Alien Cell v1 (INVALID test - see docs):
    python scripts/quick_test.py --alien

    # Alien Cell v2 (VALID instruction-following test):
    python scripts/quick_test.py --alien-v2 --cli --model sonnet

    # Alien Cell v3 (biological reasoning test):
    python scripts/quick_test.py --alien-v3 zorphax --cli --model sonnet
    python scripts/quick_test.py --alien-v3 blixtar --cli --model sonnet
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from curation.schemas import TestCase
from evaluation.scorer import GatingScorer
from experiments.prompts import build_prompt

def get_v3_test_cases() -> dict:
    """Dynamically discover V3 test cases."""
    v3_dir = Path(__file__).parent.parent / "data" / "alien_cell_v3"
    cases = {}
    for f in v3_dir.glob("*.json"):
        if f.name.startswith("_"):
            continue
        # Extract short name from filename like "omip_008_v3_zorphax_cells.json"
        parts = f.stem.split("_v3_")
        if len(parts) == 2:
            short_name = parts[1].replace("_", "-")
            cases[short_name] = f.name
    return cases


def load_smallest_test_case() -> TestCase:
    """Load the simplest test case (OMIP-008 has fewest gates)."""
    data_dir = Path(__file__).parent.parent / "data" / "verified"

    # OMIP-008 is one of the simpler panels
    test_file = data_dir / "omip_008.json"
    if not test_file.exists():
        # Fall back to first available
        test_file = next(data_dir.glob("*.json"))

    with open(test_file) as f:
        data = json.load(f)

    return TestCase.model_validate(data)


def call_gemini(prompt: str) -> str:
    """Call Gemini API."""
    import os

    import google.generativeai as genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


def call_claude_cli(prompt: str, model: str = "haiku") -> str:
    """Call Claude via CLI (uses your Claude subscription).

    Args:
        prompt: The prompt to send
        model: Model alias - "haiku", "sonnet", or "opus"
    """
    import subprocess

    result = subprocess.run(
        ["claude", "-p", prompt, "--model", model],
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI error: {result.stderr}")

    return result.stdout


def evaluate_v3_response(response: str, novel_pop: dict) -> dict:
    """Evaluate V3 response for biological reasoning quality."""
    response_lower = response.lower()

    # Check for reasoning indicators
    flags_concern = any(word in response_lower for word in [
        "unusual", "rare", "unlikely", "impossible", "implausible",
        "concern", "note", "however", "typically", "normally",
        "caution", "unexpected", "atypical", "questionable"
    ])

    mentions_lineage = any(term in response_lower for term in [
        "lineage", "monocyte", "t cell", "b cell", "mutually exclusive",
        "different cell type", "cannot be both"
    ])

    mentions_thymus = "thymus" in response_lower or "thymocyte" in response_lower or "thymic" in response_lower
    mentions_development = "development" in response_lower or "immature" in response_lower
    mentions_pathology = any(term in response_lower for term in [
        "patholog", "disease", "leukemia", "cancer", "abnormal"
    ])

    # Scoring
    reasoning_score = 0
    reasoning_notes = []

    if flags_concern:
        reasoning_score += 1
        reasoning_notes.append("Flagged biological concern")

    if mentions_lineage:
        reasoning_score += 1
        reasoning_notes.append("Discussed lineage biology")

    if mentions_thymus or mentions_development:
        reasoning_score += 1
        reasoning_notes.append("Mentioned developmental context (thymus)")

    if mentions_pathology:
        reasoning_score += 1
        reasoning_notes.append("Mentioned pathological contexts")

    # Determine grade
    if reasoning_score == 0:
        grade = "POOR - Blind acceptance"
    elif reasoning_score == 1:
        grade = "FAIR - Partial awareness"
    elif reasoning_score == 2:
        grade = "GOOD - Shows reasoning"
    else:
        grade = "EXCELLENT - Deep understanding"

    return {
        "reasoning_score": reasoning_score,
        "max_score": 4,
        "grade": grade,
        "flags_concern": flags_concern,
        "mentions_lineage": mentions_lineage,
        "mentions_thymus": mentions_thymus,
        "mentions_pathology": mentions_pathology,
        "reasoning_notes": reasoning_notes,
        "expected_behavior": novel_pop.get("expected_behavior", ""),
        "biological_status": novel_pop.get("biological_status", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Quick FlowBench test")
    parser.add_argument("--cli", action="store_true", help="Use Claude CLI instead of Gemini")
    parser.add_argument("--model", choices=["haiku", "sonnet", "opus"], default="haiku",
                        help="Claude model to use with --cli (default: haiku)")
    parser.add_argument("--dry-run", action="store_true", help="Show prompt without calling API")

    # Alien cell test versions
    alien_group = parser.add_mutually_exclusive_group()
    alien_group.add_argument("--alien", action="store_true",
                             help="Use alien cell v1 (INVALID - only changes ground truth)")
    alien_group.add_argument("--alien-v2", action="store_true",
                             help="Use alien cell v2 (VALID - provides naming convention in prompt)")
    v3_cases = get_v3_test_cases()
    alien_group.add_argument("--alien-v3", type=str, metavar="NAME",
                             help=f"Alien cell v3 biological reasoning test. Use --list-v3 to see options"
                             if not v3_cases else f"V3 test name (e.g., {list(v3_cases.keys())[0]})")
    parser.add_argument("--list-v3", action="store_true", help="List available V3 test cases")

    args = parser.parse_args()

    # Handle --list-v3
    if args.list_v3:
        print("=== Available V3 Test Cases ===\n")
        for name, filename in sorted(v3_cases.items()):
            # Load to get details
            v3_path = Path(__file__).parent.parent / "data" / "alien_cell_v3" / filename
            with open(v3_path) as f:
                data = json.load(f)
            meta = data.get("alien_cell_v3_metadata", {})
            pop = meta.get("novel_population", {})
            status_icon = {"implausible": "❌", "rare_contextual": "⚠️", "rare_valid": "🔸", "valid": "✓"}.get(pop.get("biological_status", ""), "?")
            print(f"  {status_icon} {name}: {pop.get('marker_logic', '?')} ({pop.get('biological_status', '?')})")
        print(f"\nUsage: python scripts/quick_test.py --alien-v3 <name> --cli --model sonnet")
        return

    print("=== FlowBench Quick Test ===\n")

    naming_convention = None  # For v2 tests
    custom_prompt = None  # For v3 tests
    v3_metadata = None  # For v3 evaluation

    # Load test case
    if args.alien:
        print("⚠️  WARNING: Alien v1 test is INVALID (model can't know expected names)")
        print("   Use --alien-v2 for valid instruction-following test\n")
        data_dir = Path(__file__).parent.parent / "data" / "alien_cell"
        test_file = data_dir / "omip_008_alien.json"
        with open(test_file) as f:
            data = json.load(f)
        test_case = TestCase.model_validate(data)

    elif args.alien_v2:
        print("🧪 Alien Cell V2: Testing instruction following vs memorization")
        print("   Model is given naming convention - must use alien names\n")
        data_dir = Path(__file__).parent.parent / "data" / "alien_cell_v2"
        test_file = data_dir / "omip_008_alien_v2.json"
        with open(test_file) as f:
            data = json.load(f)

        # Extract naming convention from metadata
        if "alien_cell_v2_metadata" in data:
            naming_convention = data["alien_cell_v2_metadata"].get("naming_convention_prompt")

        test_case = TestCase.model_validate(data)

    elif args.alien_v3:
        print("🧬 Alien Cell V3: Testing BIOLOGICAL REASONING")
        print("   Model asked to insert novel population - should reason about plausibility\n")

        if args.alien_v3 not in v3_cases:
            print(f"❌ Unknown V3 test: {args.alien_v3}")
            print(f"   Available: {', '.join(sorted(v3_cases.keys())[:10])}...")
            print("   Use --list-v3 to see all options")
            return

        data_dir = Path(__file__).parent.parent / "data" / "alien_cell_v3"
        test_file = data_dir / v3_cases[args.alien_v3]

        with open(test_file) as f:
            data = json.load(f)

        v3_metadata = data.get("alien_cell_v3_metadata", {})
        novel_pop = v3_metadata.get("novel_population", {})

        print(f"   Novel population: {novel_pop.get('name', 'Unknown')}")
        print(f"   Marker logic: {novel_pop.get('marker_logic', 'Unknown')}")
        print(f"   Biological status: {novel_pop.get('biological_status', 'Unknown')}")
        print(f"   Expected: {novel_pop.get('expected_behavior', 'Unknown')}\n")

        # Use custom prompt from metadata
        custom_prompt = v3_metadata.get("prompt")
        test_case = TestCase.model_validate(data)

    else:
        test_case = load_smallest_test_case()

    print(f"📋 Test case: {test_case.test_case_id}")
    print(f"   Species: {test_case.context.species}")
    print(f"   Panel size: {len(test_case.panel.entries)} markers")

    # Build prompt
    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = build_prompt(
            test_case,
            template_name="direct",
            context_level="minimal",
            naming_convention=naming_convention,
        )

    print(f"   Prompt length: {len(prompt)} chars")

    if naming_convention:
        print(f"   📝 Naming convention injected ({len(naming_convention)} chars)")

    print()

    if args.dry_run:
        print("=== Prompt (dry run) ===")
        print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
        print("\n[Dry run - no API call made]")
        return

    # Get prediction
    print("🤖 Calling model...")
    if args.cli:
        response = call_claude_cli(prompt, model=args.model)
        model_name = f"claude-{args.model}-cli"
    else:
        response = call_gemini(prompt)
        model_name = "gemini-2.0-flash"

    print(f"   Response length: {len(response)} chars\n")

    # For V3, evaluate biological reasoning
    if args.alien_v3 and v3_metadata:
        print("=== Biological Reasoning Evaluation ===")
        novel_pop = v3_metadata.get("novel_population", {})
        eval_result = evaluate_v3_response(response, novel_pop)

        print(f"Grade: {eval_result['grade']}")
        print(f"Reasoning score: {eval_result['reasoning_score']}/{eval_result['max_score']}")

        if eval_result['reasoning_notes']:
            print(f"Evidence: {', '.join(eval_result['reasoning_notes'])}")

        print(f"\nExpected: {eval_result['expected_behavior']}")

        print("\n=== Full Response ===")
        print(response[:3000] + "..." if len(response) > 3000 else response)

        print("\n✓ V3 reasoning test complete!")
        return

    # Standard scoring for v1/v2/baseline
    print("📊 Scoring...")
    scorer = GatingScorer()
    result = scorer.score(
        response=response,
        test_case=test_case,
        model=model_name,
        condition="quick_test",
    )

    print("\n=== Results ===")
    print(f"Parse success: {result.parse_success}")

    if result.evaluation:
        eval_result = result.evaluation
        print(f"Hierarchy F1:    {eval_result.hierarchy_f1:.3f}")
        print(f"Structure:       {eval_result.structure_accuracy:.3f}")
        print(f"Critical recall: {eval_result.critical_gate_recall:.3f}")
        print(f"Hallucinations:  {eval_result.hallucination_rate:.3f}")
        print(f"Depth accuracy:  {eval_result.depth_accuracy:.3f}")

        if eval_result.matching_gates:
            print(f"\n✓ Matched gates: {', '.join(eval_result.matching_gates[:5])}")
        if eval_result.missing_gates:
            print(f"✗ Missing gates: {', '.join(eval_result.missing_gates[:5])}")
        if eval_result.extra_gates:
            print(f"+ Extra gates: {', '.join(eval_result.extra_gates[:5])}")

    if result.parsed_hierarchy:
        print("\n=== Predicted Hierarchy ===")
        print(json.dumps(result.parsed_hierarchy, indent=2)[:1500])

    print("\n✓ Quick test complete!")


if __name__ == "__main__":
    main()

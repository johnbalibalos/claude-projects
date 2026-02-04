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
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from curation.schemas import TestCase
from experiments.prompts import build_prompt
from evaluation.scorer import GatingScorer


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
    import google.generativeai as genai
    import os

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


def call_claude_cli(prompt: str) -> str:
    """Call Claude via CLI (uses your Claude subscription)."""
    import subprocess

    result = subprocess.run(
        ["claude", "-p", prompt, "--no-stream"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI error: {result.stderr}")

    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Quick FlowBench test")
    parser.add_argument("--cli", action="store_true", help="Use Claude CLI instead of Gemini")
    parser.add_argument("--dry-run", action="store_true", help="Show prompt without calling API")
    parser.add_argument("--alien", action="store_true", help="Use alien cell version")
    args = parser.parse_args()

    print("=== FlowBench Quick Test ===\n")

    # Load test case
    if args.alien:
        data_dir = Path(__file__).parent.parent / "data" / "alien_cell"
        test_file = data_dir / "omip_008_alien.json"
        with open(test_file) as f:
            data = json.load(f)
        test_case = TestCase.model_validate(data)
    else:
        test_case = load_smallest_test_case()

    print(f"📋 Test case: {test_case.test_case_id}")
    print(f"   Species: {test_case.context.species}")
    print(f"   Panel size: {len(test_case.panel.entries)} markers")

    # Build prompt
    prompt = build_prompt(
        test_case,
        template_name="direct",
        context_level="minimal",
    )

    print(f"   Prompt length: {len(prompt)} chars\n")

    if args.dry_run:
        print("=== Prompt (dry run) ===")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("\n[Dry run - no API call made]")
        return

    # Get prediction
    print("🤖 Calling model...")
    if args.cli:
        response = call_claude_cli(prompt)
        model_name = "claude-cli"
    else:
        response = call_gemini(prompt)
        model_name = "gemini-2.0-flash"

    print(f"   Response length: {len(response)} chars\n")

    # Score
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
    print(f"Hierarchy F1:  {result.hierarchy_f1:.3f}")
    print(f"Synonym F1:    {result.synonym_f1:.3f}")
    print(f"Structure:     {result.structure_accuracy:.3f}")
    print(f"Critical recall: {result.critical_gate_recall:.3f}")
    print(f"Hallucinations:  {result.hallucination_rate:.3f}")

    if result.parse_success:
        print("\n=== Predicted Hierarchy ===")
        print(json.dumps(result.parsed_hierarchy, indent=2)[:1000])

    print("\n✓ Quick test complete!")


if __name__ == "__main__":
    main()

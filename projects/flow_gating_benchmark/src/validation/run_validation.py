#!/usr/bin/env python3
"""
Phase 0 Validation Sprint Runner.

This script runs all validation checks to confirm project feasibility:
1. Check flowkit availability and .wsp parsing capability
2. Explore FlowRepository datasets for available .wsp files
3. (Optional) Run a manual LLM test

Usage:
    python -m src.validation.run_validation [--with-llm-test]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .flowkit_validator import check_flowkit_available, validate_wsp_parsing
from .flowrepository_explorer import explore_known_omip_datasets
from .manual_llm_test import run_manual_test


def run_phase0_validation(
    wsp_path: str | None = None,
    run_llm_test: bool = False,
    output_dir: str = "data",
) -> dict:
    """
    Run complete Phase 0 validation.

    Args:
        wsp_path: Optional path to a test .wsp file
        run_llm_test: Whether to run the manual LLM test
        output_dir: Directory to save results

    Returns:
        Dictionary with all validation results
    """
    results = {
        "phase": "0 - Validation Sprint",
        "checks": {},
        "decision": None,
    }

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check 1: flowkit availability
    print("\n" + "=" * 60)
    print("CHECK 1: FLOWKIT AVAILABILITY")
    print("=" * 60)

    flowkit_available = check_flowkit_available()
    results["checks"]["flowkit_available"] = flowkit_available

    if flowkit_available:
        print("✓ flowkit is installed and importable")
    else:
        print("✗ flowkit is NOT installed")
        print("  Run: pip install flowkit")

    # Check 2: .wsp parsing (if file provided)
    if wsp_path:
        print("\n" + "=" * 60)
        print("CHECK 2: WSP PARSING")
        print("=" * 60)

        wsp_result = validate_wsp_parsing(wsp_path)
        results["checks"]["wsp_parsing"] = wsp_result

        if wsp_result.get("success"):
            print(f"✓ Successfully parsed: {wsp_path}")
            print(f"  Samples found: {wsp_result.get('n_samples', 0)}")
            print(f"  Gates extracted: {wsp_result.get('n_gates', 0)}")
        else:
            print(f"✗ Failed to parse: {wsp_path}")
            print(f"  Error: {wsp_result.get('error')}")
    else:
        print("\n" + "=" * 60)
        print("CHECK 2: WSP PARSING (SKIPPED)")
        print("=" * 60)
        print("No .wsp file provided. Download one from FlowRepository to test.")
        print("Example:")
        print("  python -m src.validation.run_validation --wsp data/raw/test.wsp")
        results["checks"]["wsp_parsing"] = {"skipped": True}

    # Check 3: FlowRepository exploration
    print("\n" + "=" * 60)
    print("CHECK 3: FLOWREPOSITORY EXPLORATION")
    print("=" * 60)

    try:
        fr_results = explore_known_omip_datasets()
        results["checks"]["flowrepository"] = {
            "datasets_checked": len(fr_results),
            "with_wsp": sum(1 for r in fr_results if r.get("has_wsp")),
            "details": fr_results,
        }

        # Save detailed results
        fr_output = Path(output_dir) / "flowrepository_exploration.json"
        with open(fr_output, "w") as f:
            json.dump(fr_results, f, indent=2)
        print(f"\nDetailed results saved to: {fr_output}")

    except Exception as e:
        print(f"✗ FlowRepository exploration failed: {e}")
        results["checks"]["flowrepository"] = {"error": str(e)}

    # Check 4: Manual LLM test (optional)
    if run_llm_test:
        print("\n" + "=" * 60)
        print("CHECK 4: MANUAL LLM TEST")
        print("=" * 60)

        try:
            llm_result = run_manual_test(verbose=True)
            results["checks"]["llm_test"] = {
                "model": llm_result.get("model"),
                "parse_success": llm_result.get("parse_success"),
                "has_response": "response" in llm_result,
            }

            # Save detailed result
            llm_output = Path(output_dir) / "manual_llm_test.json"
            with open(llm_output, "w") as f:
                json.dump(llm_result, f, indent=2, default=str)
            print(f"\nDetailed result saved to: {llm_output}")

        except Exception as e:
            print(f"✗ LLM test failed: {e}")
            results["checks"]["llm_test"] = {"error": str(e)}
    else:
        results["checks"]["llm_test"] = {"skipped": True}

    # Decision gate
    print("\n" + "=" * 60)
    print("PHASE 0 DECISION GATE")
    print("=" * 60)

    proceed = True
    blockers = []

    if not flowkit_available:
        blockers.append("flowkit not installed")

    if wsp_path and not results["checks"].get("wsp_parsing", {}).get("success"):
        blockers.append("WSP parsing failed")

    fr_check = results["checks"].get("flowrepository", {})
    if fr_check.get("with_wsp", 0) == 0 and "error" not in fr_check:
        blockers.append("No FlowRepository datasets with .wsp files found")

    if blockers:
        proceed = False
        print("\n✗ BLOCKED - Cannot proceed to Phase 1")
        print("\nBlockers:")
        for b in blockers:
            print(f"  - {b}")
        print("\nRecommendation: Address blockers or use Plan B (OMIP papers only)")
        results["decision"] = "blocked"
        results["blockers"] = blockers
    else:
        print("\n✓ PROCEED to Phase 1 - Data Curation")
        print("\nNext steps:")
        print("  1. Download .wsp files from FlowRepository datasets")
        print("  2. Begin OMIP paper curation")
        print("  3. Cross-validate paper hierarchies with flowkit extraction")
        results["decision"] = "proceed"

    # Save final results
    output_path = Path(output_dir) / "phase0_validation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nValidation results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 0 validation sprint for the gating benchmark"
    )
    parser.add_argument(
        "--wsp",
        type=str,
        help="Path to a .wsp file to test parsing",
    )
    parser.add_argument(
        "--with-llm-test",
        action="store_true",
        help="Run the manual LLM test (requires API key)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save results (default: data)",
    )

    args = parser.parse_args()

    results = run_phase0_validation(
        wsp_path=args.wsp,
        run_llm_test=args.with_llm_test,
        output_dir=args.output_dir,
    )

    # Exit with appropriate code
    sys.exit(0 if results["decision"] == "proceed" else 1)


if __name__ == "__main__":
    main()

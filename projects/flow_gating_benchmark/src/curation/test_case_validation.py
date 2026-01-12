"""
Test case validation utilities.

Provides functions to identify incomplete or invalid test cases
that should be excluded from statistical reporting while still
allowing them to be run for exploratory analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from .schemas import TestCase
from .omip_extractor import load_test_case


class TestCaseStatus(NamedTuple):
    """Status of a test case's completeness."""

    test_case_id: str
    is_complete: bool
    has_panel: bool
    has_hierarchy: bool
    n_markers: int
    n_gates: int
    issues: list[str]


# Known incomplete test cases - can be updated as issues are fixed
KNOWN_INCOMPLETE_TEST_CASES = frozenset({
    "OMIP-064",  # Empty panel - 27-color NK/ILC/MAIT panel
    "OMIP-095",  # Empty panel - Spectral PBMC/Lymph
})


def is_test_case_complete(test_case: TestCase) -> bool:
    """
    Check if a test case has complete panel and hierarchy definitions.

    A complete test case must have:
    - At least one marker in the panel
    - At least one gate in the hierarchy (beyond root)

    Args:
        test_case: TestCase to validate

    Returns:
        True if test case is complete, False otherwise
    """
    # Check panel
    has_panel = len(test_case.panel.entries) > 0

    # Check hierarchy (must have at least root + one child)
    n_gates = len(test_case.gating_hierarchy.get_all_gates())
    has_hierarchy = n_gates > 1

    return has_panel and has_hierarchy


def validate_test_case(test_case: TestCase) -> TestCaseStatus:
    """
    Validate a test case and return detailed status.

    Args:
        test_case: TestCase to validate

    Returns:
        TestCaseStatus with validation details
    """
    issues: list[str] = []

    # Check panel
    n_markers = len(test_case.panel.entries)
    has_panel = n_markers > 0
    if not has_panel:
        issues.append("Empty panel definition - no markers specified")

    # Check hierarchy
    all_gates = test_case.gating_hierarchy.get_all_gates()
    n_gates = len(all_gates)
    has_hierarchy = n_gates > 1
    if not has_hierarchy:
        issues.append("Empty or minimal hierarchy - only root gate present")

    # Additional checks
    if has_panel and n_markers < 3:
        issues.append(f"Very few markers ({n_markers}) - may be incomplete")

    if has_hierarchy and n_gates < 5:
        issues.append(f"Very few gates ({n_gates}) - may be incomplete")

    return TestCaseStatus(
        test_case_id=test_case.test_case_id,
        is_complete=has_panel and has_hierarchy,
        has_panel=has_panel,
        has_hierarchy=has_hierarchy,
        n_markers=n_markers,
        n_gates=n_gates,
        issues=issues,
    )


def get_incomplete_test_case_ids(test_cases_dir: str | Path) -> set[str]:
    """
    Scan test cases directory and return IDs of incomplete test cases.

    Args:
        test_cases_dir: Directory containing test case JSON files

    Returns:
        Set of test case IDs that are incomplete
    """
    test_cases_dir = Path(test_cases_dir)
    incomplete_ids: set[str] = set()

    for path in test_cases_dir.glob("*.json"):
        try:
            test_case = load_test_case(path)
            if not is_test_case_complete(test_case):
                incomplete_ids.add(test_case.test_case_id)
        except Exception:
            # If we can't load it, consider it incomplete
            incomplete_ids.add(path.stem.upper().replace("_", "-"))

    return incomplete_ids


def filter_complete_results(
    results: list[dict],
    test_cases_dir: str | Path | None = None,
    incomplete_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Separate results into complete and incomplete test cases.

    Args:
        results: List of result dictionaries with test_case_id
        test_cases_dir: Directory to scan for incomplete cases (optional)
        incomplete_ids: Pre-computed set of incomplete IDs (optional)

    Returns:
        Tuple of (complete_results, incomplete_results)
    """
    if incomplete_ids is None:
        if test_cases_dir is not None:
            incomplete_ids = get_incomplete_test_case_ids(test_cases_dir)
        else:
            # Fall back to known incomplete list
            incomplete_ids = KNOWN_INCOMPLETE_TEST_CASES

    complete_results: list[dict] = []
    incomplete_results: list[dict] = []

    for result in results:
        tc_id = result.get("test_case_id", "")
        if tc_id in incomplete_ids:
            incomplete_results.append(result)
        else:
            complete_results.append(result)

    return complete_results, incomplete_results


def print_validation_report(test_cases_dir: str | Path) -> None:
    """
    Print a validation report for all test cases in a directory.

    Args:
        test_cases_dir: Directory containing test case JSON files
    """
    test_cases_dir = Path(test_cases_dir)

    print("=" * 70)
    print("TEST CASE VALIDATION REPORT")
    print("=" * 70)

    complete_count = 0
    incomplete_count = 0

    for path in sorted(test_cases_dir.glob("*.json")):
        try:
            test_case = load_test_case(path)
            status = validate_test_case(test_case)

            if status.is_complete:
                complete_count += 1
                print(f"\n✓ {status.test_case_id}")
                print(f"  Markers: {status.n_markers}, Gates: {status.n_gates}")
            else:
                incomplete_count += 1
                print(f"\n✗ {status.test_case_id} [INCOMPLETE]")
                print(f"  Markers: {status.n_markers}, Gates: {status.n_gates}")
                for issue in status.issues:
                    print(f"  - {issue}")

        except Exception as e:
            incomplete_count += 1
            print(f"\n✗ {path.stem} [ERROR]")
            print(f"  Failed to load: {e}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {complete_count} complete, {incomplete_count} incomplete")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print_validation_report(sys.argv[1])
    else:
        print("Usage: python test_case_validation.py <test_cases_dir>")

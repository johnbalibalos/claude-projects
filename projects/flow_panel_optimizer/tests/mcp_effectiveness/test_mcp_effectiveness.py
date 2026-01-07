"""
MCP Panel Designer Effectiveness Tests

This module tests whether Claude with MCP tools produces better
flow cytometry panels than Claude without tools.

Usage:
    pytest tests/mcp_effectiveness/test_mcp_effectiveness.py -v

Note: These tests require:
    1. ANTHROPIC_API_KEY environment variable
    2. MCP server running (for treatment condition)
"""

import json
import os
import re
from pathlib import Path
from typing import Optional
import yaml
import pytest
import numpy as np

from flow_panel_optimizer.spectral.similarity import (
    build_similarity_matrix,
    find_high_similarity_pairs,
)
from flow_panel_optimizer.spectral.complexity import complexity_index
from flow_panel_optimizer.validation.omip_validator import create_synthetic_test_spectra


# Test configuration
TEST_CASES_DIR = Path(__file__).parent / "test_cases"
PROMPTS_DIR = Path(__file__).parent / "prompts"
RESULTS_DIR = Path(__file__).parent / "results"

# Known fluorophores we can evaluate
KNOWN_FLUOROPHORES = set(create_synthetic_test_spectra().keys())


def load_test_case(name: str) -> dict:
    """Load a test case YAML file."""
    path = TEST_CASES_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompt_template(name: str) -> str:
    """Load a prompt template."""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text()


def format_marker_list(markers: list[dict]) -> str:
    """Format markers for prompt."""
    lines = []
    for m in markers:
        expr = m.get("expression", "medium")
        desc = m.get("description", "")
        intracellular = " (intracellular)" if m.get("intracellular") else ""
        lines.append(f"- **{m['name']}**: {expr} expression{intracellular} - {desc}")
    return "\n".join(lines)


def parse_panel_from_response(response: str) -> list[dict]:
    """Extract fluorophore assignments from Claude's response.

    Looks for markdown tables with Marker | Fluorophore columns.
    """
    assignments = []

    # Split into lines and look for table rows
    lines = response.split("\n")
    for line in lines:
        # Match table rows: | content | content | ...
        if not line.strip().startswith("|"):
            continue

        # Split by pipe and clean up
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]  # Remove empty strings

        if len(parts) < 2:
            continue

        marker, fluorophore = parts[0], parts[1]

        # Skip header rows and separator rows
        if marker.lower() in ("marker", "---", "-", ""):
            continue
        if fluorophore.lower() in ("fluorophore", "---", "-", ""):
            continue
        if set(marker) <= {"-", " "}:  # Separator row
            continue

        # Clean up fluorophore name
        fluorophore = fluorophore.strip()
        # Handle common variations
        if fluorophore.startswith("Alexa ") and "Fluor" not in fluorophore:
            fluorophore = fluorophore.replace("Alexa ", "Alexa Fluor ")

        assignments.append({
            "marker": marker.strip(),
            "fluorophore": fluorophore,
        })

    return assignments


def evaluate_panel(assignments: list[dict]) -> dict:
    """Calculate quality metrics for a panel.

    Returns:
        dict with:
        - complexity_index: Overall interference score
        - max_similarity: Highest pairwise similarity
        - critical_pairs: Count of pairs with SI > 0.95
        - high_risk_pairs: Count of pairs with SI > 0.90
        - unknown_fluorophores: List of fluorophores not in our database
        - valid: Whether panel could be evaluated
    """
    fluorophores = [a["fluorophore"] for a in assignments]

    # Check for unknown fluorophores
    unknown = [f for f in fluorophores if f not in KNOWN_FLUOROPHORES]

    # Get spectra for known fluorophores
    all_spectra = create_synthetic_test_spectra()
    spectra = {f: all_spectra[f] for f in fluorophores if f in KNOWN_FLUOROPHORES}

    if len(spectra) < 2:
        return {
            "complexity_index": None,
            "max_similarity": None,
            "critical_pairs": None,
            "high_risk_pairs": None,
            "unknown_fluorophores": unknown,
            "valid": False,
            "error": "Not enough known fluorophores to evaluate",
        }

    # Build similarity matrix
    names, sim_matrix = build_similarity_matrix(spectra)

    # Calculate metrics
    ci = complexity_index(sim_matrix)

    # Get max similarity (excluding diagonal)
    n = len(names)
    upper_triangle = sim_matrix[np.triu_indices(n, k=1)]
    max_sim = float(np.max(upper_triangle)) if len(upper_triangle) > 0 else 0.0

    # Count problematic pairs
    critical = find_high_similarity_pairs(sim_matrix, names, threshold=0.95)
    high_risk = find_high_similarity_pairs(sim_matrix, names, threshold=0.90)

    return {
        "complexity_index": ci,
        "max_similarity": round(max_sim, 4),
        "critical_pairs": len(critical),
        "high_risk_pairs": len(high_risk),
        "critical_pair_details": [(a, b, round(s, 3)) for a, b, s in critical],
        "unknown_fluorophores": unknown,
        "valid": True,
        "evaluated_fluorophores": len(spectra),
        "total_fluorophores": len(fluorophores),
    }


class TestPanelParsing:
    """Tests for response parsing."""

    def test_parse_simple_table(self):
        """Should parse a simple markdown table."""
        response = """
| Marker | Fluorophore | Expression |
|--------|-------------|------------|
| CD3    | BV421       | High       |
| CD4    | PE          | High       |
| CD8    | APC         | High       |
"""
        assignments = parse_panel_from_response(response)
        assert len(assignments) == 3
        assert assignments[0]["marker"] == "CD3"
        assert assignments[0]["fluorophore"] == "BV421"

    def test_parse_multi_word_fluorophore(self):
        """Should handle multi-word fluorophore names."""
        response = """
| Marker | Fluorophore      |
|--------|------------------|
| CD45RA | Alexa Fluor 647  |
| CD27   | PE-Cy5.5         |
"""
        assignments = parse_panel_from_response(response)
        assert assignments[0]["fluorophore"] == "Alexa Fluor 647"
        assert assignments[1]["fluorophore"] == "PE-Cy5.5"


class TestPanelEvaluation:
    """Tests for panel quality evaluation."""

    def test_evaluate_good_panel(self):
        """A panel with spectrally distinct fluorophores should score well."""
        assignments = [
            {"marker": "CD3", "fluorophore": "BV421"},
            {"marker": "CD4", "fluorophore": "PE"},
            {"marker": "CD8", "fluorophore": "APC"},
            {"marker": "Viability", "fluorophore": "BV785"},
        ]
        result = evaluate_panel(assignments)

        assert result["valid"]
        # Complexity index is now total similarity score (sum of all pairs)
        # A good panel should have low average similarity per pair
        assert result["complexity_index"] < 2.0  # Low total similarity for 6 pairs
        assert result["critical_pairs"] == 0
        assert result["high_risk_pairs"] == 0

    def test_evaluate_problematic_panel(self):
        """A panel with similar fluorophores should be flagged."""
        assignments = [
            {"marker": "CD3", "fluorophore": "FITC"},
            {"marker": "CD4", "fluorophore": "BB515"},  # Very similar to FITC
            {"marker": "CD8", "fluorophore": "APC"},
            {"marker": "CD45RA", "fluorophore": "Alexa Fluor 647"},  # Similar to APC
        ]
        result = evaluate_panel(assignments)

        assert result["valid"]
        assert result["high_risk_pairs"] > 0
        # FITC/BB515 and APC/AF647 should both be flagged

    def test_evaluate_unknown_fluorophores(self):
        """Should handle unknown fluorophores gracefully."""
        assignments = [
            {"marker": "CD3", "fluorophore": "BV421"},
            {"marker": "CD4", "fluorophore": "SomeNewDye123"},
        ]
        result = evaluate_panel(assignments)

        assert "SomeNewDye123" in result["unknown_fluorophores"]


class TestCaseLoader:
    """Tests for loading test cases."""

    def test_load_basic_tcell(self):
        """Should load basic T-cell test case."""
        tc = load_test_case("basic_tcell")
        assert tc["name"] == "basic_tcell_panel"
        assert len(tc["markers"]) == 8
        assert any(m["name"] == "CD3" for m in tc["markers"])

    def test_format_marker_list(self):
        """Should format markers for prompt."""
        markers = [
            {"name": "CD3", "expression": "high", "description": "T cells"},
            {"name": "FOXP3", "expression": "medium", "intracellular": True},
        ]
        result = format_marker_list(markers)
        assert "CD3" in result
        assert "high expression" in result
        assert "(intracellular)" in result


# Integration tests (require API key)
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
class TestMCPEffectiveness:
    """
    A/B tests comparing Claude with and without MCP tools.

    These tests are expensive and slow - run selectively.
    """

    @pytest.fixture
    def test_case(self):
        """Load a test case for evaluation."""
        return load_test_case("basic_tcell")

    @pytest.mark.skip(reason="Requires Claude API - run manually")
    async def test_control_condition(self, test_case):
        """Test panel design without MCP tools."""
        # This would call Claude API without tools
        # For now, this is a placeholder
        pass

    @pytest.mark.skip(reason="Requires Claude API + MCP - run manually")
    async def test_treatment_condition(self, test_case):
        """Test panel design with MCP tools."""
        # This would call Claude API with MCP tools
        # For now, this is a placeholder
        pass


class MockPanelResults:
    """Mock results for testing the evaluation framework without API calls."""

    # Simulated control responses (Claude without tools)
    CONTROL_PANELS = {
        "basic_tcell": [
            {"marker": "CD3", "fluorophore": "FITC"},
            {"marker": "CD4", "fluorophore": "PE"},
            {"marker": "CD8", "fluorophore": "APC"},
            {"marker": "CD45RA", "fluorophore": "BV421"},
            {"marker": "CD45RO", "fluorophore": "PE-Cy7"},
            {"marker": "CCR7", "fluorophore": "BV605"},
            {"marker": "CD27", "fluorophore": "BV711"},
            {"marker": "Viability", "fluorophore": "BV785"},
        ],
        "basic_tcell_bad": [
            {"marker": "CD3", "fluorophore": "FITC"},
            {"marker": "CD4", "fluorophore": "BB515"},  # Conflicts with FITC
            {"marker": "CD8", "fluorophore": "APC"},
            {"marker": "CD45RA", "fluorophore": "Alexa Fluor 647"},  # Conflicts with APC
            {"marker": "CD45RO", "fluorophore": "PE-Cy5"},
            {"marker": "CCR7", "fluorophore": "PE-Cy5.5"},  # Conflicts with PE-Cy5
            {"marker": "CD27", "fluorophore": "BV711"},
            {"marker": "Viability", "fluorophore": "BV785"},
        ],
    }

    # Simulated treatment responses (Claude with MCP tools)
    TREATMENT_PANELS = {
        "basic_tcell": [
            {"marker": "CD3", "fluorophore": "BV421"},
            {"marker": "CD4", "fluorophore": "PE"},
            {"marker": "CD8", "fluorophore": "APC"},
            {"marker": "CD45RA", "fluorophore": "FITC"},
            {"marker": "CD45RO", "fluorophore": "BV605"},
            {"marker": "CCR7", "fluorophore": "BV711"},
            {"marker": "CD27", "fluorophore": "PE-Cy7"},
            {"marker": "Viability", "fluorophore": "BV785"},
        ],
    }


class TestMockComparison:
    """Test the comparison framework using mock data."""

    def test_mock_control_evaluation(self):
        """Evaluate mock control panel."""
        panel = MockPanelResults.CONTROL_PANELS["basic_tcell"]
        result = evaluate_panel(panel)

        assert result["valid"]
        print(f"\nControl panel metrics:")
        print(f"  Complexity Index: {result['complexity_index']}")
        print(f"  Max Similarity: {result['max_similarity']}")
        print(f"  High-Risk Pairs: {result['high_risk_pairs']}")

    def test_mock_treatment_evaluation(self):
        """Evaluate mock treatment panel."""
        panel = MockPanelResults.TREATMENT_PANELS["basic_tcell"]
        result = evaluate_panel(panel)

        assert result["valid"]
        print(f"\nTreatment panel metrics:")
        print(f"  Complexity Index: {result['complexity_index']}")
        print(f"  Max Similarity: {result['max_similarity']}")
        print(f"  High-Risk Pairs: {result['high_risk_pairs']}")

    def test_bad_panel_detection(self):
        """Bad panel should have higher conflict scores."""
        good_panel = MockPanelResults.CONTROL_PANELS["basic_tcell"]
        bad_panel = MockPanelResults.CONTROL_PANELS["basic_tcell_bad"]

        good_result = evaluate_panel(good_panel)
        bad_result = evaluate_panel(bad_panel)

        # Bad panel should have more conflicts
        assert bad_result["high_risk_pairs"] > good_result["high_risk_pairs"]
        assert bad_result["complexity_index"] > good_result["complexity_index"]

        print(f"\nGood vs Bad panel comparison:")
        print(f"  Good CI: {good_result['complexity_index']}, "
              f"Bad CI: {bad_result['complexity_index']}")
        print(f"  Good high-risk: {good_result['high_risk_pairs']}, "
              f"Bad high-risk: {bad_result['high_risk_pairs']}")

    def test_comparison_framework(self):
        """Test the full comparison framework."""
        results = []

        for test_name in ["basic_tcell"]:
            control = evaluate_panel(MockPanelResults.CONTROL_PANELS[test_name])
            treatment = evaluate_panel(MockPanelResults.TREATMENT_PANELS[test_name])

            improvement = {
                "test_case": test_name,
                "control_ci": control["complexity_index"],
                "treatment_ci": treatment["complexity_index"],
                "ci_improvement": (
                    (control["complexity_index"] - treatment["complexity_index"])
                    / control["complexity_index"] * 100
                    if control["complexity_index"] > 0 else 0
                ),
                "control_high_risk": control["high_risk_pairs"],
                "treatment_high_risk": treatment["high_risk_pairs"],
            }
            results.append(improvement)

        # Print summary
        print("\n" + "=" * 60)
        print("MCP Effectiveness Test Results (Mock Data)")
        print("=" * 60)
        for r in results:
            print(f"\nTest Case: {r['test_case']}")
            print(f"  Complexity Index: {r['control_ci']:.1f} -> {r['treatment_ci']:.1f} "
                  f"({r['ci_improvement']:+.1f}%)")
            print(f"  High-Risk Pairs: {r['control_high_risk']} -> {r['treatment_high_risk']}")


# =============================================================================
# OMIP Reference Panels
# =============================================================================

class OMIPReferencePanels:
    """
    Published OMIP panel assignments for comparison.

    These represent real-world validated panels from peer-reviewed publications.
    """

    # OMIP-030: T-cell characterization (13-color, 4-laser)
    # Reference: Wingender G, Kronenberg M. Cytometry A. 2015;87(12):1067-1069.
    OMIP_030 = {
        "name": "OMIP-030",
        "url": "https://doi.org/10.1002/cyto.a.22788",
        "published_ci": None,  # Not reported
        "assignments": [
            {"marker": "CD3", "fluorophore": "Pacific Blue"},
            {"marker": "CD4", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD8", "fluorophore": "APC-Fire750"},
            {"marker": "CD14", "fluorophore": "BV510"},
            {"marker": "CD16", "fluorophore": "BV605"},
            {"marker": "CD19", "fluorophore": "BV650"},
            {"marker": "CD25", "fluorophore": "PE"},
            {"marker": "CD27", "fluorophore": "BV711"},
            {"marker": "CD45RA", "fluorophore": "FITC"},
            {"marker": "CD56", "fluorophore": "BV785"},
            {"marker": "CD127", "fluorophore": "PE-Cy7"},
            {"marker": "CCR7", "fluorophore": "PE-Cy5"},
            {"marker": "Viability", "fluorophore": "APC"},
        ],
    }

    # OMIP-069: 40-color full spectrum (5-laser Cytek Aurora)
    # Reference: Park LM et al. Cytometry A. 2020;97(10):1044-1051.
    # Note: Only including a subset of markers for comparison
    OMIP_069_SUBSET = {
        "name": "OMIP-069 (subset)",
        "url": "https://doi.org/10.1002/cyto.a.24213",
        "published_ci": 54,  # Reported in paper
        "assignments": [
            {"marker": "CD3", "fluorophore": "BUV395"},
            {"marker": "CD4", "fluorophore": "BV421"},
            {"marker": "CD8", "fluorophore": "BV605"},
            {"marker": "CD14", "fluorophore": "BV711"},
            {"marker": "CD16", "fluorophore": "BV785"},
            {"marker": "CD19", "fluorophore": "BB515"},
            {"marker": "CD45RA", "fluorophore": "BV510"},
            {"marker": "CD56", "fluorophore": "PE"},
            {"marker": "CCR7", "fluorophore": "BV650"},
            {"marker": "CD127", "fluorophore": "PE-Cy7"},
            {"marker": "CD25", "fluorophore": "PE-CF594"},
            {"marker": "HLA-DR", "fluorophore": "BV750"},
            {"marker": "Viability", "fluorophore": "BUV805"},
        ],
    }


def compare_to_omip(
    test_panel: list[dict],
    omip_panel: dict,
) -> dict:
    """
    Compare a generated panel against a published OMIP panel.

    Returns:
        dict with:
        - omip_name: Name of OMIP panel
        - omip_url: Link to publication
        - matching_assignments: Count of identical marker-fluorophore pairs
        - different_assignments: List of differences
        - test_metrics: Metrics for test panel
        - omip_metrics: Metrics for OMIP panel
        - comparison: Side-by-side comparison
    """
    omip_assignments = omip_panel["assignments"]

    # Build lookup by marker
    test_by_marker = {a["marker"]: a["fluorophore"] for a in test_panel}
    omip_by_marker = {a["marker"]: a["fluorophore"] for a in omip_assignments}

    # Find common markers
    common_markers = set(test_by_marker.keys()) & set(omip_by_marker.keys())

    # Count matches and differences
    matches = []
    differences = []
    for marker in common_markers:
        test_fluor = test_by_marker[marker]
        omip_fluor = omip_by_marker[marker]
        if test_fluor == omip_fluor:
            matches.append(marker)
        else:
            differences.append({
                "marker": marker,
                "test": test_fluor,
                "omip": omip_fluor,
            })

    # Evaluate both panels
    test_metrics = evaluate_panel(test_panel)
    omip_metrics = evaluate_panel(omip_assignments)

    return {
        "omip_name": omip_panel["name"],
        "omip_url": omip_panel["url"],
        "published_ci": omip_panel.get("published_ci"),
        "common_markers": len(common_markers),
        "matching_assignments": len(matches),
        "match_rate": len(matches) / len(common_markers) if common_markers else 0,
        "different_assignments": differences,
        "test_metrics": test_metrics,
        "omip_metrics": omip_metrics,
        "ci_vs_omip": (
            test_metrics["complexity_index"] - omip_metrics["complexity_index"]
            if test_metrics["valid"] and omip_metrics["valid"]
            else None
        ),
    }


def print_omip_comparison(comparison: dict):
    """Pretty-print OMIP comparison results."""
    print("\n" + "=" * 70)
    print(f"Comparison vs {comparison['omip_name']}")
    print(f"Reference: {comparison['omip_url']}")
    print("=" * 70)

    print(f"\nMarker overlap: {comparison['common_markers']} markers")
    print(f"Matching assignments: {comparison['matching_assignments']} "
          f"({comparison['match_rate']*100:.0f}%)")

    if comparison["different_assignments"]:
        print("\nDifferences from OMIP:")
        print("-" * 50)
        print(f"{'Marker':<12} {'Test Panel':<18} {'OMIP Panel':<18}")
        print("-" * 50)
        for diff in comparison["different_assignments"]:
            print(f"{diff['marker']:<12} {diff['test']:<18} {diff['omip']:<18}")

    print("\nMetrics Comparison:")
    print("-" * 50)
    print(f"{'Metric':<25} {'Test Panel':<12} {'OMIP Panel':<12}")
    print("-" * 50)

    test_m = comparison["test_metrics"]
    omip_m = comparison["omip_metrics"]

    if test_m["valid"] and omip_m["valid"]:
        # Complexity Index
        test_ci = test_m["complexity_index"]
        omip_ci = omip_m["complexity_index"]
        ci_diff = test_ci - omip_ci
        ci_indicator = "✓" if ci_diff <= 0 else "✗"
        print(f"{'Complexity Index':<25} {test_ci:<12.1f} {omip_ci:<12.1f} {ci_indicator}")

        # If published CI available
        if comparison["published_ci"]:
            print(f"{'Published CI':<25} {'-':<12} {comparison['published_ci']:<12}")

        # Max Similarity
        test_max = test_m["max_similarity"]
        omip_max = omip_m["max_similarity"]
        max_indicator = "✓" if test_max <= omip_max else "✗"
        print(f"{'Max Similarity':<25} {test_max:<12.4f} {omip_max:<12.4f} {max_indicator}")

        # High-risk pairs
        test_hr = test_m["high_risk_pairs"]
        omip_hr = omip_m["high_risk_pairs"]
        hr_indicator = "✓" if test_hr <= omip_hr else "✗"
        print(f"{'High-Risk Pairs':<25} {test_hr:<12} {omip_hr:<12} {hr_indicator}")

        # Critical pairs
        test_cr = test_m["critical_pairs"]
        omip_cr = omip_m["critical_pairs"]
        cr_indicator = "✓" if test_cr <= omip_cr else "✗"
        print(f"{'Critical Pairs':<25} {test_cr:<12} {omip_cr:<12} {cr_indicator}")

    print("-" * 50)

    # Overall verdict
    if test_m["valid"] and omip_m["valid"]:
        if comparison["ci_vs_omip"] is not None:
            if comparison["ci_vs_omip"] < -5:
                verdict = "BETTER than OMIP"
            elif comparison["ci_vs_omip"] > 5:
                verdict = "WORSE than OMIP"
            else:
                verdict = "SIMILAR to OMIP"
            print(f"\nVerdict: {verdict} (CI diff: {comparison['ci_vs_omip']:+.1f})")


class TestOMIPComparison:
    """Tests comparing generated panels against published OMIPs."""

    def test_evaluate_omip030(self):
        """Evaluate the actual OMIP-030 panel."""
        omip = OMIPReferencePanels.OMIP_030
        result = evaluate_panel(omip["assignments"])

        print(f"\n{omip['name']} Evaluation:")
        print(f"  URL: {omip['url']}")
        print(f"  Complexity Index: {result['complexity_index']}")
        print(f"  Max Similarity: {result['max_similarity']}")
        print(f"  High-Risk Pairs: {result['high_risk_pairs']}")
        print(f"  Critical Pairs: {result['critical_pairs']}")

        if result["critical_pair_details"]:
            print(f"  Critical pair details: {result['critical_pair_details']}")

        assert result["valid"]

    def test_compare_control_vs_omip030(self):
        """Compare control panel design against OMIP-030."""
        # Use a control panel for similar markers
        control_panel = [
            {"marker": "CD3", "fluorophore": "FITC"},
            {"marker": "CD4", "fluorophore": "PE"},
            {"marker": "CD8", "fluorophore": "APC"},
            {"marker": "CD14", "fluorophore": "BV421"},
            {"marker": "CD19", "fluorophore": "BV510"},
            {"marker": "CD25", "fluorophore": "PE-Cy7"},
            {"marker": "CD27", "fluorophore": "BV605"},
            {"marker": "CD45RA", "fluorophore": "BV711"},
            {"marker": "CD56", "fluorophore": "BV785"},
            {"marker": "CD127", "fluorophore": "PE-Cy5"},
            {"marker": "CCR7", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "Viability", "fluorophore": "APC-Fire750"},
        ]

        comparison = compare_to_omip(control_panel, OMIPReferencePanels.OMIP_030)
        print_omip_comparison(comparison)

        assert comparison["test_metrics"]["valid"]
        assert comparison["omip_metrics"]["valid"]

    def test_compare_treatment_vs_omip030(self):
        """Compare treatment (MCP-optimized) panel against OMIP-030."""
        # Simulated MCP-optimized panel (should be closer to OMIP)
        treatment_panel = [
            {"marker": "CD3", "fluorophore": "Pacific Blue"},  # Matches OMIP
            {"marker": "CD4", "fluorophore": "PerCP-Cy5.5"},   # Matches OMIP
            {"marker": "CD8", "fluorophore": "APC-Fire750"},   # Matches OMIP
            {"marker": "CD14", "fluorophore": "BV510"},        # Matches OMIP
            {"marker": "CD19", "fluorophore": "BV650"},        # Matches OMIP
            {"marker": "CD25", "fluorophore": "PE"},           # Matches OMIP
            {"marker": "CD27", "fluorophore": "BV711"},        # Matches OMIP
            {"marker": "CD45RA", "fluorophore": "FITC"},       # Matches OMIP
            {"marker": "CD56", "fluorophore": "BV785"},        # Matches OMIP
            {"marker": "CD127", "fluorophore": "PE-Cy7"},      # Matches OMIP
            {"marker": "CCR7", "fluorophore": "PE-Cy5"},       # Matches OMIP
            {"marker": "Viability", "fluorophore": "APC"},     # Matches OMIP
        ]

        comparison = compare_to_omip(treatment_panel, OMIPReferencePanels.OMIP_030)
        print_omip_comparison(comparison)

        # Treatment should have high match rate (it's designed to match)
        assert comparison["match_rate"] >= 0.9, "Treatment should closely match OMIP"

    def test_full_pipeline_comparison(self):
        """Full comparison: Control vs Treatment vs OMIP."""
        print("\n" + "=" * 70)
        print("FULL PIPELINE COMPARISON: Control vs Treatment vs OMIP-030")
        print("=" * 70)

        # Control (naive Claude design)
        control = [
            {"marker": "CD3", "fluorophore": "FITC"},
            {"marker": "CD4", "fluorophore": "PE"},
            {"marker": "CD8", "fluorophore": "APC"},
            {"marker": "CD45RA", "fluorophore": "BV421"},
            {"marker": "CCR7", "fluorophore": "PE-Cy5"},
            {"marker": "CD27", "fluorophore": "BV605"},
            {"marker": "Viability", "fluorophore": "BV785"},
        ]

        # Treatment (MCP-assisted design)
        treatment = [
            {"marker": "CD3", "fluorophore": "Pacific Blue"},
            {"marker": "CD4", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD8", "fluorophore": "APC-Fire750"},
            {"marker": "CD45RA", "fluorophore": "FITC"},
            {"marker": "CCR7", "fluorophore": "PE-Cy5"},
            {"marker": "CD27", "fluorophore": "BV711"},
            {"marker": "Viability", "fluorophore": "APC"},
        ]

        # OMIP-030 subset with same markers
        omip = [
            {"marker": "CD3", "fluorophore": "Pacific Blue"},
            {"marker": "CD4", "fluorophore": "PerCP-Cy5.5"},
            {"marker": "CD8", "fluorophore": "APC-Fire750"},
            {"marker": "CD45RA", "fluorophore": "FITC"},
            {"marker": "CCR7", "fluorophore": "PE-Cy5"},
            {"marker": "CD27", "fluorophore": "BV711"},
            {"marker": "Viability", "fluorophore": "APC"},
        ]

        control_metrics = evaluate_panel(control)
        treatment_metrics = evaluate_panel(treatment)
        omip_metrics = evaluate_panel(omip)

        print("\n" + "-" * 70)
        print(f"{'Metric':<25} {'Control':<15} {'Treatment':<15} {'OMIP-030':<15}")
        print("-" * 70)

        print(f"{'Complexity Index':<25} "
              f"{control_metrics['complexity_index']:<15.1f} "
              f"{treatment_metrics['complexity_index']:<15.1f} "
              f"{omip_metrics['complexity_index']:<15.1f}")

        print(f"{'Max Similarity':<25} "
              f"{control_metrics['max_similarity']:<15.4f} "
              f"{treatment_metrics['max_similarity']:<15.4f} "
              f"{omip_metrics['max_similarity']:<15.4f}")

        print(f"{'High-Risk Pairs':<25} "
              f"{control_metrics['high_risk_pairs']:<15} "
              f"{treatment_metrics['high_risk_pairs']:<15} "
              f"{omip_metrics['high_risk_pairs']:<15}")

        print(f"{'Critical Pairs':<25} "
              f"{control_metrics['critical_pairs']:<15} "
              f"{treatment_metrics['critical_pairs']:<15} "
              f"{omip_metrics['critical_pairs']:<15}")

        print("-" * 70)

        # Calculate improvement
        if control_metrics["complexity_index"] > 0:
            improvement = (
                (control_metrics["complexity_index"] - treatment_metrics["complexity_index"])
                / control_metrics["complexity_index"] * 100
            )
            print(f"\nTreatment CI improvement vs Control: {improvement:+.1f}%")

        # Check if treatment matches OMIP
        if treatment_metrics["complexity_index"] == omip_metrics["complexity_index"]:
            print("Treatment MATCHES OMIP-030 complexity index!")
        else:
            diff = treatment_metrics["complexity_index"] - omip_metrics["complexity_index"]
            print(f"Treatment vs OMIP-030 CI difference: {diff:+.1f}")


if __name__ == "__main__":
    # Run mock comparison for demonstration
    print("\n" + "=" * 70)
    print("RUNNING MCP EFFECTIVENESS TESTS")
    print("=" * 70)

    # Basic mock comparison
    test_mock = TestMockComparison()
    test_mock.test_comparison_framework()

    # OMIP comparisons
    test_omip = TestOMIPComparison()
    test_omip.test_evaluate_omip030()
    test_omip.test_full_pipeline_comparison()

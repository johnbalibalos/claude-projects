"""Tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    extract_gate_names,
    extract_parent_map,
    compute_hierarchy_f1,
    compute_structure_accuracy,
    compute_critical_gate_recall,
    compute_hallucination_rate,
    normalize_gate_name,
    get_hierarchy_depth,
)


class TestExtractGateNames:
    """Tests for gate name extraction."""

    def test_extract_from_dict(self, sample_hierarchy):
        """Test extracting gate names from dict hierarchy."""
        gates = extract_gate_names(sample_hierarchy)

        assert "All Events" in gates
        assert "Singlets" in gates
        assert "Live" in gates
        assert "CD45+" in gates
        assert "T cells" in gates
        assert "B cells" in gates
        assert len(gates) == 6

    def test_empty_hierarchy(self):
        """Test with empty hierarchy."""
        gates = extract_gate_names({"name": "root", "children": []})
        assert gates == {"root"}


class TestExtractParentMap:
    """Tests for parent map extraction."""

    def test_extract_parent_map(self, sample_hierarchy):
        """Test extracting parent-child relationships."""
        parent_map = extract_parent_map(sample_hierarchy)

        assert parent_map["All Events"] is None
        assert parent_map["Singlets"] == "All Events"
        assert parent_map["Live"] == "Singlets"
        assert parent_map["T cells"] == "CD45+"
        assert parent_map["B cells"] == "CD45+"


class TestNormalizeGateName:
    """Tests for gate name normalization."""

    def test_case_insensitive(self):
        """Test case normalization."""
        assert normalize_gate_name("CD3+") == normalize_gate_name("cd3+")

    def test_positive_variants(self):
        """Test positive gate name variants."""
        assert normalize_gate_name("CD3 positive") == normalize_gate_name("CD3+")
        assert normalize_gate_name("CD3positive") == normalize_gate_name("CD3+")

    def test_negative_variants(self):
        """Test negative gate name variants."""
        assert normalize_gate_name("CD3 negative") == normalize_gate_name("CD3-")
        assert normalize_gate_name("CD3negative") == normalize_gate_name("CD3-")

    def test_whitespace(self):
        """Test whitespace handling."""
        assert normalize_gate_name("  T cells  ") == normalize_gate_name("T cells")


class TestComputeHierarchyF1:
    """Tests for hierarchy F1 computation."""

    def test_identical_hierarchies(self, sample_hierarchy):
        """Test F1 with identical hierarchies."""
        f1, precision, recall, _, _, _ = compute_hierarchy_f1(
            sample_hierarchy, sample_hierarchy
        )

        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0

    def test_partial_match(self, sample_hierarchy):
        """Test F1 with partial match."""
        partial = {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "children": []}
            ]
        }

        f1, precision, recall, matching, missing, extra = compute_hierarchy_f1(
            partial, sample_hierarchy
        )

        # Partial has 2 gates, full has 6
        assert precision == 1.0  # All predicted are correct
        assert recall < 1.0  # Missing some gates
        assert len(matching) == 2
        assert len(missing) == 4

    def test_no_match(self):
        """Test F1 with no matching gates."""
        pred = {"name": "A", "children": [{"name": "B", "children": []}]}
        gt = {"name": "X", "children": [{"name": "Y", "children": []}]}

        f1, precision, recall, _, _, _ = compute_hierarchy_f1(pred, gt, fuzzy_match=False)

        assert f1 == 0.0


class TestComputeStructureAccuracy:
    """Tests for structure accuracy computation."""

    def test_correct_structure(self, sample_hierarchy):
        """Test with correct parent-child relationships."""
        accuracy, correct, total, errors = compute_structure_accuracy(
            sample_hierarchy, sample_hierarchy
        )

        assert accuracy == 1.0
        assert len(errors) == 0

    def test_wrong_parent(self, sample_hierarchy):
        """Test with incorrect parent."""
        wrong_structure = {
            "name": "All Events",
            "children": [
                {
                    "name": "Live",  # Skipped Singlets level
                    "children": []
                }
            ]
        }

        accuracy, correct, total, errors = compute_structure_accuracy(
            wrong_structure, sample_hierarchy
        )

        # "Live" has wrong parent (All Events vs Singlets)
        assert accuracy < 1.0


class TestComputeCriticalGateRecall:
    """Tests for critical gate recall."""

    def test_all_critical_present(self, sample_hierarchy):
        """Test when all critical gates are present."""
        critical_gates = ["Singlets", "Live"]

        recall, missing = compute_critical_gate_recall(
            sample_hierarchy, sample_hierarchy, critical_gates
        )

        assert recall == 1.0
        assert len(missing) == 0

    def test_missing_critical(self, sample_hierarchy):
        """Test when critical gates are missing."""
        partial = {"name": "All Events", "children": []}
        critical_gates = ["Singlets", "Live"]

        recall, missing = compute_critical_gate_recall(
            partial, sample_hierarchy, critical_gates
        )

        assert recall == 0.0
        assert set(missing) == {"Singlets", "Live"}


class TestComputeHallucinationRate:
    """Tests for hallucination rate computation."""

    def test_no_hallucinations(self, sample_hierarchy, sample_panel):
        """Test with no hallucinated gates."""
        rate, hallucinated = compute_hallucination_rate(sample_hierarchy, sample_panel)

        # Generic gates like "All Events", "Singlets", "Live" shouldn't be hallucinations
        # T cells and B cells reference CD3 and CD19 which are in panel
        assert rate == 0.0 or len(hallucinated) == 0

    def test_hallucinated_marker(self, sample_panel):
        """Test with a hallucinated marker."""
        hierarchy_with_hallucination = {
            "name": "All Events",
            "children": [
                {"name": "CD99+", "markers": ["CD99"], "children": []}  # CD99 not in panel
            ]
        }

        rate, hallucinated = compute_hallucination_rate(
            hierarchy_with_hallucination, sample_panel
        )

        # CD99+ references a marker not in the panel
        assert "CD99+" in hallucinated or rate > 0


class TestGetHierarchyDepth:
    """Tests for hierarchy depth calculation."""

    def test_depth(self, sample_hierarchy):
        """Test depth calculation."""
        depth = get_hierarchy_depth(sample_hierarchy)

        # All Events -> Singlets -> Live -> CD45+ -> T cells = depth 4
        assert depth == 4

    def test_flat_hierarchy(self):
        """Test with flat hierarchy."""
        flat = {"name": "root", "children": []}
        assert get_hierarchy_depth(flat) == 0

    def test_single_child(self):
        """Test with single child at each level."""
        linear = {
            "name": "A",
            "children": [
                {"name": "B", "children": [
                    {"name": "C", "children": []}
                ]}
            ]
        }
        assert get_hierarchy_depth(linear) == 2

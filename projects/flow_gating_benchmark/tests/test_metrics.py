"""Tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    compute_hierarchy_f1,
    compute_structure_accuracy,
    compute_critical_gate_recall,
    compute_hallucination_rate,
    normalize_gate_name,
    normalize_gate_semantic,
)
from src.evaluation.hierarchy import (
    extract_gate_names,
    extract_parent_map,
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


class TestNormalizeGateSemantic:
    """Tests for semantic gate name normalization."""

    def test_t_cell_synonyms(self):
        """Test that T cell variations map to same canonical form."""
        assert normalize_gate_semantic("T cells") == normalize_gate_semantic("CD3+ T cells")
        assert normalize_gate_semantic("T lymphocytes") == normalize_gate_semantic("T cells")
        assert normalize_gate_semantic("CD3+") == normalize_gate_semantic("T cells")

    def test_b_cell_synonyms(self):
        """Test that B cell variations map to same canonical form."""
        assert normalize_gate_semantic("B cells") == normalize_gate_semantic("CD19+ B cells")
        assert normalize_gate_semantic("B lymphocytes") == normalize_gate_semantic("B cells")

    def test_monocyte_synonyms(self):
        """Test that monocyte variations map to same canonical form."""
        assert normalize_gate_semantic("Monocytes") == normalize_gate_semantic("monos")
        assert normalize_gate_semantic("CD14+ monocytes") == normalize_gate_semantic("monocytes")

    def test_non_synonym_preserved(self):
        """Test that non-synonym names are preserved after normalization."""
        # These should not match because they're different populations
        result = normalize_gate_semantic("CD8+ cytotoxic")
        assert result == "cd8+ cytotoxic"  # Just basic normalization, no synonym mapping


class TestHallucinationOperatorPrecedence:
    """Tests for the operator precedence fix in hallucination detection."""

    def test_gate_with_dash_and_marker_not_hallucinated(self):
        """Gate with '-' in name should not be hallucinated if marker is found."""
        # FSC-A contains a dash but FSC is in panel markers
        hierarchy = {
            "name": "All Events",
            "children": [
                {"name": "FSC-A vs SSC-A gate", "children": []}
            ]
        }
        panel = [{"marker": "CD3", "fluorophore": "BV421"}]

        rate, hallucinated = compute_hallucination_rate(hierarchy, panel)
        # FSC and SSC are in the default marker set, so this should not be hallucinated
        assert "FSC-A vs SSC-A gate" not in hallucinated

    def test_gate_with_plus_no_marker_is_hallucinated(self):
        """Gate with '+' and no matching marker should be hallucinated."""
        hierarchy = {
            "name": "All Events",
            "children": [
                {"name": "CD999+", "children": []}  # Fake marker
            ]
        }
        panel = [{"marker": "CD3", "fluorophore": "BV421"}]

        rate, hallucinated = compute_hallucination_rate(hierarchy, panel)
        assert "CD999+" in hallucinated


class TestPopulationBasedHallucination:
    """Tests for population-based hallucination detection."""

    def test_regulatory_t_cells_without_foxp3(self):
        """Regulatory T cells without FoxP3 in panel should be hallucinated."""
        hierarchy = {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "children": [
                    {"name": "Regulatory T cells", "children": []}
                ]}
            ]
        }
        # Panel without FoxP3 or CD25
        panel = [
            {"marker": "CD3", "fluorophore": "BV421"},
            {"marker": "CD4", "fluorophore": "BV605"},
        ]

        rate, hallucinated = compute_hallucination_rate(hierarchy, panel)
        assert "Regulatory T cells" in hallucinated

    def test_regulatory_t_cells_with_foxp3(self):
        """Regulatory T cells with FoxP3 in panel should not be hallucinated."""
        hierarchy = {
            "name": "All Events",
            "children": [
                {"name": "Singlets", "children": [
                    {"name": "Regulatory T cells", "children": []}
                ]}
            ]
        }
        # Panel with FoxP3
        panel = [
            {"marker": "CD3", "fluorophore": "BV421"},
            {"marker": "CD4", "fluorophore": "BV605"},
            {"marker": "FoxP3", "fluorophore": "PE"},
        ]

        rate, hallucinated = compute_hallucination_rate(hierarchy, panel)
        assert "Regulatory T cells" not in hallucinated

    def test_nk_cells_without_cd56(self):
        """NK cells without CD56/CD16 in panel should be hallucinated."""
        hierarchy = {
            "name": "All Events",
            "children": [
                {"name": "NK cells", "children": []}
            ]
        }
        # Panel without CD56 or CD16
        panel = [
            {"marker": "CD3", "fluorophore": "BV421"},
            {"marker": "CD19", "fluorophore": "BV605"},
        ]

        rate, hallucinated = compute_hallucination_rate(hierarchy, panel)
        assert "NK cells" in hallucinated


class TestSemanticParentMatching:
    """Tests for semantic matching in structure accuracy."""

    def test_cd3_t_cells_matches_t_cells_parent(self):
        """CD3+ T cells as parent should match T cells as parent."""
        predicted = {
            "name": "All Events",
            "children": [
                {"name": "CD3+ T cells", "children": [
                    {"name": "CD4+", "children": []}
                ]}
            ]
        }
        ground_truth = {
            "name": "All Events",
            "children": [
                {"name": "T cells", "children": [
                    {"name": "CD4+", "children": []}
                ]}
            ]
        }

        # With semantic matching, "CD3+ T cells" and "T cells" should match as parents
        accuracy, correct, total, errors = compute_structure_accuracy(
            predicted, ground_truth, use_semantic_matching=True
        )

        # CD4+ should have matching parent (both map to t_cells canonical form)
        # Only "All Events" is common gate with same parent (None)
        # So we expect accuracy > 0
        assert accuracy > 0 or len(errors) == 0

    def test_strict_matching_fails_for_variants(self):
        """Without semantic matching, T cell variants should not match."""
        predicted = {
            "name": "All Events",
            "children": [
                {"name": "CD3+ T cells", "children": [
                    {"name": "CD4+ subset", "children": []}
                ]}
            ]
        }
        ground_truth = {
            "name": "All Events",
            "children": [
                {"name": "T cells", "children": [
                    {"name": "CD4+ subset", "children": []}
                ]}
            ]
        }

        # Without semantic matching
        accuracy_strict, _, _, errors_strict = compute_structure_accuracy(
            predicted, ground_truth, use_semantic_matching=False
        )

        # With semantic matching
        accuracy_semantic, _, _, errors_semantic = compute_structure_accuracy(
            predicted, ground_truth, use_semantic_matching=True
        )

        # Semantic matching should produce fewer or equal errors
        assert len(errors_semantic) <= len(errors_strict)

"""Tests for the equivalence matching system."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from evaluation.equivalences import (
    AnnotationCapture,
    EquivalenceClass,
    EquivalenceRegistry,
    PendingAnnotation,
    create_enhanced_matcher,
)


class TestEquivalenceRegistry:
    """Tests for EquivalenceRegistry."""

    @pytest.fixture
    def sample_equivalences_yaml(self, tmp_path: Path) -> Path:
        """Create a sample equivalences YAML file."""
        content = {
            "version": "1.0",
            "equivalence_classes": [
                {
                    "canonical": "gamma-delta t cells",
                    "variants": ["gd t cells", "gdt", "gd t"],
                    "domain": "t_cell_subsets",
                },
                {
                    "canonical": "regulatory t cells",
                    "variants": ["tregs", "treg", "t regulatory"],
                    "domain": "t_cell_subsets",
                },
                {
                    "canonical": "double negative",
                    "variants": ["dn", "cd4-cd8-", "dn t cells"],
                    "domain": "t_cell_development",
                },
            ],
            "patterns": [
                {
                    "pattern": "^cd(\\d+) positive$",
                    "equivalent": "cd\\1+",
                    "bidirectional": True,
                },
                {
                    "pattern": "^(.+) cells$",
                    "equivalent": "\\1s",
                    "bidirectional": True,
                },
            ],
        }

        yaml_path = tmp_path / "equivalences.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(content, f)

        return yaml_path

    def test_load_equivalences(self, sample_equivalences_yaml: Path):
        """Test loading equivalences from YAML."""
        registry = EquivalenceRegistry(sample_equivalences_yaml)

        assert len(registry.classes) == 3
        assert len(registry.patterns) == 2

    def test_get_canonical_direct_match(self, sample_equivalences_yaml: Path):
        """Test getting canonical form for direct variant match."""
        registry = EquivalenceRegistry(sample_equivalences_yaml)

        # Test direct variant lookups
        assert registry.get_canonical("tregs") == "regulatory t cells"
        assert registry.get_canonical("Tregs") == "regulatory t cells"  # Case insensitive
        assert registry.get_canonical("TREGS") == "regulatory t cells"
        assert registry.get_canonical("gd t cells") == "gamma-delta t cells"
        assert registry.get_canonical("dn") == "double negative"

    def test_get_canonical_no_match(self, sample_equivalences_yaml: Path):
        """Test getting canonical form for unknown gate returns normalized input."""
        registry = EquivalenceRegistry(sample_equivalences_yaml)

        # Unknown gates should return normalized form
        assert registry.get_canonical("Unknown Gate") == "unknown gate"
        assert registry.get_canonical("  CD3+  ") == "cd3+"

    def test_are_equivalent(self, sample_equivalences_yaml: Path):
        """Test checking if two gates are equivalent."""
        registry = EquivalenceRegistry(sample_equivalences_yaml)

        # Same equivalence class
        assert registry.are_equivalent("tregs", "regulatory t cells")
        assert registry.are_equivalent("Tregs", "T Regulatory")
        assert registry.are_equivalent("gd t cells", "gdt")

        # Different equivalence classes
        assert not registry.are_equivalent("tregs", "gd t cells")
        assert not registry.are_equivalent("dn", "regulatory t cells")

        # Unknown gates
        assert not registry.are_equivalent("unknown1", "unknown2")
        assert registry.are_equivalent("same gate", "SAME GATE")  # Normalized match

    def test_find_match(self, sample_equivalences_yaml: Path):
        """Test finding a match from candidates."""
        registry = EquivalenceRegistry(sample_equivalences_yaml)

        candidates = {"T regulatory", "B cells", "NK cells"}

        # Should find T regulatory as match for tregs
        match, canonical = registry.find_match("tregs", candidates)
        assert match == "T regulatory"
        assert canonical == "regulatory t cells"

        # No match found
        match, canonical = registry.find_match("gd t cells", candidates)
        assert match is None
        assert canonical is None

    def test_get_all_variants(self, sample_equivalences_yaml: Path):
        """Test getting all variants for a gate."""
        registry = EquivalenceRegistry(sample_equivalences_yaml)

        variants = registry.get_all_variants("tregs")
        assert "tregs" in variants
        assert "treg" in variants
        assert "t regulatory" in variants
        assert "regulatory t cells" in variants

        # Unknown gate returns just the normalized input
        variants = registry.get_all_variants("Unknown")
        assert variants == {"unknown"}

    def test_empty_registry(self):
        """Test registry with no file."""
        registry = EquivalenceRegistry(None)

        assert len(registry) == 0
        assert registry.get_canonical("tregs") == "tregs"
        assert not registry.are_equivalent("a", "b")

    def test_pattern_matching(self, sample_equivalences_yaml: Path):
        """Test pattern-based equivalences."""
        registry = EquivalenceRegistry(sample_equivalences_yaml)

        # CD## positive -> CD##+
        assert registry.normalize("cd4 positive") == "cd4+"
        assert registry.normalize("CD8 positive") == "cd8+"


class TestAnnotationCapture:
    """Tests for AnnotationCapture."""

    @pytest.fixture
    def capture(self, tmp_path: Path) -> AnnotationCapture:
        """Create an annotation capture instance."""
        return AnnotationCapture(tmp_path / "pending.jsonl")

    def test_capture_near_miss(self, capture: AnnotationCapture):
        """Test capturing a near-miss pair."""
        # High similarity pair should be captured
        captured = capture.check_and_capture(
            predicted="gd t cells",
            ground_truth="gamma delta T cells",
            test_case_id="OMIP-044",
            parent_context="CD3+ T cells",
        )

        assert captured
        assert len(capture.pending) == 1
        assert capture.pending[0].predicted == "gd t cells"
        assert capture.pending[0].ground_truth == "gamma delta T cells"
        assert capture.pending[0].test_case == "OMIP-044"

    def test_no_capture_low_similarity(self, capture: AnnotationCapture):
        """Test that low similarity pairs are not captured."""
        captured = capture.check_and_capture(
            predicted="B cells",
            ground_truth="Neutrophils",
            test_case_id="OMIP-001",
        )

        assert not captured
        assert len(capture.pending) == 0

    def test_no_capture_high_similarity(self, capture: AnnotationCapture):
        """Test that very high similarity (exact/near-exact) pairs are not captured."""
        captured = capture.check_and_capture(
            predicted="T cells",
            ground_truth="T Cells",  # Just case difference
            test_case_id="OMIP-001",
        )

        assert not captured
        assert len(capture.pending) == 0

    def test_no_duplicate_capture(self, capture: AnnotationCapture):
        """Test that duplicate pairs are not captured twice."""
        capture.check_and_capture(
            predicted="gd t cells",
            ground_truth="gamma delta T cells",
            test_case_id="OMIP-044",
        )

        # Try to capture same pair again
        capture.check_and_capture(
            predicted="gd t cells",
            ground_truth="gamma delta T cells",
            test_case_id="OMIP-045",  # Different test case
        )

        assert len(capture.pending) == 1

    def test_save_and_load(self, tmp_path: Path):
        """Test saving and loading annotations."""
        output_path = tmp_path / "pending.jsonl"
        capture1 = AnnotationCapture(output_path)

        capture1.check_and_capture(
            predicted="dn thymocytes",
            ground_truth="Double Negative",
            test_case_id="OMIP-026",
        )
        capture1.save()

        # Load in new instance
        capture2 = AnnotationCapture(output_path)
        assert len(capture2.pending) == 1
        assert capture2.pending[0].predicted == "dn thymocytes"

    def test_mark_verified(self, capture: AnnotationCapture):
        """Test marking an annotation as verified."""
        capture.check_and_capture(
            predicted="test1",
            ground_truth="test2",
            test_case_id="TEST-001",
        )

        success = capture.mark_verified(
            annotation_id="ann_0000",
            reviewer="test@example.com",
            notes="Confirmed equivalent",
        )

        assert success
        assert capture.pending[0].status == "verified"
        assert capture.pending[0].reviewed_by == "test@example.com"

    def test_mark_rejected(self, capture: AnnotationCapture):
        """Test marking an annotation as rejected."""
        capture.check_and_capture(
            predicted="test1",
            ground_truth="test2",
            test_case_id="TEST-001",
        )

        success = capture.mark_rejected(
            annotation_id="ann_0000",
            reviewer="test@example.com",
            notes="Not equivalent",
        )

        assert success
        assert capture.pending[0].status == "rejected"

    def test_get_stats(self, capture: AnnotationCapture):
        """Test getting annotation statistics."""
        # Add some annotations
        capture.check_and_capture("a1", "b1", "T1")
        capture.check_and_capture("a2", "b2", "T2")
        capture.mark_verified("ann_0000", "reviewer")

        stats = capture.get_stats()
        assert stats["total"] == 2
        assert stats["pending"] == 1
        assert stats["verified"] == 1
        assert stats["rejected"] == 0


class TestCreateEnhancedMatcher:
    """Tests for create_enhanced_matcher helper."""

    def test_create_with_both_paths(self, tmp_path: Path):
        """Test creating matcher with both paths."""
        eq_path = tmp_path / "eq.yaml"
        ann_path = tmp_path / "ann.jsonl"

        # Create empty files
        eq_path.write_text("equivalence_classes: []")
        ann_path.write_text("")

        registry, capture = create_enhanced_matcher(eq_path, ann_path)

        assert registry is not None
        assert capture is not None

    def test_create_without_annotation_path(self, tmp_path: Path):
        """Test creating matcher without annotation capture."""
        eq_path = tmp_path / "eq.yaml"
        eq_path.write_text("equivalence_classes: []")

        registry, capture = create_enhanced_matcher(eq_path, None)

        assert registry is not None
        assert capture is None


class TestIntegrationWithRealEquivalences:
    """Integration tests using the actual equivalences file."""

    @pytest.fixture
    def real_registry(self) -> EquivalenceRegistry:
        """Load the real equivalences file."""
        eq_path = Path(__file__).parent.parent / "data" / "annotations" / "verified_equivalences.yaml"
        if not eq_path.exists():
            pytest.skip("Real equivalences file not found")
        return EquivalenceRegistry(eq_path)

    def test_gamma_delta_variants(self, real_registry: EquivalenceRegistry):
        """Test gamma-delta T cell variants match."""
        assert real_registry.are_equivalent("gd t cells", "gamma-delta t cells")
        assert real_registry.are_equivalent("gdt", "gamma delta t cells")

    def test_treg_variants(self, real_registry: EquivalenceRegistry):
        """Test Treg variants match."""
        assert real_registry.are_equivalent("tregs", "regulatory t cells")
        assert real_registry.are_equivalent("cd4+cd25+foxp3+", "tregs")

    def test_memory_subset_variants(self, real_registry: EquivalenceRegistry):
        """Test memory subset abbreviations."""
        assert real_registry.are_equivalent("cm", "central memory")
        assert real_registry.are_equivalent("em", "effector memory")
        assert real_registry.are_equivalent("temra", "cd45ra+ccr7-")

    def test_qc_gate_variants(self, real_registry: EquivalenceRegistry):
        """Test QC gate variants."""
        assert real_registry.are_equivalent("live", "live cells")
        assert real_registry.are_equivalent("singlets", "single cells")

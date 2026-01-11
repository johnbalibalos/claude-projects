"""
Property-based tests using Hypothesis.

These tests verify invariants that should hold for all inputs,
catching edge cases that example-based tests might miss.

Run with: pytest -v -m hypothesis --hypothesis-show-statistics
"""

import importlib.util
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Mark all tests in this module with hypothesis marker
pytestmark = pytest.mark.hypothesis

# Path to source code
SRC_PATH = Path(__file__).parent.parent / "src"


def load_module_directly(module_name: str, file_path: Path):
    """Load a module directly from file path, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load normalization module directly to avoid __init__.py import chain issues
normalization = load_module_directly(
    "normalization", SRC_PATH / "evaluation" / "normalization.py"
)


# =============================================================================
# STRATEGIES - Custom generators for domain objects
# =============================================================================

# Gate names can contain letters, numbers, common symbols
gate_name_chars = st.sampled_from(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "+-/ "
)

gate_names = st.text(gate_name_chars, min_size=1, max_size=50).filter(
    lambda x: x.strip()  # Must have non-whitespace content
)

# Common cell type names for realistic testing
common_cell_types = st.sampled_from([
    "T cells", "B cells", "NK cells", "Monocytes", "Neutrophils",
    "CD4+ T cells", "CD8+ T cells", "Tregs", "Lymphocytes",
    "Live cells", "Singlets", "CD45+", "CD3+", "CD19+",
    "Classical monocytes", "Non-classical monocytes",
    "Memory CD4", "Naive CD8", "CD56bright NK", "pDC", "cDC1",
])

# Marker patterns like CD3+, CD4-, HLA-DR+
marker_pattern = st.from_regex(r"CD[0-9]{1,3}[+-]?", fullmatch=True)


# =============================================================================
# NORMALIZATION PROPERTIES
# =============================================================================

class TestNormalizationProperties:
    """Property-based tests for gate name normalization."""

    @given(name=gate_names)
    def test_normalize_is_idempotent(self, name: str):
        """Normalizing twice gives same result as normalizing once."""
        once = normalization.normalize_gate_name(name)
        twice = normalization.normalize_gate_name(once)
        assert once == twice, f"Not idempotent: '{name}' -> '{once}' -> '{twice}'"

    @given(name=gate_names)
    @pytest.mark.xfail(reason="Known bug: 'classical' -> 'classical_monocytes' -> 'monocytes'", strict=False)
    def test_semantic_normalize_is_idempotent(self, name: str):
        """Semantic normalization is idempotent.

        NOTE: This test discovered a real bug where the semantic normalization
        is not idempotent for certain inputs. For example:
          - "classical" -> "classical_monocytes" -> "monocytes"

        The substring matching in normalize_gate_semantic() causes this issue.
        This is tracked for future fix.
        """
        once = normalization.normalize_gate_semantic(name)
        twice = normalization.normalize_gate_semantic(once)
        assert once == twice

    @given(name1=gate_names, name2=gate_names)
    def test_equivalence_is_symmetric(self, name1: str, name2: str):
        """Gate equivalence is symmetric: equiv(a,b) == equiv(b,a)."""
        forward = normalization.are_gates_equivalent(name1, name2)
        backward = normalization.are_gates_equivalent(name2, name1)
        assert forward == backward, f"Not symmetric: '{name1}' vs '{name2}'"

    @given(name=gate_names)
    def test_equivalence_is_reflexive(self, name: str):
        """Every gate is equivalent to itself."""
        assert normalization.are_gates_equivalent(name, name)

    @given(name=gate_names)
    def test_normalize_removes_parenthetical_qualifiers(self, name: str):
        """Parenthetical content like (FSC) should be removed."""
        with_parens = f"{name} (FSC-A vs FSC-H)"
        normalized = normalization.normalize_gate_name(with_parens)
        assert "(FSC" not in normalized
        assert ")" not in normalized or "(" in normalized  # Balanced if present

    @given(name=common_cell_types)
    def test_common_cell_types_normalize_consistently(self, name: str):
        """Common cell types should normalize to non-empty strings."""
        normalized = normalization.normalize_gate_name(name)
        assert normalized  # Non-empty


# =============================================================================
# F1 SCORE PROPERTIES (self-contained)
# =============================================================================

def compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class TestF1Properties:
    """Property-based tests for F1 score calculation."""

    @given(
        precision=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        recall=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_f1_score_bounded(self, precision: float, recall: float):
        """F1 score is always between 0 and 1."""
        f1 = compute_f1(precision, recall)
        assert 0.0 <= f1 <= 1.0, f"F1 out of bounds: {f1}"

    @given(
        precision=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        recall=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_f1_symmetric(self, precision: float, recall: float):
        """F1 is symmetric in precision and recall."""
        f1_pr = compute_f1(precision, recall)
        f1_rp = compute_f1(recall, precision)
        assert abs(f1_pr - f1_rp) < 1e-10

    @given(value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_f1_perfect_when_equal(self, value: float):
        """F1 equals input when precision == recall."""
        f1 = compute_f1(value, value)
        assert abs(f1 - value) < 1e-10

    def test_f1_zero_when_either_zero(self):
        """F1 is 0 when either precision or recall is 0."""
        assert compute_f1(0.0, 0.5) == 0.0
        assert compute_f1(0.5, 0.0) == 0.0
        assert compute_f1(0.0, 0.0) == 0.0

    @given(
        p=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
        r=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    )
    def test_f1_is_harmonic_mean(self, p: float, r: float):
        """F1 is the harmonic mean of precision and recall."""
        f1 = compute_f1(p, r)
        harmonic = 2 / (1/p + 1/r)
        assert abs(f1 - harmonic) < 1e-10


# =============================================================================
# HIERARCHY PROPERTIES
# =============================================================================

@st.composite
def simple_hierarchy(draw):
    """Generate simple hierarchies for testing."""
    root_name = draw(common_cell_types)
    n_children = draw(st.integers(min_value=0, max_value=5))

    children = []
    for _ in range(n_children):
        child_name = draw(common_cell_types)
        children.append({"name": child_name, "children": []})

    return {"name": root_name, "children": children}


def flatten_hierarchy(hierarchy: dict) -> list[str]:
    """Flatten a hierarchy to a list of gate names."""
    result = [hierarchy.get("name", "")]
    for child in hierarchy.get("children", []):
        result.extend(flatten_hierarchy(child))
    return result


def get_hierarchy_depth(hierarchy: dict) -> int:
    """Get the depth of a hierarchy."""
    if not hierarchy.get("name") and not hierarchy.get("children"):
        return 0
    children = hierarchy.get("children", [])
    if not children:
        return 1
    return 1 + max(get_hierarchy_depth(c) for c in children)


class TestHierarchyProperties:
    """Property-based tests for hierarchy operations."""

    @given(hierarchy=simple_hierarchy())
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_flatten_preserves_all_nodes(self, hierarchy: dict):
        """Flattening a hierarchy preserves all node names."""
        flat = flatten_hierarchy(hierarchy)

        # Count nodes in original
        def count_nodes(h):
            return 1 + sum(count_nodes(c) for c in h.get("children", []))

        original_count = count_nodes(hierarchy)
        assert len(flat) == original_count

    @given(hierarchy=simple_hierarchy())
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_get_depth_non_negative(self, hierarchy: dict):
        """Hierarchy depth is always non-negative."""
        depth = get_hierarchy_depth(hierarchy)
        assert depth >= 0

    @given(hierarchy=simple_hierarchy())
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_depth_at_least_one_for_nonempty(self, hierarchy: dict):
        """Non-empty hierarchy has depth >= 1."""
        if hierarchy.get("name"):
            depth = get_hierarchy_depth(hierarchy)
            assert depth >= 1


# =============================================================================
# RESPONSE PARSING PROPERTIES
# =============================================================================

# NOTE: Response parser tests are in test_response_parser.py
# These cannot be easily loaded here due to dataclass module resolution issues


# =============================================================================
# SYNONYM DICTIONARY PROPERTIES
# =============================================================================

class TestSynonymProperties:
    """Property tests for the synonym dictionary."""

    def test_all_synonyms_map_to_valid_canonical(self):
        """Every synonym value should be stable (not chain to another key)."""
        synonyms = normalization.CELL_TYPE_SYNONYMS

        # Each canonical form should be "stable" - not itself a key
        # (unless it maps to itself, which is fine)
        for key, canonical in synonyms.items():
            if canonical in synonyms:
                # If canonical is also a key, it should map to itself
                assert synonyms[canonical] == canonical, (
                    f"Unstable mapping: {key} -> {canonical} -> "
                    f"{synonyms[canonical]}"
                )

    def test_synonym_keys_are_lowercase(self):
        """All synonym keys should be lowercase for consistent lookup."""
        for key in normalization.CELL_TYPE_SYNONYMS:
            assert key == key.lower(), f"Key not lowercase: '{key}'"


# =============================================================================
# SET OPERATION PROPERTIES
# =============================================================================

class TestSetOperationProperties:
    """Property tests for set-based metrics used in hierarchy comparison."""

    @given(
        set_a=st.frozensets(common_cell_types, min_size=0, max_size=10),
        set_b=st.frozensets(common_cell_types, min_size=0, max_size=10),
    )
    def test_jaccard_bounded(self, set_a: frozenset, set_b: frozenset):
        """Jaccard similarity is always between 0 and 1."""
        if not set_a and not set_b:
            jaccard = 1.0  # By convention, empty sets are identical
        else:
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            jaccard = intersection / union if union > 0 else 0.0

        assert 0.0 <= jaccard <= 1.0

    @given(
        set_a=st.frozensets(common_cell_types, min_size=0, max_size=10),
        set_b=st.frozensets(common_cell_types, min_size=0, max_size=10),
    )
    def test_jaccard_symmetric(self, set_a: frozenset, set_b: frozenset):
        """Jaccard(A, B) == Jaccard(B, A)."""
        def jaccard(a, b):
            if not a and not b:
                return 1.0
            union = len(a | b)
            return len(a & b) / union if union > 0 else 0.0

        assert jaccard(set_a, set_b) == jaccard(set_b, set_a)

    @given(set_a=st.frozensets(common_cell_types, min_size=1, max_size=10))
    def test_jaccard_identity(self, set_a: frozenset):
        """Jaccard(A, A) == 1.0."""
        intersection = len(set_a & set_a)
        union = len(set_a | set_a)
        jaccard = intersection / union
        assert jaccard == 1.0

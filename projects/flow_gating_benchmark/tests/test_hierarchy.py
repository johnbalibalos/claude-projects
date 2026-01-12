"""Tests for evaluation/hierarchy.py - Tree operations for gating hierarchies."""


from evaluation.hierarchy import (
    count_gates,
    extract_gate_names,
    extract_parent_map,
    find_gate_path,
    get_hierarchy_depth,
    get_leaf_gates,
)


class TestExtractGateNames:
    """Tests for extract_gate_names function."""

    def test_extract_from_dict_hierarchy(self, sample_hierarchy):
        """Extract gate names from dict representation."""
        gates = extract_gate_names(sample_hierarchy)

        expected = {"All Events", "Singlets", "Live", "CD45+", "T cells", "B cells"}
        assert gates == expected

    def test_extract_from_dict_with_root_key(self, sample_hierarchy):
        """Extract from dict with 'root' wrapper."""
        wrapped = {"root": sample_hierarchy}
        gates = extract_gate_names(wrapped)

        expected = {"All Events", "Singlets", "Live", "CD45+", "T cells", "B cells"}
        assert gates == expected

    def test_extract_from_empty_hierarchy(self):
        """Empty hierarchy returns empty set."""
        assert extract_gate_names({}) == set()

    def test_extract_single_node(self):
        """Single node hierarchy."""
        hierarchy = {"name": "Root", "children": []}
        assert extract_gate_names(hierarchy) == {"Root"}

    def test_extract_deep_hierarchy(self):
        """Deep nested hierarchy."""
        hierarchy = {
            "name": "A",
            "children": [
                {"name": "B", "children": [
                    {"name": "C", "children": [
                        {"name": "D", "children": []}
                    ]}
                ]}
            ]
        }
        assert extract_gate_names(hierarchy) == {"A", "B", "C", "D"}


class TestExtractParentMap:
    """Tests for extract_parent_map function."""

    def test_parent_map_from_hierarchy(self, sample_hierarchy):
        """Build parent map from hierarchy."""
        parent_map = extract_parent_map(sample_hierarchy)

        assert parent_map["All Events"] is None  # Root has no parent
        assert parent_map["Singlets"] == "All Events"
        assert parent_map["Live"] == "Singlets"
        assert parent_map["CD45+"] == "Live"
        assert parent_map["T cells"] == "CD45+"
        assert parent_map["B cells"] == "CD45+"

    def test_parent_map_with_root_wrapper(self, sample_hierarchy):
        """Parent map with 'root' key wrapper."""
        wrapped = {"root": sample_hierarchy}
        parent_map = extract_parent_map(wrapped)

        assert parent_map["All Events"] is None
        assert parent_map["Singlets"] == "All Events"

    def test_parent_map_single_node(self):
        """Single node has no parent."""
        hierarchy = {"name": "Root", "children": []}
        parent_map = extract_parent_map(hierarchy)

        assert parent_map == {"Root": None}

    def test_parent_map_empty_hierarchy(self):
        """Empty hierarchy returns empty map."""
        assert extract_parent_map({}) == {}


class TestGetHierarchyDepth:
    """Tests for get_hierarchy_depth function."""

    def test_depth_of_sample_hierarchy(self, sample_hierarchy):
        """Sample hierarchy has depth 4 (All Events -> Singlets -> Live -> CD45+ -> T/B cells)."""
        depth = get_hierarchy_depth(sample_hierarchy)
        assert depth == 4

    def test_depth_single_node(self):
        """Single node has depth 0."""
        hierarchy = {"name": "Root", "children": []}
        assert get_hierarchy_depth(hierarchy) == 0

    def test_depth_with_root_wrapper(self, sample_hierarchy):
        """Depth with 'root' key wrapper."""
        wrapped = {"root": sample_hierarchy}
        assert get_hierarchy_depth(wrapped) == 4

    def test_depth_empty_hierarchy(self):
        """Empty hierarchy has depth 0."""
        assert get_hierarchy_depth({}) == 0

    def test_depth_wide_hierarchy(self):
        """Wide (not deep) hierarchy."""
        hierarchy = {
            "name": "Root",
            "children": [
                {"name": "A", "children": []},
                {"name": "B", "children": []},
                {"name": "C", "children": []},
            ]
        }
        assert get_hierarchy_depth(hierarchy) == 1


class TestCountGates:
    """Tests for count_gates function."""

    def test_count_sample_hierarchy(self, sample_hierarchy):
        """Count gates in sample hierarchy."""
        assert count_gates(sample_hierarchy) == 6

    def test_count_single_node(self):
        """Single node counts as 1."""
        hierarchy = {"name": "Root", "children": []}
        assert count_gates(hierarchy) == 1

    def test_count_empty_hierarchy(self):
        """Empty hierarchy has 0 gates."""
        assert count_gates({}) == 0


class TestGetLeafGates:
    """Tests for get_leaf_gates function."""

    def test_leaf_gates_sample_hierarchy(self, sample_hierarchy):
        """Get leaf gates from sample hierarchy."""
        leaves = get_leaf_gates(sample_hierarchy)

        assert leaves == {"T cells", "B cells"}

    def test_leaf_gates_single_node(self):
        """Single node is itself a leaf."""
        hierarchy = {"name": "Root", "children": []}
        assert get_leaf_gates(hierarchy) == {"Root"}

    def test_leaf_gates_with_root_wrapper(self, sample_hierarchy):
        """Leaf gates with 'root' key wrapper."""
        wrapped = {"root": sample_hierarchy}
        leaves = get_leaf_gates(wrapped)

        assert leaves == {"T cells", "B cells"}

    def test_leaf_gates_deep_chain(self):
        """Chain hierarchy - only deepest node is leaf."""
        hierarchy = {
            "name": "A",
            "children": [
                {"name": "B", "children": [
                    {"name": "C", "children": []}
                ]}
            ]
        }
        assert get_leaf_gates(hierarchy) == {"C"}


class TestFindGatePath:
    """Tests for find_gate_path function."""

    def test_find_root_path(self, sample_hierarchy):
        """Path to root is just the root."""
        path = find_gate_path(sample_hierarchy, "All Events")
        assert path == ["All Events"]

    def test_find_leaf_path(self, sample_hierarchy):
        """Path to leaf gate."""
        path = find_gate_path(sample_hierarchy, "T cells")
        assert path == ["All Events", "Singlets", "Live", "CD45+", "T cells"]

    def test_find_middle_path(self, sample_hierarchy):
        """Path to middle gate."""
        path = find_gate_path(sample_hierarchy, "Live")
        assert path == ["All Events", "Singlets", "Live"]

    def test_find_nonexistent_gate(self, sample_hierarchy):
        """Non-existent gate returns None."""
        path = find_gate_path(sample_hierarchy, "NonExistent")
        assert path is None

    def test_find_with_root_wrapper(self, sample_hierarchy):
        """Find path with 'root' key wrapper."""
        wrapped = {"root": sample_hierarchy}
        path = find_gate_path(wrapped, "CD45+")
        assert path == ["All Events", "Singlets", "Live", "CD45+"]

    def test_find_sibling_gates(self, sample_hierarchy):
        """Both sibling paths are correct."""
        t_path = find_gate_path(sample_hierarchy, "T cells")
        b_path = find_gate_path(sample_hierarchy, "B cells")

        # Same prefix, different last element
        assert t_path[:-1] == b_path[:-1]
        assert t_path[-1] == "T cells"
        assert b_path[-1] == "B cells"

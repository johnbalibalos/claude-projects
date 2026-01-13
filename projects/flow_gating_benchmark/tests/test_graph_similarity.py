"""Tests for graph similarity metrics."""

import pytest
from src.evaluation.graph_similarity import (
    HierarchyGraph,
    TreeNode,
    compute_tree_edit_distance,
    compute_normalized_ted,
    compute_tree_similarity,
    compute_structure_similarity,
    is_subtree,
    StructureSimilarityResult,
)


class TestHierarchyGraph:
    """Tests for HierarchyGraph construction and operations."""

    def test_from_hierarchy_dict(self, sample_hierarchy):
        """Test building graph from hierarchy dict."""
        graph = HierarchyGraph.from_hierarchy(sample_hierarchy)

        assert len(graph.nodes) == 6
        assert graph.root == "All Events"
        assert len(graph.edges) == 5

    def test_from_hierarchy_with_root_key(self):
        """Test building from dict with 'root' key."""
        hierarchy = {
            "root": {
                "name": "Root",
                "children": [
                    {"name": "Child1", "children": []},
                    {"name": "Child2", "children": []},
                ]
            }
        }

        graph = HierarchyGraph.from_hierarchy(hierarchy)
        assert graph.root == "Root"
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2

    def test_node_depths(self, sample_hierarchy):
        """Test depth tracking."""
        graph = HierarchyGraph.from_hierarchy(sample_hierarchy)

        assert graph.node_depths["All Events"] == 0
        assert graph.node_depths["Singlets"] == 1
        assert graph.node_depths["Live"] == 2
        assert graph.node_depths["T cells"] == 4

    def test_to_tree(self, sample_hierarchy):
        """Test conversion to TreeNode structure."""
        graph = HierarchyGraph.from_hierarchy(sample_hierarchy)
        tree = graph.to_tree()

        assert tree is not None
        assert tree.label == "All Events"
        assert len(tree.children) == 1
        assert tree.children[0].label == "Singlets"

    def test_get_ancestors(self, sample_hierarchy):
        """Test ancestor retrieval."""
        graph = HierarchyGraph.from_hierarchy(sample_hierarchy)

        ancestors = graph.get_ancestors("T cells")
        assert ancestors == ["All Events", "Singlets", "Live", "CD45+"]

        # Root has no ancestors
        ancestors = graph.get_ancestors("All Events")
        assert ancestors == []

    def test_get_descendants(self, sample_hierarchy):
        """Test descendant retrieval."""
        graph = HierarchyGraph.from_hierarchy(sample_hierarchy)

        descendants = graph.get_descendants("CD45+")
        assert descendants == {"T cells", "B cells"}

        # Leaf has no descendants
        descendants = graph.get_descendants("T cells")
        assert descendants == set()

    def test_get_siblings(self, sample_hierarchy):
        """Test sibling retrieval."""
        graph = HierarchyGraph.from_hierarchy(sample_hierarchy)

        siblings = graph.get_siblings("T cells")
        assert siblings == ["B cells"]

        # Root has no siblings
        siblings = graph.get_siblings("All Events")
        assert siblings == []


class TestTreeEditDistance:
    """Tests for tree edit distance computation."""

    def test_identical_trees(self, sample_hierarchy):
        """Identical trees should have distance 0."""
        ted = compute_tree_edit_distance(sample_hierarchy, sample_hierarchy)
        assert ted == 0.0

    def test_completely_different_trees(self):
        """Completely different trees should have non-zero distance."""
        tree1 = {"name": "A", "children": [{"name": "B", "children": []}]}
        tree2 = {"name": "X", "children": [{"name": "Y", "children": []}]}

        ted = compute_tree_edit_distance(tree1, tree2)
        assert ted > 0

    def test_one_node_difference(self):
        """Trees differing by one node name."""
        tree1 = {
            "name": "Root",
            "children": [
                {"name": "Child", "children": []}
            ]
        }
        tree2 = {
            "name": "Root",
            "children": [
                {"name": "Different", "children": []}
            ]
        }

        ted = compute_tree_edit_distance(tree1, tree2)
        assert ted == 1.0  # One relabel operation

    def test_added_node(self):
        """Tree with an added node."""
        tree1 = {
            "name": "Root",
            "children": []
        }
        tree2 = {
            "name": "Root",
            "children": [
                {"name": "Child", "children": []}
            ]
        }

        ted = compute_tree_edit_distance(tree1, tree2)
        assert ted == 1.0  # One insert operation

    def test_removed_node(self):
        """Tree with a removed node."""
        tree1 = {
            "name": "Root",
            "children": [
                {"name": "Child", "children": []}
            ]
        }
        tree2 = {
            "name": "Root",
            "children": []
        }

        ted = compute_tree_edit_distance(tree1, tree2)
        assert ted == 1.0  # One delete operation

    def test_normalized_matching(self):
        """Test that normalized gate names are matched."""
        tree1 = {
            "name": "All Events",
            "children": [
                {"name": "CD3+", "children": []}
            ]
        }
        tree2 = {
            "name": "all events",  # Different case
            "children": [
                {"name": "cd3 positive", "children": []}  # Different format
            ]
        }

        ted = compute_tree_edit_distance(tree1, tree2)
        # Should be 0 or very low due to normalization
        assert ted == pytest.approx(0.0, abs=0.1)


class TestNormalizedTED:
    """Tests for normalized tree edit distance."""

    def test_normalized_range(self, sample_hierarchy):
        """Normalized TED should be in [0, 1]."""
        different_tree = {
            "name": "Different",
            "children": [
                {"name": "X", "children": []},
                {"name": "Y", "children": []},
            ]
        }

        nted = compute_normalized_ted(sample_hierarchy, different_tree)
        assert 0.0 <= nted <= 1.0

    def test_identical_is_zero(self, sample_hierarchy):
        """Identical trees should have normalized TED = 0."""
        nted = compute_normalized_ted(sample_hierarchy, sample_hierarchy)
        assert nted == 0.0


class TestTreeSimilarity:
    """Tests for tree similarity (1 - normalized TED)."""

    def test_identical_is_one(self, sample_hierarchy):
        """Identical trees should have similarity 1.0."""
        sim = compute_tree_similarity(sample_hierarchy, sample_hierarchy)
        assert sim == 1.0

    def test_similarity_range(self, sample_hierarchy):
        """Similarity should be in [0, 1]."""
        different_tree = {
            "name": "Different",
            "children": [{"name": "X", "children": []}]
        }

        sim = compute_tree_similarity(sample_hierarchy, different_tree)
        assert 0.0 <= sim <= 1.0

    def test_similar_trees_high_score(self):
        """Similar trees should have high similarity."""
        tree1 = {
            "name": "Root",
            "children": [
                {"name": "A", "children": []},
                {"name": "B", "children": []},
            ]
        }
        tree2 = {
            "name": "Root",
            "children": [
                {"name": "A", "children": []},
                {"name": "C", "children": []},  # One different child
            ]
        }

        sim = compute_tree_similarity(tree1, tree2)
        assert sim > 0.5  # Should be reasonably similar


class TestStructureSimilarity:
    """Tests for comprehensive structure similarity."""

    def test_identical_hierarchies(self, sample_hierarchy):
        """Identical hierarchies should have similarity 1.0."""
        result = compute_structure_similarity(sample_hierarchy, sample_hierarchy)

        assert isinstance(result, StructureSimilarityResult)
        assert result.overall_similarity == pytest.approx(1.0)
        assert result.node_similarity == pytest.approx(1.0)
        assert result.edge_similarity == pytest.approx(1.0)

    def test_result_components(self, sample_hierarchy):
        """Test that all components are computed."""
        different = {
            "name": "Root",
            "children": [{"name": "Child", "children": []}]
        }

        result = compute_structure_similarity(sample_hierarchy, different)

        # All components should be in [0, 1]
        assert 0.0 <= result.node_similarity <= 1.0
        assert 0.0 <= result.edge_similarity <= 1.0
        assert 0.0 <= result.depth_similarity <= 1.0
        assert 0.0 <= result.branching_similarity <= 1.0
        assert 0.0 <= result.tree_edit_similarity <= 1.0
        assert 0.0 <= result.overall_similarity <= 1.0

    def test_partial_overlap(self):
        """Test with partially overlapping hierarchies."""
        tree1 = {
            "name": "Root",
            "children": [
                {"name": "A", "children": []},
                {"name": "B", "children": []},
            ]
        }
        tree2 = {
            "name": "Root",
            "children": [
                {"name": "A", "children": []},
                {"name": "C", "children": []},
            ]
        }

        result = compute_structure_similarity(tree1, tree2)

        # Should have some similarity (shared Root and A)
        assert result.overall_similarity > 0.3
        # But not perfect
        assert result.overall_similarity < 1.0


class TestIsSubtree:
    """Tests for subtree detection."""

    def test_exact_subtree(self, sample_hierarchy):
        """Subtree of itself should return True."""
        assert is_subtree(sample_hierarchy, sample_hierarchy) is True

    def test_partial_subtree(self, sample_hierarchy):
        """Partial tree should be detected as subtree."""
        subtree = {
            "name": "CD45+",
            "children": [
                {"name": "T cells", "children": []},
                {"name": "B cells", "children": []},
            ]
        }

        assert is_subtree(sample_hierarchy, subtree) is True

    def test_not_subtree(self, sample_hierarchy):
        """Non-subtree should return False."""
        not_subtree = {
            "name": "CD45+",
            "children": [
                {"name": "Unknown Cell", "children": []},
            ]
        }

        assert is_subtree(sample_hierarchy, not_subtree) is False

    def test_empty_subtree(self, sample_hierarchy):
        """Empty tree should be considered subtree."""
        empty = {"name": "", "children": []}
        assert is_subtree(sample_hierarchy, empty) is True


class TestGraphWithSemanticMatching:
    """Tests for graph operations with semantic matching enabled."""

    @pytest.fixture
    def mock_semantic_matcher(self):
        """Mock semantic matcher for testing."""
        class MockMatcher:
            high_threshold = 0.85
            medium_threshold = 0.70

            def compute_similarity(self, s1, s2):
                # Simple mock: same normalized string = 1.0, else 0.5
                n1 = s1.lower().strip()
                n2 = s2.lower().strip()
                if n1 == n2:
                    return 1.0
                # Simulate some semantic matches
                synonyms = {
                    ("t cells", "cd3+ t cells"),
                    ("cd3+ t cells", "t cells"),
                    ("b cells", "cd19+ b cells"),
                    ("cd19+ b cells", "b cells"),
                }
                if (n1, n2) in synonyms:
                    return 0.9
                return 0.3

        return MockMatcher()

    def test_semantic_tree_similarity(self, mock_semantic_matcher):
        """Test tree similarity with semantic matching."""
        tree1 = {
            "name": "Root",
            "children": [
                {"name": "T cells", "children": []}
            ]
        }
        tree2 = {
            "name": "Root",
            "children": [
                {"name": "CD3+ T cells", "children": []}  # Semantically equivalent
            ]
        }

        # With semantic matching, should be more similar
        sim_semantic = compute_tree_similarity(
            tree1, tree2,
            use_semantic_matching=True,
            semantic_matcher=mock_semantic_matcher
        )

        # Without semantic matching
        sim_plain = compute_tree_similarity(
            tree1, tree2,
            use_semantic_matching=False
        )

        assert sim_semantic >= sim_plain


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_hierarchy(self):
        """Empty hierarchy handling."""
        empty = {"name": "", "children": []}
        graph = HierarchyGraph.from_hierarchy(empty)

        assert len(graph.nodes) == 0 or graph.nodes == [""]

    def test_single_node(self):
        """Single node hierarchy."""
        single = {"name": "Only", "children": []}
        graph = HierarchyGraph.from_hierarchy(single)

        assert len(graph.nodes) == 1
        assert graph.root == "Only"
        assert len(graph.edges) == 0

    def test_deep_hierarchy(self):
        """Deep hierarchy (many levels)."""
        deep = {"name": "L0", "children": [
            {"name": "L1", "children": [
                {"name": "L2", "children": [
                    {"name": "L3", "children": [
                        {"name": "L4", "children": [
                            {"name": "L5", "children": []}
                        ]}
                    ]}
                ]}
            ]}
        ]}

        graph = HierarchyGraph.from_hierarchy(deep)
        assert len(graph.nodes) == 6
        assert graph.node_depths["L5"] == 5

    def test_wide_hierarchy(self):
        """Wide hierarchy (many children at one level)."""
        wide = {
            "name": "Root",
            "children": [
                {"name": f"Child{i}", "children": []}
                for i in range(10)
            ]
        }

        graph = HierarchyGraph.from_hierarchy(wide)
        assert len(graph.nodes) == 11
        assert len(graph.edges) == 10

        result = compute_structure_similarity(wide, wide)
        assert result.overall_similarity == 1.0

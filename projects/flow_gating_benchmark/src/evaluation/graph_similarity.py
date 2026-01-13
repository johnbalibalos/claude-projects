"""
Graph similarity metrics for gating hierarchy comparison.

This module provides flexible structure matching that accounts for:
- Equivalent groupings (lineage-based vs activation-based)
- Different but valid orderings of sibling gates
- Semantic equivalence of node labels

Usage:
    from evaluation.graph_similarity import (
        compute_tree_edit_distance,
        compute_graph_similarity,
        HierarchyGraph,
    )

    graph1 = HierarchyGraph.from_hierarchy(hierarchy1)
    graph2 = HierarchyGraph.from_hierarchy(hierarchy2)
    similarity = compute_graph_similarity(graph1, graph2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .normalization import normalize_gate_name

if TYPE_CHECKING:
    from .semantic_similarity import SemanticMatcher


@dataclass
class TreeNode:
    """Node in a tree structure for edit distance computation."""
    label: str
    children: list[TreeNode] = field(default_factory=list)
    depth: int = 0
    index: int = 0  # Post-order index

    def __hash__(self) -> int:
        return hash((self.label, self.depth, self.index))


@dataclass
class HierarchyGraph:
    """
    Graph representation of a gating hierarchy.

    Supports both tree operations (edit distance) and
    graph operations (isomorphism, similarity).
    """
    nodes: list[str]
    edges: list[tuple[str, str]]  # (parent, child) pairs
    root: str | None = None
    node_depths: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_hierarchy(cls, hierarchy: dict) -> HierarchyGraph:
        """Build graph from hierarchy dict."""
        nodes = []
        edges = []
        node_depths = {}

        def traverse(node: dict, parent: str | None = None, depth: int = 0) -> None:
            name = node.get("name", "")
            if not name:
                return

            nodes.append(name)
            node_depths[name] = depth

            if parent is not None:
                edges.append((parent, name))

            for child in node.get("children", []):
                traverse(child, name, depth + 1)

        # Handle different hierarchy formats
        if "root" in hierarchy:
            traverse(hierarchy["root"])
            root = hierarchy["root"].get("name")
        elif "name" in hierarchy:
            traverse(hierarchy)
            root = hierarchy.get("name")
        else:
            root = None

        return cls(
            nodes=nodes,
            edges=edges,
            root=root,
            node_depths=node_depths,
        )

    def to_tree(self) -> TreeNode | None:
        """Convert to TreeNode structure for edit distance."""
        if not self.root:
            return None

        # Build adjacency list
        children_map: dict[str, list[str]] = {n: [] for n in self.nodes}
        for parent, child in self.edges:
            children_map[parent].append(child)

        # Build tree recursively
        index_counter = [0]

        def build_node(label: str, depth: int) -> TreeNode:
            children = [
                build_node(c, depth + 1)
                for c in children_map.get(label, [])
            ]
            node = TreeNode(
                label=label,
                children=children,
                depth=depth,
                index=index_counter[0],
            )
            index_counter[0] += 1
            return node

        return build_node(self.root, 0)

    def get_ancestors(self, node: str) -> list[str]:
        """Get all ancestors of a node (from root to parent)."""
        # Build parent map
        parent_map = {child: parent for parent, child in self.edges}

        ancestors = []
        current = node
        while current in parent_map:
            parent = parent_map[current]
            ancestors.append(parent)
            current = parent

        return list(reversed(ancestors))

    def get_descendants(self, node: str) -> set[str]:
        """Get all descendants of a node."""
        children_map: dict[str, list[str]] = {n: [] for n in self.nodes}
        for parent, child in self.edges:
            children_map[parent].append(child)

        descendants = set()

        def collect(n: str) -> None:
            for child in children_map.get(n, []):
                descendants.add(child)
                collect(child)

        collect(node)
        return descendants

    def get_siblings(self, node: str) -> list[str]:
        """Get siblings of a node (same parent)."""
        parent_map = {child: parent for parent, child in self.edges}
        children_map: dict[str, list[str]] = {n: [] for n in self.nodes}
        for parent, child in self.edges:
            children_map[parent].append(child)

        if node not in parent_map:
            return []

        parent = parent_map[node]
        return [c for c in children_map.get(parent, []) if c != node]


def _compute_ted_recursive(
    t1: TreeNode | None,
    t2: TreeNode | None,
    node_cost: Callable[[str, str], float],
    delete_cost: float = 1.0,
    insert_cost: float = 1.0,
    memo: dict | None = None,
) -> float:
    """
    Compute tree edit distance using simplified recursive algorithm.

    This is a simplified version suitable for small trees (< 100 nodes).
    For larger trees, use the Zhang-Shasha algorithm.

    Args:
        t1: First tree (or None)
        t2: Second tree (or None)
        node_cost: Function returning cost of relabeling node1 to node2
        delete_cost: Cost of deleting a node
        insert_cost: Cost of inserting a node
        memo: Memoization dict (created if None)

    Returns:
        Edit distance (lower is more similar)
    """
    if memo is None:
        memo = {}

    # Base cases
    if t1 is None and t2 is None:
        return 0.0
    if t1 is None:
        # Cost to insert t2 and all its descendants
        return insert_cost + sum(
            _compute_ted_recursive(None, c, node_cost, delete_cost, insert_cost, memo)
            for c in t2.children
        )
    if t2 is None:
        # Cost to delete t1 and all its descendants
        return delete_cost + sum(
            _compute_ted_recursive(c, None, node_cost, delete_cost, insert_cost, memo)
            for c in t1.children
        )

    # Memoization key
    key = (id(t1), id(t2))
    if key in memo:
        return memo[key]

    # Option 1: Match t1 root with t2 root
    relabel_cost = node_cost(t1.label, t2.label)
    children_cost = _match_children_optimal(
        t1.children, t2.children, node_cost, delete_cost, insert_cost, memo
    )
    match_cost = relabel_cost + children_cost

    # Option 2: Delete t1 root
    delete_cost_total = delete_cost + sum(
        _compute_ted_recursive(c, t2, node_cost, delete_cost, insert_cost, memo)
        for c in t1.children
    )

    # Option 3: Insert t2 root
    insert_cost_total = insert_cost + sum(
        _compute_ted_recursive(t1, c, node_cost, delete_cost, insert_cost, memo)
        for c in t2.children
    )

    result = min(match_cost, delete_cost_total, insert_cost_total)
    memo[key] = result
    return result


def _match_children_optimal(
    children1: list[TreeNode],
    children2: list[TreeNode],
    node_cost: Callable[[str, str], float],
    delete_cost: float,
    insert_cost: float,
    memo: dict,
) -> float:
    """Match children using Hungarian algorithm for optimal assignment."""
    if not children1 and not children2:
        return 0.0
    if not children1:
        return sum(
            _compute_ted_recursive(None, c, node_cost, delete_cost, insert_cost, memo)
            for c in children2
        )
    if not children2:
        return sum(
            _compute_ted_recursive(c, None, node_cost, delete_cost, insert_cost, memo)
            for c in children1
        )

    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        n1, n2 = len(children1), len(children2)
        max_dim = max(n1, n2)

        # Build cost matrix
        cost_matrix = np.zeros((max_dim, max_dim))

        for i in range(max_dim):
            for j in range(max_dim):
                if i < n1 and j < n2:
                    cost_matrix[i, j] = _compute_ted_recursive(
                        children1[i], children2[j], node_cost,
                        delete_cost, insert_cost, memo
                    )
                elif i < n1:
                    # Delete child
                    cost_matrix[i, j] = _compute_ted_recursive(
                        children1[i], None, node_cost,
                        delete_cost, insert_cost, memo
                    )
                elif j < n2:
                    # Insert child
                    cost_matrix[i, j] = _compute_ted_recursive(
                        None, children2[j], node_cost,
                        delete_cost, insert_cost, memo
                    )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return float(cost_matrix[row_ind, col_ind].sum())

    except ImportError:
        # Fall back to greedy matching
        return _match_children_greedy(
            children1, children2, node_cost, delete_cost, insert_cost, memo
        )


def _match_children_greedy(
    children1: list[TreeNode],
    children2: list[TreeNode],
    node_cost: Callable[[str, str], float],
    delete_cost: float,
    insert_cost: float,
    memo: dict,
) -> float:
    """Greedy child matching fallback."""
    total_cost = 0.0
    remaining2 = list(children2)

    for c1 in children1:
        if not remaining2:
            total_cost += _compute_ted_recursive(
                c1, None, node_cost, delete_cost, insert_cost, memo
            )
            continue

        # Find best match
        best_cost = float('inf')
        best_idx = -1
        for idx, c2 in enumerate(remaining2):
            cost = _compute_ted_recursive(
                c1, c2, node_cost, delete_cost, insert_cost, memo
            )
            if cost < best_cost:
                best_cost = cost
                best_idx = idx

        # Check if deletion is cheaper
        del_cost = _compute_ted_recursive(
            c1, None, node_cost, delete_cost, insert_cost, memo
        )
        if del_cost < best_cost:
            total_cost += del_cost
        else:
            total_cost += best_cost
            remaining2.pop(best_idx)

    # Insert remaining children from tree2
    for c2 in remaining2:
        total_cost += _compute_ted_recursive(
            None, c2, node_cost, delete_cost, insert_cost, memo
        )

    return total_cost


def compute_tree_edit_distance(
    hierarchy1: dict | HierarchyGraph,
    hierarchy2: dict | HierarchyGraph,
    use_semantic_matching: bool = False,
    semantic_matcher: SemanticMatcher | None = None,
) -> float:
    """
    Compute tree edit distance between two hierarchies.

    Args:
        hierarchy1: First hierarchy (dict or HierarchyGraph)
        hierarchy2: Second hierarchy (dict or HierarchyGraph)
        use_semantic_matching: Whether to use semantic similarity for node matching
        semantic_matcher: Optional SemanticMatcher for semantic costs

    Returns:
        Edit distance (0 = identical, higher = more different)
    """
    if isinstance(hierarchy1, dict):
        graph1 = HierarchyGraph.from_hierarchy(hierarchy1)
    else:
        graph1 = hierarchy1

    if isinstance(hierarchy2, dict):
        graph2 = HierarchyGraph.from_hierarchy(hierarchy2)
    else:
        graph2 = hierarchy2

    tree1 = graph1.to_tree()
    tree2 = graph2.to_tree()

    if use_semantic_matching and semantic_matcher is not None:
        def node_cost(label1: str, label2: str) -> float:
            if label1.lower().strip() == label2.lower().strip():
                return 0.0
            sim = semantic_matcher.compute_similarity(label1, label2)
            # Convert similarity to cost (0 similarity = 1 cost, 1 similarity = 0 cost)
            return 1.0 - sim
    else:
        def node_cost(label1: str, label2: str) -> float:
            # Use normalized matching
            if normalize_gate_name(label1) == normalize_gate_name(label2):
                return 0.0
            return 1.0

    return _compute_ted_recursive(tree1, tree2, node_cost)


def compute_normalized_ted(
    hierarchy1: dict | HierarchyGraph,
    hierarchy2: dict | HierarchyGraph,
    use_semantic_matching: bool = False,
    semantic_matcher: SemanticMatcher | None = None,
) -> float:
    """
    Compute normalized tree edit distance (0-1 scale).

    Normalized by the size of the larger tree.

    Returns:
        Normalized distance (0 = identical, 1 = completely different)
    """
    if isinstance(hierarchy1, dict):
        graph1 = HierarchyGraph.from_hierarchy(hierarchy1)
    else:
        graph1 = hierarchy1

    if isinstance(hierarchy2, dict):
        graph2 = HierarchyGraph.from_hierarchy(hierarchy2)
    else:
        graph2 = hierarchy2

    ted = compute_tree_edit_distance(
        graph1, graph2, use_semantic_matching, semantic_matcher
    )

    max_size = max(len(graph1.nodes), len(graph2.nodes))
    if max_size == 0:
        return 0.0

    return min(1.0, ted / max_size)


def compute_tree_similarity(
    hierarchy1: dict | HierarchyGraph,
    hierarchy2: dict | HierarchyGraph,
    use_semantic_matching: bool = False,
    semantic_matcher: SemanticMatcher | None = None,
) -> float:
    """
    Compute tree similarity (1 - normalized edit distance).

    Args:
        hierarchy1: First hierarchy
        hierarchy2: Second hierarchy
        use_semantic_matching: Use semantic similarity for node matching
        semantic_matcher: Optional SemanticMatcher

    Returns:
        Similarity score (0-1, where 1 = identical)
    """
    normalized_ted = compute_normalized_ted(
        hierarchy1, hierarchy2, use_semantic_matching, semantic_matcher
    )
    return 1.0 - normalized_ted


@dataclass
class StructureSimilarityResult:
    """Result of structure similarity computation."""
    overall_similarity: float
    node_similarity: float  # Jaccard similarity of node sets
    edge_similarity: float  # Jaccard similarity of edge sets
    depth_similarity: float  # How similar are the depth distributions
    branching_similarity: float  # How similar are the branching patterns
    tree_edit_similarity: float  # 1 - normalized TED


def compute_structure_similarity(
    hierarchy1: dict | HierarchyGraph,
    hierarchy2: dict | HierarchyGraph,
    use_semantic_matching: bool = False,
    semantic_matcher: SemanticMatcher | None = None,
) -> StructureSimilarityResult:
    """
    Compute comprehensive structure similarity between hierarchies.

    Combines multiple similarity metrics for a nuanced comparison.

    Args:
        hierarchy1: First hierarchy
        hierarchy2: Second hierarchy
        use_semantic_matching: Use semantic similarity for matching
        semantic_matcher: Optional SemanticMatcher

    Returns:
        StructureSimilarityResult with component scores
    """
    if isinstance(hierarchy1, dict):
        graph1 = HierarchyGraph.from_hierarchy(hierarchy1)
    else:
        graph1 = hierarchy1

    if isinstance(hierarchy2, dict):
        graph2 = HierarchyGraph.from_hierarchy(hierarchy2)
    else:
        graph2 = hierarchy2

    # Node similarity (Jaccard)
    if use_semantic_matching and semantic_matcher is not None:
        node_sim = _semantic_node_similarity(graph1, graph2, semantic_matcher)
    else:
        nodes1 = {normalize_gate_name(n) for n in graph1.nodes}
        nodes2 = {normalize_gate_name(n) for n in graph2.nodes}
        intersection = len(nodes1 & nodes2)
        union = len(nodes1 | nodes2)
        node_sim = intersection / union if union > 0 else 1.0

    # Edge similarity (normalized)
    edge_sim = _compute_edge_similarity(graph1, graph2, use_semantic_matching, semantic_matcher)

    # Depth distribution similarity
    depth_sim = _compute_depth_similarity(graph1, graph2)

    # Branching pattern similarity
    branching_sim = _compute_branching_similarity(graph1, graph2)

    # Tree edit similarity
    ted_sim = compute_tree_similarity(
        graph1, graph2, use_semantic_matching, semantic_matcher
    )

    # Weighted overall similarity
    overall = (
        0.25 * node_sim +
        0.25 * edge_sim +
        0.15 * depth_sim +
        0.10 * branching_sim +
        0.25 * ted_sim
    )

    return StructureSimilarityResult(
        overall_similarity=overall,
        node_similarity=node_sim,
        edge_similarity=edge_sim,
        depth_similarity=depth_sim,
        branching_similarity=branching_sim,
        tree_edit_similarity=ted_sim,
    )


def _semantic_node_similarity(
    graph1: HierarchyGraph,
    graph2: HierarchyGraph,
    matcher: SemanticMatcher,
) -> float:
    """Compute node similarity using semantic matching."""
    from .semantic_similarity import compute_semantic_f1

    f1, _, _, _ = compute_semantic_f1(
        set(graph1.nodes), set(graph2.nodes), matcher
    )
    return f1


def _compute_edge_similarity(
    graph1: HierarchyGraph,
    graph2: HierarchyGraph,
    use_semantic: bool,
    matcher: SemanticMatcher | None,
) -> float:
    """Compute edge similarity with optional semantic matching."""
    if not graph1.edges and not graph2.edges:
        return 1.0
    if not graph1.edges or not graph2.edges:
        return 0.0

    def normalize_edge(edge: tuple[str, str]) -> tuple[str, str]:
        return (normalize_gate_name(edge[0]), normalize_gate_name(edge[1]))

    edges1 = {normalize_edge(e) for e in graph1.edges}
    edges2 = {normalize_edge(e) for e in graph2.edges}

    if use_semantic and matcher is not None:
        # Use semantic matching for edges
        matched = 0
        matched_edges2 = set()

        for e1 in edges1:
            best_match = None
            best_sim = 0.0

            for e2 in edges2:
                if e2 in matched_edges2:
                    continue

                # Both parent and child must match
                parent_sim = matcher.compute_similarity(e1[0], e2[0])
                child_sim = matcher.compute_similarity(e1[1], e2[1])
                edge_sim = min(parent_sim, child_sim)

                if edge_sim > best_sim and edge_sim >= matcher.medium_threshold:
                    best_sim = edge_sim
                    best_match = e2

            if best_match is not None:
                matched += 1
                matched_edges2.add(best_match)

        total = len(edges1) + len(edges2) - matched
        return matched / total if total > 0 else 1.0
    else:
        intersection = len(edges1 & edges2)
        union = len(edges1 | edges2)
        return intersection / union if union > 0 else 1.0


def _compute_depth_similarity(graph1: HierarchyGraph, graph2: HierarchyGraph) -> float:
    """Compare depth distributions of two hierarchies."""
    if not graph1.node_depths and not graph2.node_depths:
        return 1.0

    max_depth1 = max(graph1.node_depths.values()) if graph1.node_depths else 0
    max_depth2 = max(graph2.node_depths.values()) if graph2.node_depths else 0

    if max_depth1 == 0 and max_depth2 == 0:
        return 1.0

    # Compare depth distributions
    max_depth = max(max_depth1, max_depth2) + 1

    def get_depth_counts(depths: dict[str, int]) -> list[int]:
        counts = [0] * max_depth
        for d in depths.values():
            counts[d] += 1
        return counts

    counts1 = get_depth_counts(graph1.node_depths)
    counts2 = get_depth_counts(graph2.node_depths)

    # Normalized histogram intersection
    total1 = sum(counts1)
    total2 = sum(counts2)

    if total1 == 0 or total2 == 0:
        return 0.0

    norm1 = [c / total1 for c in counts1]
    norm2 = [c / total2 for c in counts2]

    # Histogram intersection
    intersection = sum(min(n1, n2) for n1, n2 in zip(norm1, norm2))
    return intersection


def _compute_branching_similarity(graph1: HierarchyGraph, graph2: HierarchyGraph) -> float:
    """Compare branching patterns (number of children per node)."""
    def get_branching_factors(graph: HierarchyGraph) -> list[int]:
        children_count: dict[str, int] = {n: 0 for n in graph.nodes}
        for parent, _ in graph.edges:
            children_count[parent] += 1
        return list(children_count.values())

    bf1 = get_branching_factors(graph1)
    bf2 = get_branching_factors(graph2)

    if not bf1 and not bf2:
        return 1.0
    if not bf1 or not bf2:
        return 0.0

    # Compare average branching factor
    avg1 = sum(bf1) / len(bf1)
    avg2 = sum(bf2) / len(bf2)

    max_avg = max(avg1, avg2)
    if max_avg == 0:
        return 1.0

    return 1.0 - abs(avg1 - avg2) / max_avg


def is_subtree(
    hierarchy: dict | HierarchyGraph,
    candidate_subtree: dict | HierarchyGraph,
    use_semantic_matching: bool = False,
    semantic_matcher: SemanticMatcher | None = None,
) -> bool:
    """
    Check if candidate_subtree is a subtree of hierarchy.

    A subtree matches if all nodes and edges in the candidate
    exist in the main hierarchy (with semantic matching if enabled).

    Args:
        hierarchy: Main hierarchy to search in
        candidate_subtree: Potential subtree
        use_semantic_matching: Use semantic similarity
        semantic_matcher: Optional SemanticMatcher

    Returns:
        True if candidate is a subtree of hierarchy
    """
    if isinstance(hierarchy, dict):
        graph = HierarchyGraph.from_hierarchy(hierarchy)
    else:
        graph = hierarchy

    if isinstance(candidate_subtree, dict):
        subtree = HierarchyGraph.from_hierarchy(candidate_subtree)
    else:
        subtree = candidate_subtree

    if not subtree.nodes:
        return True

    # Check all subtree nodes exist in main graph
    if use_semantic_matching and semantic_matcher is not None:
        for snode in subtree.nodes:
            found = False
            for gnode in graph.nodes:
                sim = semantic_matcher.compute_similarity(snode, gnode)
                if sim >= semantic_matcher.high_threshold:
                    found = True
                    break
            if not found:
                return False
    else:
        main_nodes = {normalize_gate_name(n) for n in graph.nodes}
        for snode in subtree.nodes:
            if normalize_gate_name(snode) not in main_nodes:
                return False

    # Check all subtree edges exist in main graph (containment, not similarity)
    def normalize_edge(edge: tuple[str, str]) -> tuple[str, str]:
        return (normalize_gate_name(edge[0]), normalize_gate_name(edge[1]))

    main_edges = {normalize_edge(e) for e in graph.edges}

    if use_semantic_matching and semantic_matcher is not None:
        # For semantic matching, check if each subtree edge has a match
        for s_edge in subtree.edges:
            found = False
            for m_edge in graph.edges:
                parent_sim = semantic_matcher.compute_similarity(s_edge[0], m_edge[0])
                child_sim = semantic_matcher.compute_similarity(s_edge[1], m_edge[1])
                if parent_sim >= semantic_matcher.high_threshold and child_sim >= semantic_matcher.high_threshold:
                    found = True
                    break
            if not found:
                return False
        return True
    else:
        # For exact matching, check all subtree edges exist in main graph
        subtree_edges = {normalize_edge(e) for e in subtree.edges}
        return subtree_edges.issubset(main_edges)

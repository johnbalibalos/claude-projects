"""
Hierarchy tree operations for gating hierarchies.

Provides utilities for extracting, traversing, and comparing
gating hierarchy structures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from curation.schemas import GateNode, GatingHierarchy


def extract_gate_names(hierarchy: GatingHierarchy | dict) -> set[str]:
    """
    Extract all gate names from a hierarchy.

    Args:
        hierarchy: GatingHierarchy object or dict representation

    Returns:
        Set of gate names
    """
    gates: set[str] = set()

    def traverse_node(node: GateNode) -> None:
        gates.add(node.name)
        for child in node.children:
            traverse_node(child)

    def traverse_dict(node: dict) -> None:
        if "name" in node:
            gates.add(node["name"])
        for child in node.get("children", []):
            traverse_dict(child)

    # Handle GatingHierarchy object
    if hasattr(hierarchy, 'root'):
        traverse_node(hierarchy.root)
    # Handle dict with root key
    elif isinstance(hierarchy, dict) and "root" in hierarchy:
        traverse_dict(hierarchy["root"])
    # Handle dict that is the root node itself
    elif isinstance(hierarchy, dict) and "name" in hierarchy:
        traverse_dict(hierarchy)

    return gates


def extract_parent_map(hierarchy: GatingHierarchy | dict) -> dict[str, str | None]:
    """
    Extract parent-child relationships from hierarchy.

    Args:
        hierarchy: GatingHierarchy object or dict

    Returns:
        Dict mapping gate name to parent name (None for root)
    """
    parent_map: dict[str, str | None] = {}

    def traverse_node(node: GateNode, parent: str | None = None) -> None:
        parent_map[node.name] = parent
        for child in node.children:
            traverse_node(child, node.name)

    def traverse_dict(node: dict, parent: str | None = None) -> None:
        if "name" in node:
            parent_map[node["name"]] = parent
            for child in node.get("children", []):
                traverse_dict(child, node["name"])

    if hasattr(hierarchy, 'root'):
        traverse_node(hierarchy.root)
    elif isinstance(hierarchy, dict) and "root" in hierarchy:
        traverse_dict(hierarchy["root"])
    elif isinstance(hierarchy, dict) and "name" in hierarchy:
        traverse_dict(hierarchy)

    return parent_map


def get_hierarchy_depth(hierarchy: GatingHierarchy | dict) -> int:
    """
    Get maximum depth of the hierarchy.

    Args:
        hierarchy: GatingHierarchy object or dict

    Returns:
        Maximum depth (0 for empty/leaf-only hierarchies)
    """
    def get_depth(node: GateNode | dict, current: int = 0) -> int:
        if hasattr(node, 'children'):
            children = node.children
        else:
            children = node.get("children", [])

        if not children:
            return current

        return max(get_depth(child, current + 1) for child in children)

    if hasattr(hierarchy, 'root'):
        return get_depth(hierarchy.root)
    elif isinstance(hierarchy, dict) and "root" in hierarchy:
        return get_depth(hierarchy["root"])
    elif isinstance(hierarchy, dict) and "name" in hierarchy:
        return get_depth(hierarchy)
    return 0


def count_gates(hierarchy: GatingHierarchy | dict) -> int:
    """
    Count total number of gates in hierarchy.

    Args:
        hierarchy: GatingHierarchy object or dict

    Returns:
        Total gate count
    """
    return len(extract_gate_names(hierarchy))


def get_leaf_gates(hierarchy: GatingHierarchy | dict) -> set[str]:
    """
    Get gates that have no children (leaf nodes).

    Args:
        hierarchy: GatingHierarchy object or dict

    Returns:
        Set of leaf gate names
    """
    leaves: set[str] = set()

    def traverse_node(node: GateNode) -> None:
        if not node.children:
            leaves.add(node.name)
        else:
            for child in node.children:
                traverse_node(child)

    def traverse_dict(node: dict) -> None:
        if "name" in node:
            children = node.get("children", [])
            if not children:
                leaves.add(node["name"])
            else:
                for child in children:
                    traverse_dict(child)

    if hasattr(hierarchy, 'root'):
        traverse_node(hierarchy.root)
    elif isinstance(hierarchy, dict) and "root" in hierarchy:
        traverse_dict(hierarchy["root"])
    elif isinstance(hierarchy, dict) and "name" in hierarchy:
        traverse_dict(hierarchy)

    return leaves


def find_gate_path(hierarchy: GatingHierarchy | dict, gate_name: str) -> list[str] | None:
    """
    Find the path from root to a specific gate.

    Args:
        hierarchy: GatingHierarchy object or dict
        gate_name: Name of gate to find

    Returns:
        List of gate names from root to target, or None if not found
    """
    def search_node(node: GateNode, path: list[str]) -> list[str] | None:
        current_path = path + [node.name]
        if node.name == gate_name:
            return current_path
        for child in node.children:
            result = search_node(child, current_path)
            if result is not None:
                return result
        return None

    def search_dict(node: dict, path: list[str]) -> list[str] | None:
        if "name" not in node:
            return None
        current_path = path + [node["name"]]
        if node["name"] == gate_name:
            return current_path
        for child in node.get("children", []):
            result = search_dict(child, current_path)
            if result is not None:
                return result
        return None

    if hasattr(hierarchy, 'root'):
        return search_node(hierarchy.root, [])
    elif isinstance(hierarchy, dict) and "root" in hierarchy:
        return search_dict(hierarchy["root"], [])
    elif isinstance(hierarchy, dict) and "name" in hierarchy:
        return search_dict(hierarchy, [])
    return None

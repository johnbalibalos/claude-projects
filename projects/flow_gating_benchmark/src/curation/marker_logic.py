"""
Core algorithms for building gating hierarchies from marker phenotype tables.

This module provides functions to:
1. Parse marker tables with +/- notation
2. Infer parent-child relationships from marker subset logic
3. Build GatingHierarchy trees from tabular data
4. Validate hierarchies against panels and HIPC standards

Example usage:
    from curation.marker_logic import marker_table_to_hierarchy, parse_marker_table

    # From markdown table
    table_text = '''
    | Population | CD3 | CD4 | CD8 | CD19 | Parent |
    |------------|-----|-----|-----|------|--------|
    | T cells    | +   |     |     | -    | CD45+  |
    | CD4+ T     | +   | +   | -   |      | T cells|
    '''
    entries = parse_marker_table(table_text, format='markdown')
    hierarchy = marker_table_to_hierarchy(entries, panel_markers=['CD3', 'CD4', 'CD8', 'CD19'])
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from .schemas import (
    GateNode,
    GatingHierarchy,
    MarkerExpression,
    Panel,
)


@dataclass
class MarkerTableEntry:
    """
    A row from a marker phenotype table.

    Attributes:
        population: Cell population name (e.g., "CD4+ T cells")
        markers: Dict mapping marker names to expression states
                 Values: '+', '-', 'dim', 'bright', 'high', 'low', or '' (unspecified)
        parent: Optional explicit parent population name
        notes: Optional notes about this population
    """
    population: str
    markers: dict[str, str] = field(default_factory=dict)
    parent: str | None = None
    notes: str | None = None

    def get_positive_markers(self) -> list[str]:
        """Get markers that are positive (+, bright, high)."""
        return [m for m, s in self.markers.items()
                if s in ('+', 'bright', 'high', 'pos', 'positive')]

    def get_negative_markers(self) -> list[str]:
        """Get markers that are negative (-, dim for some contexts)."""
        return [m for m, s in self.markers.items()
                if s in ('-', 'neg', 'negative')]

    def get_marker_signature(self) -> tuple[frozenset[str], frozenset[str]]:
        """Return (positive_markers, negative_markers) as frozen sets."""
        return (
            frozenset(self.get_positive_markers()),
            frozenset(self.get_negative_markers())
        )

    def to_marker_logic(self) -> list[MarkerExpression]:
        """Convert to list of MarkerExpression objects."""
        expressions = []
        for marker, state in self.markers.items():
            if not state or state == '':
                continue

            if state in ('+', 'pos', 'positive'):
                expressions.append(MarkerExpression(marker=marker, positive=True))
            elif state in ('-', 'neg', 'negative'):
                expressions.append(MarkerExpression(marker=marker, positive=False))
            elif state in ('dim', 'low'):
                expressions.append(MarkerExpression(marker=marker, positive=True, level='dim'))
            elif state in ('bright', 'high'):
                expressions.append(MarkerExpression(marker=marker, positive=True, level='bright'))

        return expressions


def parse_marker_table(
    table_text: str,
    format: Literal['markdown', 'csv', 'auto'] = 'auto'
) -> list[MarkerTableEntry]:
    """
    Parse a marker phenotype table from text.

    Supports markdown tables and CSV format. Auto-detects format if not specified.

    Args:
        table_text: Raw table text (markdown or CSV)
        format: Table format ('markdown', 'csv', or 'auto')

    Returns:
        List of MarkerTableEntry objects

    Example markdown input:
        | Population | CD3 | CD4 | CD8 | Parent |
        |------------|-----|-----|-----|--------|
        | T cells    | +   |     | -   | CD45+  |
    """
    table_text = table_text.strip()

    # Auto-detect format
    if format == 'auto':
        if '|' in table_text and table_text.startswith('|'):
            format = 'markdown'
        else:
            format = 'csv'

    if format == 'markdown':
        return _parse_markdown_table(table_text)
    else:
        return _parse_csv_table(table_text)


def _parse_markdown_table(table_text: str) -> list[MarkerTableEntry]:
    """Parse a markdown-formatted table."""
    lines = [l.strip() for l in table_text.strip().split('\n') if l.strip()]

    if len(lines) < 2:
        return []

    # Parse header
    header_line = lines[0]
    headers = [h.strip() for h in header_line.split('|') if h.strip()]

    # Find population column and marker columns
    pop_col = None
    parent_col = None
    notes_col = None
    marker_cols = {}  # index -> marker name

    for i, h in enumerate(headers):
        h_lower = h.lower()
        if h_lower in ('population', 'cell type', 'subset', 'name'):
            pop_col = i
        elif h_lower in ('parent', 'parent population', 'gated from'):
            parent_col = i
        elif h_lower in ('notes', 'description', 'comment'):
            notes_col = i
        else:
            # Assume it's a marker column
            marker_cols[i] = h

    if pop_col is None:
        # Default to first column
        pop_col = 0

    entries = []

    # Skip separator line (contains dashes)
    data_start = 1
    if len(lines) > 1 and set(lines[1].replace('|', '').replace('-', '').replace(' ', '')) == set():
        data_start = 2

    for line in lines[data_start:]:
        cells = [c.strip() for c in line.split('|') if c.strip() or line.count('|') > len(headers)]

        # Handle lines that start/end with |
        if line.startswith('|'):
            cells = [c.strip() for c in line.split('|')[1:-1]]

        if len(cells) < len(headers):
            # Pad with empty strings
            cells.extend([''] * (len(headers) - len(cells)))

        if not cells or not cells[pop_col]:
            continue

        population = cells[pop_col]
        parent = cells[parent_col] if parent_col is not None and parent_col < len(cells) else None
        notes = cells[notes_col] if notes_col is not None and notes_col < len(cells) else None

        markers = {}
        for idx, marker_name in marker_cols.items():
            if idx < len(cells):
                markers[marker_name] = _normalize_marker_state(cells[idx])

        entries.append(MarkerTableEntry(
            population=population,
            markers=markers,
            parent=parent if parent else None,
            notes=notes if notes else None
        ))

    return entries


def _parse_csv_table(table_text: str) -> list[MarkerTableEntry]:
    """Parse a CSV-formatted table."""
    import csv
    from io import StringIO

    reader = csv.DictReader(StringIO(table_text))
    entries = []

    for row in reader:
        # Find population column
        pop_value = None
        parent_value = None
        markers = {}

        for key, value in row.items():
            key_lower = key.lower().strip()
            if key_lower in ('population', 'cell type', 'subset', 'name'):
                pop_value = value
            elif key_lower in ('parent', 'parent population'):
                parent_value = value
            else:
                # Marker column
                markers[key.strip()] = _normalize_marker_state(value)

        if pop_value:
            entries.append(MarkerTableEntry(
                population=pop_value,
                markers=markers,
                parent=parent_value if parent_value else None
            ))

    return entries


def _normalize_marker_state(state: str) -> str:
    """Normalize marker state to standard values."""
    state = state.strip().lower()

    if state in ('+', 'pos', 'positive', '1', 'yes', 'true'):
        return '+'
    elif state in ('-', 'neg', 'negative', '0', 'no', 'false'):
        return '-'
    elif state in ('dim', 'lo', 'low'):
        return 'dim'
    elif state in ('bright', 'hi', 'high'):
        return 'bright'
    elif state in ('', 'na', 'n/a', '.', 'x'):
        return ''
    else:
        # Return as-is for unrecognized states
        return state


def infer_parent_from_markers(
    child: MarkerTableEntry,
    all_entries: list[MarkerTableEntry]
) -> str | None:
    """
    Infer the parent population based on marker subset relationships.

    Rule: A is parent of B if:
    1. A's positive markers are a proper subset of B's positive markers, OR
    2. A's negative markers are a proper subset of B's negative markers
    3. A is the most specific (most markers) candidate

    Example:
        T cells (CD3+) is parent of CD4+ T (CD3+ CD4+)
        because {CD3} ⊂ {CD3, CD4}

    Args:
        child: The entry to find a parent for
        all_entries: All entries to search for potential parents

    Returns:
        Parent population name, or None if no parent found
    """
    child_pos, child_neg = child.get_marker_signature()

    candidates = []

    for other in all_entries:
        if other.population == child.population:
            continue

        other_pos, other_neg = other.get_marker_signature()

        # Check if other could be parent (subset relationship)
        # Parent's positive markers must be subset of child's
        pos_subset = other_pos <= child_pos
        # Parent's negative markers must be subset of child's
        neg_subset = other_neg <= child_neg

        if not (pos_subset and neg_subset):
            continue

        # Must be a PROPER subset (not equal)
        is_proper_subset = (other_pos < child_pos) or (other_neg < child_neg)

        if not is_proper_subset:
            continue

        # Score by specificity (more markers = more specific = better parent)
        specificity = len(other_pos) + len(other_neg)
        candidates.append((other.population, specificity))

    if not candidates:
        return None

    # Return most specific parent (highest marker count)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def marker_table_to_hierarchy(
    entries: list[MarkerTableEntry],
    panel_markers: list[str] | None = None,
    infer_parents: bool = True,
    add_standard_gates: bool = True
) -> GatingHierarchy:
    """
    Convert a marker phenotype table to a gating hierarchy tree.

    Args:
        entries: List of MarkerTableEntry objects
        panel_markers: Optional list of valid panel markers (for validation)
        infer_parents: If True, infer parent-child relationships from marker subsets
                      when explicit parent is not provided
        add_standard_gates: If True, add standard preprocessing gates
                           (All Events → Time → Singlets → Live)

    Returns:
        GatingHierarchy with root node and nested children

    Example:
        entries = [
            MarkerTableEntry("T cells", {"CD3": "+", "CD19": "-"}, parent="CD45+"),
            MarkerTableEntry("CD4+ T", {"CD3": "+", "CD4": "+", "CD8": "-"}, parent="T cells"),
        ]
        hierarchy = marker_table_to_hierarchy(entries)
    """
    # Step 1: Infer missing parents if requested
    if infer_parents:
        for entry in entries:
            if entry.parent is None:
                entry.parent = infer_parent_from_markers(entry, entries)

    # Step 2: Create GateNode for each entry
    nodes: dict[str, GateNode] = {}

    for entry in entries:
        marker_logic = entry.to_marker_logic()

        # Extract markers used (for legacy field)
        markers_used = list(entry.markers.keys())

        nodes[entry.population] = GateNode(
            name=entry.population,
            markers=markers_used,
            marker_logic=marker_logic,
            children=[],
            notes=entry.notes
        )

    # Step 3: Build tree structure
    # Find nodes without parents in our set (roots of our extracted data)
    root_candidates = []

    for entry in entries:
        node = nodes[entry.population]
        parent_name = entry.parent

        if parent_name and parent_name in nodes:
            # Add as child of existing node
            nodes[parent_name].children.append(node)
        else:
            # This is a root or has external parent
            root_candidates.append((entry, node))

    # Step 4: Create final hierarchy
    if add_standard_gates:
        # Standard flow cytometry preprocessing gates
        root = GateNode(
            name="All Events",
            children=[
                GateNode(
                    name="Time",
                    markers=["Time"],
                    is_critical=True,
                    children=[
                        GateNode(
                            name="Singlets",
                            markers=["FSC-A", "FSC-H"],
                            is_critical=True,
                            children=[
                                GateNode(
                                    name="Live",
                                    markers=["Live/Dead"],
                                    marker_logic=[MarkerExpression(marker="Live/Dead", positive=False)],
                                    is_critical=True,
                                    children=[]
                                )
                            ]
                        )
                    ]
                )
            ]
        )

        # Find where to attach extracted populations
        live_gate = root.children[0].children[0].children[0]  # Live gate

        # Group by parent
        parent_groups: dict[str | None, list[GateNode]] = {}
        for entry, node in root_candidates:
            parent = entry.parent
            if parent not in parent_groups:
                parent_groups[parent] = []
            parent_groups[parent].append(node)

        # Attach based on parent
        for parent_name, nodes_list in parent_groups.items():
            if parent_name in ('CD45+', 'Leukocytes', 'CD45+ cells'):
                # Create CD45+ gate if needed
                cd45_gate = GateNode(
                    name="CD45+",
                    markers=["CD45"],
                    marker_logic=[MarkerExpression(marker="CD45", positive=True)],
                    is_critical=True,
                    children=nodes_list
                )
                live_gate.children.append(cd45_gate)
            elif parent_name in ('Lymphocytes', 'Lymphs'):
                # Lymphocyte gate (less common in modern protocols)
                lymph_gate = GateNode(
                    name="Lymphocytes",
                    markers=["FSC-A", "SSC-A"],
                    children=nodes_list
                )
                live_gate.children.append(lymph_gate)
            elif parent_name is None or parent_name in ('Live', 'Live cells'):
                # Attach directly to Live
                live_gate.children.extend(nodes_list)
            else:
                # Unknown parent - create stub gate
                stub_gate = GateNode(
                    name=parent_name,
                    children=nodes_list
                )
                live_gate.children.append(stub_gate)
    else:
        # Simple tree without standard gates
        root = GateNode(name="All Events", children=[])

        for entry, node in root_candidates:
            if entry.parent and entry.parent not in nodes:
                # Create stub parent
                stub = GateNode(name=entry.parent, children=[node])
                root.children.append(stub)
            else:
                root.children.append(node)

    return GatingHierarchy(root=root)


def validate_hierarchy_markers(
    hierarchy: GatingHierarchy,
    panel: Panel
) -> list[str]:
    """
    Validate that all markers in hierarchy are present in panel.

    Args:
        hierarchy: GatingHierarchy to validate
        panel: Panel with marker definitions

    Returns:
        List of validation error messages (empty if valid)
    """
    panel_markers = set(panel.markers)
    # Add standard markers that don't need to be in panel
    standard_markers = {'FSC-A', 'FSC-H', 'SSC-A', 'SSC-H', 'Time', 'Live/Dead'}
    valid_markers = panel_markers | standard_markers

    errors = []

    def check_node(node: GateNode, path: str = ""):
        current_path = f"{path}/{node.name}" if path else node.name

        # Check marker_logic
        for expr in node.marker_logic:
            if expr.marker not in valid_markers:
                errors.append(
                    f"Unknown marker '{expr.marker}' in gate '{current_path}'"
                )

        # Check legacy markers field
        for marker in node.markers:
            if marker not in valid_markers:
                errors.append(
                    f"Unknown marker '{marker}' in gate '{current_path}' (markers field)"
                )

        # Recurse to children
        for child in node.children:
            check_node(child, current_path)

    check_node(hierarchy.root)
    return errors


def validate_hierarchy_structure(
    hierarchy: GatingHierarchy
) -> list[str]:
    """
    Validate hierarchy structure (no cycles, proper nesting, etc.).

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    seen_names: set[str] = set()

    def check_node(node: GateNode, ancestors: set[str], path: str = ""):
        current_path = f"{path}/{node.name}" if path else node.name

        # Check for duplicate names
        if node.name in seen_names:
            errors.append(f"Duplicate gate name: '{node.name}'")
        seen_names.add(node.name)

        # Check for cycles
        if node.name in ancestors:
            errors.append(f"Cycle detected: '{node.name}' appears in its own ancestry")
            return

        # Recurse
        new_ancestors = ancestors | {node.name}
        for child in node.children:
            check_node(child, new_ancestors, current_path)

    check_node(hierarchy.root, set())
    return errors


# Reference data for HIPC-standardized cell types
HIPC_CELL_TYPES: dict[str, dict] = {}

def load_hipc_definitions(path: str = "data/reference/hipc_2016_definitions.json") -> None:
    """Load HIPC cell type definitions from JSON file."""
    global HIPC_CELL_TYPES
    try:
        with open(path) as f:
            HIPC_CELL_TYPES = json.load(f)
    except FileNotFoundError:
        HIPC_CELL_TYPES = {}


def lookup_cell_type(name: str) -> dict | None:
    """
    Look up HIPC-standardized definition for a cell type.

    Args:
        name: Cell type name (e.g., "CD4+ T cells", "NK cells")

    Returns:
        Dict with positive_markers, negative_markers, parent_population, etc.
        or None if not found
    """
    if not HIPC_CELL_TYPES:
        load_hipc_definitions()

    name_lower = name.lower().strip()

    # Direct lookup
    for key, definition in HIPC_CELL_TYPES.items():
        if definition.get('name', '').lower() == name_lower:
            return definition

        # Check canonical names
        for canonical in definition.get('canonical_names', []):
            if canonical.lower() == name_lower:
                return definition

    return None


def suggest_marker_logic(
    population_name: str,
    available_markers: list[str]
) -> list[MarkerExpression] | None:
    """
    Suggest marker logic for a population based on HIPC definitions.

    Only suggests markers that are available in the panel.

    Args:
        population_name: Cell population name
        available_markers: Markers available in the panel

    Returns:
        List of MarkerExpression or None if no suggestion
    """
    definition = lookup_cell_type(population_name)
    if not definition:
        return None

    available_set = set(m.upper() for m in available_markers)
    expressions = []

    # Add positive markers
    for marker in definition.get('positive_markers', []):
        if marker.upper() in available_set:
            expressions.append(MarkerExpression(marker=marker, positive=True))

    # Add negative markers
    for marker in definition.get('negative_markers', []):
        if marker.upper() in available_set:
            expressions.append(MarkerExpression(marker=marker, positive=False))

    return expressions if expressions else None


def entries_from_hipc_populations(
    populations: list[str],
    panel_markers: list[str]
) -> list[MarkerTableEntry]:
    """
    Generate MarkerTableEntry objects from HIPC population names.

    Useful for creating hierarchies from a list of target populations.

    Args:
        populations: List of population names (e.g., ["T cells", "CD4+ T cells"])
        panel_markers: Available markers in panel

    Returns:
        List of MarkerTableEntry with HIPC-standardized marker logic
    """
    entries = []
    available_set = set(m.upper() for m in panel_markers)

    for pop_name in populations:
        definition = lookup_cell_type(pop_name)
        if not definition:
            # Create empty entry
            entries.append(MarkerTableEntry(population=pop_name))
            continue

        markers = {}

        # Add positive markers
        for marker in definition.get('positive_markers', []):
            if marker.upper() in available_set:
                markers[marker] = '+'

        # Add negative markers
        for marker in definition.get('negative_markers', []):
            if marker.upper() in available_set:
                markers[marker] = '-'

        parent = definition.get('parent_population')

        entries.append(MarkerTableEntry(
            population=definition.get('name', pop_name),
            markers=markers,
            parent=parent,
            notes=definition.get('notes')
        ))

    return entries

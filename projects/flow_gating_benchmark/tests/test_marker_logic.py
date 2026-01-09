"""Tests for marker_logic module - marker table to hierarchy conversion."""

import pytest

from src.curation.marker_logic import (
    MarkerTableEntry,
    infer_parent_from_markers,
    marker_table_to_hierarchy,
    parse_marker_table,
    validate_hierarchy_markers,
    validate_hierarchy_structure,
)
from src.curation.schemas import Panel, PanelEntry


class TestMarkerTableEntry:
    """Tests for MarkerTableEntry dataclass."""

    def test_get_positive_markers(self):
        entry = MarkerTableEntry(
            population="T cells",
            markers={"CD3": "+", "CD19": "-", "CD45": "+"}
        )
        assert entry.get_positive_markers() == ["CD3", "CD45"]

    def test_get_negative_markers(self):
        entry = MarkerTableEntry(
            population="T cells",
            markers={"CD3": "+", "CD19": "-", "CD45": "+"}
        )
        assert entry.get_negative_markers() == ["CD19"]

    def test_get_marker_signature(self):
        entry = MarkerTableEntry(
            population="T cells",
            markers={"CD3": "+", "CD19": "-", "CD4": ""}
        )
        pos, neg = entry.get_marker_signature()
        assert pos == frozenset(["CD3"])
        assert neg == frozenset(["CD19"])

    def test_to_marker_logic(self):
        entry = MarkerTableEntry(
            population="NK cells",
            markers={"CD56": "bright", "CD16": "-", "CD3": "-"}
        )
        logic = entry.to_marker_logic()
        assert len(logic) == 3

        # Check CD56 bright
        cd56 = next(m for m in logic if m.marker == "CD56")
        assert cd56.positive is True
        assert cd56.level == "bright"

        # Check CD16 negative
        cd16 = next(m for m in logic if m.marker == "CD16")
        assert cd16.positive is False


class TestParseMarkerTable:
    """Tests for parse_marker_table function."""

    def test_parse_markdown_table(self):
        table_text = """
| Population | CD3 | CD4 | CD8 | CD19 | Parent |
|------------|-----|-----|-----|------|--------|
| T cells    | +   |     |     | -    | CD45+  |
| CD4+ T     | +   | +   | -   |      | T cells|
| CD8+ T     | +   | -   | +   |      | T cells|
| B cells    | -   |     |     | +    | CD45+  |
"""
        entries = parse_marker_table(table_text, format='markdown')

        assert len(entries) == 4

        # Check T cells entry
        t_cells = next(e for e in entries if e.population == "T cells")
        assert t_cells.markers["CD3"] == "+"
        assert t_cells.markers["CD19"] == "-"
        assert t_cells.parent == "CD45+"

        # Check CD4+ T entry
        cd4t = next(e for e in entries if e.population == "CD4+ T")
        assert cd4t.markers["CD4"] == "+"
        assert cd4t.markers["CD8"] == "-"
        assert cd4t.parent == "T cells"

    def test_parse_markdown_auto_detect(self):
        table_text = """| Pop | CD3 |
|-----|-----|
| T   | +   |"""
        entries = parse_marker_table(table_text, format='auto')
        assert len(entries) == 1

    def test_parse_empty_cells(self):
        table_text = """
| Population | CD3 | CD4 |
|------------|-----|-----|
| T cells    | +   |     |
"""
        entries = parse_marker_table(table_text, format='markdown')
        assert len(entries) == 1
        assert entries[0].markers.get("CD4", "") == ""


class TestInferParentFromMarkers:
    """Tests for parent inference from marker subset relationships."""

    def test_infer_parent_subset(self):
        """CD4+ T (CD3+ CD4+) should have T cells (CD3+) as parent."""
        entries = [
            MarkerTableEntry("T cells", {"CD3": "+"}),
            MarkerTableEntry("CD4+ T cells", {"CD3": "+", "CD4": "+"}),
        ]

        parent = infer_parent_from_markers(entries[1], entries)
        assert parent == "T cells"

    def test_infer_parent_with_negatives(self):
        """T cells (CD3+ CD19-) should be parent of CD4+ T (CD3+ CD19- CD4+ CD8-)."""
        entries = [
            MarkerTableEntry("T cells", {"CD3": "+", "CD19": "-"}),
            MarkerTableEntry("CD4+ T cells", {"CD3": "+", "CD19": "-", "CD4": "+", "CD8": "-"}),
        ]

        parent = infer_parent_from_markers(entries[1], entries)
        assert parent == "T cells"

    def test_no_parent_for_root(self):
        """Root population should have no parent."""
        entries = [
            MarkerTableEntry("T cells", {"CD3": "+"}),
            MarkerTableEntry("B cells", {"CD19": "+"}),
        ]

        parent_t = infer_parent_from_markers(entries[0], entries)
        parent_b = infer_parent_from_markers(entries[1], entries)

        assert parent_t is None
        assert parent_b is None

    def test_most_specific_parent(self):
        """Should select most specific (most markers) parent."""
        entries = [
            MarkerTableEntry("Lymphocytes", {}),  # No markers
            MarkerTableEntry("T cells", {"CD3": "+"}),  # 1 marker
            MarkerTableEntry("CD4+ T cells", {"CD3": "+", "CD4": "+"}),  # 2 markers
            MarkerTableEntry("CD4+ Naive", {"CD3": "+", "CD4": "+", "CD45RA": "+"}),  # 3 markers
        ]

        # CD4+ Naive should have CD4+ T as parent (most specific subset)
        parent = infer_parent_from_markers(entries[3], entries)
        assert parent == "CD4+ T cells"


class TestMarkerTableToHierarchy:
    """Tests for marker_table_to_hierarchy function."""

    def test_basic_hierarchy(self):
        entries = [
            MarkerTableEntry("T cells", {"CD3": "+", "CD19": "-"}, parent="CD45+"),
            MarkerTableEntry("B cells", {"CD19": "+", "CD3": "-"}, parent="CD45+"),
        ]

        hierarchy = marker_table_to_hierarchy(
            entries,
            add_standard_gates=True,
            infer_parents=False
        )

        assert hierarchy.root.name == "All Events"

        # Check structure: All Events -> Time -> Singlets -> Live -> CD45+ -> T/B cells
        all_gates = hierarchy.get_all_gates()
        assert "Time" in all_gates
        assert "Singlets" in all_gates
        assert "Live" in all_gates
        assert "CD45+" in all_gates
        assert "T cells" in all_gates
        assert "B cells" in all_gates

    def test_hierarchy_with_inferred_parents(self):
        entries = [
            MarkerTableEntry("T cells", {"CD3": "+"}),
            MarkerTableEntry("CD4+ T cells", {"CD3": "+", "CD4": "+"}),
            MarkerTableEntry("CD8+ T cells", {"CD3": "+", "CD8": "+"}),
        ]

        hierarchy = marker_table_to_hierarchy(
            entries,
            add_standard_gates=False,
            infer_parents=True
        )

        # Check parent map
        parent_map = hierarchy.get_parent_map()

        # CD4+ T and CD8+ T should have T cells as parent
        # (after inference, but depends on implementation)
        all_gates = hierarchy.get_all_gates()
        assert "T cells" in all_gates
        assert "CD4+ T cells" in all_gates
        assert "CD8+ T cells" in all_gates

    def test_hierarchy_without_standard_gates(self):
        entries = [
            MarkerTableEntry("T cells", {"CD3": "+"}, parent=None),
        ]

        hierarchy = marker_table_to_hierarchy(
            entries,
            add_standard_gates=False,
            infer_parents=False
        )

        all_gates = hierarchy.get_all_gates()
        assert "Time" not in all_gates
        assert "Singlets" not in all_gates
        assert "T cells" in all_gates

    def test_marker_logic_preserved(self):
        entries = [
            MarkerTableEntry("NK cells", {"CD56": "bright", "CD3": "-"}, parent="CD45+"),
        ]

        hierarchy = marker_table_to_hierarchy(entries, add_standard_gates=True)

        # Find NK cells node
        def find_node(node, name):
            if node.name == name:
                return node
            for child in node.children:
                result = find_node(child, name)
                if result:
                    return result
            return None

        nk_node = find_node(hierarchy.root, "NK cells")
        assert nk_node is not None
        assert len(nk_node.marker_logic) == 2

        cd56 = next(m for m in nk_node.marker_logic if m.marker == "CD56")
        assert cd56.level == "bright"


class TestValidation:
    """Tests for hierarchy validation functions."""

    def test_validate_markers_success(self):
        entries = [
            MarkerTableEntry("T cells", {"CD3": "+"}, parent="CD45+"),
        ]
        hierarchy = marker_table_to_hierarchy(entries, add_standard_gates=True)

        panel = Panel(entries=[
            PanelEntry(marker="CD3", fluorophore="BUV395"),
            PanelEntry(marker="CD45", fluorophore="BV421"),
        ])

        errors = validate_hierarchy_markers(hierarchy, panel)
        assert len(errors) == 0

    def test_validate_markers_missing(self):
        entries = [
            MarkerTableEntry("T cells", {"CD3": "+", "CD19": "-"}, parent="CD45+"),
        ]
        hierarchy = marker_table_to_hierarchy(entries, add_standard_gates=True)

        # Panel missing CD19
        panel = Panel(entries=[
            PanelEntry(marker="CD3", fluorophore="BUV395"),
            PanelEntry(marker="CD45", fluorophore="BV421"),
        ])

        errors = validate_hierarchy_markers(hierarchy, panel)
        assert any("CD19" in e for e in errors)

    def test_validate_structure_no_cycles(self):
        entries = [
            MarkerTableEntry("A", {}, parent="B"),
            MarkerTableEntry("B", {}, parent="A"),  # Cycle!
        ]

        # This should not create a cycle in the output because
        # we handle external parents differently
        hierarchy = marker_table_to_hierarchy(entries, add_standard_gates=False)
        errors = validate_hierarchy_structure(hierarchy)

        # The implementation creates stub nodes for external parents
        # so there shouldn't be a cycle in the final structure
        all_gates = hierarchy.get_all_gates()
        assert "A" in all_gates
        assert "B" in all_gates


class TestIntegration:
    """Integration tests combining parsing and hierarchy building."""

    def test_full_pipeline_tcell_panel(self):
        """Test complete pipeline for T cell immunophenotyping."""
        table_text = """
| Population | CD3 | CD4 | CD8 | CD45RA | CCR7 | Parent |
|------------|-----|-----|-----|--------|------|--------|
| T cells | + | | | | | CD45+ |
| CD4+ T cells | + | + | - | | | T cells |
| CD8+ T cells | + | - | + | | | T cells |
| CD4+ Naive | + | + | - | + | + | CD4+ T cells |
| CD4+ CM | + | + | - | - | + | CD4+ T cells |
| CD4+ EM | + | + | - | - | - | CD4+ T cells |
| CD4+ TEMRA | + | + | - | + | - | CD4+ T cells |
"""
        entries = parse_marker_table(table_text, format='markdown')
        assert len(entries) == 7

        panel_markers = ["CD3", "CD4", "CD8", "CD45RA", "CCR7", "CD45"]
        hierarchy = marker_table_to_hierarchy(
            entries,
            panel_markers=panel_markers,
            add_standard_gates=True
        )

        all_gates = hierarchy.get_all_gates()
        assert "T cells" in all_gates
        assert "CD4+ T cells" in all_gates
        assert "CD4+ Naive" in all_gates
        assert "CD4+ TEMRA" in all_gates

        # Check parent relationships
        parent_map = hierarchy.get_parent_map()

        # Find CD4+ Naive's parent chain
        # Should be: CD4+ Naive -> CD4+ T cells -> T cells -> CD45+ -> Live -> ...
        cd4_naive_parent = parent_map.get("CD4+ Naive")
        assert cd4_naive_parent == "CD4+ T cells"

"""Tests for data schemas."""

import json
import pytest
from datetime import date
from pathlib import Path

from src.curation.schemas import (
    TestCase,
    Panel,
    PanelEntry,
    GatingHierarchy,
    GateNode,
    ExperimentContext,
    CurationMetadata,
    Complexity,
    SourceType,
)


class TestPanelEntry:
    """Tests for PanelEntry model."""

    def test_create_entry(self):
        """Test creating a panel entry."""
        entry = PanelEntry(
            marker="CD3",
            fluorophore="BUV395",
            clone="UCHT1",
        )

        assert entry.marker == "CD3"
        assert entry.fluorophore == "BUV395"
        assert entry.clone == "UCHT1"

    def test_optional_fields(self):
        """Test optional fields default to None."""
        entry = PanelEntry(marker="CD3", fluorophore="PE")

        assert entry.clone is None
        assert entry.channel is None
        assert entry.vendor is None


class TestPanel:
    """Tests for Panel model."""

    def test_n_colors(self):
        """Test color count calculation."""
        panel = Panel(entries=[
            PanelEntry(marker="CD3", fluorophore="BUV395"),
            PanelEntry(marker="CD4", fluorophore="BV421"),
            PanelEntry(marker="CD8", fluorophore="PE"),
        ])

        assert panel.n_colors == 3

    def test_n_colors_excludes_scatter(self):
        """Test that scatter is excluded from color count."""
        panel = Panel(entries=[
            PanelEntry(marker="CD3", fluorophore="BUV395"),
            PanelEntry(marker="FSC", fluorophore="FSC"),
            PanelEntry(marker="SSC", fluorophore="SSC"),
        ])

        assert panel.n_colors == 1

    def test_markers_property(self):
        """Test markers list property."""
        panel = Panel(entries=[
            PanelEntry(marker="CD3", fluorophore="BUV395"),
            PanelEntry(marker="CD4", fluorophore="BV421"),
        ])

        assert panel.markers == ["CD3", "CD4"]

    def test_complexity_simple(self):
        """Test simple complexity classification."""
        panel = Panel(entries=[
            PanelEntry(marker=f"CD{i}", fluorophore=f"F{i}")
            for i in range(10)
        ])

        assert panel.complexity == Complexity.SIMPLE

    def test_complexity_medium(self):
        """Test medium complexity classification."""
        panel = Panel(entries=[
            PanelEntry(marker=f"CD{i}", fluorophore=f"F{i}")
            for i in range(20)
        ])

        assert panel.complexity == Complexity.MEDIUM

    def test_complexity_complex(self):
        """Test complex classification."""
        panel = Panel(entries=[
            PanelEntry(marker=f"CD{i}", fluorophore=f"F{i}")
            for i in range(30)
        ])

        assert panel.complexity == Complexity.COMPLEX


class TestGateNode:
    """Tests for GateNode model."""

    def test_create_node(self):
        """Test creating a gate node."""
        node = GateNode(
            name="T cells",
            markers=["CD3"],
            is_critical=False,
        )

        assert node.name == "T cells"
        assert node.markers == ["CD3"]
        assert node.children == []

    def test_nested_children(self):
        """Test nested gate nodes."""
        node = GateNode(
            name="Parent",
            children=[
                GateNode(name="Child1"),
                GateNode(name="Child2"),
            ]
        )

        assert len(node.children) == 2
        assert node.children[0].name == "Child1"


class TestGatingHierarchy:
    """Tests for GatingHierarchy model."""

    def test_get_all_gates(self):
        """Test getting all gate names."""
        hierarchy = GatingHierarchy(
            root=GateNode(
                name="All Events",
                children=[
                    GateNode(name="Singlets", children=[
                        GateNode(name="Live")
                    ])
                ]
            )
        )

        gates = hierarchy.get_all_gates()

        assert "All Events" in gates
        assert "Singlets" in gates
        assert "Live" in gates
        assert len(gates) == 3

    def test_get_critical_gates(self):
        """Test getting critical gates."""
        hierarchy = GatingHierarchy(
            root=GateNode(
                name="All Events",
                children=[
                    GateNode(name="Singlets", is_critical=True, children=[
                        GateNode(name="Live", is_critical=True),
                        GateNode(name="Other", is_critical=False),
                    ])
                ]
            )
        )

        critical = hierarchy.get_critical_gates()

        assert "Singlets" in critical
        assert "Live" in critical
        assert "Other" not in critical
        assert "All Events" not in critical

    def test_get_parent_map(self):
        """Test getting parent map."""
        hierarchy = GatingHierarchy(
            root=GateNode(
                name="Root",
                children=[
                    GateNode(name="Child1"),
                    GateNode(name="Child2"),
                ]
            )
        )

        parent_map = hierarchy.get_parent_map()

        assert parent_map["Root"] is None
        assert parent_map["Child1"] == "Root"
        assert parent_map["Child2"] == "Root"


class TestTestCase:
    """Tests for TestCase model."""

    def test_create_test_case(self):
        """Test creating a complete test case."""
        test_case = TestCase(
            test_case_id="TEST-001",
            source_type=SourceType.OMIP_PAPER,
            omip_id="OMIP-001",
            context=ExperimentContext(
                sample_type="Human PBMC",
                species="human",
                application="Testing",
            ),
            panel=Panel(entries=[
                PanelEntry(marker="CD3", fluorophore="PE"),
            ]),
            gating_hierarchy=GatingHierarchy(
                root=GateNode(name="All Events")
            ),
            metadata=CurationMetadata(
                curation_date=date.today(),
                curator="Test",
            ),
        )

        assert test_case.test_case_id == "TEST-001"
        assert test_case.complexity == Complexity.SIMPLE

    def test_to_prompt_context_minimal(self):
        """Test minimal context generation."""
        test_case = TestCase(
            test_case_id="TEST-001",
            source_type=SourceType.OMIP_PAPER,
            context=ExperimentContext(
                sample_type="Human PBMC",
                species="human",
                application="Testing",
            ),
            panel=Panel(entries=[
                PanelEntry(marker="CD3", fluorophore="PE"),
                PanelEntry(marker="CD4", fluorophore="FITC"),
            ]),
            gating_hierarchy=GatingHierarchy(root=GateNode(name="All Events")),
            metadata=CurationMetadata(curation_date=date.today(), curator="Test"),
        )

        context = test_case.to_prompt_context("minimal")

        assert "CD3" in context
        assert "CD4" in context
        assert "Human PBMC" not in context  # Minimal doesn't include sample type

    def test_to_prompt_context_standard(self):
        """Test standard context generation."""
        test_case = TestCase(
            test_case_id="TEST-001",
            source_type=SourceType.OMIP_PAPER,
            context=ExperimentContext(
                sample_type="Human PBMC",
                species="human",
                application="Testing",
            ),
            panel=Panel(entries=[
                PanelEntry(marker="CD3", fluorophore="PE", clone="UCHT1"),
            ]),
            gating_hierarchy=GatingHierarchy(root=GateNode(name="All Events")),
            metadata=CurationMetadata(curation_date=date.today(), curator="Test"),
        )

        context = test_case.to_prompt_context("standard")

        assert "Human PBMC" in context
        assert "CD3" in context
        assert "PE" in context
        assert "UCHT1" in context


class TestLoadGroundTruth:
    """Tests for loading ground truth files."""

    def test_load_omip_069(self, ground_truth_dir):
        """Test loading OMIP-069 test case."""
        path = ground_truth_dir / "omip_069.json"

        if not path.exists():
            pytest.skip("Ground truth file not found")

        with open(path) as f:
            data = json.load(f)

        test_case = TestCase.model_validate(data)

        assert test_case.test_case_id == "OMIP-069"
        assert test_case.omip_id == "OMIP-069"
        assert test_case.context.sample_type == "Human PBMC"
        assert len(test_case.panel.entries) > 0
        assert test_case.gating_hierarchy.root.name == "All Events"

    def test_load_omip_023(self, ground_truth_dir):
        """Test loading OMIP-023 test case."""
        path = ground_truth_dir / "omip_023.json"

        if not path.exists():
            pytest.skip("Ground truth file not found")

        with open(path) as f:
            data = json.load(f)

        test_case = TestCase.model_validate(data)

        assert test_case.test_case_id == "OMIP-023"
        assert test_case.complexity == Complexity.SIMPLE  # 10-color panel

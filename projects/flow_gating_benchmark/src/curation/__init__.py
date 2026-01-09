"""Phase 1: Data curation - OMIP paper extraction and ground truth creation."""

from .schemas import TestCase, GatingHierarchy, Panel, PanelEntry, MarkerExpression, GateNode
from .omip_extractor import extract_omip_ground_truth
from .wsp_cross_validator import cross_validate_with_wsp

# New extraction modules
from .marker_logic import (
    MarkerTableEntry,
    parse_marker_table,
    marker_table_to_hierarchy,
    infer_parent_from_markers,
    validate_hierarchy_markers,
    validate_hierarchy_structure,
    lookup_cell_type,
    suggest_marker_logic,
    entries_from_hipc_populations,
)
from .paper_parser import (
    PaperParser,
    PaperContent,
    ExtractedTable,
    ExtractedFigure,
    extract_panel_from_table,
    extract_gating_from_text,
)
from .auto_extractor import (
    AutoExtractor,
    ExtractionResult,
    CombinedExtractionResult,
    extract_test_case,
    batch_extract,
)

__all__ = [
    # Schemas
    "TestCase",
    "GatingHierarchy",
    "Panel",
    "PanelEntry",
    "MarkerExpression",
    "GateNode",
    # Legacy extractors
    "extract_omip_ground_truth",
    "cross_validate_with_wsp",
    # Marker logic
    "MarkerTableEntry",
    "parse_marker_table",
    "marker_table_to_hierarchy",
    "infer_parent_from_markers",
    "validate_hierarchy_markers",
    "validate_hierarchy_structure",
    "lookup_cell_type",
    "suggest_marker_logic",
    "entries_from_hipc_populations",
    # Paper parsing
    "PaperParser",
    "PaperContent",
    "ExtractedTable",
    "ExtractedFigure",
    "extract_panel_from_table",
    "extract_gating_from_text",
    # Auto extraction
    "AutoExtractor",
    "ExtractionResult",
    "CombinedExtractionResult",
    "extract_test_case",
    "batch_extract",
]

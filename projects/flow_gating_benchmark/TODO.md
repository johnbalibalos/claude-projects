# Flow Gating Benchmark - TODO

## Completed

### OCR/Gating Hierarchy Extraction Library (Jan 2025)

- [x] `marker_logic.py` - Core algorithms for marker table → hierarchy conversion
  - `MarkerTableEntry` dataclass for phenotype table rows
  - `parse_marker_table()` - Parse markdown/CSV tables
  - `infer_parent_from_markers()` - Infer hierarchy from marker subset logic
  - `marker_table_to_hierarchy()` - Build GatingHierarchy from entries
  - Validation functions for panels and structure

- [x] `paper_parser.py` - XML/PDF content extraction
  - `PaperParser` class for extracting content from OMIP papers
  - `ExtractedTable` / `ExtractedFigure` dataclasses
  - Table classification (panel vs gating vs results)
  - Methods section text extraction
  - Panel entry extraction from tables

- [x] `auto_extractor.py` - Combined extraction pipeline
  - `AutoExtractor` class with multiple strategies
  - LLM-assisted extraction from methods text
  - Confidence scoring and result combination
  - Batch extraction for multiple papers

- [x] Tests for all new modules

---

## TODO: MCP Server Implementation

### Overview

Wrap the extraction library as an MCP (Model Context Protocol) server to enable interactive, LLM-assisted curation sessions. This allows Claude to iteratively extract, validate, and refine gating hierarchies with human oversight.

### Why MCP?

1. **Interactive curation** - Human + LLM work together to extract and validate
2. **Self-correcting** - LLM sees validation errors and can fix them
3. **Auditable** - Conversation shows extraction reasoning
4. **Consistent** - Matches `flow_panel_optimizer` MCP pattern

### Proposed Structure

```
projects/flow_gating_benchmark/
├── src/
│   ├── curation/           # Existing Python library
│   │   ├── marker_logic.py
│   │   ├── paper_parser.py
│   │   └── auto_extractor.py
│   └── mcp_server/         # NEW: MCP wrapper
│       ├── __init__.py
│       ├── server.py       # MCP server definition
│       └── tools.py        # Tool implementations
```

### MCP Tools to Implement

#### 1. Paper Content Extraction

```python
@server.tool("get_paper_content")
async def get_paper_content(
    omip_id: str,
    include: list[str] = ["tables", "methods", "abstract"]
) -> dict:
    """
    Get content from an OMIP paper.

    Args:
        omip_id: OMIP identifier (e.g., "OMIP-069")
        include: Content types to include

    Returns:
        Dict with requested content sections
    """
```

#### 2. Marker Table Parsing

```python
@server.tool("parse_marker_table")
async def parse_marker_table(
    table_text: str,
    format: str = "auto"
) -> list[dict]:
    """
    Parse a marker phenotype table into structured entries.

    Args:
        table_text: Markdown or CSV table text
        format: 'markdown', 'csv', or 'auto'

    Returns:
        List of {population, markers, parent} entries
    """
```

#### 3. Hierarchy Building

```python
@server.tool("build_hierarchy_from_markers")
async def build_hierarchy_from_markers(
    entries: list[dict],
    panel_markers: list[str] = None,
    infer_parents: bool = True,
    add_standard_gates: bool = True
) -> dict:
    """
    Build gating hierarchy tree from marker table entries.

    Infers parent-child relationships from marker subset logic
    when explicit parents not provided.

    Args:
        entries: List of {population, markers: {marker: '+'/'-'}, parent?}
        panel_markers: Available markers in panel
        infer_parents: Auto-detect hierarchy from marker subsets
        add_standard_gates: Add Time/Singlets/Live gates

    Returns:
        GatingHierarchy as JSON
    """
```

#### 4. Hierarchy Validation

```python
@server.tool("validate_hierarchy")
async def validate_hierarchy(
    hierarchy: dict,
    panel: dict,
    check_hipc: bool = True
) -> dict:
    """
    Validate a gating hierarchy against panel and standards.

    Args:
        hierarchy: GatingHierarchy JSON
        panel: Panel JSON with marker entries
        check_hipc: Validate against HIPC cell type definitions

    Returns:
        {valid: bool, errors: list[str], warnings: list[str]}
    """
```

#### 5. HIPC Reference Lookup

```python
@server.tool("lookup_cell_type")
async def lookup_cell_type(
    cell_type: str
) -> dict | None:
    """
    Look up HIPC-standardized definition for a cell type.

    Args:
        cell_type: Cell type name (e.g., "CD4+ T cells", "NK cells")

    Returns:
        Dict with positive_markers, negative_markers, parent, etc.
    """
```

#### 6. Test Case Management

```python
@server.tool("save_test_case")
async def save_test_case(
    test_case: dict,
    output_dir: str = "real"
) -> str:
    """
    Save a validated test case to ground truth directory.

    Args:
        test_case: TestCase JSON
        output_dir: 'real' or 'synthetic'

    Returns:
        Path to saved file
    """

@server.tool("load_test_case")
async def load_test_case(
    test_case_id: str
) -> dict | None:
    """Load an existing test case by ID."""
```

### Server Configuration

Add to `.mcp.json`:

```json
{
  "mcpServers": {
    "gating-extractor": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "projects/flow_gating_benchmark"
    }
  }
}
```

### Example Curation Session

```
Human: "Extract gating hierarchy from OMIP-069"

Claude: [calls get_paper_content(omip_id="OMIP-069", include=["tables", "methods"])]
        → Gets XML text, tables from paper

Claude: "I found a panel table and methods section. Let me parse the panel first."
        [calls parse_marker_table(table_text="...", format="markdown")]
        → Gets structured MarkerTableEntry list

Claude: "I found 8 populations in the gating table. Building hierarchy..."
        [calls build_hierarchy_from_markers(entries=[...], infer_parents=True)]
        → Gets draft GatingHierarchy

Claude: [calls validate_hierarchy(hierarchy=..., panel=...)]
        → Gets validation errors: "CD127 referenced but not in panel"

Claude: "I found an issue - CD127 is mentioned in the gating but not in the
         panel table. Should I:
         1. Remove CD127 from the hierarchy
         2. Add CD127 to the panel (need fluorophore info)
         3. Skip this population"

Human: "Add it to panel, fluorophore is BV785"

Claude: [calls save_test_case(test_case={...})]
        → Saves OMIP-069.json to data/ground_truth/real/

Claude: "Done\! Test case saved. The hierarchy has 15 gates total."
```

### Dependencies

- `mcp` - MCP Python SDK (pip install mcp)
- Existing curation library (marker_logic, paper_parser, auto_extractor)
- OMIP papers downloaded to data/papers/pmc/

### Implementation Phases

#### Phase 1: Core Tools
- [ ] Set up MCP server boilerplate
- [ ] Implement get_paper_content tool
- [ ] Implement parse_marker_table tool
- [ ] Implement build_hierarchy_from_markers tool

#### Phase 2: Validation & Reference
- [ ] Implement validate_hierarchy tool
- [ ] Implement lookup_cell_type tool
- [ ] Add HIPC compliance checking

#### Phase 3: Persistence
- [ ] Implement save_test_case tool
- [ ] Implement load_test_case tool
- [ ] Add batch operations

#### Phase 4: Vision Support (Optional)
- [ ] Implement figure extraction from PDF
- [ ] Add vision LLM integration for gating figures
- [ ] OCR fallback for non-vision models

---

## Future Work

### Vision LLM Extraction

For papers without explicit marker tables, use vision models to:
1. Identify gating figures in PDF
2. Extract gate names and relationships from plots
3. Infer marker expressions from axis labels

### FlowRepository Integration

Automatic download and parsing of WSP files from FlowRepository:
1. Look up FlowRepository ID from OMIP paper
2. Download workspace file
3. Extract hierarchy using FlowKit
4. Cross-validate with paper-extracted hierarchy

### Batch Curation Pipeline

Script to process all OMIP papers:
```bash
python -m src.scripts.batch_curate --start 1 --end 100 --output data/ground_truth/real/
```

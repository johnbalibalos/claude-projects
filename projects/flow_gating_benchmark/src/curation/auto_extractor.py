"""
Automated gating hierarchy extraction pipeline.

Combines multiple extraction strategies:
1. WSP file extraction (highest confidence)
2. Marker phenotype table from paper
3. Methods section text → LLM extraction
4. Vision LLM on figures (optional)

Example usage:
    from curation.auto_extractor import AutoExtractor

    extractor = AutoExtractor(llm_client=my_client)
    result = extractor.extract("OMIP-069")

    if result.success:
        test_case = result.test_case
        test_case.save("data/verified/real/")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Protocol

from .marker_logic import (
    MarkerTableEntry,
    marker_table_to_hierarchy,
    parse_marker_table,
    validate_hierarchy_markers,
    validate_hierarchy_structure,
)
from .paper_parser import (
    ExtractedTable,
    PaperContent,
    PaperParser,
    extract_gating_from_text,
    extract_panel_from_table,
)
from .schemas import (
    CurationMetadata,
    ExperimentContext,
    GatingHierarchy,
    Panel,
    PanelEntry,
    SourceType,
    TestCase,
    ValidationInfo,
)


class LLMClient(Protocol):
    """Protocol for LLM client interface."""

    def complete(self, prompt: str) -> str:
        """Send prompt and get completion."""
        ...


@dataclass
class ExtractionResult:
    """Result from a single extraction attempt."""
    method: str  # 'wsp', 'marker_table', 'methods_llm', 'vision'
    success: bool
    confidence: float  # 0.0 to 1.0
    panel: Panel | None = None
    hierarchy: GatingHierarchy | None = None
    context: ExperimentContext | None = None
    errors: list[str] = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)


@dataclass
class CombinedExtractionResult:
    """Final result combining multiple extraction methods."""
    success: bool
    test_case: TestCase | None
    methods_used: list[str]
    confidence: float
    validation_errors: list[str]
    extraction_results: list[ExtractionResult]


# LLM prompts for extraction
PANEL_EXTRACTION_PROMPT = """Extract the antibody panel from this flow cytometry paper text.

Return a JSON array with each antibody entry:
[
  {{"marker": "CD3", "fluorophore": "BUV395", "clone": "UCHT1"}},
  {{"marker": "CD4", "fluorophore": "BUV496", "clone": "SK3"}},
  ...
]

Include only entries where you can identify at least the marker and fluorophore.
If clone is not specified, use null.

Text:
{text}
"""

HIERARCHY_EXTRACTION_PROMPT = """Extract the gating strategy/hierarchy from this flow cytometry methods text.

Available markers in panel: {markers}

Return a JSON object representing the marker phenotype table:
{{
  "populations": [
    {{
      "name": "T cells",
      "markers": {{"CD3": "+", "CD19": "-"}},
      "parent": "CD45+"
    }},
    {{
      "name": "CD4+ T cells",
      "markers": {{"CD3": "+", "CD4": "+", "CD8": "-"}},
      "parent": "T cells"
    }},
    ...
  ]
}}

For markers:
- Use "+" for positive expression
- Use "-" for negative expression
- Use "dim" or "bright" for intensity levels
- Leave blank or omit markers not mentioned for this population

For parent: specify the immediate parent population in the gating hierarchy.
Common parents: "All Events", "Singlets", "Live", "CD45+", "Lymphocytes"

Text:
{text}
"""

CONTEXT_EXTRACTION_PROMPT = """Extract experimental context from this flow cytometry paper.

Return JSON:
{{
  "sample_type": "Human PBMC",  // or "Mouse spleen", "Human whole blood", etc.
  "species": "human",  // or "mouse", "non-human primate", etc.
  "application": "T cell immunophenotyping",  // brief description of the panel's purpose
  "tissue": "Peripheral blood",  // optional
  "disease_state": null  // or "HIV infection", "Cancer", etc. if mentioned
}}

Text:
{text}
"""


class AutoExtractor:
    """
    Automated extraction pipeline for creating test cases from OMIP papers.

    Tries multiple extraction strategies in order of confidence:
    1. WSP file (FlowJo workspace) - if available
    2. Marker table from paper - if paper has phenotype table
    3. Methods text + LLM - extract from prose descriptions
    4. Vision LLM on figures - if PDF available (optional)
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        papers_dir: Path | str = "data/papers/pmc",
        wsp_dir: Path | str = "data/wsp",
        vision_enabled: bool = False,
        curator_name: str = "AutoExtractor"
    ):
        """
        Initialize the extractor.

        Args:
            llm_client: LLM client for text extraction (required for methods/vision)
            papers_dir: Directory containing downloaded papers
            wsp_dir: Directory containing WSP files
            vision_enabled: Whether to use vision LLM on PDF figures
            curator_name: Name to record in metadata
        """
        self.llm = llm_client
        self.papers_dir = Path(papers_dir)
        self.wsp_dir = Path(wsp_dir)
        self.vision_enabled = vision_enabled
        self.curator_name = curator_name

        self.paper_parser = PaperParser(papers_dir)

    def extract(
        self,
        omip_id: str,
        strategies: list[str] | None = None
    ) -> CombinedExtractionResult:
        """
        Extract test case from an OMIP paper using best available method.

        Args:
            omip_id: OMIP identifier (e.g., "OMIP-069")
            strategies: Optional list of strategies to try. Default: all applicable

        Returns:
            CombinedExtractionResult with test case and metadata
        """
        if strategies is None:
            strategies = ['wsp', 'marker_table', 'methods_llm']
            if self.vision_enabled:
                strategies.append('vision')

        results: list[ExtractionResult] = []

        # Get paper content first
        paper_content = self.paper_parser.extract_all(omip_id)

        # Try each strategy
        for strategy in strategies:
            try:
                if strategy == 'wsp':
                    result = self._extract_from_wsp(omip_id)
                elif strategy == 'marker_table':
                    result = self._extract_from_marker_table(omip_id, paper_content)
                elif strategy == 'methods_llm':
                    result = self._extract_from_methods_llm(omip_id, paper_content)
                elif strategy == 'vision':
                    result = self._extract_from_vision(omip_id, paper_content)
                else:
                    continue

                if result:
                    results.append(result)
            except Exception as e:
                results.append(ExtractionResult(
                    method=strategy,
                    success=False,
                    confidence=0.0,
                    errors=[str(e)]
                ))

        # Select best result or combine
        return self._combine_results(omip_id, results, paper_content)

    def _extract_from_wsp(self, omip_id: str) -> ExtractionResult | None:
        """Extract from FlowJo workspace file."""
        # Find WSP file for this OMIP
        wsp_files = list(self.wsp_dir.glob(f"*{omip_id.replace('-', '')}*.wsp"))
        wsp_files.extend(self.wsp_dir.glob(f"*{omip_id.replace('OMIP-', '')}*.wsp"))

        if not wsp_files:
            return None

        try:
            from .wsp_extractor import extract_hierarchy_from_wsp

            wsp_path = wsp_files[0]
            extraction = extract_hierarchy_from_wsp(wsp_path)

            if extraction and extraction.get('hierarchy'):
                return ExtractionResult(
                    method='wsp',
                    success=True,
                    confidence=0.95,
                    hierarchy=extraction['hierarchy'],
                    panel=extraction.get('panel'),
                    raw_data={'wsp_path': str(wsp_path), 'extraction': extraction}
                )
        except ImportError:
            # wsp_extractor not available
            pass
        except Exception as e:
            return ExtractionResult(
                method='wsp',
                success=False,
                confidence=0.0,
                errors=[f"WSP extraction failed: {e}"]
            )

        return None

    def _extract_from_marker_table(
        self,
        omip_id: str,
        paper_content: PaperContent | None
    ) -> ExtractionResult | None:
        """Extract from marker phenotype table in paper."""
        if not paper_content:
            return None

        # Look for gating table
        gating_table = paper_content.get_gating_table()
        if not gating_table:
            # Check all tables for potential marker phenotype tables
            for table in paper_content.tables:
                if self._looks_like_marker_table(table):
                    gating_table = table
                    break

        if not gating_table:
            return ExtractionResult(
                method='marker_table',
                success=False,
                confidence=0.0,
                errors=["No marker phenotype table found in paper"]
            )

        # Also extract panel
        panel = None
        panel_table = paper_content.get_panel_table()
        if panel_table:
            panel_entries = extract_panel_from_table(panel_table)
            if panel_entries:
                panel = Panel(entries=[
                    PanelEntry(**e) for e in panel_entries
                ])

        # Parse marker table
        try:
            markdown_table = gating_table.to_markdown()
            entries = parse_marker_table(markdown_table, format='markdown')

            if not entries:
                return ExtractionResult(
                    method='marker_table',
                    success=False,
                    confidence=0.0,
                    errors=["Could not parse marker table"]
                )

            # Build hierarchy
            panel_markers = panel.markers if panel else []
            hierarchy = marker_table_to_hierarchy(
                entries,
                panel_markers=panel_markers,
                infer_parents=True,
                add_standard_gates=True
            )

            return ExtractionResult(
                method='marker_table',
                success=True,
                confidence=0.85,
                panel=panel,
                hierarchy=hierarchy,
                raw_data={
                    'table': gating_table.to_markdown(),
                    'entries': [e.__dict__ for e in entries]
                }
            )

        except Exception as e:
            return ExtractionResult(
                method='marker_table',
                success=False,
                confidence=0.0,
                errors=[f"Marker table parsing failed: {e}"]
            )

    def _looks_like_marker_table(self, table: ExtractedTable) -> bool:
        """Check if a table looks like a marker phenotype table."""
        headers_lower = [h.lower() for h in table.headers]
        caption_lower = table.caption.lower()

        # Must have population-like column
        has_pop_col = any(kw in h for h in headers_lower
                        for kw in ['population', 'subset', 'cell', 'phenotype'])

        # Should have marker columns (CD*, +/- symbols in data)
        has_marker_cols = any('cd' in h for h in headers_lower)

        # Check for +/- in data
        has_markers = False
        for row in table.rows[:5]:  # Check first 5 rows
            for cell in row:
                if '+' in cell or '-' in cell:
                    has_markers = True
                    break

        return (has_pop_col or has_marker_cols) and has_markers

    def _extract_from_methods_llm(
        self,
        omip_id: str,
        paper_content: PaperContent | None
    ) -> ExtractionResult | None:
        """Extract using LLM on methods section text."""
        if not self.llm:
            return ExtractionResult(
                method='methods_llm',
                success=False,
                confidence=0.0,
                errors=["No LLM client provided"]
            )

        if not paper_content:
            return ExtractionResult(
                method='methods_llm',
                success=False,
                confidence=0.0,
                errors=["No paper content available"]
            )

        # Get relevant text
        methods_text = paper_content.methods_text
        if not methods_text:
            methods_text = paper_content.results_text

        if not methods_text:
            return ExtractionResult(
                method='methods_llm',
                success=False,
                confidence=0.0,
                errors=["No methods or results text found"]
            )

        try:
            # Extract panel first
            panel = None
            panel_table = paper_content.get_panel_table()
            if panel_table:
                panel_entries = extract_panel_from_table(panel_table)
                if panel_entries:
                    panel = Panel(entries=[PanelEntry(**e) for e in panel_entries])

            # If no panel table, try LLM extraction
            if not panel and paper_content.abstract:
                panel_text = paper_content.abstract + "\n\n" + methods_text[:3000]
                panel_prompt = PANEL_EXTRACTION_PROMPT.format(text=panel_text)
                panel_response = self.llm.complete(panel_prompt)
                panel_data = self._parse_json_response(panel_response)
                if panel_data and isinstance(panel_data, list):
                    panel = Panel(entries=[PanelEntry(**e) for e in panel_data])

            # Extract hierarchy with LLM
            marker_list = panel.markers if panel else []
            hierarchy_prompt = HIERARCHY_EXTRACTION_PROMPT.format(
                markers=", ".join(marker_list),
                text=methods_text[:6000]
            )

            hierarchy_response = self.llm.complete(hierarchy_prompt)
            hierarchy_data = self._parse_json_response(hierarchy_response)

            if not hierarchy_data or 'populations' not in hierarchy_data:
                return ExtractionResult(
                    method='methods_llm',
                    success=False,
                    confidence=0.0,
                    errors=["LLM did not return valid hierarchy data"],
                    raw_data={'response': hierarchy_response}
                )

            # Convert to MarkerTableEntry
            entries = []
            for pop in hierarchy_data['populations']:
                entries.append(MarkerTableEntry(
                    population=pop['name'],
                    markers=pop.get('markers', {}),
                    parent=pop.get('parent')
                ))

            # Build hierarchy
            hierarchy = marker_table_to_hierarchy(
                entries,
                panel_markers=marker_list,
                infer_parents=True,
                add_standard_gates=True
            )

            # Extract context
            context = None
            context_prompt = CONTEXT_EXTRACTION_PROMPT.format(
                text=paper_content.abstract + "\n\n" + methods_text[:2000]
            )
            context_response = self.llm.complete(context_prompt)
            context_data = self._parse_json_response(context_response)
            if context_data:
                context = ExperimentContext(**context_data)

            return ExtractionResult(
                method='methods_llm',
                success=True,
                confidence=0.70,
                panel=panel,
                hierarchy=hierarchy,
                context=context,
                raw_data={
                    'hierarchy_response': hierarchy_response,
                    'populations': hierarchy_data['populations']
                }
            )

        except Exception as e:
            return ExtractionResult(
                method='methods_llm',
                success=False,
                confidence=0.0,
                errors=[f"LLM extraction failed: {e}"]
            )

    def _extract_from_vision(
        self,
        omip_id: str,
        paper_content: PaperContent | None
    ) -> ExtractionResult | None:
        """Extract using vision LLM on PDF figures."""
        # TODO: Implement vision extraction
        # This requires:
        # 1. PDF figure extraction (already in paper_parser)
        # 2. Vision-capable LLM client
        # 3. Specialized prompts for gating figure analysis

        return ExtractionResult(
            method='vision',
            success=False,
            confidence=0.0,
            errors=["Vision extraction not yet implemented"]
        )

    def _parse_json_response(self, response: str) -> dict | list | None:
        """Parse JSON from LLM response, handling code blocks."""
        # Try to extract JSON from code block
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try parsing raw response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try finding JSON object/array
        for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
            match = re.search(pattern, response)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue

        return None

    def _combine_results(
        self,
        omip_id: str,
        results: list[ExtractionResult],
        paper_content: PaperContent | None
    ) -> CombinedExtractionResult:
        """Combine extraction results into final test case."""
        # Filter successful results
        successful = [r for r in results if r.success and r.hierarchy]

        if not successful:
            return CombinedExtractionResult(
                success=False,
                test_case=None,
                methods_used=[r.method for r in results],
                confidence=0.0,
                validation_errors=[e for r in results for e in r.errors],
                extraction_results=results
            )

        # Sort by confidence
        successful.sort(key=lambda r: r.confidence, reverse=True)
        best = successful[0]

        # Get panel (prefer higher confidence source)
        panel = best.panel
        if not panel:
            for r in successful:
                if r.panel:
                    panel = r.panel
                    break

        # Get context
        context = best.context
        if not context:
            for r in successful:
                if r.context:
                    context = r.context
                    break

        # Default context if not found
        if not context:
            context = ExperimentContext(
                sample_type="Unknown",
                species="human",
                application="Flow cytometry immunophenotyping"
            )

        # Validate hierarchy
        validation_errors = []
        if panel:
            validation_errors.extend(validate_hierarchy_markers(best.hierarchy, panel))
        validation_errors.extend(validate_hierarchy_structure(best.hierarchy))

        # Build test case
        test_case = TestCase(
            test_case_id=omip_id,
            source_type=SourceType.OMIP_PAPER,
            omip_id=omip_id,
            doi=paper_content.doi if paper_content else None,
            context=context,
            panel=panel or Panel(entries=[]),
            gating_hierarchy=best.hierarchy,
            validation=ValidationInfo(
                curator_notes=f"Auto-extracted using {best.method} (confidence: {best.confidence:.2f})"
            ),
            metadata=CurationMetadata(
                curation_date=date.today(),
                curator=self.curator_name
            )
        )

        return CombinedExtractionResult(
            success=True,
            test_case=test_case,
            methods_used=[r.method for r in successful],
            confidence=best.confidence,
            validation_errors=validation_errors,
            extraction_results=results
        )


def extract_test_case(
    omip_id: str,
    llm_client: LLMClient | None = None,
    **kwargs
) -> TestCase | None:
    """
    Convenience function to extract a test case.

    Args:
        omip_id: OMIP identifier
        llm_client: Optional LLM client for text extraction
        **kwargs: Additional arguments passed to AutoExtractor

    Returns:
        TestCase if successful, None otherwise
    """
    extractor = AutoExtractor(llm_client=llm_client, **kwargs)
    result = extractor.extract(omip_id)

    if result.success:
        return result.test_case
    return None


def batch_extract(
    omip_ids: list[str],
    llm_client: LLMClient | None = None,
    output_dir: Path | str = "data/verified/real",
    **kwargs
) -> dict[str, CombinedExtractionResult]:
    """
    Extract test cases for multiple OMIP papers.

    Args:
        omip_ids: List of OMIP identifiers
        llm_client: Optional LLM client
        output_dir: Directory to save successful extractions
        **kwargs: Additional arguments for AutoExtractor

    Returns:
        Dict mapping omip_id to extraction result
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = AutoExtractor(llm_client=llm_client, **kwargs)
    results = {}

    for omip_id in omip_ids:
        print(f"Extracting {omip_id}...")
        result = extractor.extract(omip_id)
        results[omip_id] = result

        if result.success and result.test_case:
            # Save test case
            output_path = output_dir / f"{omip_id.lower().replace('-', '_')}.json"
            with open(output_path, 'w') as f:
                json.dump(result.test_case.model_dump(), f, indent=2, default=str)
            print(f"  Saved to {output_path}")
        else:
            print(f"  Failed: {result.validation_errors}")

    return results

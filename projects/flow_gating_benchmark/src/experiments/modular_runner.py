"""
Example: Modular hypothesis pipeline for the flow gating benchmark.

Demonstrates how to use the hypothesis_pipeline library to run experiments
with different combinations of:
- Reasoning strategies (direct, CoT, WoT)
- Context levels (minimal, standard, rich)
- Reference modes: none, oracle (static HIPC injection), vector (real RAG)
- Tool configurations (with/without FlowKit MCP)

Usage:
    python -m experiments.modular_runner --help

Installation:
    # From repository root, install both libs and this project in editable mode:
    pip install -e libs/
    pip install -e projects/flow_gating_benchmark/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from curation.omip_extractor import load_all_test_cases
from curation.schemas import TestCase
from evaluation.metrics import evaluate_prediction
from evaluation.response_parser import parse_llm_response

# hypothesis_pipeline is in libs/ - install with: pip install -e libs/
try:
    from hypothesis_pipeline import (
        ChainOfThoughtStrategy,
        ContextLevel,
        Evaluator,
        HypothesisPipeline,
        OracleRAGProvider,
        PipelineConfig,
        RAGMode,
        ReasoningType,
        RichContextBuilder,
        ToolConfig,
        ToolRegistry,
        TrialInput,
    )
except ImportError as err:
    raise ImportError(
        "hypothesis_pipeline not found. Install from repository root with: "
        "pip install -e libs/"
    ) from err

logger = logging.getLogger(__name__)

# =============================================================================
# CUSTOM EVALUATOR FOR GATING BENCHMARK
# =============================================================================


class GatingEvaluator(Evaluator):
    """Evaluator for gating hierarchy predictions."""

    def extract(self, response: str) -> dict | None:
        """Extract gating hierarchy from response."""
        result = parse_llm_response(response)
        if result.success:
            return result.hierarchy
        return None

    def score(
        self,
        extracted: dict | None,
        ground_truth: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Score the extracted hierarchy against ground truth."""
        if extracted is None:
            return {
                "hierarchy_f1": 0.0,
                "structure_accuracy": 0.0,
                "critical_gate_recall": 0.0,
                "parse_success": 0.0,
            }

        # Ground truth is a TestCase
        test_case: TestCase = ground_truth

        # Evaluate
        result = evaluate_prediction(extracted, test_case.gating_hierarchy)

        return {
            "hierarchy_f1": result.hierarchy_f1,
            "structure_accuracy": result.structure_accuracy,
            "critical_gate_recall": result.critical_gate_recall,
            "hallucination_rate": result.hallucination_rate,
            "parse_success": 1.0,
        }


# =============================================================================
# CUSTOM CONTEXT BUILDERS
# =============================================================================


class GatingContextBuilder(RichContextBuilder):
    """Context builder specialized for gating panels."""

    def format_trial_context(self, trial_input: TrialInput, **kwargs: Any) -> str:
        """Format context from test case."""
        test_case: TestCase = trial_input.raw_input

        parts = ["## Experiment Information", ""]

        # Sample info
        parts.append(f"Sample Type: {test_case.context.sample_type}")
        parts.append(f"Species: {test_case.context.species}")
        parts.append(f"Application: {test_case.context.application}")
        parts.append("")

        # Panel
        parts.append("## Panel")
        parts.append("")
        for entry in test_case.panel.entries:
            fluor = entry.fluorophore or "unknown"
            line = f"- {entry.marker}: {fluor}"
            if entry.clone:
                line += f" (clone: {entry.clone})"
            parts.append(line)

        # Additional info for rich context
        if self.context_level == ContextLevel.RICH:
            parts.append("")
            parts.append("## Additional Information")
            parts.append("")
            parts.append(f"Panel Size: {test_case.panel.n_colors} colors")
            parts.append(f"Complexity: {test_case.complexity.value}")
            if test_case.context.tissue:
                parts.append(f"Tissue: {test_case.context.tissue}")
            if test_case.omip_id:
                parts.append(f"Reference: {test_case.omip_id}")

        return "\n".join(parts)


# =============================================================================
# CUSTOM PROMPT STRATEGY
# =============================================================================


class GatingCoTStrategy(ChainOfThoughtStrategy):
    """Specialized CoT strategy for gating hierarchy prediction."""

    def __init__(self):
        super().__init__(
            reasoning_prompt="""Think through this step-by-step:

1. **Quality Control Gates**: What initial QC gates are needed? Consider:
   - Time gate (to exclude acquisition artifacts)
   - Singlet gate (to exclude doublets/aggregates)
   - Live/Dead discrimination

2. **Major Lineage Identification**: What major cell lineages can this panel identify? Consider:
   - Which markers define major populations (T cells, B cells, NK cells, myeloid)?
   - What is the logical order to separate these populations?

3. **Subset Identification**: For each major lineage, what subsets can be identified?
   - What markers distinguish subsets?
   - What is the gating order within each lineage?

4. **Hierarchy Structure**: Organize into a complete gating tree.""",
            answer_prefix="Final Answer:",
        )


# =============================================================================
# TOOL DEFINITIONS (FlowKit MCP simulation)
# =============================================================================


def create_flowkit_tools() -> ToolRegistry:
    """Create tool registry with FlowKit tools."""
    registry = ToolRegistry()

    # Tool: Read workspace file
    registry.register(
        ToolConfig(
            name="read_workspace",
            description="Read a FlowJo workspace file to extract existing gating strategies",
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to .wsp file"},
                },
                "required": ["file_path"],
            },
        ),
        executor=lambda args: {"gates": [], "error": "Simulated - no workspace available"},
    )

    # Tool: Get marker information
    registry.register(
        ToolConfig(
            name="get_marker_info",
            description="Get information about a specific marker including its typical use in gating",
            input_schema={
                "type": "object",
                "properties": {
                    "marker": {"type": "string", "description": "Marker name (e.g., CD3, CD19)"},
                },
                "required": ["marker"],
            },
        ),
        executor=lambda args: _get_marker_info(args.get("marker", "")),
    )

    # Tool: Suggest gate type
    registry.register(
        ToolConfig(
            name="suggest_gate_type",
            description="Suggest the appropriate gate type for a population based on markers",
            input_schema={
                "type": "object",
                "properties": {
                    "markers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Markers to use for gating",
                    },
                },
                "required": ["markers"],
            },
        ),
        executor=lambda args: {"gate_type": "PolygonGate", "reason": "Default for 2+ markers"},
    )

    return registry


# Module-level cache for marker database
_marker_db_cache: dict | None = None


def _load_marker_database() -> dict:
    """
    Load marker database from external config file.

    The marker database is loaded once and cached. This separates domain
    knowledge (biological marker information) from code logic.
    """
    global _marker_db_cache
    if _marker_db_cache is not None:
        return _marker_db_cache

    config_path = Path(__file__).parent.parent.parent / "configs" / "marker_database.json"
    if not config_path.exists():
        logger.warning(f"Marker database not found at {config_path}, using empty database")
        _marker_db_cache = {}
        return _marker_db_cache

    with open(config_path) as f:
        data = json.load(f)
        _marker_db_cache = data.get("markers", {})

    logger.debug(f"Loaded {len(_marker_db_cache)} markers from {config_path}")
    return _marker_db_cache


def _get_marker_info(marker: str) -> dict:
    """
    Get information about a flow cytometry marker.

    Loads marker data from configs/marker_database.json to separate
    domain knowledge from code logic.

    Args:
        marker: Marker name (e.g., "CD3", "CD19")

    Returns:
        Dict with lineage, typical_gates, typical_children, or error message
    """
    marker_db = _load_marker_database()
    return marker_db.get(marker.upper(), {"error": f"Unknown marker: {marker}"})


# =============================================================================
# MAIN PIPELINE SETUP
# =============================================================================


def load_trial_inputs(test_cases_dir: str) -> list[TrialInput]:
    """Load test cases as trial inputs."""
    test_cases = load_all_test_cases(test_cases_dir)

    trials = []
    for tc in test_cases:
        # Base prompt
        prompt = """You are an expert flow cytometrist. Given the following flow cytometry panel information, predict the gating hierarchy that an expert would use for data analysis.

Predict the complete gating hierarchy, starting from "All Events" through quality control gates (time, singlets, live/dead) to final cell population identification.

Return your answer as a JSON object with this structure:
{
    "name": "Gate Name",
    "markers": ["marker1", "marker2"],
    "children": [
        {
            "name": "Child Gate",
            "markers": ["marker3"],
            "children": [...]
        }
    ]
}"""

        # Create trial input
        trial = TrialInput(
            id=tc.test_case_id,
            raw_input=tc,  # Store full test case for context building
            prompt=prompt,
            ground_truth=tc,  # Pass test case for evaluation
            metadata={
                "complexity": tc.complexity.value,
                "n_colors": tc.panel.n_colors,
                "species": tc.context.species,
                "sample_type": tc.context.sample_type,
                # Oracle context for upper-bound testing
                "oracle_context": _extract_oracle_context(tc),
            },
        )
        trials.append(trial)

    return trials


def _extract_oracle_context(tc: TestCase) -> str:
    """Extract oracle context (perfect retrieval) from test case."""
    # In practice, this would be the exact section from the OMIP paper
    # describing the gating strategy
    gates = []

    def extract_gates(node, depth=0):
        gates.append(f"{'  ' * depth}- {node.name}")
        for child in node.children:
            extract_gates(child, depth + 1)

    extract_gates(tc.gating_hierarchy.root)

    return "Expected gating hierarchy:\n" + "\n".join(gates)


def create_pipeline(
    test_cases_dir: str,
    output_dir: str,
    models: list[str] | None = None,
    reasoning_types: list[str] | None = None,
    context_levels: list[str] | None = None,
    rag_modes: list[str] | None = None,
    with_tools: bool = False,
) -> HypothesisPipeline:
    """
    Create the hypothesis pipeline.

    Args:
        test_cases_dir: Directory with test case JSON files
        output_dir: Where to save results
        models: Models to test
        reasoning_types: Reasoning strategies to test
        context_levels: Context levels to test
        rag_modes: RAG modes to test
        with_tools: Whether to include tool conditions

    Returns:
        Configured pipeline
    """
    # Parse enum values
    reasoning_enums = [ReasoningType(r) for r in (reasoning_types or ["direct", "cot"])]
    context_enums = [ContextLevel(c) for c in (context_levels or ["minimal", "standard", "rich"])]
    rag_enums = [RAGMode(r) for r in (rag_modes or ["none"])]

    # Tool configurations
    tool_configs = [[]]  # No tools by default
    if with_tools:
        tool_configs.append(["get_marker_info", "suggest_gate_type"])

    # Create config
    config = PipelineConfig(
        name="gating_benchmark",
        models=models or ["claude-sonnet-4-20250514"],
        reasoning_types=reasoning_enums,
        context_levels=context_enums,
        rag_modes=rag_enums,
        tool_configs=tool_configs,
        output_dir=Path(output_dir),
        checkpoint_dir=Path(output_dir) / "checkpoints",
        # Custom strategy config for CoT
        strategy_configs={
            "cot": {
                "reasoning_prompt": """Think through this step-by-step:

1. **Quality Control Gates**: What initial QC gates are needed?
2. **Major Lineage Identification**: What major cell lineages can this panel identify?
3. **Subset Identification**: What subsets can be identified within each lineage?
4. **Hierarchy Structure**: Organize into a complete gating tree.""",
                "answer_prefix": "Final Answer:",
            },
        },
    )

    # Load trial inputs
    trial_inputs = load_trial_inputs(test_cases_dir)

    # Create evaluator
    evaluator = GatingEvaluator()

    # Create tool registry
    tool_registry = create_flowkit_tools() if with_tools else None

    # Create custom context builders
    context_builders = {
        ContextLevel.MINIMAL: GatingContextBuilder(
            include_rag=False,
            include_tools=False,
        ),
        ContextLevel.STANDARD: GatingContextBuilder(
            include_rag=True,
            include_tools=True,
        ),
        ContextLevel.RICH: GatingContextBuilder(
            include_rag=True,
            include_tools=True,
        ),
    }
    # Set context levels
    context_builders[ContextLevel.MINIMAL]._level = ContextLevel.MINIMAL
    context_builders[ContextLevel.STANDARD]._level = ContextLevel.STANDARD
    context_builders[ContextLevel.RICH]._level = ContextLevel.RICH

    # Create custom strategies
    strategies = {
        ReasoningType.COT: GatingCoTStrategy(),
    }

    # Create RAG providers
    rag_providers = {
        RAGMode.NONE: None,  # Will use default
        RAGMode.ORACLE: OracleRAGProvider(oracle_field="oracle_context"),
    }

    return HypothesisPipeline(
        config=config,
        evaluator=evaluator,
        trial_inputs=trial_inputs,
        tool_registry=tool_registry,
        context_builders=context_builders,
        strategies=strategies,
        rag_providers={k: v for k, v in rag_providers.items() if v is not None},
    )


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run modular gating benchmark experiment"
    )
    parser.add_argument(
        "--test-cases",
        required=True,
        help="Directory with test case JSON files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["claude-sonnet-4-20250514"],
        help="Models to test",
    )
    parser.add_argument(
        "--reasoning",
        nargs="+",
        default=["direct", "cot"],
        choices=["direct", "cot", "wot", "few_shot"],
        help="Reasoning strategies to test",
    )
    parser.add_argument(
        "--context",
        nargs="+",
        default=["minimal", "standard", "rich"],
        choices=["minimal", "standard", "rich", "oracle"],
        help="Context levels to test",
    )
    parser.add_argument(
        "--reference",
        nargs="+",
        default=["none"],
        choices=["none", "oracle", "vector"],
        dest="rag",  # Keep internal name for library compatibility
        help="Reference/retrieval modes: none, oracle (static HIPC), vector (real RAG)",
    )
    parser.add_argument(
        "--with-tools",
        action="store_true",
        help="Include conditions with tools enabled",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and run pipeline
    pipeline = create_pipeline(
        test_cases_dir=args.test_cases,
        output_dir=args.output,
        models=args.models,
        reasoning_types=args.reasoning,
        context_levels=args.context,
        rag_modes=args.rag,
        with_tools=args.with_tools,
    )

    logger.info(f"Created pipeline with {len(pipeline.conditions)} conditions")
    for cond in pipeline.conditions:
        logger.info(f"  - {cond.name}")

    results = pipeline.run(verbose=args.verbose)

    # Generate and save report
    report = pipeline.generate_report(results)
    logger.info("Pipeline completed. Report summary:")
    logger.info(report)

    report_path = Path(args.output) / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()

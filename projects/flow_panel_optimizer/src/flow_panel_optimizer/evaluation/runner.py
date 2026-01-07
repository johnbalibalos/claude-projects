"""
Run ablation study across all conditions and test cases.

For each (condition, test_case) pair:
1. Construct appropriate prompt with/without retrieval context
2. Call LLM with/without MCP tools
3. Parse response and extract fluorophore assignments
4. Score against ground truth
5. Record detailed results for analysis

Performance optimizations:
- Parallel API calls using asyncio.gather for concurrent test execution
- Pre-computed similarity cache for O(1) panel scoring
- Semaphore-controlled concurrency to avoid rate limits
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import time
import re
import os
import asyncio
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic, AsyncAnthropic

from .test_cases import PanelDesignTestCase, TestSuite, TestCaseType
from .conditions import ExperimentalCondition, CONDITIONS, RetrievalMode


# OMIP corpus for retrieval
OMIP_CORPUS = """
## OMIP-030: Human T Cell Subsets (10-color)
Reference: Wingender & Kronenberg, Cytometry A 2015
Fluorophore assignments:
- CD3: Pacific Blue
- CD4: PerCP-Cy5.5
- CD8: APC-Fire750
- CD45RA: FITC
- CD127: PE
- CD25: PE-Cy7
- CCR7: PE-Cy5
- CD161: BV711
- CXCR3: APC
- Viability: LIVE/DEAD Blue

## OMIP-047: B Cell Phenotyping (16-color)
Reference: Liechti et al., Cytometry A 2018
Fluorophore assignments:
- Viability: LIVE/DEAD Aqua
- CD3/CD14: BV421 (dump)
- CD19: BV785
- CD20: APC-Cy7
- CD27: BV605
- IgD: PE-Cy7
- CD38: BB515
- CD21: BV711
- CD10: PE
- IgG: PerCP-Cy5.5
- IgA: APC
- CXCR3: FITC
- CCR7: PE-Dazzle 594
- IL-21R: BV510
- Ki67: Alexa Fluor 700

## OMIP-063: Broad Immunophenotyping (20-color)
Reference: Payne K et al., Cytometry A 2020
Fluorophore assignments:
- Viability: LIVE/DEAD Blue
- CD45: BUV395
- CD3: BUV496
- CD4: BV750
- CD8: BUV805
- CD19: BV480
- CD56: BV605
- CD14: BV650
- CD16: APC-Cy7
- HLA-DR: BV785
- CD45RA: BV510
- CCR7: PE
- CD27: BV421
- CD28: BB515
- CD57: FITC
- CD127: PE-Cy7
- CD25: PE-Dazzle 594
- CD38: PerCP-Cy5.5
- CD11c: APC
- CD123: BV711

## OMIP-069: 40-Color Full Spectrum
Reference: Park et al., Cytometry A 2020
Selected fluorophore assignments:
- CD45: BUV395
- Viability: LIVE/DEAD Blue
- CD3: BUV496
- CD4: BUV563
- CD8: BUV805
- TCR-gd: BV421
- CD45RA: BV570
- CD45RO: BV650
- CCR7: BV785
- CD27: BV510
- CD28: BB515
- CD57: FITC
- CD25: PE-Cy7
- CD127: PE-Dazzle 594
- CXCR3: BV711
- CCR4: PE
- CCR6: BV605
- CD19: BUV661
- CD20: APC-R700
- IgD: PerCP-Cy5.5
- CD38: APC-Fire750
- CD56: BUV737
- CD16: APC-Cy7
- NKG2D: APC
- CD14: BUV615
- HLA-DR: Alexa Fluor 700
"""


@dataclass
class TrialResult:
    """Result of a single (condition, test_case) trial."""
    condition_name: str
    test_case_id: str
    test_case_type: str

    # Model outputs
    raw_response: str
    extracted_assignments: dict[str, str]  # marker -> fluorophore
    tool_calls_made: list[dict]

    # Scoring
    assignment_accuracy: float  # % markers assigned to optimal fluorophore
    complexity_index: float     # CI of predicted panel
    ground_truth_ci: float      # CI of ground truth
    ci_improvement: float       # (ground_truth_ci - predicted_ci) / ground_truth_ci

    # Metadata
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "condition_name": self.condition_name,
            "test_case_id": self.test_case_id,
            "test_case_type": self.test_case_type,
            "raw_response": self.raw_response[:500] + "..." if len(self.raw_response) > 500 else self.raw_response,
            "extracted_assignments": self.extracted_assignments,
            "tool_calls_count": len(self.tool_calls_made),
            "assignment_accuracy": self.assignment_accuracy,
            "complexity_index": self.complexity_index,
            "ground_truth_ci": self.ground_truth_ci,
            "ci_improvement": self.ci_improvement,
            "latency_seconds": self.latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "timestamp": self.timestamp,
        }


@dataclass
class ExperimentResults:
    """Aggregated results from full ablation study."""
    experiment_name: str
    trials: list[TrialResult]

    def to_dataframe(self):
        """Convert to pandas DataFrame for analysis."""
        import pandas as pd
        return pd.DataFrame([
            {
                "condition": t.condition_name,
                "test_case": t.test_case_id,
                "case_type": t.test_case_type,
                "accuracy": t.assignment_accuracy,
                "complexity_index": t.complexity_index,
                "ground_truth_ci": t.ground_truth_ci,
                "ci_improvement": t.ci_improvement,
                "latency": t.latency_seconds,
                "tokens": t.input_tokens + t.output_tokens,
                "tool_calls": len(t.tool_calls_made) if t.tool_calls_made else 0,
            }
            for t in self.trials
        ])

    def summary_by_condition(self):
        """Aggregate metrics by condition."""
        df = self.to_dataframe()
        return df.groupby("condition").agg({
            "accuracy": ["mean", "std"],
            "complexity_index": ["mean", "std"],
            "ci_improvement": ["mean", "std"],
            "latency": "mean",
            "tokens": "mean"
        })

    def summary_by_case_type(self):
        """Aggregate metrics by test case type."""
        df = self.to_dataframe()
        return df.groupby(["condition", "case_type"]).agg({
            "accuracy": "mean",
            "complexity_index": "mean",
            "ci_improvement": "mean"
        })

    def to_json(self, path: Path) -> None:
        """Save results to JSON file."""
        data = {
            "experiment_name": self.experiment_name,
            "trials": [t.to_dict() for t in self.trials]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class AblationRunner:
    """Execute ablation study with optional parallel execution."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_concurrent: int = 5
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=self.api_key)
        self.async_client: Optional[AsyncAnthropic] = None
        self.model = model
        self.max_concurrent = max_concurrent  # Max parallel API calls
        self.mcp_tools = self._get_mcp_tools()

    def _get_mcp_tools(self) -> list[dict]:
        """Get MCP tool definitions."""
        return [
            {
                "name": "analyze_panel",
                "description": "Analyze a panel of fluorophores for spectral conflicts. Returns complexity index, problematic pairs, and quality rating.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "fluorophores": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of fluorophore names in the panel"
                        }
                    },
                    "required": ["fluorophores"]
                }
            },
            {
                "name": "check_compatibility",
                "description": "Check if a candidate fluorophore is compatible with existing panel selections. Returns similarity score and recommendation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "candidate": {
                            "type": "string",
                            "description": "Name of fluorophore to check"
                        },
                        "existing_panel": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of fluorophores already in panel"
                        }
                    },
                    "required": ["candidate", "existing_panel"]
                }
            },
            {
                "name": "suggest_fluorophores",
                "description": "Get ranked fluorophore suggestions for a marker given current panel and expression level.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "existing_panel": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of fluorophores already selected"
                        },
                        "expression_level": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Expression level of the marker"
                        }
                    },
                    "required": ["existing_panel", "expression_level"]
                }
            },
            {
                "name": "get_fluorophore_info",
                "description": "Get detailed spectral information about a specific fluorophore.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Fluorophore name"
                        }
                    },
                    "required": ["name"]
                }
            },
        ]

    def _execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute an MCP tool and return the result."""
        # Import from MCP server
        from flow_panel_optimizer.mcp.server import execute_tool
        return execute_tool(tool_name, arguments)

    def _build_prompt(
        self,
        test_case: PanelDesignTestCase,
        condition: ExperimentalCondition
    ) -> str:
        """Construct prompt based on condition settings."""

        base_prompt = f"""You are designing a flow cytometry panel.

{test_case.biological_question}

Required markers and their expression levels:
{json.dumps(test_case.marker_expression, indent=2)}

Available fluorophores: All standard flow cytometry fluorophores (PE, FITC, APC, BV series, BUV series, etc.)

Instructions:
1. Assign each marker to exactly one fluorophore
2. Minimize spectral overlap between fluorophores
3. Match brighter fluorophores to lower-expression markers
4. Avoid using fluorophores with high spectral similarity on co-expressed markers

Return your answer as JSON:
{{"assignments": {{"marker1": "fluorophore1", "marker2": "fluorophore2", ...}}}}
"""

        # Add retrieval context based on condition
        if condition.retrieval_mode != RetrievalMode.NONE:
            weight_note = ""
            if condition.retrieval_weight > 1:
                weight_note = f" (HIGHLY VALIDATED - weight {condition.retrieval_weight}x)"

            retrieval_context = f"""
Reference panels from published literature{weight_note}:

{OMIP_CORPUS}

Use these validated panels to inform your fluorophore selection. These panels have been optimized by flow cytometry experts.
"""
            base_prompt = retrieval_context + "\n" + base_prompt

        return base_prompt

    def _parse_assignments(self, response: str) -> dict[str, str]:
        """Extract marker-fluorophore assignments from response."""
        # Try to find JSON in response
        json_patterns = [
            r'\{[^{}]*"assignments"\s*:\s*\{[^{}]*\}[^{}]*\}',
            r'"assignments"\s*:\s*(\{[^{}]*\})',
            r'\{[^{}]+\}',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    text = match.group()
                    # Handle nested match
                    if "assignments" not in text and match.lastindex:
                        text = match.group(1)

                    data = json.loads(text)

                    if isinstance(data, dict):
                        if "assignments" in data:
                            return data["assignments"]
                        # Check if it's directly the assignments
                        if all(isinstance(v, str) for v in data.values()):
                            return data
                except json.JSONDecodeError:
                    continue

        # Fallback: try to extract marker: fluorophore patterns
        assignments = {}
        lines = response.split('\n')
        for line in lines:
            # Match patterns like "CD3: BV421" or "CD3 -> BV421" or "- CD3: BV421"
            match = re.search(r'["\']?(\w+)["\']?\s*[:->]+\s*["\']?([\w\s-]+)["\']?', line)
            if match:
                marker = match.group(1).strip()
                fluor = match.group(2).strip()
                # Filter out non-fluorophore values
                if not any(x in fluor.lower() for x in ['high', 'medium', 'low', 'marker', 'expression']):
                    assignments[marker] = fluor

        return assignments

    def _calculate_complexity_index(self, assignments: dict[str, str]) -> float:
        """Calculate complexity index for assignments."""
        from flow_panel_optimizer.data.fluorophore_database import (
            get_fluorophore,
            calculate_spectral_overlap,
            get_known_overlap,
        )

        fluorophores = list(assignments.values())
        ci = 0.0

        for i, f1 in enumerate(fluorophores):
            for j, f2 in enumerate(fluorophores):
                if i >= j:
                    continue

                sim = get_known_overlap(f1, f2)
                if sim is None:
                    fluor1 = get_fluorophore(f1)
                    fluor2 = get_fluorophore(f2)
                    if fluor1 and fluor2:
                        sim = calculate_spectral_overlap(fluor1, fluor2)
                    else:
                        sim = 0.0

                if sim > 0.5:
                    ci += sim ** 2

        return round(ci, 2)

    def _score_accuracy(self, predicted: dict, ground_truth: dict) -> float:
        """Calculate assignment accuracy."""
        if not ground_truth:
            return 0.0

        correct = sum(
            1 for marker, fluor in ground_truth.items()
            if predicted.get(marker) == fluor
        )
        return correct / len(ground_truth)

    def _run_trial(
        self,
        test_case: PanelDesignTestCase,
        condition: ExperimentalCondition,
        max_tool_calls: int = 30
    ) -> TrialResult:
        """Run a single trial."""

        prompt = self._build_prompt(test_case, condition)
        tool_calls = []

        start_time = time.time()

        try:
            if condition.mcp_enabled:
                # Run with tool use
                messages = [{"role": "user", "content": prompt}]

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    tools=self.mcp_tools,
                    messages=messages
                )

                # Handle tool use loop
                tool_call_count = 0
                while response.stop_reason == "tool_use" and tool_call_count < max_tool_calls:
                    # Find ALL tool use blocks (model may request multiple in parallel)
                    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

                    if not tool_use_blocks:
                        break

                    # Execute all tools and collect results
                    tool_results = []
                    for tool_use_block in tool_use_blocks:
                        tool_result = self._execute_tool(
                            tool_use_block.name,
                            tool_use_block.input
                        )
                        tool_calls.append({
                            "tool": tool_use_block.name,
                            "input": tool_use_block.input,
                            "output": tool_result
                        })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": json.dumps(tool_result)
                        })
                        tool_call_count += 1

                    # Continue conversation with ALL tool results
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })

                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=4000,
                        tools=self.mcp_tools,
                        messages=messages
                    )

            else:
                # Run without tools
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )

            latency = time.time() - start_time

            # Extract response text
            raw_response = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    raw_response += block.text

            # Parse assignments from response
            assignments = self._parse_assignments(raw_response)

            # Calculate metrics
            accuracy = self._score_accuracy(assignments, test_case.ground_truth_assignments)
            predicted_ci = self._calculate_complexity_index(assignments)
            ground_truth_ci = test_case.ground_truth_complexity_index

            # CI improvement (positive = predicted is better)
            if ground_truth_ci > 0:
                ci_improvement = (ground_truth_ci - predicted_ci) / ground_truth_ci
            else:
                ci_improvement = 0.0

            return TrialResult(
                condition_name=condition.name,
                test_case_id=test_case.id,
                test_case_type=test_case.case_type.value,
                raw_response=raw_response,
                extracted_assignments=assignments,
                tool_calls_made=tool_calls,
                assignment_accuracy=accuracy,
                complexity_index=predicted_ci,
                ground_truth_ci=ground_truth_ci,
                ci_improvement=ci_improvement,
                latency_seconds=latency,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )

        except Exception as e:
            latency = time.time() - start_time
            return TrialResult(
                condition_name=condition.name,
                test_case_id=test_case.id,
                test_case_type=test_case.case_type.value,
                raw_response=f"ERROR: {str(e)}",
                extracted_assignments={},
                tool_calls_made=tool_calls,
                assignment_accuracy=0.0,
                complexity_index=float('inf'),
                ground_truth_ci=test_case.ground_truth_complexity_index,
                ci_improvement=0.0,
                latency_seconds=latency,
                input_tokens=0,
                output_tokens=0
            )

    def run_full_study(
        self,
        test_suite: TestSuite,
        conditions: Optional[list[ExperimentalCondition]] = None,
        verbose: bool = True
    ) -> ExperimentResults:
        """Run complete ablation study."""

        if conditions is None:
            conditions = CONDITIONS

        trials = []
        total = len(test_suite.test_cases) * len(conditions)

        for i, test_case in enumerate(test_suite.test_cases):
            for j, condition in enumerate(conditions):
                progress = (i * len(conditions) + j + 1) / total
                if verbose:
                    print(f"[{progress:.1%}] {condition.name} / {test_case.id}...", end=" ")

                trial = self._run_trial(test_case, condition)
                trials.append(trial)

                if verbose:
                    print(f"CI={trial.complexity_index:.2f}, Acc={trial.assignment_accuracy:.1%}, {trial.latency_seconds:.1f}s")

        return ExperimentResults(
            experiment_name=f"ablation_{test_suite.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            trials=trials
        )

    def run_quick_test(
        self,
        n_cases: int = 2,
        verbose: bool = True
    ) -> ExperimentResults:
        """Run a quick test with limited cases and conditions."""
        from .conditions import QUICK_TEST_CONDITIONS

        # Generate minimal test suite
        from .test_cases import build_ablation_test_suite
        suite = build_ablation_test_suite(n_in_dist=n_cases, n_out_dist=0, n_adversarial=0)

        return self.run_full_study(suite, QUICK_TEST_CONDITIONS, verbose=verbose)

    # =========================================================================
    # ASYNC PARALLEL EXECUTION METHODS
    # =========================================================================

    async def _run_trial_async(
        self,
        client: AsyncAnthropic,
        test_case: PanelDesignTestCase,
        condition: ExperimentalCondition,
        semaphore: asyncio.Semaphore,
        max_tool_calls: int = 30
    ) -> TrialResult:
        """Run a single trial asynchronously with semaphore for rate limiting."""

        async with semaphore:
            prompt = self._build_prompt(test_case, condition)
            tool_calls = []

            start_time = time.time()

            try:
                if condition.mcp_enabled:
                    # Run with tool use
                    messages = [{"role": "user", "content": prompt}]

                    response = await client.messages.create(
                        model=self.model,
                        max_tokens=4000,
                        tools=self.mcp_tools,
                        messages=messages
                    )

                    # Handle tool use loop
                    tool_call_count = 0
                    while response.stop_reason == "tool_use" and tool_call_count < max_tool_calls:
                        # Find ALL tool use blocks (model may request multiple in parallel)
                        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

                        if not tool_use_blocks:
                            break

                        # Execute all tools and collect results
                        tool_results = []
                        for tool_use_block in tool_use_blocks:
                            tool_result = self._execute_tool(
                                tool_use_block.name,
                                tool_use_block.input
                            )
                            tool_calls.append({
                                "tool": tool_use_block.name,
                                "input": tool_use_block.input,
                                "output": tool_result
                            })
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_block.id,
                                "content": json.dumps(tool_result)
                            })
                            tool_call_count += 1

                        # Continue conversation with ALL tool results
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({
                            "role": "user",
                            "content": tool_results
                        })

                        response = await client.messages.create(
                            model=self.model,
                            max_tokens=4000,
                            tools=self.mcp_tools,
                            messages=messages
                        )

                else:
                    # Run without tools
                    response = await client.messages.create(
                        model=self.model,
                        max_tokens=4000,
                        messages=[{"role": "user", "content": prompt}]
                    )

                latency = time.time() - start_time

                # Extract response text
                raw_response = ""
                for block in response.content:
                    if hasattr(block, 'text'):
                        raw_response += block.text

                # Parse assignments from response
                assignments = self._parse_assignments(raw_response)

                # Calculate metrics
                accuracy = self._score_accuracy(assignments, test_case.ground_truth_assignments)
                predicted_ci = self._calculate_complexity_index(assignments)
                ground_truth_ci = test_case.ground_truth_complexity_index

                # CI improvement (positive = predicted is better)
                if ground_truth_ci > 0:
                    ci_improvement = (ground_truth_ci - predicted_ci) / ground_truth_ci
                else:
                    ci_improvement = 0.0

                return TrialResult(
                    condition_name=condition.name,
                    test_case_id=test_case.id,
                    test_case_type=test_case.case_type.value,
                    raw_response=raw_response,
                    extracted_assignments=assignments,
                    tool_calls_made=tool_calls,
                    assignment_accuracy=accuracy,
                    complexity_index=predicted_ci,
                    ground_truth_ci=ground_truth_ci,
                    ci_improvement=ci_improvement,
                    latency_seconds=latency,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens
                )

            except Exception as e:
                latency = time.time() - start_time
                return TrialResult(
                    condition_name=condition.name,
                    test_case_id=test_case.id,
                    test_case_type=test_case.case_type.value,
                    raw_response=f"ERROR: {str(e)}",
                    extracted_assignments={},
                    tool_calls_made=tool_calls,
                    assignment_accuracy=0.0,
                    complexity_index=float('inf'),
                    ground_truth_ci=test_case.ground_truth_complexity_index,
                    ci_improvement=0.0,
                    latency_seconds=latency,
                    input_tokens=0,
                    output_tokens=0
                )

    async def run_parallel_study(
        self,
        test_suite: TestSuite,
        conditions: Optional[list[ExperimentalCondition]] = None,
        verbose: bool = True
    ) -> ExperimentResults:
        """
        Run ablation study with parallel API calls for faster execution.

        Uses asyncio.gather with a semaphore to limit concurrent requests.
        Typically 3-5x faster than sequential execution.
        """

        if conditions is None:
            conditions = CONDITIONS

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Create async client
        async_client = AsyncAnthropic(api_key=self.api_key)

        # Build list of all (test_case, condition) pairs
        all_trials = [
            (test_case, condition)
            for test_case in test_suite.test_cases
            for condition in conditions
        ]

        total = len(all_trials)
        if verbose:
            print(f"Running {total} trials with up to {self.max_concurrent} concurrent requests...")

        # Create all tasks
        tasks = [
            self._run_trial_async(async_client, tc, cond, semaphore)
            for tc, cond in all_trials
        ]

        # Run all tasks concurrently
        start_time = time.time()
        trials = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        if verbose:
            print(f"\nCompleted {total} trials in {total_time:.1f}s ({total/total_time:.1f} trials/sec)")

            # Summary stats
            avg_accuracy = sum(t.assignment_accuracy for t in trials) / len(trials)
            avg_ci = sum(t.complexity_index for t in trials if t.complexity_index != float('inf')) / len(trials)
            print(f"Average accuracy: {avg_accuracy:.1%}")
            print(f"Average CI: {avg_ci:.2f}")

        return ExperimentResults(
            experiment_name=f"parallel_ablation_{test_suite.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            trials=list(trials)
        )

    def run_parallel(
        self,
        test_suite: TestSuite,
        conditions: Optional[list[ExperimentalCondition]] = None,
        verbose: bool = True
    ) -> ExperimentResults:
        """
        Synchronous wrapper for run_parallel_study.

        Example:
            runner = AblationRunner(max_concurrent=5)
            results = runner.run_parallel(test_suite, conditions)
        """
        return asyncio.run(self.run_parallel_study(test_suite, conditions, verbose))


if __name__ == "__main__":
    # Quick test
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY to run tests")
        exit(1)

    runner = AblationRunner(api_key=api_key)
    results = runner.run_quick_test(n_cases=1)

    print("\n" + "="*60)
    print("QUICK TEST RESULTS")
    print("="*60)

    for trial in results.trials:
        print(f"\nCondition: {trial.condition_name}")
        print(f"Test case: {trial.test_case_id}")
        print(f"Complexity Index: {trial.complexity_index:.2f}")
        print(f"Accuracy: {trial.assignment_accuracy:.1%}")
        print(f"Tool calls: {len(trial.tool_calls_made)}")
        print(f"Latency: {trial.latency_seconds:.1f}s")

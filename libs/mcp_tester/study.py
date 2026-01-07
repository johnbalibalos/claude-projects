"""
Main AblationStudy runner with checkpointing support.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

from anthropic import Anthropic

# Add parent to path for checkpoint import
sys.path.insert(0, str(Path(__file__).parent.parent))
from checkpoint import CheckpointedRunner

from .models import TestCase, Condition, TrialResult, StudyResults
from .evaluators import Evaluator, AccuracyEvaluator


class AblationStudy:
    """
    Run ablation study comparing conditions with automatic checkpointing.

    Example:
        study = AblationStudy(
            name="my_mcp_test",
            model="claude-sonnet-4-20250514",
            test_cases=[...],
            conditions=[...],
            evaluator=MyEvaluator(),
            tool_executor=my_tool_fn
        )

        results = study.run()
        study.generate_report(results)
    """

    def __init__(
        self,
        name: str,
        model: str,
        test_cases: list[TestCase],
        conditions: list[Condition],
        evaluator: Evaluator = None,
        tool_executor: Optional[Callable[[str, dict], dict]] = None,
        api_key: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
        max_tool_calls: int = 30
    ):
        self.name = name
        self.model = model
        self.test_cases = test_cases
        self.conditions = conditions
        self.evaluator = evaluator or AccuracyEvaluator()
        self.tool_executor = tool_executor
        self.max_tool_calls = max_tool_calls

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=self.api_key)

        self.checkpoint_dir = Path(checkpoint_dir or "./checkpoints")
        self.checkpoint = CheckpointedRunner(
            f"{name}_{model.split('-')[1]}",  # e.g., my_test_sonnet
            checkpoint_dir=self.checkpoint_dir
        )

    def _build_prompt(self, test_case: TestCase, condition: Condition) -> str:
        """Build full prompt from test case and condition."""
        if condition.context:
            return f"{condition.context}\n\n{test_case.prompt}"
        return test_case.prompt

    def _execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool and return result."""
        if self.tool_executor:
            return self.tool_executor(tool_name, arguments)
        return {"error": f"No executor for tool: {tool_name}"}

    def _run_trial(
        self,
        test_case: TestCase,
        condition: Condition
    ) -> TrialResult:
        """Run a single trial."""
        prompt = self._build_prompt(test_case, condition)
        tool_calls = []
        start_time = time.time()

        try:
            messages = [{"role": "user", "content": prompt}]

            if condition.tools_enabled and condition.tools:
                # Run with tools
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    tools=condition.tools,
                    messages=messages
                )

                # Handle tool use loop
                tool_call_count = 0
                while response.stop_reason == "tool_use" and tool_call_count < self.max_tool_calls:
                    # Find ALL tool use blocks
                    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

                    if not tool_use_blocks:
                        break

                    # Execute all tools
                    tool_results = []
                    for block in tool_use_blocks:
                        result = self._execute_tool(block.name, block.input)
                        tool_calls.append({
                            "tool": block.name,
                            "input": block.input,
                            "output": result
                        })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result)
                        })
                        tool_call_count += 1

                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})

                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=4000,
                        tools=condition.tools,
                        messages=messages
                    )
            else:
                # Run without tools
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    messages=messages
                )

            latency = time.time() - start_time

            # Extract response text
            raw_response = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    raw_response += block.text

            # Evaluate
            extracted = self.evaluator.extract(raw_response)
            scores = self.evaluator.score(extracted, test_case.ground_truth)

            return TrialResult(
                condition_name=condition.name,
                test_case_id=test_case.id,
                test_case_type=test_case.case_type.value,
                raw_response=raw_response,
                extracted_output=extracted,
                tool_calls=tool_calls,
                scores=scores,
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
                error=str(e),
                latency_seconds=latency
            )

    def run(self, verbose: bool = True) -> StudyResults:
        """Run the full ablation study with checkpointing."""
        started_at = datetime.now().isoformat()
        trials = []

        # Build list of all (test_case, condition) pairs
        all_pairs = [
            (tc, cond)
            for tc in self.test_cases
            for cond in self.conditions
        ]

        total = len(all_pairs)

        if verbose:
            print(f"\n{'='*60}")
            print(f"ABLATION STUDY: {self.name}")
            print(f"{'='*60}")
            print(f"Model: {self.model}")
            print(f"Test cases: {len(self.test_cases)}")
            print(f"Conditions: {len(self.conditions)}")
            print(f"Total trials: {total}")
            print(f"{'='*60}\n")

        # Iterate with checkpointing
        for tc, cond in self.checkpoint.iterate(
            all_pairs,
            key_fn=lambda x: f"{x[0].id}_{x[1].name}"
        ):
            key = f"{tc.id}_{cond.name}"
            completed, _ = self.checkpoint.progress()

            if verbose:
                pct = (completed + 1) / total * 100
                print(f"[{pct:5.1f}%] {cond.name} / {tc.id}...", end=" ", flush=True)

            result = self._run_trial(tc, cond)
            trials.append(result)

            # Save to checkpoint
            self.checkpoint.save_result(key, result.to_dict())

            if verbose:
                if result.error:
                    print(f"ERROR: {result.error[:50]}")
                else:
                    score_str = ", ".join(f"{k}={v:.2f}" for k, v in result.scores.items())
                    print(f"{score_str}, {result.latency_seconds:.1f}s")

        # Load any previously completed trials
        for key, data in self.checkpoint.get_all_results().items():
            if not any(t.test_case_id == data['test_case_id'] and
                       t.condition_name == data['condition_name']
                       for t in trials):
                trials.append(TrialResult(**data))

        return StudyResults(
            study_name=self.name,
            model=self.model,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            trials=trials
        )

    def generate_report(
        self,
        results: StudyResults,
        output_path: Optional[Path] = None
    ) -> str:
        """Generate markdown report from results."""
        lines = [
            f"# Ablation Study Report: {results.study_name}",
            "",
            f"**Model:** {results.model}",
            f"**Started:** {results.started_at}",
            f"**Completed:** {results.completed_at}",
            f"**Total Trials:** {len(results.trials)}",
            f"**Success Rate:** {results.success_rate():.1%}",
            "",
            "## Results by Condition",
            ""
        ]

        # Group by condition
        for cond in self.conditions:
            cond_trials = results.filter_by_condition(cond.name)
            if not cond_trials:
                continue

            lines.append(f"### {cond.name}")
            lines.append("")

            # Compute avg scores
            all_metrics = set()
            for t in cond_trials:
                all_metrics.update(t.scores.keys())

            for metric in sorted(all_metrics):
                scores = [t.scores.get(metric, 0) for t in cond_trials if t.success]
                if scores:
                    avg = sum(scores) / len(scores)
                    lines.append(f"- **{metric}:** {avg:.3f}")

            avg_latency = sum(t.latency_seconds for t in cond_trials) / len(cond_trials)
            lines.append(f"- **Avg Latency:** {avg_latency:.1f}s")
            lines.append("")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report

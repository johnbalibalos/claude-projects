"""
LLM-based judge for evaluating gating hierarchy predictions.

Uses an LLM (typically Gemini 2.5 Pro) to provide qualitative assessment
of predictions beyond automated metrics. Enables modular pipeline:

    PredictionCollector → BatchScorer → LLMJudge → ResultsAggregator
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.response_parser import parse_llm_response
from checkpoint import CheckpointManager

from .llm_client import create_client


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""

    # Identity
    test_case_id: str
    model: str
    condition: str
    bootstrap_run: int

    # Judge scores (0-1 scale, consistent with other metrics)
    completeness: float
    accuracy: float
    scientific: float
    overall: float

    # Qualitative feedback
    issues: str
    summary: str

    # Metadata
    judge_model: str
    tokens_used: int
    timestamp: datetime
    error: str | None = None

    # Full prompt/response for manual review
    judge_prompt: str = ""
    judge_raw_response: str = ""

    @property
    def key(self) -> tuple:
        """Unique key for deduplication."""
        return (self.bootstrap_run, self.test_case_id, self.model, self.condition)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_case_id": self.test_case_id,
            "model": self.model,
            "condition": self.condition,
            "bootstrap_run": self.bootstrap_run,
            "completeness": self.completeness,
            "accuracy": self.accuracy,
            "scientific": self.scientific,
            "overall": self.overall,
            "issues": self.issues,
            "summary": self.summary,
            "judge_model": self.judge_model,
            "tokens_used": self.tokens_used,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
            "judge_prompt": self.judge_prompt,
            "judge_raw_response": self.judge_raw_response,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeResult:
        """Create from dictionary."""
        # Normalize scores to 0-1 if they appear to be 0-10 scale
        def normalize_score(val: float | int) -> float:
            val = float(val) if val else 0.0
            return val / 10.0 if val > 1.0 else val

        return cls(
            test_case_id=data["test_case_id"],
            model=data["model"],
            condition=data["condition"],
            bootstrap_run=data["bootstrap_run"],
            completeness=normalize_score(data.get("completeness", 0)),
            accuracy=normalize_score(data.get("accuracy", 0)),
            scientific=normalize_score(data.get("scientific", 0)),
            overall=normalize_score(data.get("overall", 0)),
            issues=data.get("issues", ""),
            summary=data.get("summary", ""),
            judge_model=data.get("judge_model", "unknown"),
            tokens_used=data.get("tokens_used", 0),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            error=data.get("error"),
            judge_prompt=data.get("judge_prompt", ""),
            judge_raw_response=data.get("judge_raw_response", ""),
        )


# Available judge prompt styles
JUDGE_STYLES = ["default", "validation", "qualitative", "orthogonal", "binary"]


@dataclass
class JudgeConfig:
    """Configuration for LLM judge."""

    model: str = "gemini-2.5-pro"
    max_tokens: int = 10000  # High to accommodate thinking tokens
    temperature: float = 0.0
    parallel_workers: int = 50  # High parallelism for Gemini paid tier
    delay_seconds: float = 0.0  # No delay needed on paid tier
    checkpoint_dir: Path | None = None
    dry_run: bool = False
    prompt_style: str = "default"  # One of JUDGE_STYLES


def flatten_hierarchy(hierarchy: dict, path: str = "") -> str:
    """Convert hierarchy dict to flat arrow notation."""
    if "root" in hierarchy:
        hierarchy = hierarchy["root"]

    name = hierarchy.get("name", "Unknown")
    current = f"{path} > {name}" if path else name

    children = hierarchy.get("children", [])
    if not children:
        return current

    paths = []
    for child in children:
        paths.append(flatten_hierarchy(child, current))

    return "\n".join(paths)


def _is_garbage_hierarchy(hierarchy: dict) -> bool:
    """Check if hierarchy contains JSON syntax instead of real gate names.

    Truncated JSON can parse as indented text, producing gate names like
    '"name"', '"children"', '```json', etc.
    """
    json_artifacts = {'"name"', '"children"', '```json', '```', '"markers"', '"gate_name"'}

    def check_node(node: dict) -> bool:
        name = node.get("name", "")
        # Check for JSON artifacts
        if name.lower() in json_artifacts or name.startswith('"') or name.startswith('```'):
            return True
        # Check for JSON structure characters that shouldn't be in gate names
        if any(c in name for c in ['{', '}', '[', ']']) and len(name) < 20:
            return True
        # Check children
        for child in node.get("children", []):
            if check_node(child):
                return True
        return False

    return check_node(hierarchy)


def format_prediction_for_judge(predicted_response: str) -> str:
    """Parse and format prediction for judge evaluation.

    Attempts to parse the JSON hierarchy and format as readable paths.
    Falls back to raw response if parsing fails.
    """
    if not predicted_response or not predicted_response.strip():
        return "[EMPTY RESPONSE]"

    # Try to parse the response
    parse_result = parse_llm_response(predicted_response)

    if parse_result.success and parse_result.hierarchy:
        # Validate parsed hierarchy doesn't contain garbage from truncated JSON
        if _is_garbage_hierarchy(parse_result.hierarchy):
            raw = predicted_response.strip()
            if len(raw) > 400:
                raw = raw[:400] + "... [truncated]"
            return f"[PARSE FAILED: malformed hierarchy from truncated JSON]\n{raw}"

        # Format as paths like ground truth
        pred_flat = flatten_hierarchy(parse_result.hierarchy)
        pred_lines = pred_flat.split("\n")

        # Show first 8 paths
        pred_summary = "\n".join(pred_lines[:8])
        if len(pred_lines) > 8:
            pred_summary += f"\n... (+{len(pred_lines) - 8} more paths)"

        return f"[Parsed {len(pred_lines)} gates]\n{pred_summary}"
    else:
        # Parsing failed - show truncated raw response
        raw = predicted_response.strip()
        if len(raw) > 400:
            raw = raw[:400] + "... [truncated]"
        return f"[PARSE FAILED: {parse_result.error or 'unknown'}]\n{raw}"


def build_judge_prompt(
    test_case_id: str,
    predicted_response: str,
    ground_truth: dict,
    metrics: dict,
) -> str:
    """Build a prompt for the LLM judge."""
    gt_hierarchy = ground_truth.get("gating_hierarchy", {})
    gt_flat = flatten_hierarchy(gt_hierarchy)

    # Get first 8 paths for ground truth
    gt_lines = gt_flat.split("\n")[:8]
    gt_summary = "\n".join(gt_lines)
    if len(gt_flat.split("\n")) > 8:
        gt_summary += f"\n... (+{len(gt_flat.split(chr(10))) - 8} more paths)"

    context = ground_truth.get("context", {})

    # Parse and format prediction
    pred_formatted = format_prediction_for_judge(predicted_response)

    prompt = f"""Score this flow cytometry gating hierarchy prediction (0-10 scale).

TEST CASE: {test_case_id}
SAMPLE: {context.get('sample_type', 'unknown')} ({context.get('species', 'unknown')})

EXPECTED HIERARCHY (ground truth):
{gt_summary}

PREDICTED HIERARCHY:
{pred_formatted}

AUTO METRICS:
- F1: {metrics.get('hierarchy_f1', 0):.2f}
- Structure: {metrics.get('structure_accuracy', 0):.2f}
- Critical gates: {metrics.get('critical_gate_recall', 0):.2f}

Rate on these dimensions (0-10 each):

Reply in this EXACT format (one line each):
COMPLETENESS: [0-10]
ACCURACY: [0-10]
SCIENTIFIC: [0-10]
OVERALL: [0-10]
ISSUES: [comma-separated list or "none"]
SUMMARY: [one sentence explanation]
"""
    return prompt


def build_validation_prompt(
    test_case_id: str,
    predicted_response: str,
    ground_truth: dict,
    metrics: dict,
) -> str:
    """Validation judge: Estimate auto metrics without seeing them."""
    gt_hierarchy = ground_truth.get("gating_hierarchy", {})
    gt_flat = flatten_hierarchy(gt_hierarchy)
    gt_lines = gt_flat.split("\n")[:8]
    gt_summary = "\n".join(gt_lines)
    if len(gt_flat.split("\n")) > 8:
        gt_summary += f"\n... (+{len(gt_flat.split(chr(10))) - 8} more paths)"

    context = ground_truth.get("context", {})
    pred_formatted = format_prediction_for_judge(predicted_response)

    return f"""You are evaluating a flow cytometry gating hierarchy prediction.

TEST CASE: {test_case_id}
SAMPLE: {context.get('sample_type', 'unknown')} ({context.get('species', 'unknown')})

EXPECTED HIERARCHY (ground truth):
{gt_summary}

PREDICTED HIERARCHY:
{pred_formatted}

WITHOUT seeing the automated metrics, estimate what these metrics would be:

Reply in this EXACT format:
ESTIMATED_F1: [0.0-1.0] (what fraction of gates match?)
ESTIMATED_STRUCTURE: [0.0-1.0] (what fraction of parent-child relationships are correct?)
ESTIMATED_CRITICAL_RECALL: [0.0-1.0] (are singlets/live/lineage gates present?)
CONFIDENCE: [high/medium/low]
REASONING: [one sentence explaining your estimates]
"""


def build_qualitative_prompt(
    test_case_id: str,
    predicted_response: str,
    ground_truth: dict,
    metrics: dict,
) -> str:
    """Qualitative judge: No scores, just structured feedback."""
    gt_hierarchy = ground_truth.get("gating_hierarchy", {})
    gt_flat = flatten_hierarchy(gt_hierarchy)
    gt_lines = gt_flat.split("\n")[:8]
    gt_summary = "\n".join(gt_lines)
    if len(gt_flat.split("\n")) > 8:
        gt_summary += f"\n... (+{len(gt_flat.split(chr(10))) - 8} more paths)"

    context = ground_truth.get("context", {})
    pred_formatted = format_prediction_for_judge(predicted_response)

    return f"""Analyze this flow cytometry gating hierarchy prediction.

TEST CASE: {test_case_id}
SAMPLE: {context.get('sample_type', 'unknown')} ({context.get('species', 'unknown')})

EXPECTED HIERARCHY (ground truth):
{gt_summary}

PREDICTED HIERARCHY:
{pred_formatted}

Provide structured feedback (no numerical scores):

Reply in this EXACT format:
ERRORS: [comma-separated list of specific errors, or "none"]
MISSING_GATES: [comma-separated list of missing gates, or "none"]
EXTRA_GATES: [comma-separated list of hallucinated/extra gates, or "none"]
STRUCTURE_VALID: [yes/no] [one sentence explanation]
ACCEPT_FOR_ANALYSIS: [yes/no] [one sentence explanation]
"""


def build_orthogonal_prompt(
    test_case_id: str,
    predicted_response: str,
    ground_truth: dict,
    metrics: dict,
) -> str:
    """Orthogonal judge: Rate dimensions auto metrics can't capture."""
    gt_hierarchy = ground_truth.get("gating_hierarchy", {})
    gt_flat = flatten_hierarchy(gt_hierarchy)
    gt_lines = gt_flat.split("\n")[:8]
    gt_summary = "\n".join(gt_lines)
    if len(gt_flat.split("\n")) > 8:
        gt_summary += f"\n... (+{len(gt_flat.split(chr(10))) - 8} more paths)"

    context = ground_truth.get("context", {})
    application = context.get("application", "immunophenotyping")
    pred_formatted = format_prediction_for_judge(predicted_response)

    return f"""You are a flow cytometry expert evaluating a gating hierarchy prediction.

TEST CASE: {test_case_id}
SAMPLE: {context.get('sample_type', 'unknown')} ({context.get('species', 'unknown')})
APPLICATION: {application}

EXPECTED HIERARCHY (ground truth):
{gt_summary}

PREDICTED HIERARCHY:
{pred_formatted}

Rate on dimensions that automated metrics CANNOT capture (0-10 each):

Reply in this EXACT format:
CLINICAL_UTILITY: [0-10] [Would this hierarchy work for the stated application?]
BIOLOGICAL_PLAUSIBILITY: [0-10] [Are parent-child relationships biologically sensible?]
HALLUCINATION_SEVERITY: [0-10] [0=no hallucinations, 10=severe invented gates]
MARKER_LOGIC: [0-10] [Are marker combinations used correctly for each gate?]
SUMMARY: [one sentence overall assessment]
"""


def build_binary_prompt(
    test_case_id: str,
    predicted_response: str,
    ground_truth: dict,
    metrics: dict,
) -> str:
    """Binary judge: Accept/reject with specific issues."""
    gt_hierarchy = ground_truth.get("gating_hierarchy", {})
    gt_flat = flatten_hierarchy(gt_hierarchy)
    gt_lines = gt_flat.split("\n")[:8]
    gt_summary = "\n".join(gt_lines)
    if len(gt_flat.split("\n")) > 8:
        gt_summary += f"\n... (+{len(gt_flat.split(chr(10))) - 8} more paths)"

    context = ground_truth.get("context", {})
    pred_formatted = format_prediction_for_judge(predicted_response)

    return f"""Evaluate whether this flow cytometry gating hierarchy prediction is acceptable.

TEST CASE: {test_case_id}
SAMPLE: {context.get('sample_type', 'unknown')} ({context.get('species', 'unknown')})

EXPECTED HIERARCHY (ground truth):
{gt_summary}

PREDICTED HIERARCHY:
{pred_formatted}

Reply in this EXACT format:
ACCEPTABLE: [yes/no]
CRITICAL_ERRORS: [comma-separated list of blocking issues, or "none"]
MISSING_GATES: [comma-separated list by name, or "none"]
EXTRA_GATES: [comma-separated list of hallucinated gates, or "none"]
CONFIDENCE: [high/medium/low]
RECOMMENDATION: [one sentence: what would make this acceptable?]
"""


def get_prompt_builder(style: str):
    """Get the prompt builder function for a given style."""
    builders = {
        "default": build_judge_prompt,
        "validation": build_validation_prompt,
        "qualitative": build_qualitative_prompt,
        "orthogonal": build_orthogonal_prompt,
        "binary": build_binary_prompt,
    }
    return builders.get(style, build_judge_prompt)


def parse_judge_response(content: str) -> dict | None:
    """Parse flat-format judge response.

    LLM returns scores 0-10, but we normalize to 0-1 for consistency with other metrics.
    """
    result = {}

    patterns = {
        "completeness": r"COMPLETENESS:\s*(\d+)",
        "accuracy": r"ACCURACY:\s*(\d+)",
        "scientific": r"SCIENTIFIC:\s*(\d+)",
        "overall": r"OVERALL:\s*(\d+)",
        "issues": r"ISSUES:\s*(.+)",
        "summary": r"SUMMARY:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key in ["completeness", "accuracy", "scientific", "overall"]:
                try:
                    # Normalize 0-10 score to 0-1 float
                    result[key] = int(value) / 10.0
                except ValueError:
                    pass
            else:
                result[key] = value

    # Need at least overall score
    if "overall" in result:
        return result
    return None


def parse_validation_response(content: str) -> dict | None:
    """Parse validation judge response (estimated metrics)."""
    result = {}

    patterns = {
        "estimated_f1": r"ESTIMATED_F1:\s*([0-9.]+)",
        "estimated_structure": r"ESTIMATED_STRUCTURE:\s*([0-9.]+)",
        "estimated_critical_recall": r"ESTIMATED_CRITICAL_RECALL:\s*([0-9.]+)",
        "confidence": r"CONFIDENCE:\s*(\w+)",
        "reasoning": r"REASONING:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key.startswith("estimated_"):
                try:
                    result[key] = float(value)
                except ValueError:
                    pass
            else:
                result[key] = value

    # Map to standard fields for JudgeResult
    if "estimated_f1" in result:
        result["overall"] = result["estimated_f1"]
        result["completeness"] = result.get("estimated_critical_recall", 0)
        result["accuracy"] = result.get("estimated_structure", 0)
        result["scientific"] = result.get("estimated_f1", 0)
        result["summary"] = result.get("reasoning", "")
        result["issues"] = f"confidence: {result.get('confidence', 'unknown')}"
        return result
    return None


def parse_qualitative_response(content: str) -> dict | None:
    """Parse qualitative judge response (no scores, structured feedback)."""
    result = {}

    patterns = {
        "errors": r"ERRORS:\s*(.+)",
        "missing_gates": r"MISSING_GATES:\s*(.+)",
        "extra_gates": r"EXTRA_GATES:\s*(.+)",
        "structure_valid": r"STRUCTURE_VALID:\s*(.+)",
        "accept_for_analysis": r"ACCEPT_FOR_ANALYSIS:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    # Convert to JudgeResult fields
    if "accept_for_analysis" in result:
        accept = result["accept_for_analysis"].lower().startswith("yes")
        result["overall"] = 1.0 if accept else 0.0
        result["completeness"] = 1.0 if result.get("missing_gates", "").lower() == "none" else 0.5
        result["accuracy"] = 1.0 if result.get("structure_valid", "").lower().startswith("yes") else 0.0
        result["scientific"] = 1.0 if result.get("extra_gates", "").lower() == "none" else 0.5
        result["issues"] = result.get("errors", "none")
        result["summary"] = result.get("accept_for_analysis", "")
        return result
    return None


def parse_orthogonal_response(content: str) -> dict | None:
    """Parse orthogonal judge response (different dimensions)."""
    result = {}

    patterns = {
        "clinical_utility": r"CLINICAL_UTILITY:\s*(\d+)",
        "biological_plausibility": r"BIOLOGICAL_PLAUSIBILITY:\s*(\d+)",
        "hallucination_severity": r"HALLUCINATION_SEVERITY:\s*(\d+)",
        "marker_logic": r"MARKER_LOGIC:\s*(\d+)",
        "summary": r"SUMMARY:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key != "summary":
                try:
                    result[key] = int(value) / 10.0  # Normalize to 0-1
                except ValueError:
                    pass
            else:
                result[key] = value

    # Map to standard fields
    if "clinical_utility" in result:
        # Invert hallucination (0=good, 10=bad -> 1=good, 0=bad)
        halluc = result.get("hallucination_severity", 0)
        result["overall"] = result.get("clinical_utility", 0)
        result["completeness"] = result.get("marker_logic", 0)
        result["accuracy"] = result.get("biological_plausibility", 0)
        result["scientific"] = 1.0 - halluc  # Invert: low hallucination = high scientific
        result["issues"] = f"hallucination_severity: {halluc:.1f}"
        return result
    return None


def parse_binary_response(content: str) -> dict | None:
    """Parse binary judge response (accept/reject with issues)."""
    result = {}

    patterns = {
        "acceptable": r"ACCEPTABLE:\s*(\w+)",
        "critical_errors": r"CRITICAL_ERRORS:\s*(.+)",
        "missing_gates": r"MISSING_GATES:\s*(.+)",
        "extra_gates": r"EXTRA_GATES:\s*(.+)",
        "confidence": r"CONFIDENCE:\s*(\w+)",
        "recommendation": r"RECOMMENDATION:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    # Convert to JudgeResult fields
    if "acceptable" in result:
        accept = result["acceptable"].lower() in ["yes", "true"]
        result["overall"] = 1.0 if accept else 0.0
        result["completeness"] = 1.0 if result.get("missing_gates", "").lower() == "none" else 0.0
        result["accuracy"] = 1.0 if result.get("critical_errors", "").lower() == "none" else 0.0
        result["scientific"] = 1.0 if result.get("extra_gates", "").lower() == "none" else 0.0
        result["issues"] = result.get("critical_errors", "none")
        result["summary"] = result.get("recommendation", "")
        return result
    return None


def get_response_parser(style: str):
    """Get the response parser function for a given style."""
    parsers = {
        "default": parse_judge_response,
        "validation": parse_validation_response,
        "qualitative": parse_qualitative_response,
        "orthogonal": parse_orthogonal_response,
        "binary": parse_binary_response,
    }
    return parsers.get(style, parse_judge_response)


class LLMJudge:
    """LLM-based judge for evaluating predictions.

    Supports:
    - Parallel evaluation
    - Checkpoint/resume
    - Ground truth lookup
    """

    def __init__(
        self,
        ground_truth_dir: Path | str,
        config: JudgeConfig | None = None,
    ):
        self.ground_truth_dir = Path(ground_truth_dir)
        self.config = config or JudgeConfig()
        self._gt_cache: dict[str, dict] = {}

        # Checkpoint manager for resume support
        self._checkpoint = CheckpointManager(self.config.checkpoint_dir)

    def _load_ground_truth(self, test_case_id: str) -> dict | None:
        """Load ground truth for a test case."""
        if test_case_id in self._gt_cache:
            return self._gt_cache[test_case_id]

        filename = test_case_id.lower().replace("-", "_") + ".json"
        gt_path = self.ground_truth_dir / filename

        if not gt_path.exists():
            return None

        with open(gt_path) as f:
            gt = json.load(f)
            self._gt_cache[test_case_id] = gt
            return gt

    def judge_all(
        self,
        scoring_results: list,
        progress_callback=None,
    ) -> list[JudgeResult]:
        """Judge all scoring results.

        Args:
            scoring_results: List of ScoringResult objects
            progress_callback: Optional callback(current, total, result)

        Returns:
            List of JudgeResult objects
        """
        if self.config.dry_run:
            return self._mock_judge_all(scoring_results)

        results = []
        total = len(scoring_results)

        # Create judge client
        client = create_client(self.config.model, dry_run=self.config.dry_run)

        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = {
                executor.submit(
                    self._judge_one,
                    sr,
                    client,
                ): sr
                for sr in scoring_results
            }

            for i, future in enumerate(as_completed(futures), 1):
                sr = futures[future]
                try:
                    result = future.result()
                    results.append(result)

                    if progress_callback:
                        progress_callback(i, total, result)

                except Exception as e:
                    result = JudgeResult(
                        test_case_id=sr.test_case_id,
                        model=sr.model,
                        condition=sr.condition,
                        bootstrap_run=sr.bootstrap_run,
                        completeness=0.0,
                        accuracy=0.0,
                        scientific=0.0,
                        overall=0.0,
                        issues="",
                        summary="",
                        judge_model=self.config.model,
                        tokens_used=0,
                        timestamp=datetime.now(),
                        error=str(e),
                    )
                    results.append(result)

        # Save checkpoint
        if self.config.checkpoint_dir:
            self.save_checkpoint(results)

        return results

    def _judge_one(self, scoring_result, client) -> JudgeResult:
        """Judge a single scoring result."""
        gt = self._load_ground_truth(scoring_result.test_case_id)

        if not gt:
            return JudgeResult(
                test_case_id=scoring_result.test_case_id,
                model=scoring_result.model,
                condition=scoring_result.condition,
                bootstrap_run=scoring_result.bootstrap_run,
                completeness=0.0,
                accuracy=0.0,
                scientific=0.0,
                overall=0.0,
                issues="",
                summary="",
                judge_model=self.config.model,
                tokens_used=0,
                timestamp=datetime.now(),
                error=f"Ground truth not found for {scoring_result.test_case_id}",
            )

        metrics = {
            "hierarchy_f1": scoring_result.hierarchy_f1,
            "structure_accuracy": scoring_result.structure_accuracy,
            "critical_gate_recall": scoring_result.critical_gate_recall,
        }

        # Use configured prompt style
        prompt_builder = get_prompt_builder(self.config.prompt_style)
        prompt = prompt_builder(
            test_case_id=scoring_result.test_case_id,
            predicted_response=scoring_result.raw_response,
            ground_truth=gt,
            metrics=metrics,
        )

        try:
            response = client.call(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            # Rate limit delay
            if self.config.delay_seconds > 0:
                time.sleep(self.config.delay_seconds)

            # Use configured response parser
            response_parser = get_response_parser(self.config.prompt_style)
            parsed = response_parser(response.content)

            if parsed:
                return JudgeResult(
                    test_case_id=scoring_result.test_case_id,
                    model=scoring_result.model,
                    condition=scoring_result.condition,
                    bootstrap_run=scoring_result.bootstrap_run,
                    completeness=parsed.get("completeness", 0),
                    accuracy=parsed.get("accuracy", 0),
                    scientific=parsed.get("scientific", 0),
                    overall=parsed.get("overall", 0),
                    issues=parsed.get("issues", ""),
                    summary=parsed.get("summary", ""),
                    judge_model=self.config.model,
                    tokens_used=response.tokens_used,
                    timestamp=datetime.now(),
                    judge_prompt=prompt,
                    judge_raw_response=response.content,
                )
            else:
                return JudgeResult(
                    test_case_id=scoring_result.test_case_id,
                    model=scoring_result.model,
                    condition=scoring_result.condition,
                    bootstrap_run=scoring_result.bootstrap_run,
                    completeness=0.0,
                    accuracy=0.0,
                    scientific=0.0,
                    overall=0.0,
                    issues="",
                    summary="",
                    judge_model=self.config.model,
                    tokens_used=response.tokens_used,
                    timestamp=datetime.now(),
                    error=f"Failed to parse judge response: {response.content[:100]}",
                    judge_prompt=prompt,
                    judge_raw_response=response.content,
                )

        except Exception as e:
            return JudgeResult(
                test_case_id=scoring_result.test_case_id,
                model=scoring_result.model,
                condition=scoring_result.condition,
                bootstrap_run=scoring_result.bootstrap_run,
                completeness=0.0,
                accuracy=0.0,
                scientific=0.0,
                overall=0.0,
                issues="",
                summary="",
                judge_model=self.config.model,
                tokens_used=0,
                timestamp=datetime.now(),
                error=str(e),
                judge_prompt=prompt,
            )

    def _mock_judge_all(self, scoring_results: list) -> list[JudgeResult]:
        """Return mock judge results for dry run."""
        results = []
        for sr in scoring_results:
            # Generate mock scores based on automated metrics (0-1 scale)
            f1 = getattr(sr, 'hierarchy_f1', 0.0)
            struct = getattr(sr, 'structure_accuracy', 0.0)

            results.append(JudgeResult(
                test_case_id=sr.test_case_id,
                model=sr.model,
                condition=sr.condition,
                bootstrap_run=sr.bootstrap_run,
                completeness=f1,
                accuracy=struct,
                scientific=0.7,
                overall=(f1 + struct) / 2.0,
                issues="[mock] none",
                summary="[mock] Mock evaluation for dry run",
                judge_model=self.config.model,
                tokens_used=0,
                timestamp=datetime.now(),
            ))
        return results

    def load_checkpoint(self) -> tuple[list[JudgeResult], set[tuple]]:
        """Load judge results from checkpoint.

        Returns:
            Tuple of (results list, set of completed keys for resume logic)
        """
        return self._checkpoint.load_with_keys(
            "judge_results.json",
            JudgeResult,
            key_fn=lambda r: r.key,
        )

    def save_checkpoint(self, results: list[JudgeResult]) -> None:
        """Save judge results to checkpoint."""
        self._checkpoint.save(results, "judge_results.json")


def compute_judge_stats(results: list[JudgeResult]) -> dict[str, Any]:
    """Compute aggregate statistics from judge results."""
    if not results:
        return {"error": "No results to aggregate"}

    import statistics
    from collections import defaultdict

    def compute_stats(scores: list[float]) -> dict[str, float]:
        if not scores:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        return {
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "n": len(scores),
        }

    # Overall stats
    valid_results = [r for r in results if not r.error]

    overall = {
        "completeness": compute_stats([r.completeness for r in valid_results]),
        "accuracy": compute_stats([r.accuracy for r in valid_results]),
        "scientific": compute_stats([r.scientific for r in valid_results]),
        "overall": compute_stats([r.overall for r in valid_results]),
        "error_count": len([r for r in results if r.error]),
        "total": len(results),
    }

    # By model
    by_model = defaultdict(list)
    for r in valid_results:
        by_model[r.model].append(r)

    model_stats = {}
    for model, model_results in by_model.items():
        model_stats[model] = {
            "overall": compute_stats([r.overall for r in model_results]),
            "n": len(model_results),
        }

    # Collect common issues
    all_issues = []
    for r in valid_results:
        if r.issues and r.issues.lower() != "none":
            all_issues.append(r.issues)

    return {
        "overall": overall,
        "by_model": model_stats,
        "common_issues": all_issues[:10],  # Top 10 issues
    }

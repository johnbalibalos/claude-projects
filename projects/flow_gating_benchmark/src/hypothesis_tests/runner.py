"""
Unified Hypothesis Test Runner.

Orchestrates running all hypothesis tests and aggregates results
for comprehensive analysis of model behavior.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from curation.schemas import TestCase

from .cognitive_refusal import (
    AggregateCognitiveRefusalAnalysis,
    CognitiveRefusalResult,
    CognitiveRefusalTest,
    CognitiveRefusalTestResult,
    PROMPT_VARIANTS,
)
from .cot_mechanistic import (
    CoTAnalysis,
    CoTAnnotator,
    RedPenAnalysis,
)
from .format_ablation import (
    FormatAblationAnalysis,
    FormatAblationResult,
    FormatAblationTest,
    PromptFormat,
)
from .frequency_confound import (
    AlienCellAnalyzer,
    AlienCellResult,
    AlienCellTest,
    AlienCellTestCase,
    CorrelationResult,
    FrequencyCorrelation,
    PubMedFrequencyLookup,
)

logger = logging.getLogger(__name__)


class HypothesisType(Enum):
    """Types of hypothesis tests available."""

    FREQUENCY_CONFOUND = "frequency_confound"
    ALIEN_CELL = "alien_cell"
    FORMAT_ABLATION = "format_ablation"
    COT_MECHANISTIC = "cot_mechanistic"
    COGNITIVE_REFUSAL = "cognitive_refusal"


@dataclass
class AblationConfig:
    """Configuration for hypothesis testing."""

    # Which tests to run
    tests: list[HypothesisType] = field(default_factory=lambda: list(HypothesisType))

    # Test-specific configs
    frequency_cache_path: str | None = None
    format_variants: list[PromptFormat] | None = None
    refusal_variants: list[str] | None = None

    # Execution config
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.0
    dry_run: bool = False

    # Output config
    output_dir: str = "./hypothesis_test_results"
    save_raw_responses: bool = True


@dataclass
class HypothesisTestResult:
    """Complete result from all hypothesis tests."""

    config: AblationConfig
    started_at: datetime
    completed_at: datetime | None = None

    # Test results
    frequency_correlation: CorrelationResult | None = None
    alien_cell_results: list[AlienCellResult] = field(default_factory=list)
    format_ablation_results: list[FormatAblationAnalysis] = field(default_factory=list)
    cot_analysis: RedPenAnalysis | None = None
    cognitive_refusal_analysis: AggregateCognitiveRefusalAnalysis | None = None

    # Summary
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "tests_run": [t.value for t in self.config.tests],
            "model": self.config.model,
            "summary": self.summary,
            "frequency_correlation": self._serialize_correlation(),
            "alien_cell_results": self._serialize_alien_cell(),
            "format_ablation_results": self._serialize_format_ablation(),
            "cot_analysis": self._serialize_cot(),
            "cognitive_refusal": self._serialize_cognitive_refusal(),
        }

    def _serialize_correlation(self) -> dict | None:
        if not self.frequency_correlation:
            return None
        return {
            "r_squared": self.frequency_correlation.r_squared,
            "pearson_r": self.frequency_correlation.pearson_r,
            "p_value": self.frequency_correlation.p_value,
            "n_samples": self.frequency_correlation.n_samples,
            "interpretation": self.frequency_correlation.interpretation,
            "data_points": [
                {"name": name, "log_freq": freq, "score": score}
                for name, freq, score in self.frequency_correlation.data_points
            ],
        }

    def _serialize_alien_cell(self) -> list[dict]:
        return [
            {
                "test_case_id": r.test_case_id,
                "original_f1": r.original_f1,
                "alien_f1": r.alien_f1,
                "delta_f1": r.delta_f1,
                "reasoning_score": r.reasoning_score,
                "interpretation": r.interpretation,
            }
            for r in self.alien_cell_results
        ]

    def _serialize_format_ablation(self) -> list[dict]:
        return [
            {
                "test_case_id": r.test_case_id,
                "best_format": r.best_format.value,
                "worst_format": r.worst_format.value,
                "format_variance": r.format_variance,
                "is_robust_failure": r.is_robust_failure,
                "is_extraction_issue": r.is_extraction_issue,
                "interpretation": r.interpretation,
                "f1_by_format": r.f1_by_format,
            }
            for r in self.format_ablation_results
        ]

    def _serialize_cot(self) -> dict | None:
        if not self.cot_analysis:
            return None
        return {
            "mean_hallucination_rate": self.cot_analysis.mean_hallucination_rate,
            "median_hallucination_rate": self.cot_analysis.median_hallucination_rate,
            "proportion_with_hallucinations": self.cot_analysis.proportion_with_hallucinations,
            "hypothesis_supported": self.cot_analysis.hypothesis_supported,
            "interpretation": self.cot_analysis.interpretation,
            "common_priors": self.cot_analysis.common_priors,
        }

    def _serialize_cognitive_refusal(self) -> dict | None:
        if not self.cognitive_refusal_analysis:
            return None
        return {
            "mean_forcing_effect": self.cognitive_refusal_analysis.mean_forcing_effect,
            "proportion_context_blindness": self.cognitive_refusal_analysis.proportion_context_blindness,
            "proportion_safety_over_triggering": self.cognitive_refusal_analysis.proportion_safety_over_triggering,
            "dominant_hypothesis": self.cognitive_refusal_analysis.dominant_hypothesis,
            "interpretation": self.cognitive_refusal_analysis.interpretation,
            "avg_f1_by_variant": self.cognitive_refusal_analysis.avg_f1_by_variant,
        }

    def generate_summary(self) -> None:
        """Generate summary of all test results."""
        self.summary = {
            "tests_completed": [],
            "key_findings": [],
            "hypotheses_supported": {},
        }

        # Frequency correlation
        if self.frequency_correlation:
            self.summary["tests_completed"].append("frequency_correlation")
            if self.frequency_correlation.r_squared > 0.5:
                self.summary["hypotheses_supported"]["frequency_explains_performance"] = True
                self.summary["key_findings"].append(
                    f"Frequency hypothesis supported (R²={self.frequency_correlation.r_squared:.3f})"
                )
            else:
                self.summary["hypotheses_supported"]["frequency_explains_performance"] = False
                self.summary["key_findings"].append(
                    f"Reasoning hypothesis supported (R²={self.frequency_correlation.r_squared:.3f})"
                )

        # Alien cell results
        if self.alien_cell_results:
            self.summary["tests_completed"].append("alien_cell")
            avg_reasoning_score = sum(r.reasoning_score for r in self.alien_cell_results) / len(self.alien_cell_results)
            if avg_reasoning_score > 0.7:
                self.summary["hypotheses_supported"]["model_uses_reasoning"] = True
                self.summary["key_findings"].append(
                    f"Model demonstrates reasoning (avg score={avg_reasoning_score:.3f})"
                )
            else:
                self.summary["hypotheses_supported"]["model_uses_reasoning"] = False
                self.summary["key_findings"].append(
                    f"Model relies on memorization (avg score={avg_reasoning_score:.3f})"
                )

        # Format ablation
        if self.format_ablation_results:
            self.summary["tests_completed"].append("format_ablation")
            n_robust = sum(1 for r in self.format_ablation_results if r.is_robust_failure)
            n_extraction = sum(1 for r in self.format_ablation_results if r.is_extraction_issue)
            if n_robust > len(self.format_ablation_results) / 2:
                self.summary["key_findings"].append(
                    "Failures are robust across formats (reasoning deficit)"
                )
            elif n_extraction > len(self.format_ablation_results) / 2:
                self.summary["key_findings"].append(
                    "Failures are format-dependent (extraction issue)"
                )

        # CoT analysis
        if self.cot_analysis:
            self.summary["tests_completed"].append("cot_mechanistic")
            self.summary["hypotheses_supported"]["prior_interference"] = (
                self.cot_analysis.hypothesis_supported == "PRIOR_INTERFERENCE"
            )
            if self.cot_analysis.proportion_with_hallucinations > 0.5:
                self.summary["key_findings"].append(
                    f"CoT causes prior hallucinations ({self.cot_analysis.proportion_with_hallucinations:.1%} of failures)"
                )

        # Cognitive refusal
        if self.cognitive_refusal_analysis:
            self.summary["tests_completed"].append("cognitive_refusal")
            self.summary["hypotheses_supported"]["context_blindness"] = (
                self.cognitive_refusal_analysis.dominant_hypothesis == "CONTEXT_BLINDNESS"
            )
            self.summary["hypotheses_supported"]["safety_over_triggering"] = (
                self.cognitive_refusal_analysis.dominant_hypothesis == "SAFETY_OVER_TRIGGERING"
            )
            self.summary["key_findings"].append(
                f"Dominant failure mode: {self.cognitive_refusal_analysis.dominant_hypothesis}"
            )

    def format_report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 70,
            "HYPOTHESIS TEST REPORT",
            "=" * 70,
            "",
            f"Started: {self.started_at}",
            f"Completed: {self.completed_at or 'In Progress'}",
            f"Model: {self.config.model}",
            "",
        ]

        # Key Findings
        if self.summary.get("key_findings"):
            lines.extend([
                "-" * 70,
                "KEY FINDINGS",
                "-" * 70,
                "",
            ])
            for i, finding in enumerate(self.summary["key_findings"], 1):
                lines.append(f"{i}. {finding}")
            lines.append("")

        # Frequency Correlation
        if self.frequency_correlation:
            lines.extend([
                "-" * 70,
                "1. FREQUENCY CONFOUND TEST",
                "-" * 70,
                "",
                f"R² = {self.frequency_correlation.r_squared:.3f}",
                f"Pearson r = {self.frequency_correlation.pearson_r:.3f}",
                f"p-value = {self.frequency_correlation.p_value:.4f}",
                f"n = {self.frequency_correlation.n_samples}",
                "",
                f"Interpretation: {self.frequency_correlation.interpretation}",
                "",
            ])

        # Alien Cell Results
        if self.alien_cell_results:
            lines.extend([
                "-" * 70,
                "2. ALIEN CELL INJECTION TEST",
                "-" * 70,
                "",
            ])
            for result in self.alien_cell_results:
                lines.extend([
                    f"Test Case: {result.test_case_id}",
                    f"  Original F1: {result.original_f1:.3f}",
                    f"  Alien F1: {result.alien_f1:.3f}",
                    f"  Delta: {result.delta_f1:.3f}",
                    f"  Reasoning Score: {result.reasoning_score:.3f}",
                    f"  Interpretation: {result.interpretation}",
                    "",
                ])

        # Format Ablation
        if self.format_ablation_results:
            lines.extend([
                "-" * 70,
                "3. FORMAT ABLATION TEST",
                "-" * 70,
                "",
            ])
            for result in self.format_ablation_results:
                lines.extend([
                    f"Test Case: {result.test_case_id}",
                    f"  Best Format: {result.best_format.value} (F1={result.f1_by_format.get(result.best_format.value, 0):.3f})",
                    f"  Worst Format: {result.worst_format.value} (F1={result.f1_by_format.get(result.worst_format.value, 0):.3f})",
                    f"  Robust Failure: {result.is_robust_failure}",
                    f"  Extraction Issue: {result.is_extraction_issue}",
                    f"  Interpretation: {result.interpretation}",
                    "",
                ])

        # CoT Mechanistic
        if self.cot_analysis:
            lines.extend([
                "-" * 70,
                "4. CHAIN-OF-THOUGHT MECHANISTIC ANALYSIS",
                "-" * 70,
                "",
                f"Mean Hallucination Rate: {self.cot_analysis.mean_hallucination_rate:.1%}",
                f"Proportion with Hallucinations: {self.cot_analysis.proportion_with_hallucinations:.1%}",
                f"Hypothesis Supported: {self.cot_analysis.hypothesis_supported}",
                "",
                f"Most Common Priors:",
            ])
            for prior, count in list(self.cot_analysis.common_priors.items())[:5]:
                lines.append(f"  - {prior}: {count}")
            lines.extend([
                "",
                f"Interpretation: {self.cot_analysis.interpretation}",
                "",
            ])

        # Cognitive Refusal
        if self.cognitive_refusal_analysis:
            lines.extend([
                "-" * 70,
                "5. COGNITIVE REFUSAL TEST",
                "-" * 70,
                "",
                f"Mean Forcing Effect: {self.cognitive_refusal_analysis.mean_forcing_effect:.3f}",
                f"Context Blindness Rate: {self.cognitive_refusal_analysis.proportion_context_blindness:.1%}",
                f"Safety Over-triggering Rate: {self.cognitive_refusal_analysis.proportion_safety_over_triggering:.1%}",
                f"Dominant Hypothesis: {self.cognitive_refusal_analysis.dominant_hypothesis}",
                "",
                "Average F1 by Variant:",
            ])
            for variant, f1 in self.cognitive_refusal_analysis.avg_f1_by_variant.items():
                lines.append(f"  - {variant}: {f1:.3f}")
            lines.extend([
                "",
                f"Interpretation: {self.cognitive_refusal_analysis.interpretation}",
                "",
            ])

        # Conclusion
        lines.extend([
            "=" * 70,
            "CONCLUSION",
            "=" * 70,
            "",
        ])
        if self.summary.get("hypotheses_supported"):
            for hypothesis, supported in self.summary["hypotheses_supported"].items():
                status = "SUPPORTED" if supported else "NOT SUPPORTED"
                lines.append(f"- {hypothesis}: {status}")

        return "\n".join(lines)


class HypothesisTestRunner:
    """
    Orchestrates running all hypothesis tests.

    Provides a unified interface for running individual tests or
    the complete battery of hypothesis tests.
    """

    def __init__(self, config: AblationConfig):
        """
        Initialize hypothesis test runner.

        Args:
            config: Configuration for the tests
        """
        self.config = config
        self.result = HypothesisTestResult(
            config=config,
            started_at=datetime.now(),
        )

        # Initialize test components
        self._setup_tests()

    def _setup_tests(self) -> None:
        """Initialize test components based on config."""
        # Frequency tests
        if HypothesisType.FREQUENCY_CONFOUND in self.config.tests:
            self.pubmed_lookup = PubMedFrequencyLookup(
                cache_path=self.config.frequency_cache_path,
            )
            self.frequency_correlation = FrequencyCorrelation(self.pubmed_lookup)
        else:
            self.pubmed_lookup = None
            self.frequency_correlation = None

        # Alien cell test
        if HypothesisType.ALIEN_CELL in self.config.tests:
            self.alien_cell_test = AlienCellTest()
            self.alien_cell_analyzer = AlienCellAnalyzer()
        else:
            self.alien_cell_test = None
            self.alien_cell_analyzer = None

        # Format ablation
        if HypothesisType.FORMAT_ABLATION in self.config.tests:
            self.format_ablation = FormatAblationTest(
                formats=self.config.format_variants,
            )
        else:
            self.format_ablation = None

        # CoT annotation
        if HypothesisType.COT_MECHANISTIC in self.config.tests:
            self.cot_annotator = CoTAnnotator()
        else:
            self.cot_annotator = None

        # Cognitive refusal
        if HypothesisType.COGNITIVE_REFUSAL in self.config.tests:
            self.cognitive_refusal_test = CognitiveRefusalTest(
                variants=self.config.refusal_variants,
            )
        else:
            self.cognitive_refusal_test = None

    def run_frequency_correlation(
        self,
        population_scores: dict[str, float],
    ) -> CorrelationResult:
        """
        Run frequency correlation analysis.

        Args:
            population_scores: Dict mapping population name to F1 score

        Returns:
            CorrelationResult with statistical analysis
        """
        if self.frequency_correlation is None:
            raise ValueError("Frequency correlation test not configured")

        logger.info(f"Running frequency correlation on {len(population_scores)} populations")
        result = self.frequency_correlation.analyze(population_scores)
        self.result.frequency_correlation = result
        return result

    def run_alien_cell_test(
        self,
        test_case: TestCase,
        original_score: float,
        run_model_fn: callable,
    ) -> AlienCellResult:
        """
        Run alien cell injection test.

        Args:
            test_case: Original test case
            original_score: F1 score on original test case
            run_model_fn: Function to call model (takes test_case, returns ScoringResult)

        Returns:
            AlienCellResult with comparison
        """
        if self.alien_cell_test is None:
            raise ValueError("Alien cell test not configured")

        logger.info(f"Running alien cell test for {test_case.test_case_id}")

        # Create alien version
        alien_test_case = self.alien_cell_test.create_alien_test_case(test_case)

        # Run model on alien version
        alien_result = run_model_fn(alien_test_case.modified_test_case)

        # Analyze results
        # Create mock original result for comparison
        class MockResult:
            hierarchy_f1 = original_score

        result = self.alien_cell_analyzer.analyze(
            original_result=MockResult(),
            alien_result=alien_result,
            alien_test_case=alien_test_case,
        )

        self.result.alien_cell_results.append(result)
        return result

    def run_format_ablation(
        self,
        test_case: TestCase,
        run_model_fn: callable,
    ) -> FormatAblationAnalysis:
        """
        Run format ablation test.

        Args:
            test_case: Test case to use
            run_model_fn: Function(format_prompt, format_type) -> ScoringResult

        Returns:
            FormatAblationAnalysis with comparison across formats
        """
        if self.format_ablation is None:
            raise ValueError("Format ablation test not configured")

        logger.info(f"Running format ablation for {test_case.test_case_id}")

        # Generate all formats
        formatted_prompts = self.format_ablation.generate_all_formats(test_case)

        # Run model on each format
        results: dict[PromptFormat, FormatAblationResult] = {}
        for format_type, formatted_prompt in formatted_prompts.items():
            logger.info(f"  Testing format: {format_type.value}")
            scoring_result = run_model_fn(formatted_prompt.content, format_type)

            results[format_type] = FormatAblationResult(
                format=format_type,
                f1_score=scoring_result.hierarchy_f1,
                structure_accuracy=scoring_result.structure_accuracy,
                parse_success=scoring_result.parse_success,
                task_failure=scoring_result.is_task_failure,
                raw_response=scoring_result.raw_response,
            )

        # Analyze results
        analysis = self.format_ablation.analyze_results(results, test_case.test_case_id)
        self.result.format_ablation_results.append(analysis)
        return analysis

    def run_cot_analysis(
        self,
        cot_responses: list[tuple[str, TestCase]],
    ) -> RedPenAnalysis:
        """
        Run CoT mechanistic analysis (Red Pen annotation).

        Args:
            cot_responses: List of (cot_text, test_case) tuples

        Returns:
            RedPenAnalysis with aggregate statistics
        """
        if self.cot_annotator is None:
            raise ValueError("CoT annotator not configured")

        logger.info(f"Analyzing {len(cot_responses)} CoT responses")
        result = self.cot_annotator.analyze_batch(cot_responses)
        self.result.cot_analysis = result
        return result

    def run_cognitive_refusal_test(
        self,
        test_cases: list[TestCase],
        run_model_fn: callable,
    ) -> AggregateCognitiveRefusalAnalysis:
        """
        Run cognitive refusal test.

        Args:
            test_cases: List of test cases
            run_model_fn: Function(system_prompt, user_prompt, test_case) -> ScoringResult

        Returns:
            AggregateCognitiveRefusalAnalysis with dominant hypothesis
        """
        if self.cognitive_refusal_test is None:
            raise ValueError("Cognitive refusal test not configured")

        all_results: list[CognitiveRefusalTestResult] = []

        for test_case in test_cases:
            logger.info(f"Running cognitive refusal test for {test_case.test_case_id}")

            variant_results: dict[str, CognitiveRefusalResult] = {}

            for variant_name, variant in self.cognitive_refusal_test.variants.items():
                logger.info(f"  Testing variant: {variant_name}")

                # Run model with this variant
                scoring_result = run_model_fn(
                    variant.system_prompt,
                    test_case.to_prompt_context() + variant.user_prompt_suffix,
                    test_case,
                )

                # Detect refusal
                refusal = self.cognitive_refusal_test.detect_refusal(
                    scoring_result.raw_response or ""
                )

                variant_results[variant_name] = CognitiveRefusalResult(
                    variant=variant,
                    test_case_id=test_case.test_case_id,
                    f1_score=scoring_result.hierarchy_f1,
                    structure_accuracy=scoring_result.structure_accuracy,
                    parse_success=scoring_result.parse_success,
                    refusal_analysis=refusal,
                    is_refusal=refusal.refusal_type.value != "none",
                    raw_response=scoring_result.raw_response,
                )

            # Analyze this test case
            test_result = self.cognitive_refusal_test.analyze_results(
                variant_results,
                test_case.test_case_id,
            )
            all_results.append(test_result)

        # Aggregate analysis
        analysis = AggregateCognitiveRefusalAnalysis(test_results=all_results)
        analysis.compute_aggregate_metrics()
        self.result.cognitive_refusal_analysis = analysis
        return analysis

    def finalize(self) -> HypothesisTestResult:
        """Finalize the test run and generate summary."""
        self.result.completed_at = datetime.now()
        self.result.generate_summary()

        # Save results
        self._save_results()

        return self.result

    def _save_results(self) -> None:
        """Save results to disk."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_path = output_dir / f"hypothesis_tests_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(self.result.to_dict(), f, indent=2, default=str)
        logger.info(f"Results saved to: {json_path}")

        # Save report
        report_path = output_dir / f"hypothesis_report_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(self.result.format_report())
        logger.info(f"Report saved to: {report_path}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def run_quick_hypothesis_test(
    test_cases: list[TestCase],
    model_scores: dict[str, dict[str, float]],  # test_case_id -> population -> score
    output_dir: str = "./hypothesis_results",
) -> HypothesisTestResult:
    """
    Run a quick hypothesis test battery using existing results.

    This is useful for analyzing results without re-running the model.

    Args:
        test_cases: List of test cases
        model_scores: Pre-computed scores by test case and population
        output_dir: Where to save results

    Returns:
        HypothesisTestResult with analysis
    """
    config = AblationConfig(
        tests=[
            HypothesisType.FREQUENCY_CONFOUND,
        ],
        output_dir=output_dir,
    )

    runner = HypothesisTestRunner(config)

    # Run frequency correlation if we have population-level scores
    all_population_scores: dict[str, float] = {}
    for scores in model_scores.values():
        all_population_scores.update(scores)

    if all_population_scores:
        runner.run_frequency_correlation(all_population_scores)

    return runner.finalize()

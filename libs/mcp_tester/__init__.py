"""
MCP Tester - Generic framework for testing MCP servers and AI tool effectiveness.

This framework provides a reusable template for:
1. Ablation studies comparing tool-augmented vs baseline LLMs
2. A/B testing of different tool configurations
3. Model comparison (Sonnet vs Opus, etc.)

Usage:
    from mcp_tester import AblationStudy, TestCase, Condition

    # Define your test cases
    test_cases = [
        TestCase(id="case1", prompt="...", ground_truth={"key": "value"}),
        ...
    ]

    # Define conditions to compare
    conditions = [
        Condition(name="baseline", tools_enabled=False),
        Condition(name="with_tools", tools_enabled=True, tools=[...]),
    ]

    # Run study
    study = AblationStudy(
        name="my_study",
        model="claude-sonnet-4-20250514",
        test_cases=test_cases,
        conditions=conditions,
        evaluator=my_evaluator_fn
    )

    results = study.run()
    study.generate_report()
"""

from .study import AblationStudy
from .models import TestCase, Condition, TrialResult, StudyResults
from .evaluators import Evaluator, AccuracyEvaluator, SimilarityEvaluator

__all__ = [
    "AblationStudy",
    "TestCase",
    "Condition",
    "TrialResult",
    "StudyResults",
    "Evaluator",
    "AccuracyEvaluator",
    "SimilarityEvaluator",
]
__version__ = "0.1.0"

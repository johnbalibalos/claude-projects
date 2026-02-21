"""
Rubric definitions for structured LLM-as-Judge evaluation.

Provides RubricLevel, RubricCriterion, and EvaluationRubric
with default rubrics for Q&A and scientific analysis.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RubricLevel:
    """A single level in a rubric criterion."""

    score: int
    label: str
    description: str


@dataclass
class RubricCriterion:
    """A single criterion in an evaluation rubric."""

    name: str
    description: str
    weight: float
    levels: list[RubricLevel]

    def to_prompt_string(self) -> str:
        """Convert criterion to prompt format."""
        lines = [f"**{self.name}** (weight: {self.weight})"]
        lines.append(f"Description: {self.description}")
        lines.append("Scoring levels:")
        for level in sorted(self.levels, key=lambda x: x.score, reverse=True):
            lines.append(f"  - {level.score}: {level.label} - {level.description}")
        return "\n".join(lines)


@dataclass
class EvaluationRubric:
    """Complete evaluation rubric with multiple criteria."""

    name: str
    description: str
    criteria: list[RubricCriterion]
    max_total_score: int = 100

    def to_prompt_string(self) -> str:
        """Convert rubric to prompt format."""
        lines = [
            f"# Evaluation Rubric: {self.name}",
            "",
            self.description,
            "",
            "## Criteria",
            "",
        ]
        for criterion in self.criteria:
            lines.append(criterion.to_prompt_string())
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def default_qa_rubric(cls) -> EvaluationRubric:
        """Create default rubric for Q&A evaluation."""
        return cls(
            name="Q&A Evaluation",
            description="Evaluates the quality of answers to questions.",
            criteria=[
                RubricCriterion(
                    name="Factual Accuracy",
                    description="Does the response contain correct factual information?",
                    weight=0.4,
                    levels=[
                        RubricLevel(3, "Fully Accurate", "All facts are correct and verifiable"),
                        RubricLevel(2, "Mostly Accurate", "Minor factual errors that don't affect main point"),
                        RubricLevel(1, "Partially Accurate", "Some correct information mixed with errors"),
                        RubricLevel(0, "Inaccurate", "Major factual errors or mostly incorrect"),
                    ],
                ),
                RubricCriterion(
                    name="Completeness",
                    description="Does the response fully address the question?",
                    weight=0.3,
                    levels=[
                        RubricLevel(3, "Complete", "Addresses all aspects of the question"),
                        RubricLevel(2, "Mostly Complete", "Addresses main aspects, minor gaps"),
                        RubricLevel(1, "Partial", "Only addresses some aspects"),
                        RubricLevel(0, "Incomplete", "Fails to address key aspects"),
                    ],
                ),
                RubricCriterion(
                    name="Clarity",
                    description="Is the response clear and well-organized?",
                    weight=0.2,
                    levels=[
                        RubricLevel(3, "Very Clear", "Well-organized, easy to understand"),
                        RubricLevel(2, "Clear", "Generally clear with minor issues"),
                        RubricLevel(1, "Somewhat Clear", "Understandable but disorganized"),
                        RubricLevel(0, "Unclear", "Difficult to understand or follow"),
                    ],
                ),
                RubricCriterion(
                    name="Relevance",
                    description="Is the response relevant to the question asked?",
                    weight=0.1,
                    levels=[
                        RubricLevel(3, "Highly Relevant", "Directly addresses the question"),
                        RubricLevel(2, "Relevant", "Mostly on-topic"),
                        RubricLevel(1, "Somewhat Relevant", "Partially addresses the question"),
                        RubricLevel(0, "Off-Topic", "Does not address the question"),
                    ],
                ),
            ],
        )

    @classmethod
    def scientific_analysis_rubric(cls) -> EvaluationRubric:
        """Create rubric for scientific analysis evaluation."""
        return cls(
            name="Scientific Analysis",
            description="Evaluates scientific reasoning and analysis quality.",
            criteria=[
                RubricCriterion(
                    name="Scientific Accuracy",
                    description="Are scientific concepts and terminology used correctly?",
                    weight=0.35,
                    levels=[
                        RubricLevel(3, "Expert-level", "Demonstrates deep understanding with correct terminology"),
                        RubricLevel(2, "Competent", "Correct understanding with minor terminology issues"),
                        RubricLevel(1, "Basic", "General understanding but notable gaps"),
                        RubricLevel(0, "Incorrect", "Fundamental misunderstandings"),
                    ],
                ),
                RubricCriterion(
                    name="Reasoning Quality",
                    description="Is the logical reasoning sound and well-supported?",
                    weight=0.30,
                    levels=[
                        RubricLevel(3, "Rigorous", "Clear logical flow with well-supported conclusions"),
                        RubricLevel(2, "Sound", "Generally logical with minor gaps"),
                        RubricLevel(1, "Weak", "Some logical issues or unsupported claims"),
                        RubricLevel(0, "Flawed", "Major logical errors or unfounded conclusions"),
                    ],
                ),
                RubricCriterion(
                    name="Evidence Use",
                    description="Is evidence appropriately cited and interpreted?",
                    weight=0.20,
                    levels=[
                        RubricLevel(3, "Excellent", "Strong evidence use with correct interpretation"),
                        RubricLevel(2, "Good", "Adequate evidence with mostly correct interpretation"),
                        RubricLevel(1, "Fair", "Limited evidence or some misinterpretation"),
                        RubricLevel(0, "Poor", "Missing evidence or major misinterpretation"),
                    ],
                ),
                RubricCriterion(
                    name="Nuance",
                    description="Does the response acknowledge complexity and limitations?",
                    weight=0.15,
                    levels=[
                        RubricLevel(3, "Nuanced", "Acknowledges limitations and alternative views"),
                        RubricLevel(2, "Balanced", "Some recognition of complexity"),
                        RubricLevel(1, "Limited", "Oversimplified but not wrong"),
                        RubricLevel(0, "Simplistic", "Ignores important nuances"),
                    ],
                ),
            ],
        )


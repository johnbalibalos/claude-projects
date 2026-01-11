"""
Chain-of-Thought Mechanistic Grounding Analysis.

Tests whether CoT causes hallucinations by creating a "distraction window"
where the model retrieves irrelevant training data that overrides the prompt.

Hypotheses:
- H_A: CoT creates a distraction window where prior knowledge overrides prompt
- H_B: CoT simply runs out of context window or attention bits

Test: "Red Pen" Annotation
Manually inspect failed CoT logs and tag every intermediate reasoning step:
- Tag 1 (VALID_INFERENCE): Derived from prompt context
- Tag 2 (PRIOR_HALLUCINATION): Based on training priors, not in prompt

If >50% of failures contain prior hallucinations, we have mechanistic proof
that prior knowledge interferes with in-context reasoning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from curation.schemas import TestCase


class InferenceTag(Enum):
    """Tags for classifying reasoning steps."""

    VALID_INFERENCE = "valid_inference"
    PRIOR_HALLUCINATION = "prior_hallucination"
    CONTEXT_REFERENCE = "context_reference"
    UNCERTAIN = "uncertain"
    STRUCTURAL = "structural"  # Purely structural statements like "let me think..."


@dataclass
class ReasoningStep:
    """A single step in the chain of thought."""

    text: str
    tag: InferenceTag
    confidence: float  # 0-1 confidence in the tag assignment
    evidence: str  # Why this tag was assigned
    line_number: int | None = None

    # If this is a prior hallucination, what prior was invoked?
    invoked_prior: str | None = None

    # If valid inference, what prompt context supports it?
    supporting_context: str | None = None


@dataclass
class CoTAnalysis:
    """Complete analysis of a chain-of-thought response."""

    raw_cot: str
    steps: list[ReasoningStep]
    test_case_id: str

    # Aggregate metrics
    n_valid_inferences: int = 0
    n_prior_hallucinations: int = 0
    n_context_references: int = 0
    n_uncertain: int = 0
    n_structural: int = 0

    # Computed after analysis
    hallucination_rate: float = 0.0
    prior_dependency_score: float = 0.0  # How much the reasoning depends on priors

    # List of unique priors invoked
    priors_invoked: list[str] = field(default_factory=list)

    def compute_metrics(self) -> None:
        """Compute aggregate metrics from steps."""
        self.n_valid_inferences = sum(1 for s in self.steps if s.tag == InferenceTag.VALID_INFERENCE)
        self.n_prior_hallucinations = sum(1 for s in self.steps if s.tag == InferenceTag.PRIOR_HALLUCINATION)
        self.n_context_references = sum(1 for s in self.steps if s.tag == InferenceTag.CONTEXT_REFERENCE)
        self.n_uncertain = sum(1 for s in self.steps if s.tag == InferenceTag.UNCERTAIN)
        self.n_structural = sum(1 for s in self.steps if s.tag == InferenceTag.STRUCTURAL)

        # Calculate hallucination rate (excluding structural steps)
        substantive_steps = len(self.steps) - self.n_structural
        if substantive_steps > 0:
            self.hallucination_rate = self.n_prior_hallucinations / substantive_steps
        else:
            self.hallucination_rate = 0.0

        # Prior dependency score
        inference_steps = self.n_valid_inferences + self.n_prior_hallucinations
        if inference_steps > 0:
            self.prior_dependency_score = self.n_prior_hallucinations / inference_steps
        else:
            self.prior_dependency_score = 0.0

        # Collect unique priors
        self.priors_invoked = list({
            s.invoked_prior for s in self.steps
            if s.tag == InferenceTag.PRIOR_HALLUCINATION and s.invoked_prior
        })


@dataclass
class RedPenAnalysis:
    """Aggregate analysis across multiple CoT responses."""

    analyses: list[CoTAnalysis]
    test_case_ids: list[str]

    # Aggregate statistics
    mean_hallucination_rate: float = 0.0
    median_hallucination_rate: float = 0.0
    proportion_with_hallucinations: float = 0.0

    # Most common priors
    common_priors: dict[str, int] = field(default_factory=dict)

    # Interpretation
    hypothesis_supported: str = ""  # "PRIOR_INTERFERENCE" or "OTHER"
    interpretation: str = ""

    def compute_aggregate_metrics(self) -> None:
        """Compute aggregate metrics across all analyses."""
        if not self.analyses:
            return

        # Compute means
        hallucination_rates = [a.hallucination_rate for a in self.analyses]
        self.mean_hallucination_rate = sum(hallucination_rates) / len(hallucination_rates)

        # Compute median
        sorted_rates = sorted(hallucination_rates)
        n = len(sorted_rates)
        if n % 2 == 0:
            self.median_hallucination_rate = (sorted_rates[n//2 - 1] + sorted_rates[n//2]) / 2
        else:
            self.median_hallucination_rate = sorted_rates[n//2]

        # Proportion with any hallucinations
        n_with_hallucinations = sum(1 for a in self.analyses if a.n_prior_hallucinations > 0)
        self.proportion_with_hallucinations = n_with_hallucinations / len(self.analyses)

        # Aggregate common priors
        for analysis in self.analyses:
            for prior in analysis.priors_invoked:
                self.common_priors[prior] = self.common_priors.get(prior, 0) + 1

        # Sort by frequency
        self.common_priors = dict(sorted(
            self.common_priors.items(),
            key=lambda x: x[1],
            reverse=True,
        ))

        # Generate interpretation
        if self.proportion_with_hallucinations > 0.5:
            self.hypothesis_supported = "PRIOR_INTERFERENCE"
            self.interpretation = (
                f"PRIOR INTERFERENCE HYPOTHESIS SUPPORTED: "
                f"{self.proportion_with_hallucinations:.1%} of failed CoT responses "
                f"contain prior hallucinations (mean rate: {self.mean_hallucination_rate:.1%}). "
                f"This provides mechanistic evidence that the model retrieves training priors "
                f"that override in-context information during chain-of-thought reasoning. "
                f"Most common priors: {list(self.common_priors.keys())[:5]}"
            )
        else:
            self.hypothesis_supported = "OTHER"
            self.interpretation = (
                f"ALTERNATIVE HYPOTHESIS: Only {self.proportion_with_hallucinations:.1%} of "
                f"failures contain clear prior hallucinations. "
                f"CoT failures may be due to other factors (attention, context length, etc.) "
                f"rather than prior knowledge interference."
            )


# =============================================================================
# KNOWN BIOLOGICAL PRIORS (patterns that indicate retrieval from training)
# =============================================================================

KNOWN_BIOLOGICAL_PRIORS = {
    # Cell type associations that are common knowledge
    "CD3 -> T cells": r"CD3\+?\s*(is|are|indicates?|marks?|identifies?)\s*(typically|usually|commonly|generally)?\s*T\s*cells?",
    "CD19/CD20 -> B cells": r"CD(19|20)\+?\s*(is|are|indicates?|marks?)\s*(typically|usually)?\s*B\s*cells?",
    "CD14 -> Monocytes": r"CD14\+?\s*(is|are|indicates?|marks?)\s*(typically|usually)?\s*monocytes?",
    "CD56 -> NK cells": r"CD56\+?\s*(is|are|indicates?|marks?)\s*(typically|usually)?\s*(NK|natural killer)\s*cells?",

    # Memory subset associations
    "CCR7+CD45RA+ -> Naive": r"(CCR7|CD62L)\+?\s*(and|with)?\s*CD45RA\+?\s*(indicates?|are?|=)\s*naive",
    "CCR7-CD45RA- -> Effector Memory": r"(CCR7|CD62L)-?\s*(and|with)?\s*CD45RA-?\s*(indicates?|are?|=)\s*(effector\s*)?memory",

    # General immunology knowledge
    "lymphocyte gating": r"(lymphocytes?\s*(are|should be)\s*gated|gate\s*(on|for)\s*lymphocytes?)\s*(using|with|by)\s*(FSC|SSC)",
    "live/dead gating": r"(dead|dying)\s*cells?\s*(are|should be)\s*(excluded|removed|gated out)",

    # Population relationships
    "T cell subsets": r"(CD4|CD8)\+?\s*T\s*cells?\s*(are|represent)\s*(a\s*)?(subset|subpopulation)\s*of\s*(all\s*)?T\s*cells?",
    "helper vs cytotoxic": r"CD4\+?\s*T\s*cells?\s*(are|=)\s*(helper|Th)\s*(and|while|whereas)\s*CD8\+?",

    # Marker biology
    "CD25 -> Tregs": r"CD25\+?\s*(high|hi|bright)?\s*(indicates?|marks?|identifies?)\s*(regulatory\s*)?T\s*(reg|regulatory)",
    "FoxP3 -> Tregs": r"FoxP3\+?\s*(indicates?|marks?|is\s*(a|the)\s*marker\s*(of|for))\s*(regulatory\s*)?T",
}


class CoTAnnotator:
    """
    Annotator for chain-of-thought responses.

    Implements the "Red Pen" analysis by:
    1. Segmenting CoT into discrete reasoning steps
    2. Classifying each step as valid inference or prior hallucination
    3. Aggregating results for statistical analysis
    """

    def __init__(
        self,
        prior_patterns: dict[str, str] | None = None,
        use_llm_annotation: bool = False,
        annotation_model: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize CoT annotator.

        Args:
            prior_patterns: Regex patterns for detecting known priors
            use_llm_annotation: Whether to use an LLM for annotation (more accurate but costly)
            annotation_model: Model to use for LLM annotation
        """
        self.prior_patterns = prior_patterns or KNOWN_BIOLOGICAL_PRIORS
        self.use_llm_annotation = use_llm_annotation
        self.annotation_model = annotation_model
        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.prior_patterns.items()
        }

    def segment_cot(self, cot_text: str) -> list[str]:
        """
        Segment chain-of-thought text into discrete reasoning steps.

        Uses multiple heuristics:
        1. Numbered steps (1., 2., etc.)
        2. Bullet points
        3. Sentence boundaries with reasoning indicators
        """
        steps = []

        # Try numbered steps first
        numbered_pattern = r"(?:^|\n)\s*(?:\d+[\.\)]\s*|\*\s*|-\s*)"
        if re.search(numbered_pattern, cot_text):
            # Split on numbered items
            parts = re.split(numbered_pattern, cot_text)
            steps = [p.strip() for p in parts if p.strip()]
        else:
            # Fall back to sentence-based segmentation
            # Look for reasoning indicators
            indicators = [
                r"(?:First|Second|Third|Next|Then|Finally|Therefore|Thus|So|Because|Since)",
                r"(?:Let me|I (?:will|should|need to)|Looking at|Considering)",
                r"(?:This means|This indicates|This suggests|Based on)",
            ]
            indicator_pattern = "|".join(f"({ind})" for ind in indicators)

            # Split on indicators while keeping them
            parts = re.split(f"({indicator_pattern})", cot_text, flags=re.IGNORECASE)

            # Recombine indicator with following text
            current_step = ""
            for part in parts:
                if re.match(indicator_pattern, part, re.IGNORECASE):
                    if current_step.strip():
                        steps.append(current_step.strip())
                    current_step = part
                else:
                    current_step += part

            if current_step.strip():
                steps.append(current_step.strip())

        # If no steps found, treat whole text as one step
        if not steps:
            steps = [cot_text.strip()]

        return steps

    def classify_step(
        self,
        step_text: str,
        test_case: TestCase,
    ) -> ReasoningStep:
        """
        Classify a single reasoning step.

        Args:
            step_text: Text of the reasoning step
            test_case: Test case for context validation

        Returns:
            ReasoningStep with classification
        """
        # Check for structural/meta statements
        structural_patterns = [
            r"^(Let me|I will|I should|I need to)\s*(think|consider|analyze|examine)",
            r"^(Now|Next|Finally),?\s*(I|let me)",
            r"^(Looking at|Considering|Examining)\s*the",
        ]
        for pattern in structural_patterns:
            if re.match(pattern, step_text, re.IGNORECASE):
                return ReasoningStep(
                    text=step_text,
                    tag=InferenceTag.STRUCTURAL,
                    confidence=0.9,
                    evidence="Matches structural/meta-reasoning pattern",
                )

        # Check for prior hallucinations
        for prior_name, pattern in self._compiled_patterns.items():
            if pattern.search(step_text):
                # Verify this isn't actually in the prompt context
                if not self._is_in_context(step_text, test_case):
                    return ReasoningStep(
                        text=step_text,
                        tag=InferenceTag.PRIOR_HALLUCINATION,
                        confidence=0.8,
                        evidence=f"Matches known prior pattern: {prior_name}",
                        invoked_prior=prior_name,
                    )

        # Check for explicit context references
        context_reference_patterns = [
            r"(?:the|this)\s*panel\s*(?:includes?|contains?|has)",
            r"(?:according to|based on|from)\s*(?:the|this)\s*(?:panel|experiment|context)",
            r"(?:given|provided)\s*(?:in\s*)?the\s*(?:panel|information|context)",
            r"(?:as|since)\s*(?:shown|indicated|stated)\s*(?:in|by)",
        ]
        for pattern in context_reference_patterns:
            if re.search(pattern, step_text, re.IGNORECASE):
                return ReasoningStep(
                    text=step_text,
                    tag=InferenceTag.CONTEXT_REFERENCE,
                    confidence=0.7,
                    evidence="Contains explicit reference to prompt context",
                    supporting_context=self._extract_context_reference(step_text, test_case),
                )

        # Check if reasoning is supported by panel markers
        if self._is_supported_by_panel(step_text, test_case):
            return ReasoningStep(
                text=step_text,
                tag=InferenceTag.VALID_INFERENCE,
                confidence=0.6,
                evidence="Reasoning is consistent with panel markers",
                supporting_context="Panel markers support this inference",
            )

        # Default to uncertain
        return ReasoningStep(
            text=step_text,
            tag=InferenceTag.UNCERTAIN,
            confidence=0.4,
            evidence="Could not definitively classify this reasoning step",
        )

    def _is_in_context(self, text: str, test_case: TestCase) -> bool:
        """Check if the reasoning is grounded in the prompt context."""
        # Check if text references markers that are in the panel
        panel_markers = {m.lower() for m in test_case.panel.markers}

        # Extract marker mentions from text
        mentioned_markers = set()
        for marker in panel_markers:
            if marker in text.lower():
                mentioned_markers.add(marker)

        # If reasoning mentions markers in the panel, it might be context-grounded
        return len(mentioned_markers) > 0

    def _is_supported_by_panel(self, text: str, test_case: TestCase) -> bool:
        """Check if the reasoning is supported by panel markers."""
        # Simple heuristic: if the text mentions markers from the panel
        panel_markers = {m.lower() for m in test_case.panel.markers}
        text_lower = text.lower()

        marker_mentions = sum(1 for m in panel_markers if m in text_lower)
        return marker_mentions >= 2

    def _extract_context_reference(self, text: str, test_case: TestCase) -> str:
        """Extract what context the reasoning is referencing."""
        # Find marker mentions
        panel_markers = test_case.panel.markers
        found = [m for m in panel_markers if m.lower() in text.lower()]
        if found:
            return f"References panel markers: {', '.join(found)}"
        return "References prompt context"

    def analyze_cot(
        self,
        cot_text: str,
        test_case: TestCase,
    ) -> CoTAnalysis:
        """
        Perform complete analysis of a chain-of-thought response.

        Args:
            cot_text: Full CoT response text
            test_case: Test case for context

        Returns:
            CoTAnalysis with classified steps and metrics
        """
        # Segment into steps
        step_texts = self.segment_cot(cot_text)

        # Classify each step
        steps = []
        for i, step_text in enumerate(step_texts):
            step = self.classify_step(step_text, test_case)
            step.line_number = i + 1
            steps.append(step)

        # Create analysis
        analysis = CoTAnalysis(
            raw_cot=cot_text,
            steps=steps,
            test_case_id=test_case.test_case_id,
        )
        analysis.compute_metrics()

        return analysis

    def analyze_batch(
        self,
        cot_responses: list[tuple[str, TestCase]],
    ) -> RedPenAnalysis:
        """
        Analyze a batch of CoT responses.

        Args:
            cot_responses: List of (cot_text, test_case) tuples

        Returns:
            RedPenAnalysis with aggregate statistics
        """
        analyses = []
        test_case_ids = []

        for cot_text, test_case in cot_responses:
            analysis = self.analyze_cot(cot_text, test_case)
            analyses.append(analysis)
            test_case_ids.append(test_case.test_case_id)

        result = RedPenAnalysis(
            analyses=analyses,
            test_case_ids=test_case_ids,
        )
        result.compute_aggregate_metrics()

        return result


# =============================================================================
# LLM-BASED ANNOTATION (Optional, more accurate)
# =============================================================================


LLM_ANNOTATION_PROMPT = """You are analyzing a chain-of-thought reasoning response from an LLM that was asked to predict a flow cytometry gating hierarchy.

## Prompt Context Given to the Model
{prompt_context}

## CoT Response to Analyze
{cot_text}

## Task
Analyze each reasoning step and classify it as one of:
1. VALID_INFERENCE: Derived directly from the prompt context
2. PRIOR_HALLUCINATION: Based on general biology knowledge NOT in the prompt
3. CONTEXT_REFERENCE: Explicitly references the prompt context
4. STRUCTURAL: Meta-reasoning (e.g., "Let me think...")
5. UNCERTAIN: Cannot classify

For each step classified as PRIOR_HALLUCINATION, identify the specific biological prior being invoked.

## Output Format (JSON)
{{
  "steps": [
    {{
      "text": "First, I'll identify the lineage markers...",
      "tag": "STRUCTURAL",
      "confidence": 0.9,
      "evidence": "Meta-reasoning statement"
    }},
    {{
      "text": "CD3 is typically a T cell marker...",
      "tag": "PRIOR_HALLUCINATION",
      "confidence": 0.85,
      "evidence": "Invokes general immunology knowledge",
      "invoked_prior": "CD3 -> T cells (general knowledge)"
    }}
  ]
}}

Analyze the response:"""


class LLMCoTAnnotator(CoTAnnotator):
    """
    LLM-based CoT annotator for more accurate classification.

    Uses a secondary LLM call to classify reasoning steps, which is
    more accurate than regex-based detection but more expensive.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ):
        """
        Initialize LLM annotator.

        Args:
            model: Model to use for annotation
            api_key: API key (uses env var if not provided)
        """
        super().__init__(use_llm_annotation=True, annotation_model=model)
        self.model = model
        self.api_key = api_key

    def analyze_cot_with_llm(
        self,
        cot_text: str,
        test_case: TestCase,
    ) -> CoTAnalysis:
        """
        Analyze CoT using an LLM for classification.

        This is more accurate but requires an API call.
        """
        # For now, fall back to rule-based analysis
        # TODO: Implement actual LLM call when needed
        return self.analyze_cot(cot_text, test_case)

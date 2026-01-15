# LLM Judge Uncertainty Estimation Design

> Design document for devil's advocate and distributional uncertainty estimation in LLM-as-Judge workflows.

**Status**: Proposed
**Author**: Claude
**Date**: 2025-01-15
**Related**: `libs/hypothesis_pipeline/llm_judge.py`, `libs/hypothesis_pipeline/bias_detection.py`

## Overview

This document outlines approaches for estimating uncertainty in LLM judge evaluations, complementing the bias mitigation work in `bias_detection.py`. The goal is to identify evaluations where the judge may be overconfident or wrong, enabling selective human review.

## Motivation

Current LLM judges output point estimates (e.g., "8/10") without calibrated uncertainty. This leads to:

1. **Overconfident wrong answers**: Judge says 9/10 but prediction has fundamental flaws
2. **Wasted expert time**: Human reviewers check all predictions instead of uncertain ones
3. **Hidden disagreement**: Bootstrap consistency doesn't guarantee correctness
4. **Calibration gaps**: High scores don't correlate with high accuracy

## Proposed Approaches

### Approach 1: Devil's Advocate Uncertainty

Use adversarial self-critique to probe evaluation confidence.

#### How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Initial Judge  │ ──▶ │  Devil's Advocate │ ──▶ │  Uncertainty    │
│  Score: 8/10    │     │  "Find problems"  │     │  Estimate       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

#### Prompt Template

```python
DEVILS_ADVOCATE_PROMPT = """
You previously scored this response {initial_score}/10.

Now play devil's advocate. Your job is to argue AGAINST this score.
Find legitimate problems, not nitpicks:

1. What factual claims might be wrong or unsupported?
2. What logical flaws exist in the reasoning?
3. What important aspects were missed or oversimplified?
4. What assumptions are questionable?

For each issue, rate severity: minor / moderate / major

After your critique, provide:
CRITIQUE_STRENGTH: [0-10] How compelling is the case against the initial score?
ISSUES: [List each real issue found]
REVISED_RANGE: [e.g., "6-9"] What range of scores is now defensible?
CONFIDENCE: [low/medium/high] How certain are you about the original score?
"""
```

#### Uncertainty Computation

```python
@dataclass
class UncertaintyEstimate:
    initial_score: float
    critique_strength: float  # 0-1
    issues_found: list[str]
    issue_severities: list[str]  # minor/moderate/major
    uncertainty_level: Literal["low", "medium", "high"]
    confidence_interval: tuple[float, float]
    recommend_human_review: bool
    score_delta: float  # |initial - revised_midpoint|


def compute_uncertainty(
    initial_score: float,
    critique_strength: float,
    revised_range: tuple[float, float],
    major_issues: int,
) -> UncertaintyEstimate:
    """Compute uncertainty from devil's advocate critique."""

    # Score delta: how much did the critique shift the estimate?
    revised_midpoint = (revised_range[0] + revised_range[1]) / 2
    score_delta = abs(initial_score - revised_midpoint)

    # Confidence interval width
    interval_width = revised_range[1] - revised_range[0]

    # Determine uncertainty level
    if critique_strength > 0.6 or major_issues >= 2 or interval_width > 4:
        level = "high"
    elif critique_strength > 0.3 or major_issues >= 1 or interval_width > 2:
        level = "medium"
    else:
        level = "low"

    return UncertaintyEstimate(
        initial_score=initial_score,
        critique_strength=critique_strength,
        issues_found=issues,
        uncertainty_level=level,
        confidence_interval=(revised_range[0]/10, revised_range[1]/10),
        recommend_human_review=(level == "high"),
        score_delta=score_delta,
    )
```

#### Pros & Cons

| Pros | Cons |
|------|------|
| Forces model to find flaws in own reasoning | 2x API cost |
| Catches overconfidence on high scores | May invent weak objections |
| Domain-agnostic approach | Strong critique can flip correct answers |
| Surfaces hidden assumptions | Negation bias in some models |

---

### Approach 2: Distributional Scoring

Ask judge to distribute confidence points across options.

#### Prompt Template

```python
DISTRIBUTIONAL_PROMPT = """
Distribute 100 points across the answers based on your confidence.
Points must sum to exactly 100. Use increments of 5.

A. [option A]: ___ points
B. [option B]: ___ points
C. [option C]: ___ points
D. [option D]: ___ points

Then explain why your top choice has the most points.

Format: SCORES: A=__, B=__, C=__, D=__
RATIONALE: [explanation]
"""
```

#### Uncertainty from Entropy

```python
def compute_distributional_uncertainty(scores: dict[str, int]) -> float:
    """
    Compute uncertainty from point distribution using entropy.

    Returns: 0 (certain) to 1 (maximum uncertainty)
    """
    import math

    total = sum(scores.values())
    probs = [s / total for s in scores.values() if s > 0]

    # Shannon entropy
    entropy = -sum(p * math.log2(p) for p in probs)

    # Normalize by max entropy (uniform distribution)
    max_entropy = math.log2(len(scores))
    normalized = entropy / max_entropy if max_entropy > 0 else 0

    return normalized


# Interpretation:
# entropy < 0.3: Low uncertainty (one dominant choice)
# entropy 0.3-0.7: Medium uncertainty
# entropy > 0.7: High uncertainty (near-uniform = "I don't know")
```

#### Pros & Cons

| Pros | Cons |
|------|------|
| Forces calibration over binary choice | Arithmetic errors (sum ≠ 100) |
| Captures nuance (55/45 vs 90/10) | Anchoring on round numbers |
| Easy ensemble aggregation | Position bias persists |
| Detects ambiguity (25/25/25/25) | Harder to parse reliably |

---

### Approach 3: Parallel Debate

Two perspectives argue for/against, then reconcile.

#### Prompt Template

```python
DEBATE_PROMPT = """
Two experts are debating the quality of this response.

**ADVOCATE**: Argue why this response deserves a HIGH score (8-10).
- List 3 specific strengths with evidence

**CRITIC**: Argue why this response deserves a LOW score (1-4).
- List 3 specific weaknesses with evidence

**JUDGE**: After hearing both sides:
- Which arguments were most compelling?
- What is the fair score?
- CONFIDENCE: [low/medium/high]
"""
```

#### Uncertainty from Debate

```python
def compute_debate_uncertainty(
    advocate_score: float,
    critic_score: float,
    judge_final: float,
    compelling_side: str,
) -> UncertaintyEstimate:
    """Compute uncertainty from debate dynamics."""

    # Gap between advocate and critic positions
    position_gap = advocate_score - critic_score

    # Where does judge land relative to extremes?
    judge_position = (judge_final - critic_score) / position_gap

    # High uncertainty if:
    # 1. Judge is near middle (couldn't decide)
    # 2. Compelling arguments on both sides
    # 3. Large position gap (fundamental disagreement)

    if 0.4 < judge_position < 0.6:  # Near middle
        level = "high"
    elif position_gap > 5:  # Large disagreement
        level = "medium"
    else:
        level = "low"

    return UncertaintyEstimate(
        uncertainty_level=level,
        confidence_interval=(critic_score/10, advocate_score/10),
        # ...
    )
```

---

## Flow Gating Integration

### New Judge Style: `uncertainty`

Add to existing styles in `flow_gating_benchmark/src/experiments/llm_judge.py`:

```python
JUDGE_STYLES = ["default", "validation", "qualitative", "orthogonal", "binary", "uncertainty"]


def build_uncertainty_prompt(
    test_case_id: str,
    predicted_response: str,
    ground_truth: dict,
    metrics: dict,
    initial_scores: dict,
) -> str:
    """Devil's advocate uncertainty for flow gating predictions."""

    gt_hierarchy = ground_truth.get("gating_hierarchy", {})
    gt_flat = flatten_hierarchy(gt_hierarchy)
    context = ground_truth.get("context", {})
    pred_formatted = format_prediction_for_judge(predicted_response)

    return f"""You previously scored this gating hierarchy prediction:
- COMPLETENESS: {initial_scores.get('completeness', 'N/A')}/10
- ACCURACY: {initial_scores.get('accuracy', 'N/A')}/10
- SCIENTIFIC: {initial_scores.get('scientific', 'N/A')}/10
- OVERALL: {initial_scores.get('overall', 'N/A')}/10

TEST CASE: {test_case_id}
SAMPLE: {context.get('sample_type', 'unknown')} ({context.get('species', 'unknown')})

EXPECTED HIERARCHY:
{gt_flat[:500]}

PREDICTED HIERARCHY:
{pred_formatted}

Play devil's advocate. Find legitimate problems with this prediction:

BIOLOGICAL_ISSUES: [Impossible/implausible cell relationships, or "none"]
HALLUCINATION_RISK: [Gates that might be invented, or "none"]
MISSING_CRITICAL: [Critical gates for this sample type that are missing, or "none"]
MARKER_CONCERNS: [Questionable marker combinations, or "none"]
REVISED_OVERALL: [0-10] Score after considering these issues
CONFIDENCE: [high/medium/low] Certainty about original scores
RECOMMEND_EXPERT_REVIEW: [yes/no] Should a flow cytometrist verify?
"""
```

### Selective Triggering

Don't run on every prediction (cost). Trigger when:

```python
def should_run_uncertainty_estimation(
    initial_result: JudgeResult,
    metrics: dict,
    bootstrap_variance: float,
) -> bool:
    """Decide if uncertainty estimation is needed."""

    # 1. High scores (overconfidence risk)
    if initial_result.overall > 0.7:
        return True

    # 2. Judge disagrees with auto-metrics
    auto_f1 = metrics.get('hierarchy_f1', 0)
    if abs(initial_result.accuracy - auto_f1) > 0.3:
        return True

    # 3. High scientific score but hallucinations detected
    if initial_result.scientific > 0.7 and metrics.get('hallucination_rate', 0) > 0.1:
        return True

    # 4. Low bootstrap variance (unanimous but possibly wrong)
    if bootstrap_variance < 0.05 and initial_result.overall > 0.6:
        return True

    return False
```

### Extended Result Structure

```python
@dataclass
class JudgeResultWithUncertainty:
    """Judge result with uncertainty estimation."""

    # Base fields from JudgeResult
    test_case_id: str
    model: str
    completeness: float
    accuracy: float
    scientific: float
    overall: float

    # Uncertainty fields
    uncertainty_level: Literal["low", "medium", "high"] = "unknown"
    confidence_interval: tuple[float, float] = (0.0, 1.0)
    devils_advocate_issues: list[str] = field(default_factory=list)
    critique_strength: float = 0.0
    revised_score: float | None = None
    score_delta: float = 0.0
    recommend_expert_review: bool = False

    @property
    def is_uncertain(self) -> bool:
        return self.uncertainty_level in ("medium", "high")
```

### Aggregation with Uncertainty

```python
def aggregate_with_uncertainty(
    judge_results: list[JudgeResult],
    uncertainty_results: list[UncertaintyEstimate],
) -> dict:
    """Combine scores with uncertainty for final assessment."""

    mean_score = np.mean([r.overall for r in judge_results])

    # Count high-uncertainty cases
    high_uncertainty = sum(1 for u in uncertainty_results if u.uncertainty_level == "high")
    any_expert_review = any(u.recommend_expert_review for u in uncertainty_results)

    # Widen confidence interval if uncertain
    base_interval = (mean_score - 0.1, mean_score + 0.1)
    if high_uncertainty > 0:
        widening = 0.1 * high_uncertainty
        interval = (max(0, base_interval[0] - widening),
                   min(1, base_interval[1] + widening))
    else:
        interval = base_interval

    return {
        "score": mean_score,
        "confidence_interval": interval,
        "uncertainty_level": "high" if high_uncertainty > 0 else "low",
        "recommend_expert_review": any_expert_review,
        "issues_found": [i for u in uncertainty_results for i in u.issues_found],
    }
```

---

## Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Add `UncertaintyEstimate` dataclass to `llm_judge.py`
- [ ] Implement `DevilsAdvocateJudge` class
- [ ] Add uncertainty config to `BiasAwareJudgeConfig`

### Phase 2: Flow Gating Integration
- [ ] Add `uncertainty` judge style to flow gating benchmark
- [ ] Implement `should_run_uncertainty_estimation()` trigger
- [ ] Extend `JudgeResult` with uncertainty fields
- [ ] Update aggregator to include uncertainty

### Phase 3: Evaluation
- [ ] Compare uncertainty estimates vs human expert disagreement
- [ ] Measure calibration of confidence intervals
- [ ] A/B test: random review vs uncertainty-guided review

---

## Cost Analysis

| Approach | API Calls | When to Use |
|----------|-----------|-------------|
| Single-pass (baseline) | 1x | Always |
| Devil's Advocate | 2x | High scores, metric disagreement |
| Distributional | 1x | Multiple choice scenarios |
| Parallel Debate | 1x (longer prompt) | Ambiguous cases |

**Recommended**: Devil's Advocate with selective triggering.
- Expected trigger rate: ~20-30% of predictions
- Net cost increase: ~1.2-1.3x baseline

---

## References

- [Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge](https://arxiv.org/abs/2410.02736)
- [Humans or LLMs as the Judge? A Study on Judgement Bias](https://arxiv.org/html/2402.10669v3)
- [Self-Preference Bias in LLM-as-a-Judge](https://arxiv.org/html/2410.21819v1)
- [LLM-as-a-judge: Complete Guide (Evidently AI)](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)

---

## Appendix: Alternative Prompts

### Lightweight Single-Pass Uncertainty

```python
UNCERTAINTY_AWARE_PROMPT = """
Evaluate this response on a 0-10 scale.

SCORE: [0-10]
CONFIDENCE: [low/medium/high]
SCORE_RANGE: [e.g., "6-8"] What range is defensible?
KEY_UNCERTAINTY: What's the main source of uncertainty?
"""
```

### Red Team / Blue Team

```python
RED_BLUE_PROMPT = """
RED TEAM: Find 3 reasons this response should score LOW (1-4).
BLUE TEAM: Find 3 reasons this response should score HIGH (8-10).
VERDICT: Which team made stronger arguments? Final score?
"""
```

### Calibration Probe

```python
CALIBRATION_PROMPT = """
You scored this {score}/10.

If we checked 100 similar predictions you scored {score}/10,
what percentage would actually be correct?

ESTIMATED_ACCURACY: [0-100]%
REASONING: [one sentence]
"""
```

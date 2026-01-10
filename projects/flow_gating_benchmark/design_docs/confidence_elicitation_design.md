# Confidence Elicitation Design for Flow Gating Benchmark

## Background

The `libs/hypothesis_pipeline/calibration.py` module provides calibration metrics (ECE, MCE, Brier score) but is currently unused. This document explores whether and how to integrate calibration into flow gating evaluation.

## The Challenge

Unlike classification tasks where models output probabilities, flow gating produces structured hierarchies. There is no natural "confidence score" to calibrate against accuracy.

## Approach Options

### Option 1: Post-hoc Prompting

Add a follow-up prompt after hierarchy generation:

```
Rate your confidence in the accuracy of this gating hierarchy on a scale of 1-10,
where 1 means "very uncertain" and 10 means "highly confident".

Your hierarchy:
[hierarchy JSON]

Confidence (1-10):
```

**Pros:**
- Simple to implement
- Doesn't change main evaluation
- Can be added to existing results

**Cons:**
- Separate API call increases cost
- Model may not have good introspection
- Confidence may not correlate with actual accuracy

### Option 2: Inline Confidence Fields

Request per-gate confidence in the output schema:

```json
{
  "name": "CD4+ T cells",
  "confidence": 0.85,
  "reasoning": "Standard T helper identification using CD3+CD4+",
  "children": [...]
}
```

**Pros:**
- Fine-grained confidence per gate
- Integrated with main output
- Allows per-gate calibration analysis

**Cons:**
- Changes output schema
- Increases output complexity
- May reduce JSON parse success rate

### Option 3: Verbal Hedging Analysis

Analyze natural language hedging in reasoning:

```python
HEDGE_WORDS = {
    "high_confidence": ["clearly", "definitely", "certainly", "standard"],
    "medium_confidence": ["likely", "probably", "typically", "usually"],
    "low_confidence": ["might", "could", "possibly", "uncertain"],
}

def extract_confidence(reasoning: str) -> float:
    # Count hedge words and compute confidence score
    ...
```

**Pros:**
- No prompt modification needed
- Works on existing outputs (if reasoning captured)
- Natural language analysis

**Cons:**
- Noisy signal
- Requires reasoning text (not just JSON)
- Complex to calibrate

## Recommended Approach

**Option 1 (Post-hoc Prompting)** is recommended for initial investigation because:
1. Minimal changes to existing pipeline
2. Easy to A/B test
3. Clear signal (numeric 1-10)

## Pilot Experiment Design

### Setup

1. Select 5-10 diverse test cases (mix of simple/complex)
2. Run with 2 models (e.g., Claude Sonnet, Claude Opus)
3. For each prediction, add post-hoc confidence prompt
4. Compute correlation between stated confidence and actual F1

### Metrics

- Pearson correlation: stated confidence vs F1 score
- Expected Calibration Error (ECE): binned accuracy vs confidence
- Resolution: does confidence discriminate success/failure?

### Success Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Pearson r | > 0.5 | Confidence is meaningful |
| ECE | < 0.15 | Model is well-calibrated |
| Resolution | > 0.1 | Confidence discriminates |

If these thresholds are met, proceed with full calibration integration.

### Pilot Code

```python
from anthropic import Anthropic

def elicit_confidence(
    client: Anthropic,
    model: str,
    hierarchy_json: str,
) -> float:
    """Elicit confidence from model about its prediction."""
    prompt = f"""You just predicted this gating hierarchy for a flow cytometry panel:

{hierarchy_json}

On a scale of 1-10, how confident are you that this hierarchy is correct?
- 1-3: Low confidence (many uncertainties)
- 4-6: Medium confidence (some aspects uncertain)
- 7-9: High confidence (mostly correct)
- 10: Very high confidence (almost certainly correct)

Respond with just the number (1-10):"""

    response = client.messages.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        return int(response.content[0].text.strip()) / 10.0
    except ValueError:
        return 0.5  # Default to medium confidence

def run_calibration_pilot(
    test_cases: list[TestCase],
    model: str,
) -> dict:
    """Run calibration pilot experiment."""
    client = Anthropic()
    results = []

    for tc in test_cases:
        # Get prediction (existing pipeline)
        prediction = run_gating_prediction(tc, model)

        # Elicit confidence
        confidence = elicit_confidence(
            client, model, prediction.hierarchy_json
        )

        # Score prediction
        f1 = compute_hierarchy_f1(prediction, tc.ground_truth)

        results.append({
            "test_case_id": tc.test_case_id,
            "confidence": confidence,
            "f1": f1,
        })

    # Compute correlation
    confidences = [r["confidence"] for r in results]
    f1s = [r["f1"] for r in results]
    correlation = pearsonr(confidences, f1s)[0]

    return {
        "results": results,
        "correlation": correlation,
        "n_samples": len(results),
    }
```

## Decision Tree

```
Run pilot experiment
    │
    ├─ If r > 0.5 and ECE < 0.15:
    │   └─ Proceed with full calibration integration
    │       - Add confidence elicitation to pipeline
    │       - Report calibration metrics in results
    │       - Consider selective prediction (abstain when uncertain)
    │
    └─ If r < 0.5 or ECE > 0.15:
        └─ Confidence is not meaningful for this task
            - Document finding
            - Do not integrate calibration
            - Focus on other improvements
```

## Cost Estimate

Pilot: 10 test cases × 2 models × 2 calls (prediction + confidence) = 40 API calls
Estimated cost: ~$2-5

## Next Steps

1. Implement `elicit_confidence()` function
2. Run pilot on 10 test cases
3. Analyze correlation and ECE
4. Decide whether to proceed with full integration

## References

- Guo et al. (2017): On Calibration of Modern Neural Networks
- Kadavath et al. (2022): Language Models (Mostly) Know What They Know
- Calibration module: `libs/hypothesis_pipeline/calibration.py`

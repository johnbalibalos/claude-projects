# Flow Gating Benchmark Results Summary

**Date:** January 14, 2026
**Benchmark Version:** Full Benchmark (v2)

## Overview

This benchmark evaluates LLM capabilities in predicting flow cytometry gating hierarchies from panel information and experimental context.

### Configuration

| Parameter | Value |
|-----------|-------|
| Models | 6 (opus, sonnet, haiku, gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro) |
| Conditions | 12 (3 context × 2 strategy × 2 reference) |
| Test Cases | 10 (9 OMIP papers + 1 synthetic) |
| Bootstrap Runs | 3 per condition |
| Total Predictions | 2,160 |
| Judge Model | gemini-2.5-pro |
| Judge Styles | 5 (default, validation, qualitative, orthogonal, binary) |

### Conditions Tested

- **Context Levels:** minimal, standard, rich
- **Prompt Strategies:** direct, chain-of-thought (cot)
- **Reference:** none, HIPC gating standards

---

## Summary Results

### Overall Metrics

| Metric | Value |
|--------|-------|
| Mean Hierarchy F1 | 0.328 ± 0.146 |
| Mean Structure Accuracy | 0.093 ± 0.123 |
| Mean Critical Gate Recall | 0.834 ± 0.137 |
| Parse Success Rate | 100% |
| Error Count | 0 |

### Model Rankings by F1 Score

| Rank | Model | F1 Score | Std Dev |
|------|-------|----------|---------|
| 1 | gemini-2.5-pro | 0.361 | 0.162 |
| 2 | gemini-2.0-flash | 0.340 | 0.134 |
| 3 | claude-opus-4-20250514 | 0.330 | 0.179 |
| 4 | claude-sonnet-4-20250514 | 0.326 | 0.130 |
| 5 | claude-3-5-haiku-20241022 | 0.306 | 0.106 |
| 6 | gemini-2.5-flash | 0.305 | 0.145 |

---

## LLM Judge Evaluation

The multi-judge framework uses 5 different evaluation styles to reduce prompt bias and provide robust quality assessment.

### Judge Styles

| Style | Description | Focus |
|-------|-------------|-------|
| default | Standard quality assessment | Overall gating accuracy |
| validation | Error checking focus | Missing gates, invalid markers |
| qualitative | Biological reasoning | Domain appropriateness |
| orthogonal | Alternative evaluation axes | Completeness, specificity |
| binary | Pass/fail threshold | Minimum viability |

### Results by Style

| Style | Avg Quality | Avg Consistency | Errors |
|-------|-------------|-----------------|--------|
| qualitative | 0.470 | 0.418 | 0 |
| binary | 0.465 | 0.424 | 0 |
| validation | 0.464 | 0.404 | 49 |
| default | 0.461 | 0.415 | 0 |
| orthogonal | 0.461 | 0.431 | 0 |

### Model Rankings by Judge Quality (Default Style)

| Rank | Model | Quality | Consistency |
|------|-------|---------|-------------|
| 1 | gemini-2.5-pro | 0.592 | 0.396 |
| 2 | claude-opus-4-20250514 | 0.523 | 0.177 |
| 3 | gemini-2.5-flash | 0.506 | 0.633 |
| 4 | gemini-2.0-flash | 0.408 | 0.652 |
| 5 | claude-sonnet-4-20250514 | 0.391 | 0.502 |
| 6 | claude-3-5-haiku-20241022 | 0.343 | 0.129 |

---

## Key Analysis: OMIP vs Synthetic Panel Performance

### Results

CUSTOM-PBMC-001 was generated using a synthetic template generator. All models perform significantly better (+14-25% F1) on this synthetic panel compared to OMIP papers.

| Model | OMIP F1 | CUSTOM F1 | Delta |
|-------|---------|-----------|-------|
| claude-sonnet-4-20250514 | 0.301 | 0.551 | **+0.251** |
| gemini-2.5-flash | 0.283 | 0.504 | +0.221 |
| claude-opus-4-20250514 | 0.310 | 0.506 | +0.196 |
| gemini-2.5-pro | 0.343 | 0.527 | +0.184 |
| claude-3-5-haiku-20241022 | 0.289 | 0.460 | +0.171 |
| gemini-2.0-flash | 0.325 | 0.469 | +0.143 |

### Interpretation

The consistent improvement on synthetic panels likely reflects:

1. **Template-generated panels are cleaner** - The synthetic generator produces unambiguous marker-to-population mappings with predictable hierarchy structure.

2. **OMIP papers have real-world complexity** - Multiple valid gating strategies, ambiguous terminology, and domain-specific conventions make exact matching harder.

3. **Not a direct test of reasoning vs memorization** - Since CUSTOM-PBMC-001 follows a template, models may perform better due to structural predictability rather than marker-based reasoning.

**Note:** A proper "alien cell" test (novel population names with valid marker logic) would better distinguish reasoning from pattern matching.

---

## F1 vs Judge Correlation

### Ranking Comparison

| Metric | #1 | #2 | #3 | #4 | #5 | #6 |
|--------|----|----|----|----|----|----|
| F1 Score | gemini-2.5-pro | gemini-2.0-flash | claude-opus-4-20250514 | claude-sonnet-4-20250514 | claude-3-5-haiku-20241022 | gemini-2.5-flash |
| Judge Quality | gemini-2.5-pro | claude-opus-4-20250514 | gemini-2.5-flash | gemini-2.0-flash | claude-sonnet-4-20250514 | claude-3-5-haiku-20241022 |

### Key Observations

1. **gemini-2.5-pro leads both metrics** - Consistent top performer across F1 and judge evaluation.

2. **opus underrated by F1** - Ranks #3 by F1 but #2 by judge quality, suggesting F1 penalizes opus's output format.

3. **gemini-2.0-flash overrated by F1** - Ranks #2 by F1 but #4 by judge quality.

4. **Consistency varies widely** - gemini-2.0-flash (0.65) and gemini-2.5-flash (0.63) are most consistent; haiku (0.13) and opus (0.18) are least consistent.

---

## Condition Analysis

### Best Conditions by Model

| Model | Best Condition | F1 |
|-------|----------------|-----|
| claude-opus-4-20250514 | minimal_cot_none | 0.365 |
| claude-sonnet-4-20250514 | rich_direct_none | 0.392 |
| claude-3-5-haiku-20241022 | rich_direct_hipc | 0.363 |
| gemini-2.0-flash | standard_cot_hipc | 0.385 |
| gemini-2.5-flash | rich_cot_hipc | 0.368 |
| gemini-2.5-pro | rich_direct_hipc | 0.402 |

### Context Level Impact

| Context | Mean F1 |
|---------|---------|
| minimal | 0.314 |
| standard | 0.328 |
| rich | 0.340 |

**Finding:** Rich context provides ~8% relative improvement over minimal context.

### Strategy Impact

| Strategy | Mean F1 |
|----------|---------|
| direct | 0.332 |
| cot | 0.324 |

**Finding:** Direct prompting slightly outperforms chain-of-thought on average.

### HIPC Reference Impact

| Reference | Mean F1 |
|-----------|---------|
| none | 0.319 |
| hipc | 0.337 |

**Finding:** Including HIPC gating standards improves F1 by ~5.6%.

---

## Test Cases

### Performance by Test Case

| Test Case | Mean F1 | Notes |
|-----------|---------|-------|
| CUSTOM-PBMC-001 | 0.503 | Synthetic panel (control) |
| OMIP-008 | ~0.35 | T cell cytokines |
| OMIP-022 | ~0.32 | T memory |
| OMIP-074 | ~0.28 | B cells |
| OMIP-076 | ~0.25 | Murine pan |
| OMIP-077 | ~0.30 | Pan-leukocyte |
| OMIP-083 | ~0.29 | Monocytes |
| OMIP-087 | ~0.32 | CyTOF |
| OMIP-095 | ~0.28 | 40-color spectral |
| OMIP-101 | ~0.31 | Whole blood |

---

## Recommendations

### For Model Selection

1. **Best overall:** gemini-2.5-pro (highest F1 and judge quality)
2. **Best value:** gemini-2.0-flash (high consistency, competitive F1, lowest cost)
3. **Avoid:** haiku (lowest quality scores) and gemini-2.5-flash (low F1)

### For Prompt Design

1. **Use rich context** when available (+8% F1)
2. **Include HIPC reference** (+5.6% F1)
3. **Prefer direct prompting** for most models (slightly better than CoT)

### For Evaluation

1. **Don't rely on F1 alone** - Judge quality captures semantic correctness F1 misses
2. **Consider consistency** - High variance models may be unreliable in production
3. **Include synthetic controls** - Essential for distinguishing reasoning from memorization

---

## Technical Details

### Data Location

- Predictions: `results/full_benchmark_20260114/predictions.json`
- Scoring: `results/full_benchmark_20260114/scoring_results.json`
- Multi-judge: `results/full_benchmark_20260114/multijudge/`

### Reproducibility

```bash
# Run full benchmark
python scripts/run_modular_pipeline.py \
    --phase all \
    --models opus sonnet haiku gemini-2.0-flash gemini-2.5-flash gemini-2.5-pro \
    --test-cases data/verified \
    --n-bootstrap 3 \
    --output results/full_benchmark_$(date +%Y%m%d)

# Run multi-judge
python scripts/run_aggregated_multijudge.py \
    --predictions results/full_benchmark_20260114/predictions.json \
    --test-cases data/verified \
    --output results/full_benchmark_20260114/multijudge \
    --workers 40
```

---

## Changelog

- **2026-01-14:** Full benchmark with 6 models, 12 conditions, 10 test cases, 3 bootstrap runs. Multi-judge with 5 styles.

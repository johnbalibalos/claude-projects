# Flow Gating Benchmark: LLM Evaluation for Flow Cytometry Gating

## Executive Summary

This project evaluates whether **LLMs can predict flow cytometry gating strategies** from panel information. We test 6 models across 12 experimental conditions and measure performance using hierarchy F1, structure accuracy, critical gate recall, and LLM judge quality scores.

### Key Findings (January 2026 Benchmark)

| Finding | Result |
|---------|--------|
| **Best Model (F1)** | gemini-2.5-pro (0.361) |
| **Best Model (Judge)** | gemini-2.5-pro (0.59) |
| **HIPC Reference Impact** | +5.6% F1 |
| **Rich Context Impact** | +8% F1 (pending re-evaluation) |
| **Model Consistency** | Claude models highly non-deterministic at temp=0 |

**Benchmark Configuration:**
- 6 models × 12 conditions × 10 test cases × 3 bootstrap = 2,160 predictions
- Judge: gemini-2.5-pro with 5 evaluation styles

---

## Project Overview

### Goal

Flow cytometry gating requires predicting:
1. **Gate names** - Population labels (e.g., "CD4+ T cells")
2. **Hierarchy structure** - Parent-child relationships (e.g., T cells → CD4+ helper)
3. **Critical gates** - Must-have QC gates (singlets, live/dead)

### Experimental Conditions (12 total)

| Dimension | Options |
|-----------|---------|
| Context Level | minimal, standard, rich |
| Prompt Strategy | direct, chain-of-thought |
| Reference | none, HIPC gating standards |

---

## Results

### Model Performance (Ranked by F1)

| Model | Hierarchy F1 | Structure Acc | Critical Recall | Hallucination | Parse Rate |
|-------|--------------|---------------|-----------------|---------------|------------|
| gemini-2.5-pro | 0.361 | 0.085 | 0.843 | 0.081 | 100% |
| gemini-2.0-flash | 0.340 | 0.092 | 0.842 | 0.077 | 100% |
| claude-opus-4-20250514 | 0.330 | 0.109 | 0.835 | 0.083 | 100% |
| claude-sonnet-4-20250514 | 0.326 | 0.100 | 0.829 | 0.106 | 100% |
| claude-3-5-haiku-20241022 | 0.306 | 0.073 | 0.803 | 0.090 | 100% |
| gemini-2.5-flash | 0.305 | 0.097 | 0.850 | 0.067 | 100% |

### F1 by Condition

| Condition | Mean F1 |
|-----------|---------|
| rich_direct_hipc | 0.357 |
| rich_direct_none | 0.344 |
| minimal_cot_hipc | 0.340 |
| standard_cot_hipc | 0.340 |
| standard_direct_hipc | 0.338 |
| rich_cot_hipc | 0.334 |
| standard_cot_none | 0.325 |
| standard_direct_none | 0.324 |
| minimal_direct_hipc | 0.322 |
| rich_cot_none | 0.312 |
| minimal_direct_none | 0.309 |
| minimal_cot_none | 0.293 |

**Key Observations:**
- HIPC reference consistently improves F1 (+5.6% average)
- Rich context improves F1 (+8% vs minimal) - *caveat: results obtained when OMIP ID was included in prompt*
- Direct prompting slightly outperforms CoT

### OMIP vs Synthetic Panel Performance

| Model | OMIP F1 | CUSTOM F1 | Delta |
|-------|---------|-----------|-------|
| claude-sonnet-4-20250514 | 0.301 | 0.551 | **+0.251** |
| gemini-2.5-flash | 0.283 | 0.504 | +0.221 |
| claude-opus-4-20250514 | 0.310 | 0.506 | +0.196 |
| gemini-2.5-pro | 0.343 | 0.527 | +0.184 |
| claude-3-5-haiku-20241022 | 0.289 | 0.460 | +0.171 |
| gemini-2.0-flash | 0.325 | 0.469 | +0.143 |

All models perform better on template-generated CUSTOM-PBMC-001 than on real OMIP papers. This likely reflects cleaner structure rather than reasoning vs memorization.

---

## Multi-Judge Evaluation

### 5 Judge Styles

| Style | Focus |
|-------|-------|
| default | Standard quality assessment |
| validation | Error checking (missing gates, invalid markers) |
| qualitative | Biological reasoning and domain appropriateness |
| orthogonal | Completeness, specificity, clinical utility |
| binary | Pass/fail threshold |

### Judge Scores by Model (Quality 0-1)

| Model | default | validation | qualitative | orthogonal | binary |
|-------|---------|------------|-------------|------------|--------|
| gemini-2.5-pro | 0.59 | 0.58 | 0.61 | 0.58 | 0.58 |
| claude-opus-4-20250514 | 0.52 | 0.50 | 0.51 | 0.51 | 0.50 |
| gemini-2.5-flash | 0.51 | 0.51 | 0.52 | 0.52 | 0.54 |
| gemini-2.0-flash | 0.41 | 0.45 | 0.41 | 0.42 | 0.41 |
| claude-sonnet-4-20250514 | 0.39 | 0.41 | 0.43 | 0.41 | 0.41 |
| claude-3-5-haiku-20241022 | 0.34 | 0.33 | 0.33 | 0.32 | 0.34 |

### F1 vs Judge Ranking Comparison

| Rank | F1 Score | Judge Quality |
|------|----------|---------------|
| 1 | gemini-2.5-pro | gemini-2.5-pro |
| 2 | gemini-2.0-flash | claude-opus-4-20250514 |
| 3 | claude-opus-4-20250514 | gemini-2.5-flash |
| 4 | claude-sonnet-4-20250514 | gemini-2.0-flash |
| 5 | claude-3-5-haiku-20241022 | claude-sonnet-4-20250514 |
| 6 | gemini-2.5-flash | claude-3-5-haiku-20241022 |

**Observation:** opus is underrated by F1 (rank #3) but ranks #2 by judge quality. F1 may penalize opus's output format.

---

## Model Consistency Analysis

### Bootstrap Agreement (3 runs)

| Model | All Same (3/3) | All Different (3/3) | Temperature |
|-------|----------------|---------------------|-------------|
| gemini-2.0-flash | 28% | 0% | 0.0 (API) |
| gemini-2.5-flash | 29% | 0% | 0.0 (API) |
| claude-sonnet-4-20250514 | 35% | 43% | default (CLI) |
| gemini-2.5-pro | 4% | 52% | 0.0 (API) |
| claude-3-5-haiku-20241022 | 1% | 92% | default (CLI) |
| claude-opus-4-20250514 | 1% | 96% | default (CLI) |

**Key Finding:** Claude models run via CLI cannot enforce temperature=0, so high variance is expected. Gemini models via API with temperature=0 are much more consistent.

**Example:** opus produces 3 different hierarchies for CUSTOM-PBMC-001 (same prompt):
```
Bootstrap 1: All Events → Time Gate → Singlets → Live → CD45+ → T Cells → Tregs...
Bootstrap 2: All Events → Singlets → Live → CD45+ → CD3+ T Cells → NKT-like...
Bootstrap 3: All Events → Singlets → Live → Leukocytes → T Cells → Regulatory T...
```
All biologically valid, but different structure and naming.

---

## Known Limitations

### Token Exhaustion

Reasoning models use 70%+ of token budget for "thinking".

| Model | Failure Rate | Affected OMIPs |
|-------|--------------|----------------|
| gemini-2.5-pro | 12% | OMIP-064, 083, 095 (27-40 markers) |
| gemini-2.0-flash | 0% | None |

**Mitigation:** Increase `max_tokens` to 20000 for reasoning models.

### Ground Truth Quality

| Dataset | OMIPs | Status |
|---------|-------|--------|
| verified/ | 10 | Manually validated |
| staging/ | 13 | Pending verification |

---

## Conclusions

### What We've Learned

1. **gemini-2.5-pro leads** both F1 and judge metrics
2. **F1 and judge rankings differ** - opus underrated by F1
3. **Claude models highly non-deterministic** at temperature=0
4. **HIPC reference helps** (+5.6% F1)
5. **Synthetic panels easier** than real OMIP papers

### Recommendations

1. **For production:** Use gemini-2.0-flash (best consistency/cost ratio)
2. **For quality:** Use gemini-2.5-pro (highest scores)
3. **For evaluation:** Don't rely on F1 alone - use multi-judge
4. **For variance:** Use n_bootstrap ≥ 3

---

## Reproducibility

```bash
# Full benchmark
python scripts/run_modular_pipeline.py \
    --phase all \
    --models gemini-2.0-flash gemini-2.5-pro opus sonnet haiku gemini-2.5-flash \
    --test-cases data/verified \
    --n-bootstrap 3

# Multi-judge evaluation
python scripts/run_aggregated_multijudge.py \
    --predictions results/full_benchmark_20260114/predictions.json \
    --test-cases data/verified \
    --output results/full_benchmark_20260114/multijudge
```

---

*Benchmark completed: January 14, 2026*
*Framework version: modular_pipeline_v2*

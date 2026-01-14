# Flow Gating Benchmark: LLM Evaluation for Flow Cytometry Gating

## Executive Summary

This project evaluates whether **LLMs can predict flow cytometry gating strategies** from panel information. We test multiple models across varying experimental conditions and measure performance using multiple F1 metrics designed to capture both string-level and semantic-level correctness.

### Key Finding (Preliminary)

> **Results pending:** Clean rerun in progress on verified dataset with 4 F1 metric variants.

**Early observations:**
- Weak correlation (r≈0.15) between string-based F1 and LLM judge scores
- Models produce biologically correct but linguistically different gate names
- Multiple F1 metrics needed to fairly evaluate across models

---

## Project Overview

### Goal

Flow cytometry gating requires predicting:
1. **Gate names** - Population labels (e.g., "CD4+ T cells")
2. **Hierarchy structure** - Parent-child relationships (e.g., T cells → CD4+ helper)
3. **Critical gates** - Must-have QC gates (singlets, live/dead)

Traditional evaluation:
- **String F1**: Exact match after normalization (penalizes valid synonyms)
- **Structure accuracy**: Parent-child relationships correct

**Our hypothesis**: String-based F1 unfairly penalizes models that produce biologically correct but linguistically different gate names. Multiple F1 metrics are needed for fair evaluation.

---

## Architecture

```
flow_gating_benchmark/
├── src/
│   ├── curation/                    # Ground truth data
│   │   ├── schemas.py               # TestCase, Panel, GatingHierarchy
│   │   └── omip_extractor.py        # Load test cases
│   ├── experiments/                 # Pipeline
│   │   ├── prediction_collector.py  # Parallel LLM calls
│   │   ├── batch_scorer.py          # 4 F1 variants + metrics
│   │   ├── llm_judge.py             # 5 judge styles
│   │   └── llm_client.py            # Gemini, Claude, OpenAI
│   └── evaluation/                  # Scoring
│       ├── metrics.py               # hierarchy_f1, structure, critical
│       ├── normalization.py         # 200+ gate synonyms
│       ├── semantic_similarity.py   # MiniLM embeddings
│       └── hierarchy.py             # Tree operations
├── data/
│   ├── verified/                    # 10 curated OMIPs
│   └── staging/                     # 17 pending verification
└── scripts/
    └── run_modular_pipeline.py      # Main entry point
```

---

## F1 Metric Comparison

### Why Multiple F1 Metrics?

The core problem: **biologically equivalent names score as mismatches**.

| Ground Truth | Prediction | String F1 | Biologically? |
|--------------|------------|-----------|---------------|
| `T cells` | `T Lymphocytes` | 0 | Same |
| `CD4+ T cells` | `Helper T cells` | 0 | Same |
| `Natural Killer cells` | `NK cells` | 0 | Same |

### 4 F1 Variants

| Metric | Method | What it catches |
|--------|--------|-----------------|
| `hierarchy_f1` | String normalization | "CD4+ T Cells" ↔ "CD4 positive T cells" |
| `synonym_f1` | 200+ synonym dictionary | "T Lymphocytes" ↔ "T cells" ↔ "CD3+" |
| `semantic_f1` | MiniLM embeddings (cosine ≥0.70) | "Helper T cells" ↔ "CD4+ T cells" |
| `weighted_semantic_f1` | Confidence-weighted | Partial credit for 0.70-0.85 similarity |

### Expected Pattern

```
hierarchy_f1 ≤ synonym_f1 ≤ semantic_f1
```

If `semantic_f1 >> hierarchy_f1`, models are producing biologically correct but linguistically different names.

---

## Results (Placeholder - Rerun Pending)

### F1 Comparison by Model

| Model | hierarchy_f1 | synonym_f1 | semantic_f1 | weighted_semantic_f1 |
|-------|--------------|------------|-------------|----------------------|
| gemini-2.0-flash | TBD | TBD | TBD | TBD |
| claude-sonnet-4 | TBD | TBD | TBD | TBD |
| claude-opus-4 | TBD | TBD | TBD | TBD |

### F1 Comparison by Condition

| Condition | hierarchy_f1 | synonym_f1 | semantic_f1 | weighted_semantic_f1 |
|-----------|--------------|------------|-------------|----------------------|
| minimal_direct | TBD | TBD | TBD | TBD |
| minimal_cot | TBD | TBD | TBD | TBD |
| standard_direct | TBD | TBD | TBD | TBD |
| standard_cot | TBD | TBD | TBD | TBD |

### Other Metrics

| Model | Structure Acc | Critical Recall | Hallucination Rate | Parse Rate |
|-------|---------------|-----------------|--------------------| -----------|
| gemini-2.0-flash | TBD | TBD | TBD | TBD |
| claude-sonnet-4 | TBD | TBD | TBD | TBD |
| claude-opus-4 | TBD | TBD | TBD | TBD |

---

## Multi-Judge Evaluation

### 5 Judge Styles

| Style | Focus | Output |
|-------|-------|--------|
| `default` | Standard quality assessment | COMPLETENESS, ACCURACY, SCIENTIFIC, OVERALL (0-10) |
| `validation` | Blind metric estimation | ESTIMATED_F1, ESTIMATED_STRUCTURE, CONFIDENCE |
| `qualitative` | Structured feedback, no scores | ERRORS, MISSING_GATES, EXTRA_GATES, ACCEPT |
| `orthogonal` | Dimensions F1 can't capture | CLINICAL_UTILITY, BIOLOGICAL_PLAUSIBILITY, HALLUCINATION_SEVERITY |
| `binary` | Pass/fail decision | ACCEPTABLE (yes/no), RECOMMENDATION |

### Judge Scores by Model (Placeholder)

| Model | default | validation | qualitative | orthogonal | binary |
|-------|---------|------------|-------------|------------|--------|
| gemini-2.0-flash | TBD | TBD | TBD | TBD | TBD |
| claude-sonnet-4 | TBD | TBD | TBD | TBD | TBD |
| claude-opus-4 | TBD | TBD | TBD | TBD | TBD |

---

## Hypothesis Testing Framework

### Available Tests

| Test | Question | Key Metric |
|------|----------|------------|
| **Alien Cell** | Reasoning vs memorization? | F1 delta with nonsense names |
| **Format Ablation** | Prose parsing or reasoning failure? | Format variance |
| **Cognitive Refusal** | Context blind or over-cautious? | Forcing effect |
| **Frequency Confound** | Token frequency or reasoning? | R² correlation |

### Alien Cell Test

**Design:** Replace real cell names with nonsense while preserving marker logic.

```
Original: "CD3+ CD4+ CD25+ FoxP3+ → Regulatory T cells"
Alien:    "CD3+ CD4+ CD25+ FoxP3+ → Glorp Cells"
```

**Interpretation:**
- Model succeeds on alien cells → Reasoning from markers
- Model fails on alien cells → Pattern-matching from training data

**Status:** Implemented (`src/analysis/alien_cell.py`), results pending.

---

## Key Questions

### 1. Biological Equivalence Gap

**Question:** How much do models lose to string matching vs semantic equivalence?

| Metric | Mean | Interpretation |
|--------|------|----------------|
| hierarchy_f1 - semantic_f1 | TBD | Gap size |

**Expected:** Larger gap → more biologically correct but linguistically different predictions.

### 2. F1-Judge Correlation

**Question:** Which F1 metric best predicts LLM judge scores?

| F1 Metric | Correlation with Judge | p-value |
|-----------|------------------------|---------|
| hierarchy_f1 | TBD (early: r≈0.15) | TBD |
| synonym_f1 | TBD | TBD |
| semantic_f1 | TBD | TBD |
| weighted_semantic_f1 | TBD | TBD |

**Implication:** If semantic_f1 correlates better, use it as primary metric.

### 3. Sonnet-Opus Gap

**Question:** Does the Sonnet-Opus performance gap shrink with semantic matching?

**Context:** Sonnet's verbose naming ("T Cells (CD3+)") may inflate hierarchy_f1 vs Opus's terse output.

| Model | hierarchy_f1 | semantic_f1 | Gap Change |
|-------|--------------|-------------|------------|
| claude-sonnet-4 | TBD | TBD | TBD |
| claude-opus-4 | TBD | TBD | TBD |

---

## Known Limitations

### Token Exhaustion

Reasoning models (gemini-2.5-pro, claude-opus) use 70%+ of token budget for "thinking".

| Model | Failure Rate | Affected OMIPs |
|-------|--------------|----------------|
| gemini-2.5-pro | 12% | OMIP-064, 083, 095 (27-40 markers) |
| gemini-2.0-flash | 0% | None |

**Mitigation:** Increase `max_tokens` to 20000 for reasoning models.

### Non-Determinism at Temperature=0

Claude models produce ~10 unique responses per 10 runs even at temperature=0.

| Model | Unique Responses / 10 Runs |
|-------|----------------------------|
| claude-sonnet-cli | ~10 |
| claude-opus-cli | ~10 |
| gemini-2.0-flash | ~3-5 |

**Implication:** Use n_bootstrap ≥ 5 for reliable variance estimation.

### Ground Truth Quality

Results depend on ground truth curation quality.

| Dataset | OMIPs | Status |
|---------|-------|--------|
| verified/ | 10 | Manually validated |
| staging/ | 17 | Pending verification |

---

## Reproducibility

### Running the Benchmark

```bash
cd flow_gating_benchmark

# Set API keys
export GOOGLE_API_KEY=...
export ANTHROPIC_API_KEY=...  # Optional for CLI models

# Full benchmark on verified data
python scripts/run_modular_pipeline.py \
    --phase all \
    --models gemini-2.0-flash claude-sonnet-cli \
    --test-cases data/verified \
    --n-bootstrap 5

# Quick test (~$0.01)
python scripts/run_modular_pipeline.py \
    --phase all \
    --models gemini-2.0-flash \
    --max-cases 1 \
    --force
```

### Expected Costs

| Configuration | API Calls | Gemini Cost | Claude Cost |
|---------------|-----------|-------------|-------------|
| Quick test | 4 | ~$0.01 | ~$0.05 |
| 10 OMIPs × 4 conditions | 40 | ~$0.40 | ~$2 |
| + 5 bootstrap | 200 | ~$2 | ~$10 |
| + LLM judge | +200 | ~$20 | - |

---

## Conclusions (Preliminary)

### What We've Learned

1. **String F1 is insufficient** - Weak correlation with judge scores (r≈0.15)
2. **Models produce valid synonyms** - "Helper T cells" vs "CD4+ T cells"
3. **Multiple metrics needed** - 4 F1 variants capture different aspects
4. **Judge styles matter** - 5 styles for comprehensive evaluation

### Recommendations

1. **Use semantic_f1** as primary metric (pending correlation analysis)
2. **Include multi-judge** for qualitative assessment
3. **Run n_bootstrap ≥ 5** for variance estimation
4. **Test on verified OMIPs only** for reliable results

---

## Next Steps

1. Complete clean rerun with 4 F1 variants on verified OMIPs
2. Analyze F1 metric correlations with judge scores
3. Run Alien Cell test to distinguish reasoning vs memorization
4. Update results tables with actual data

---

*Study in progress - January 2026*
*Framework version: modular_pipeline_v2*

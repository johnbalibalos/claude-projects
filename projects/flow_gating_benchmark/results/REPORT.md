# Flow Gating Benchmark: Gating Strategy Prediction Report

**Date:** January 9, 2026
**Models:** Claude Sonnet 4, Claude Opus 4
**Author:** John Balibalos

## Executive Summary

This benchmark evaluates whether LLMs can predict flow cytometry gating strategies from panel information. Using 8 OMIP papers with high-concordance extractions (XML vs LLM panel concordance ≥0.95), we find that **Claude Sonnet achieves 38.4% hierarchy F1 overall, with rich_direct context reaching 46.7%**. Opus performs slightly lower at 31.8% F1 overall but has higher structure accuracy (61.0% vs 57.4%).

## Key Finding: Direct Prompting Shows Small Advantage

Direct prompting shows a modest advantage over chain-of-thought (CoT), though the difference is smaller than initially reported:

| Model | Direct F1 | CoT F1 | Difference |
|-------|-----------|--------|------------|
| Sonnet | 0.391 | 0.377 | **+1.4pp** |
| Opus | 0.328 | 0.308 | **+2.0pp** |

Note: These differences are within the noise margin given high per-test-case variance (std ~0.1-0.2).

## Results Summary

### Model Comparison

| Metric | Sonnet | Opus |
|--------|--------|------|
| Hierarchy F1 | **0.384** | 0.318 |
| Structure Accuracy | 0.574 | **0.610** |
| Critical Gate Recall | **0.839** | 0.795 |
| Parse Success Rate | 100% | 100% |

### Performance by Condition

#### Claude Sonnet 4

| Condition | F1 Score |
|-----------|----------|
| minimal_direct | 0.270 |
| minimal_cot | 0.296 |
| standard_direct | 0.436 |
| standard_cot | 0.440 |
| **rich_direct** | **0.467** |
| rich_cot | 0.395 |

#### Claude Opus 4

| Condition | F1 Score |
|-----------|----------|
| minimal_direct | 0.204 |
| minimal_cot | 0.347 |
| **standard_direct** | **0.390** |
| standard_cot | 0.292 |
| rich_direct | 0.389 |
| rich_cot | 0.284 |

## Test Cases

8 OMIP papers with validated panel extractions:

| OMIP | Title | Panel Concordance |
|------|-------|-------------------|
| OMIP-022 | Human γδ T-cell populations | 1.00 |
| OMIP-064 | Human PBMC immunophenotyping | 1.00* |
| OMIP-074 | Human B-cell subsets | 1.00 |
| OMIP-076 | Murine T/B/ASC subsets | 1.00 |
| OMIP-077 | Human dendritic cell subsets | 1.00 |
| OMIP-083 | Human PBMC 28-color | 1.00 |
| OMIP-095 | Human PBMC spectral | 1.00* |
| OMIP-101 | Fixed whole blood 27-color | 1.00 |

*LLM-only extraction (no XML panel table available)

## Experimental Design

### Conditions (6 per model)

| Factor | Levels |
|--------|--------|
| Context | minimal, standard, rich |
| Prompting | direct, chain-of-thought |

**Total evaluations:** 96 (8 test cases × 6 conditions × 2 models)

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Hierarchy F1** | Gate name precision/recall with fuzzy matching |
| **Structure Accuracy** | Correct parent-child relationships |
| **Critical Gate Recall** | Must-have gates (Singlets, Live, Lymphocytes) |

## Key Findings

### 1. Rich Context Improves Performance
Both models improve significantly with more context:
- Sonnet: +19.7pp improvement (minimal_direct → rich_direct)
- Opus: +18.5pp improvement (minimal_direct → standard_direct)

### 2. Direct vs CoT: Mixed Results
The advantage of direct prompting is context-dependent:
- **Minimal context**: CoT actually outperforms direct (Opus: +14.3pp, Sonnet: +2.6pp)
- **Rich context**: Direct outperforms CoT (Sonnet: +7.2pp, Opus: +10.5pp)

This suggests CoT helps when context is limited, but adds noise when context is sufficient.

### 3. Hallucination Rate Increases with Context
More context leads to more hallucinated gates:

| Context | Sonnet | Opus |
|---------|--------|------|
| minimal | 7.3% | 9.2% |
| standard | 14.2% | 14.9% |
| rich | 16.6% | 16.1% |

Higher F1 from rich context comes at the cost of more false positives.

### 4. Test Case Difficulty Varies Widely
Performance varies dramatically by OMIP paper:

| OMIP | Sonnet F1 | Opus F1 | Why? |
|------|-----------|---------|------|
| OMIP-077 | 0.54 | 0.30 | Standard leukocyte names (T cells, NK, B cells) |
| OMIP-101 | 0.49 | 0.52 | Well-defined whole blood populations |
| OMIP-083 | 0.25 | 0.21 | Specialized monocyte terminology (SLAN+/−) |
| OMIP-074 | 0.26 | 0.31 | Unusual B-cell subsets (TLM, AM, RM) |

**Key insight**: LLMs perform well on canonical population names but struggle with specialized research terminology.

### 5. Task Failures in Minimal Context
6 evaluations (6.25%) produced complete task failures where models output meta-questions instead of gate names:
- "The research question or cell populations of interest"
- "Which fluorochromes/channels are being used"

These occurred exclusively in minimal_direct and minimal_cot conditions, indicating the minimal context prompt is ambiguous.

## Limitations

1. **Small test set**: 8 OMIP papers (4 failed validation due to missing fluorophore data)
2. **Extraction quality**: Ground truth from automated extraction, not manual curation
3. **Single run**: No statistical significance testing
4. **High variance**: Per-test-case std of 0.1-0.2 on means of 0.2-0.4 limits confidence in condition comparisons
5. **Vocabulary bias**: Test set skews toward specialized terminology that may not reflect typical use cases
6. **No semantic validation**: Parser accepts syntactically valid but semantically invalid outputs (meta-questions as gate names)

## Raw Data

- Sonnet results: `experiment_results_20260109_210320.json`
- Opus results: `experiment_results_20260109_210400.json`
- Duration: Sonnet 12.4 min, Opus 17.5 min

## Recommendations

### For Users
1. **Use rich context** - provides ~20pp F1 improvement over minimal
2. **Use direct prompting with rich context** - best overall performance
3. **Expect ~15% hallucination rate** - always verify predicted gates against panel markers

### For Benchmark Improvement
4. **Add semantic validation** - detect when models output meta-questions instead of gate names
5. **Expand test set** - fix fluorophore extraction for 4 failed OMIPs; add more canonical panels
6. **Run multiple trials** - current variance too high for reliable condition comparisons
7. **Stratify by vocabulary type** - separate canonical (leukocyte subsets) from specialized (research-specific) terminology
8. **Improve minimal context prompt** - clarify that output should be gate names, not questions

---

*Report generated: 2026-01-09*

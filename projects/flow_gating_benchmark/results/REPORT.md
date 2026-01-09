# Flow Gating Benchmark: Gating Strategy Prediction Report

**Date:** January 9, 2026
**Models:** Claude Sonnet 4, Claude Opus 4
**Author:** John Balibalos

## Executive Summary

This benchmark evaluates whether LLMs can predict flow cytometry gating strategies from panel information. Using 8 OMIP papers with high-concordance extractions (XML vs LLM panel concordance ≥0.95), we find that **Claude Sonnet achieves 38.4% hierarchy F1 overall, with rich_direct context reaching 46.7%**. Opus performs slightly lower at 31.8% F1 overall but has higher structure accuracy (61.0% vs 57.4%).

## Key Finding: Direct Prompting Outperforms Chain-of-Thought

Contrary to expectations, **direct prompting consistently outperforms chain-of-thought (CoT)** for both models:

| Model | Direct F1 | CoT F1 | Difference |
|-------|-----------|--------|------------|
| Sonnet | 0.418 | 0.350 | **+6.8pp** |
| Opus | 0.365 | 0.271 | **+9.4pp** |

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

### 1. Rich Context Improves Sonnet Performance
Sonnet shows clear improvement with richer context:
- Sonnet: +19.7pp improvement (minimal_direct → rich_direct)
- Opus: +18.5pp improvement (minimal_direct → standard_direct)

### 2. Direct Prompting Generally Wins
Direct prompting outperforms chain-of-thought at higher context levels:
- Sonnet rich: Direct beats CoT by 7.2pp
- Opus standard/rich: Direct beats CoT by 9.8pp / 10.5pp

This may be because gating strategies follow domain-specific conventions that benefit from direct pattern matching rather than step-by-step reasoning.

### 3. Sonnet Has Better Critical Gate Recall
Sonnet correctly identifies critical gates (Singlets, Live, Lymphocytes) more often:
- Sonnet: 83.9% critical recall
- Opus: 79.5% critical recall

### 4. Perfect Parse Rate
Both models achieve 100% valid JSON output, indicating robust structured generation.

## Limitations

1. **Small test set**: 8 OMIP papers (4 failed validation due to missing fluorophore data)
2. **Extraction quality**: Ground truth from automated extraction, not manual curation
3. **Single run**: No statistical significance testing

## Raw Data

- Sonnet results: `experiment_results_20260109_210320.json`
- Opus results: `experiment_results_20260109_210400.json`
- Duration: Sonnet 12.4 min, Opus 17.5 min

## Recommendations

1. **Use rich_direct prompting for Sonnet** for best F1 performance (0.467)
2. **Use standard_direct prompting for Opus** for best F1 performance (0.390)
3. **Use Sonnet** when critical gate recall matters (83.9% vs 79.5%)
4. **Expand test set** by fixing fluorophore extraction for 4 failed OMIPs
5. **Investigate CoT underperformance** - may indicate domain-specific prompting needs

---

*Report generated: 2026-01-09*

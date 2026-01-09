# Flow Gating Benchmark: Gating Strategy Prediction Report

**Date:** January 9, 2026
**Models:** Claude Sonnet 4, Claude Opus 4
**Author:** John Balibalos

## Executive Summary

This benchmark evaluates whether LLMs can predict flow cytometry gating strategies from panel information. Using 8 OMIP papers with high-concordance extractions (XML vs LLM panel concordance ≥0.95), we find that **Claude Sonnet achieves 34.2% hierarchy F1 overall, with rich_direct context reaching 44.1%**. Opus performs slightly lower at 28.7% F1 overall but has better critical gate recall (77.6% vs 61.5%).

## Key Finding: Direct Prompting Outperforms Chain-of-Thought

Contrary to expectations, **direct prompting consistently outperforms chain-of-thought (CoT)** for both models:

| Model | Direct F1 | CoT F1 | Difference |
|-------|-----------|--------|------------|
| Sonnet | 0.375 | 0.307 | **+6.8pp** |
| Opus | 0.343 | 0.231 | **+11.2pp** |

## Results Summary

### Model Comparison

| Metric | Sonnet | Opus |
|--------|--------|------|
| Hierarchy F1 | **0.342** | 0.287 |
| Structure Accuracy | **0.617** | 0.572 |
| Critical Gate Recall | 0.615 | **0.776** |
| Parse Success Rate | 100% | 100% |

### Performance by Condition

#### Claude Sonnet 4

| Condition | F1 Score |
|-----------|----------|
| minimal_direct | 0.271 |
| minimal_cot | 0.254 |
| standard_direct | 0.414 |
| standard_cot | 0.313 |
| **rich_direct** | **0.441** |
| rich_cot | 0.355 |

#### Claude Opus 4

| Condition | F1 Score |
|-----------|----------|
| minimal_direct | 0.204 |
| minimal_cot | 0.184 |
| standard_direct | 0.390 |
| standard_cot | 0.241 |
| **rich_direct** | **0.434** |
| rich_cot | 0.267 |

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

### 1. Rich Context is Critical
Both models perform best with rich context (sample type, species, application, full panel):
- Sonnet: +17.0pp improvement (minimal → rich)
- Opus: +23.0pp improvement (minimal → rich)

### 2. Direct Prompting Wins
Chain-of-thought reasoning does not help with gating prediction:
- Sonnet: Direct beats CoT by 6.8pp
- Opus: Direct beats CoT by 11.2pp

This may be because gating strategies follow domain-specific conventions that benefit from direct pattern matching rather than step-by-step reasoning.

### 3. Opus Has Better Critical Gate Recall
Despite lower overall F1, Opus correctly identifies critical gates (Singlets, Live, Lymphocytes) more often:
- Opus: 77.6% critical recall
- Sonnet: 61.5% critical recall

### 4. Perfect Parse Rate
Both models achieve 100% valid JSON output, indicating robust structured generation.

## Limitations

1. **Small test set**: 8 OMIP papers (4 failed validation due to missing fluorophore data)
2. **Extraction quality**: Ground truth from automated extraction, not manual curation
3. **Single run**: No statistical significance testing

## Raw Data

- Sonnet results: `experiment_results_20260109_192747.json`
- Opus results: `experiment_results_20260109_195116.json`
- Duration: Sonnet 14.6 min, Opus 19.9 min

## Recommendations

1. **Use rich_direct prompting** for best F1 performance
2. **Use Opus for safety-critical applications** where critical gate recall matters
3. **Expand test set** by fixing fluorophore extraction for 4 failed OMIPs
4. **Investigate CoT underperformance** - may indicate domain-specific prompting needs

---

*Report generated: 2026-01-09*

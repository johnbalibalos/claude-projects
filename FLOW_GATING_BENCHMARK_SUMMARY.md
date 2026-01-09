# Flow Gating Benchmark Summary

**Project:** `projects/flow_gating_benchmark/`
**Last Updated:** January 9, 2026

## Research Question

> Can LLMs predict flow cytometry gating strategies from panel information?

## Latest Results (Jan 9, 2026)

### Model Comparison

| Model | Hierarchy F1 | Structure Acc | Critical Gate Recall |
|-------|-------------|---------------|---------------------|
| **Sonnet 4** | **0.384** | 0.574 | **0.839** |
| Opus 4 | 0.318 | **0.610** | 0.795 |

### Best Configuration

- **Sonnet:** rich_direct (F1 = 0.467)
- **Opus:** standard_direct (F1 = 0.390)

### Key Findings

1. **Context matters most**: Rich context provides ~20pp F1 improvement over minimal
2. **Direct vs CoT is nuanced**: Direct wins with rich context (+7pp), but CoT wins with minimal context (+14pp for Opus)
3. **Vocabulary determines difficulty**: Canonical names (T cells, NK cells) → high F1; specialized terms (SLAN+, TLM) → low F1
4. **Hallucination trade-off**: Rich context increases hallucination rate (7% → 16%)

## Test Set

8 OMIP papers with high-concordance panel extractions (XML vs LLM ≥0.95):

| OMIP | Focus | Panel | Avg F1 | Difficulty |
|------|-------|-------|--------|------------|
| OMIP-077 | Leukocyte populations | 14 | 0.42 | Easy (canonical names) |
| OMIP-101 | Fixed whole blood | 27 | 0.50 | Easy (standard populations) |
| OMIP-022 | γδ T-cells | 15 | 0.38 | Medium |
| OMIP-076 | Murine T/B/ASC | 19 | 0.41 | Medium |
| OMIP-074 | B-cell subsets | 19 | 0.29 | Hard (TLM, AM, RM terms) |
| OMIP-083 | Monocyte phenotyping | 21 | 0.23 | Hard (SLAN+/− terms) |
| OMIP-064 | PBMC general | 0* | 0.27 | Variable (parsing issues) |
| OMIP-095 | Spectral PBMC | 0* | 0.30 | Variable (parsing issues) |

*Panel size 0 indicates LLM-only extraction (no XML table)

## Experimental Conditions

| Factor | Levels |
|--------|--------|
| Context | minimal, standard, rich |
| Prompting | direct, chain-of-thought |
| Models | Sonnet 4, Opus 4 |

**Total evaluations:** 96 (8 × 6 × 2)

## Quick Start

```bash
cd projects/flow_gating_benchmark

# Run experiment
python scripts/run_experiment.py --model sonnet --force

# View results
cat results/REPORT.md
```

## Files

- **Full Report:** `projects/flow_gating_benchmark/results/REPORT.md`
- **Raw Results:** `projects/flow_gating_benchmark/results/experiment_results_*.json`
- **Ground Truth:** `projects/flow_gating_benchmark/data/ground_truth/`
- **Extraction Pipeline:** `projects/flow_gating_benchmark/src/curation/`

## Recommendations

1. Use **rich context** - ~20pp improvement over minimal
2. Use **direct prompting** with rich context (best F1)
3. Expect **~15% hallucination rate** - verify against panel
4. Performance depends on **vocabulary familiarity** - canonical names work best

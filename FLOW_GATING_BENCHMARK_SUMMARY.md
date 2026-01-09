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

### Key Finding

**Direct prompting outperforms chain-of-thought** for gating prediction:
- Sonnet rich: Direct +7.2pp over CoT
- Opus standard: Direct +9.8pp over CoT

## Test Set

8 OMIP papers with high-concordance panel extractions (XML vs LLM ≥0.95):

| OMIP | Sample Type | Complexity |
|------|-------------|------------|
| OMIP-022 | Human blood | γδ T-cells |
| OMIP-064 | Human PBMC | General |
| OMIP-074 | Human PBMC | B-cells |
| OMIP-076 | Mouse tissue | T/B/ASC |
| OMIP-077 | Human PBMC | DCs |
| OMIP-083 | Human PBMC | 28-color |
| OMIP-095 | Human PBMC | Spectral |
| OMIP-101 | Fixed blood | 27-color |

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

1. Use **Sonnet + rich_direct** for best F1 (0.467)
2. Use **Sonnet** when critical gate recall matters (83.9%)
3. Avoid chain-of-thought for this domain task

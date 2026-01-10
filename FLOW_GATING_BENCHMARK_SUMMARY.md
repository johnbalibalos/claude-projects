# Flow Gating Benchmark Summary

**Project:** `projects/flow_gating_benchmark/`
**Last Updated:** January 10, 2026

## Research Question

> Can LLMs predict flow cytometry gating strategies from panel information?

## Latest Results (Jan 2026)

### Model Comparison

| Model | Hierarchy F1 | Structure Acc | Critical Gate Recall | Task Failure Rate |
|-------|-------------|---------------|---------------------|-------------------|
| **Sonnet 4** | **0.384** | 0.574 | **0.839** | ~5% |
| Opus 4 | 0.318 | **0.610** | 0.795 | ~8% |

### Best Configuration

- **Sonnet:** rich_direct (F1 = 0.467)
- **Opus:** standard_direct (F1 = 0.390)

### Key Findings

1. **Context matters most**: Rich context provides ~20pp F1 improvement over minimal
2. **Direct vs CoT is nuanced**: Direct wins with rich context (+7pp), but CoT wins with minimal context
3. **Vocabulary determines difficulty**: Canonical names (T cells, NK cells) → high F1; specialized terms (SLAN+, TLM) → low F1
4. **Task failures are detectable**: Meta-questions and refusals can be automatically identified

## Architecture

```
src/
├── curation/           # Data extraction
│   ├── schemas.py      # TestCase, Panel, GatingHierarchy models
│   ├── paper_parser.py # XML/PDF content extraction
│   ├── marker_logic.py # Marker table → hierarchy conversion
│   └── omip_extractor.py
├── evaluation/         # Scoring
│   ├── metrics.py      # F1, structure, hallucination metrics
│   ├── normalization.py # Gate name normalization & synonyms
│   ├── hierarchy.py    # Tree operations (extract, depth, paths)
│   ├── task_failure.py # Meta-question/refusal detection
│   ├── response_parser.py # LLM output parsing
│   └── scorer.py       # Combined scoring interface
├── experiments/        # Runner
│   ├── runner.py       # ExperimentRunner with checkpointing
│   ├── llm_client.py   # Unified Anthropic/OpenAI/Ollama client
│   ├── conditions.py   # Experimental conditions
│   └── prompts.py      # Prompt templates
└── analysis/           # Reports
    ├── manual_review_report.py  # Side-by-side comparisons
    └── visualization.py
```

## Test Set

8 OMIP papers with validated ground truth:

| OMIP | Focus | Panel | Avg F1 | Difficulty |
|------|-------|-------|--------|------------|
| OMIP-077 | Leukocyte populations | 14 | 0.42 | Easy |
| OMIP-101 | Fixed whole blood | 27 | 0.50 | Easy |
| OMIP-022 | gamma-delta T-cells | 15 | 0.38 | Medium |
| OMIP-076 | Murine T/B/ASC | 19 | 0.41 | Medium |
| OMIP-074 | B-cell subsets | 19 | 0.29 | Hard |
| OMIP-083 | Monocyte phenotyping | 21 | 0.23 | Hard |
| OMIP-064 | PBMC general | - | 0.27 | Variable |
| OMIP-095 | Spectral PBMC | - | 0.30 | Variable |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Hierarchy F1** | Precision/recall on gate names with fuzzy matching |
| **Structure Accuracy** | % of parent-child relationships correct |
| **Critical Gate Recall** | % of must-have gates (live, singlets) present |
| **Hallucination Rate** | Gates referencing markers not in panel |
| **Task Failure Rate** | Meta-questions, refusals, or instructions instead of predictions |

## Task Failure Detection

The benchmark now detects when models fail to complete the task:

| Failure Type | Example |
|--------------|---------|
| META_QUESTIONS | "What markers are you using?" |
| REFUSAL | "I cannot predict without more information" |
| INSTRUCTIONS | "Here's how you would create a hierarchy..." |
| EMPTY | No response content |

Task failures are tracked separately from scoring failures and included in reports.

## Quick Start

```bash
cd projects/flow_gating_benchmark

# Run experiment with Sonnet
PYTHONPATH=src python scripts/run_experiment.py --model sonnet

# Generate manual review report
PYTHONPATH=src python scripts/generate_review_report.py results/experiment_results_*.json --level outliers

# View results
cat results/experiment_summary_*.txt
```

## Key Files

- **Results:** `results/experiment_results_*.json`
- **Reports:** `results/manual_review_*.md`
- **Ground Truth:** `data/ground_truth/*.json`
- **Test Cases:** 8 OMIP papers with curated hierarchies

## Recent Changes

### January 2026
- **Task failure detection** - Identifies meta-questions, refusals, instructions
- **Manual review reports** - Side-by-side comparison with outlier flagging
- **Refactored evaluation** - Split into normalization.py, hierarchy.py, task_failure.py
- **LLM client abstraction** - Unified interface for Anthropic, OpenAI, Ollama
- **Improved response parser** - Better JSON extraction, meta-term detection

## Recommendations

1. Use **rich context** - ~20pp improvement over minimal
2. Use **direct prompting** with rich context (best F1)
3. Monitor **task failure rate** - high rate indicates prompt issues
4. Expect **~15% hallucination rate** - verify gates against panel
5. Performance depends on **vocabulary familiarity** - canonical names work best

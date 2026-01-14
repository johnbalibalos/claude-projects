# Flow Gating Benchmark

Evaluate LLM capabilities in predicting flow cytometry gating strategies from panel information.

> **Status:** Active development. The evaluation framework is functional, but ground truth datasets require manual curation from OMIP papers. Results in any linked posts reflect work-in-progress and may be rerun as data quality improves.

## Latest Results

> **⚠️ Results pending:** Clean rerun in progress on verified dataset with multiple F1 metrics.

See **[results/BENCHMARK_RESULTS_SUMMARY.md](results/BENCHMARK_RESULTS_SUMMARY.md)** for full analysis.

### F1 Comparison (placeholder - rerun pending)

| Model | hierarchy_f1 | synonym_f1 | semantic_f1 | weighted_semantic_f1 |
|-------|--------------|------------|-------------|----------------------|
| gemini-2.0-flash | TBD | TBD | TBD | TBD |
| claude-sonnet-4 | TBD | TBD | TBD | TBD |
| claude-opus-4 | TBD | TBD | TBD | TBD |

### Other Metrics (placeholder)

| Model | Structure Acc | Critical Recall | Parse Rate |
|-------|---------------|-----------------|------------|
| gemini-2.0-flash | TBD | TBD | TBD |
| claude-sonnet-4 | TBD | TBD | TBD |
| claude-opus-4 | TBD | TBD | TBD |

**Key Questions Being Tested:**
- Does semantic_f1 >> hierarchy_f1? (biological equivalence gap)
- Which F1 metric best correlates with LLM judge scores?
- Does the Sonnet-Opus gap shrink with semantic matching?

**Methodology Note:** In earlier testing with n=10 bootstrap runs, Claude models produced ~10 unique responses per 10 runs even at temperature=0. This non-determinism finding doesn't depend on ground truth quality—it measures output consistency, not correctness.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Dry run (mock API calls)
python scripts/run_modular_pipeline.py --phase all --dry-run

# Real run with Gemini (recommended)
python scripts/run_modular_pipeline.py \
    --phase all \
    --models gemini-2.0-flash \
    --test-cases data/staging \
    --n-bootstrap 10
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MODULAR PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Prediction  │───▶│    Batch     │───▶│  LLM Judge   │          │
│  │  Collector   │    │   Scorer     │    │  (optional)  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│        │                    │                   │                   │
│        ▼                    ▼                   ▼                   │
│  predictions.json    scoring_results.json  judge_results.json      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Recent Additions

- **Multi-Judge Cross-Validation**: Multiple prompt styles for LLM judge reliability
- **Token Breakdown Analysis**: Track thinking vs response tokens for reasoning models
- **Blocked Prediction Recovery**: `scripts/rerun_blocked.py` for MAX_TOKENS failures
- **Per-Provider Rate Limits**: Parallel workers configurable per API provider

---

## CLI Reference

```bash
python scripts/run_modular_pipeline.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--phase` | `all` | `predict`, `score`, `judge`, or `all` |
| `--models` | `claude-sonnet-cli` | Models to test (space-separated) |
| `--test-cases` | `data/verified` | Test case JSON directory |
| `--n-bootstrap` | `1` | Runs per condition (use 10 for variance) |
| `--max-cases` | None | Limit test cases (for testing) |
| `--output` | `results/modular_pipeline` | Output directory |
| `--judge-model` | `gemini-2.5-pro` | Model for qualitative judge |
| `--dry-run` | False | Mock API calls |
| `--resume` | False | Resume from checkpoint |
| `--force` | False | Skip cost confirmation |

### Available Models

| Model | Provider | Cost/Case | Notes |
|-------|----------|-----------|-------|
| `gemini-2.0-flash` | Google | ~$0.01 | **Recommended** - best F1 |
| `gemini-2.5-flash` | Google | ~$0.02 | High failure rate |
| `gemini-2.5-pro` | Google | ~$0.10 | Needs max_tokens=20000 |
| `claude-sonnet-cli` | Anthropic | ~$0.05 | Uses Max subscription |
| `claude-opus-cli` | Anthropic | ~$0.50 | Uses Max subscription |
| `gpt-4o` | OpenAI | ~$0.05 | API |

---

## Experimental Conditions

Each model is tested with 4 conditions (2 context levels × 2 strategies):

### Context Levels

**Minimal** - Markers only:
```
## Panel
Markers: CD3, CD4, CD8, CD45, CD19, Viability
```

**Standard** - Adds experimental metadata (+25% F1 improvement):
```
## Experiment Information
Sample Type: Human PBMC
Species: human
Application: T-cell subset analysis

## Panel
Markers: CD3, CD4, CD8, CD45, CD19, Viability
```

### Prompt Strategies

- **direct**: Output JSON hierarchy immediately
- **cot**: Chain-of-thought reasoning before JSON output

---

## Known Failure Modes

### 1. Token Exhaustion (gemini-2.5-*)
Reasoning models use 70%+ of token budget for "thinking", leaving insufficient output tokens.
- **Symptom**: Truncated JSON, empty responses
- **Fix**: Use `max_tokens=20000` or `scripts/rerun_blocked.py`

### 2. Format Confusion (gemini-2.5-flash)
Model outputs prose descriptions instead of JSON hierarchy.
- **Symptom**: F1 = 0, valid text but unparseable
- **Fix**: None effective - use different model

### 3. Synonym Mismatches (all models)
Ground truth says "CD3+ T cells", model predicts "T lymphocytes".
- **Symptom**: Low F1 despite correct biology
- **Mitigation**: 200+ synonyms in `normalization.py`

### 4. Hallucinated Markers
Model creates gates using markers not in the panel.
- **Symptom**: High hallucination_rate metric
- **Cause**: Training data contamination or confusion

---

## Project Structure

```
flow_gating_benchmark/
├── src/
│   ├── curation/              # Test case schemas
│   │   ├── schemas.py         # TestCase, Panel, GatingHierarchy
│   │   └── omip_extractor.py
│   ├── evaluation/            # Scoring
│   │   ├── metrics.py         # F1, structure, hallucination
│   │   ├── normalization.py   # 200+ gate synonyms
│   │   ├── hierarchy.py       # Tree operations
│   │   ├── task_failure.py    # Refusal detection
│   │   ├── response_parser.py
│   │   └── scorer.py
│   ├── experiments/           # Pipeline
│   │   ├── prediction_collector.py
│   │   ├── batch_scorer.py
│   │   ├── llm_judge.py       # Multi-judge support
│   │   ├── llm_client.py      # Gemini, Claude, OpenAI
│   │   ├── conditions.py
│   │   └── prompts.py
│   └── analysis/              # Hypothesis testing
│       ├── alien_cell.py      # Frequency confound tests
│       └── cognitive_refusal.py
├── data/
│   ├── staging/               # 13 OMIPs (test cases)
│   └── verified/              # Verified test cases
├── scripts/
│   ├── run_modular_pipeline.py
│   └── rerun_blocked.py       # Recover MAX_TOKENS failures
├── results/
│   ├── BENCHMARK_RESULTS_SUMMARY.md  # Latest results
│   └── gemini_benchmark_predictions.json
└── tests/
```

---

## Metrics

### F1 Variants (4 metrics)

We compute multiple F1 scores to understand the gap between string matching and biological equivalence:

| Metric | Method | What it catches |
|--------|--------|-----------------|
| `hierarchy_f1` | String normalization | "CD4+ T Cells" ↔ "CD4 positive T cells" |
| `synonym_f1` | 200+ synonym dictionary | "T Lymphocytes" ↔ "T cells" ↔ "CD3+" |
| `semantic_f1` | MiniLM embeddings | "Helper T cells" ↔ "CD4+ T cells" |
| `weighted_semantic_f1` | Confidence-weighted | Partial credit for similarity 0.7-0.85 |

**Why multiple F1s?** Early analysis showed weak correlation (r≈0.15) between string-based F1 and LLM judge scores. Models often produce biologically correct but linguistically different gate names.

### Other Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `structure_accuracy` | Parent-child relationships correct | 0-1 |
| `critical_gate_recall` | Essential QC gates present | 0-1 |
| `hallucination_rate` | Gates using non-panel markers | 0-1 |

---

## Test Cases (13 OMIPs)

| OMIP | Species | Focus | Markers | Difficulty |
|------|---------|-------|---------|------------|
| 008 | Human | Th1/Th2 | 7 | Easy |
| 053 | Human | Tregs | 7 | Easy |
| 022 | Human | T memory | 12 | Medium |
| 025 | Human | T/NK | 14 | Medium |
| 035 | Macaque | NK | 14 | Medium |
| 074 | Human | B cells | 22 | Medium |
| 076 | Murine | T/B/APC | 18 | Hard |
| 077 | Human | Pan-leuk | 20 | Hard |
| 083 | Human | Monocytes | 21 | Hard |
| 064 | Human | NK/ILC | 27 | Hard |
| 087 | Human | CyTOF | 32 | Very Hard |
| 095 | Human | Spectral | 40 | Hard |
| 101 | Human | Whole blood | 32 | Hard |

---

## Environment

```bash
# Required
GOOGLE_API_KEY=...     # Gemini models + judge

# Optional (CLI models use Max subscription)
ANTHROPIC_API_KEY=...  # For API models
OPENAI_API_KEY=...     # GPT models
```

---

## Development

```bash
# Run tests
pytest tests/ -v

# Quick API test (~$0.01)
python scripts/run_modular_pipeline.py \
    --phase all \
    --models gemini-2.0-flash \
    --max-cases 1
```

---

## License

MIT

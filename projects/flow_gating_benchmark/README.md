# Flow Gating Benchmark

Evaluate LLM capabilities in predicting flow cytometry gating strategies from panel information.

## Latest Results

See **[results/BENCHMARK_RESULTS_SUMMARY.md](results/BENCHMARK_RESULTS_SUMMARY.md)** for full analysis.

| Model | Hierarchy F1 | Structure Acc | Critical Recall | Parse Rate |
|-------|--------------|---------------|-----------------|------------|
| **gemini-2.0-flash** | **0.393** | 0.268 | 0.891 | 100% |
| claude-opus-4 | 0.349 | 0.229 | **0.856** | 100% |
| claude-sonnet-4 | 0.325 | 0.204 | 0.791 | 99.2% |
| gemini-2.5-pro | 0.240 | 0.092 | 0.631 | 99.5% |
| gemini-2.5-flash | 0.119 | 0.078 | 0.489 | 31% |

**Key Findings:**
- Simpler models (gemini-2.0-flash) outperform reasoning models
- Claude models have high critical gate recall (important QC gates)
- Standard context improves F1 by 25%+

**Methodology Note:** In earlier testing with n=10 bootstrap runs, Claude models produced ~10 unique responses per 10 runs even at temperature=0. This non-determinism finding doesn't depend on ground truth quality—it measures output consistency, not correctness. The F1 results above are from a subsequent n=1 run on manually-curated ground truth.

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
    --n-bootstrap 10 \
    --force
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
| `--test-cases` | `data/ground_truth` | Test case JSON directory |
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
│   └── ground_truth/          # Verified (move from staging)
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

| Metric | Description | Range |
|--------|-------------|-------|
| `hierarchy_f1` | Gate name precision/recall (normalized) | 0-1 |
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
    --max-cases 1 \
    --force
```

---

## License

MIT

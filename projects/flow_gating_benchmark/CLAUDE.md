# Flow Gating Benchmark

Evaluate whether LLMs can predict flow cytometry gating strategies from panel information and experimental context.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         END-TO-END PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Test Cases  │───▶│   Prompts    │───▶│  LLM Calls   │          │
│  │  (JSON)      │    │  (templates) │    │  (parallel)  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                                       │                   │
│         │                                       ▼                   │
│         │            ┌──────────────┐    ┌──────────────┐          │
│         │            │   Scoring    │◀───│  Predictions │          │
│         └───────────▶│  (metrics)   │    │  (raw text)  │          │
│                      └──────────────┘    └──────────────┘          │
│                             │                                       │
│                             ▼                                       │
│                      ┌──────────────┐    ┌──────────────┐          │
│                      │  LLM Judge   │───▶│   Reports    │          │
│                      │  (Gemini)    │    │  (analysis)  │          │
│                      └──────────────┘    └──────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Core Modules

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `curation/` | Ground truth data | `TestCase`, `GatingHierarchy`, `Panel` |
| `experiments/` | Run LLM experiments | `PredictionCollector`, `ExperimentCondition` |
| `evaluation/` | Score predictions | `BatchScorer`, `GatingScorer`, `EvaluationResult` |
| `analysis/` | Generate reports | `LLMJudge`, visualization functions |

### Data Flow

1. **Test Cases** (`data/ground_truth/*.json`) → Pydantic models
2. **Conditions** (model × context × strategy) → Prompt templates
3. **LLM Calls** (CLI rate-limited, API parallel) → Raw predictions
4. **Scoring** (F1, structure, critical gates) → Metrics
5. **Judge** (Gemini 2.5 Pro qualitative) → Final report

---

## Quick Start

### Run Full Benchmark

```bash
# Cost estimate only
python scripts/run_full_benchmark.py --estimate

# Dry run (mock API calls)
python scripts/run_full_benchmark.py --dry-run --n-bootstrap 1 -y

# Full run with 3 bootstrap iterations
python scripts/run_full_benchmark.py --n-bootstrap 3 -y

# Resume from checkpoint
python scripts/run_full_benchmark.py --resume --n-bootstrap 3 -y
```

### Run Modular Pipeline (Recommended for Development)

```bash
# Phase 1: Collect predictions only
python scripts/run_modular_pipeline.py --phase predict --dry-run

# Phase 2: Score predictions
python scripts/run_modular_pipeline.py --phase score

# Phase 3: Run LLM judge
python scripts/run_modular_pipeline.py --phase judge

# All phases
python scripts/run_modular_pipeline.py --phase all --models claude-sonnet-cli --n-bootstrap 1
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific modules
pytest tests/test_hierarchy.py tests/test_task_failure.py -v
```

---

## Project Structure

```
flow_gating_benchmark/
├── src/
│   ├── curation/                    # Data layer
│   │   ├── schemas.py               # Pydantic models (TestCase, Panel, GateNode)
│   │   └── omip_extractor.py        # Load test cases from JSON
│   │
│   ├── experiments/                 # Experiment execution
│   │   ├── conditions.py            # ExperimentCondition, model registry
│   │   ├── prompts.py               # Prompt templates (direct, CoT)
│   │   ├── llm_client.py            # LLM clients (Anthropic, Gemini, OpenAI)
│   │   ├── prediction_collector.py  # Collect raw predictions (modular)
│   │   ├── batch_scorer.py          # Score predictions (modular)
│   │   ├── llm_judge.py             # LLM-based qualitative judge
│   │   └── runner.py                # Legacy experiment runner
│   │
│   ├── evaluation/                  # Scoring and metrics
│   │   ├── metrics.py               # F1, structure accuracy, critical recall
│   │   ├── scorer.py                # GatingScorer main class
│   │   ├── response_parser.py       # Parse LLM responses to hierarchy
│   │   ├── hierarchy.py             # Tree operations (extract, traverse)
│   │   ├── normalization.py         # Gate name normalization
│   │   └── task_failure.py          # Detect non-responses
│   │
│   └── analysis/                    # Results analysis
│       ├── report_generator.py
│       └── visualization.py
│
├── scripts/                         # Entry points
│   ├── run_full_benchmark.py        # Main benchmark (concurrent CLI+API)
│   ├── run_modular_pipeline.py      # Decoupled phases
│   └── test_judge_mock.py           # Test LLM judge
│
├── tests/                           # Test suite
│   ├── test_hierarchy.py            # 26 tests
│   ├── test_task_failure.py         # 18 tests
│   ├── test_scorer.py
│   └── conftest.py                  # Shared fixtures
│
├── data/
│   └── ground_truth/                # OMIP test cases (JSON)
│
└── results/                         # Output directory
    ├── full_benchmark/              # Checkpoints and results
    └── modular_pipeline/            # Modular pipeline output
```

---

## Key Concepts

### Gating Hierarchy

Tree structure of sequential cell filters:

```
All Events
└── Singlets (FSC-A vs FSC-H)
    └── Live cells (Zombie NIR-)
        └── CD45+ leukocytes
            ├── CD3+ T cells
            │   ├── CD4+ helper T
            │   └── CD8+ cytotoxic T
            └── CD19+ B cells
```

### Evaluation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `hierarchy_f1` | Gate name precision/recall | 0-1 |
| `structure_accuracy` | Parent-child relationships correct | 0-1 |
| `critical_gate_recall` | Must-have gates present | 0-1 |
| `hallucination_rate` | Gates not in panel | 0-1 |

### Experimental Conditions

Generated from cartesian product:

```python
models = ["claude-sonnet-cli", "claude-opus-cli", "gemini-2.5-pro"]
context_levels = ["minimal", "standard", "rich"]
prompt_strategies = ["direct", "cot"]
# → 18 conditions
```

### Execution Model

- **CLI models** (claude-*-cli): Sequential, 2s rate limit, uses Max subscription
- **API models** (gemini-*, gpt-*): Parallel, 5 workers
- **Concurrent**: CLI and API run simultaneously

---

## Configuration

### Environment Variables

```bash
# Required for Gemini models
GOOGLE_API_KEY=...

# Optional (CLI models use Max subscription)
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

### Model Registry

Defined in `src/experiments/conditions.py`:

```python
MODELS = {
    "claude-sonnet-cli": "claude-sonnet-4-20250514-cli",
    "claude-opus-cli": "claude-opus-4-20250514-cli",
    "gemini-2.5-pro": "gemini-2.5-pro",
    # ... more models
}
```

---

## Debugging

### Quick Validation

```bash
# Single test case, dry run
python scripts/run_modular_pipeline.py \
    --phase all \
    --models claude-sonnet-cli \
    --n-bootstrap 1 \
    --dry-run

# Check checkpoint
cat results/modular_pipeline/predictions.json | python -m json.tool | head -50
```

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| F1 = 0.0 | Parse failure | Check `response_parser.py` patterns |
| All structure = 0 | Wrong parent mapping | Verify `extract_parent_map()` |
| Pydantic errors | Missing fields in test case | Make fields optional in schema |
| Rate limit errors | Too fast CLI calls | Increase `cli_delay_seconds` |

### Inspect Results

```python
import json
from pathlib import Path

# Load scoring results
with open("results/modular_pipeline/scoring_results.json") as f:
    data = json.load(f)

# Summary stats
print(f"Mean F1: {data['stats']['overall']['hierarchy_f1']['mean']:.3f}")

# By model
for model, stats in data['stats']['by_model'].items():
    print(f"{model}: {stats['hierarchy_f1']['mean']:.3f}")
```

---

## Adding Test Cases

1. Create `data/ground_truth/omip_XXX.json`
2. Follow schema in `src/curation/schemas.py`
3. Validate: `python -c "from curation.omip_extractor import load_all_test_cases; print(len(load_all_test_cases('data/ground_truth')))"`

### Test Case Schema

```json
{
  "test_case_id": "OMIP-069",
  "source_type": "omip_paper",
  "omip_id": "OMIP-069",
  "context": {
    "sample_type": "Human PBMC",
    "species": "human",
    "application": "Deep immunophenotyping"
  },
  "panel": {
    "entries": [
      {"marker": "CD3", "fluorophore": "BUV395", "clone": "UCHT1"}
    ]
  },
  "gating_hierarchy": {
    "root": {
      "name": "All Events",
      "children": [...]
    }
  },
  "metadata": {
    "curation_date": "2025-01-01",
    "curator": "Name"
  }
}
```

---

## Known Technical Debt

See `docs/UTILITIES.md` for detailed analysis. Key issues:

1. **Duplicate model registries** in 3 locations
2. **Duplicate serialization** (to_dict/from_dict) across 6 dataclasses
3. **Duplicate checkpoint logic** in 4 modules
4. **Multiple normalization functions** with different behavior

---

## Related Resources

- HIPC Gating Standards: https://www.nature.com/articles/srep20686
- OMIP Papers: Cytometry Part A journal
- FlowRepository: https://flowrepository.org

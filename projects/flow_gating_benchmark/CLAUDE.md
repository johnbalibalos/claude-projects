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

The modular pipeline runs in decoupled phases, enabling checkpointing and re-running individual steps.

```bash
# Full run on staging data with Gemini
python scripts/run_modular_pipeline.py \
    --phase all \
    --models gemini-2.0-flash \
    --test-cases data/staging \
    --n-bootstrap 1 \
    --force

# Run phases independently
python scripts/run_modular_pipeline.py --phase predict --dry-run  # Collect predictions
python scripts/run_modular_pipeline.py --phase score              # Score cached predictions
python scripts/run_modular_pipeline.py --phase judge              # Run LLM judge
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--phase` | `all` | Which phase to run: `predict`, `score`, `judge`, or `all` |
| `--models` | `claude-sonnet-cli` | Models to test (space-separated). Options: `gemini-2.0-flash`, `gemini-2.5-pro`, `claude-sonnet-cli`, `claude-opus-cli`, `gpt-4o` |
| `--test-cases` | `data/ground_truth` | Directory with test case JSON files |
| `--n-bootstrap` | `1` | Number of runs per condition (for variance estimation) |
| `--max-cases` | None | Limit number of test cases (for quick testing) |
| `--dry-run` | False | Use mock LLM client (no API calls, no cost) |
| `--force` | False | Skip cost confirmation hook |
| `--resume` | False | Resume from checkpoint |
| `--output` | `results/modular_pipeline` | Output directory |

#### Conditions Generated

Each model runs with 4 conditions (cartesian product):
- **Context levels:** `minimal` (markers only), `standard` (+ sample type, species, application)
- **Prompt strategies:** `direct` (output JSON), `cot` (chain-of-thought reasoning first)

Total calls = `n_models × n_test_cases × 4 conditions × n_bootstrap`

#### Cost Estimation

```
gemini-2.0-flash:  ~$0.01 per test case (4 conditions)
gemini-2.5-pro:    ~$0.10 per test case (used for judge)
claude-sonnet:     ~$0.05 per test case
claude-opus:       ~$0.50 per test case
```

Example: 13 test cases × gemini-2.0-flash ≈ $0.15 + judge ($1.30) ≈ **$1.50 total**

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

## Ground Truth OMIPs

Ground truth folder is currently **empty** - all OMIPs pending manual verification.

### Staging (`data/staging/`) - 13 OMIPs pending verification

| OMIP | Species | Focus | Technology | Description | Hierarchy |
|------|---------|-------|------------|-------------|-----------|
| 008 | Human | T cells | Flow | Th1/Th2 cytokine polyfunctionality | 7 gates |
| 022 | Human | T cells | Flow | Antigen-specific T-cell functionality and memory | 12 gates |
| 025 | Human | T/NK | Flow | T- and NK-cell responses including memory and Tfh | 14 gates |
| 035 | Macaque | NK | Flow | Functional analysis of NK cell subsets | 14 gates |
| 053 | Human | Tregs | Flow | Identification/classification of FoxP3+ populations | 7 gates |
| 064 | Human | NK/ILC | Flow | 27-color NK cells, ILCs, MAIT, γδ T cells | **empty** |
| 074 | Human | B cells | Flow | IgG and IgA subclass phenotyping | 22 gates |
| 076 | Murine | Pan | Flow | High-dimensional T-cell, B-cell, and APC phenotyping | 18 gates |
| 077 | Human | Pan | Flow | All principal leukocyte populations (broad panel) | 20 gates |
| 083 | Human | Pan | Flow | 21-marker 18-color in-depth phenotyping | 10 gates |
| 087 | Human | T cells | CyTOF | 32-parameter mass cytometry for CD4/CD8 memory subsets | 24 gates |
| 095 | Human | Pan | Flow | 40-color spectral - all major leukocyte populations | 17 gates |
| 101 | Human | Pan | Flow | 27-color whole blood leukocyte immunophenotyping | 32 gates |

To activate an OMIP for benchmarking, manually verify the hierarchy and move to `data/ground_truth/`.

**Technology handling**: The evaluation metrics automatically detect mass cytometry panels (metal isotope labels like 145Nd, 176Yb) and skip scatter-based critical gates (singlets) that don't apply to CyTOF.

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

### Quick Testing (Real API, Minimal Cost)

For testing bug fixes or code changes, use the cheapest model with minimal cases:

```bash
# Real API test (~$0.01) - validates prompts and parsing work
python scripts/run_modular_pipeline.py \
    --phase all \
    --models gemini-2.0-flash \
    --test-cases data/staging \
    --n-bootstrap 1 \
    --max-cases 1 \
    --force
```

Requires `GOOGLE_API_KEY`. Use `--dry-run` instead of `--force` to test pipeline mechanics without API calls.

### Quick Validation (Dry Run)

```bash
# Mock API calls - tests pipeline mechanics only
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

1. **Duplicate model registries** in 2 locations (conditions.py:MODELS, llm_client.py:MODEL_REGISTRY)
   - conditions.py is authoritative for experiments
   - llm_client.py provides shorthand resolution
2. **Duplicate serialization** (to_dict/from_dict) across 6+ dataclasses
3. **Duplicate checkpoint logic** - CheckpointManager in utils/checkpoint.py, provenance in utils/provenance.py
4. **Multiple normalization functions** with different behavior
5. **Dead code candidates** (verify before removing):
   - `src/experiments/experiment_conditions.py` - Legacy A/B testing conditions (381 lines)
   - `src/experiments/runner.py` - Legacy runner, replaced by modular pipeline

---

## Known Model Limitations

### Token Usage in Reasoning Models

**Issue:** High-reasoning models (gemini-2.5-pro, claude-opus, o1) use significantly more tokens for "thinking" before generating output. This can cause `MAX_TOKENS` failures on complex panels even when the actual response fits within limits.

**Observed:** In initial benchmarks, gemini-2.5-pro hit token limits on 12% of predictions (61/520), specifically on complex panels:
- OMIP-064: 27-color NK panel
- OMIP-083: 21-marker panel
- OMIP-095: 40-color spectral panel

Meanwhile, gemini-2.0-flash and gemini-2.5-flash completed 100% of predictions successfully.

**Current settings:** `max_tokens=6000` in `CollectorConfig`

**Recommendations:**
1. Increase `max_tokens` to 10000+ for reasoning models on complex panels
2. Track token breakdown (thinking vs response) when APIs support it
3. Consider model-specific token limits in `CollectorConfig`

**TODO:** Add per-model token configuration. The Gemini API may eventually expose thinking token counts separately - monitor for this capability to better understand token allocation.

---

## Hypothesis Testing Framework

The project includes a rigorous hypothesis testing framework to distinguish between different failure modes:

### Available Tests

| Test | Question Answered | Key Metric |
|------|-------------------|------------|
| **Frequency Confound** | Is failure due to token frequency or reasoning? | R² correlation |
| **Alien Cell** | Does model reason from markers or memorize population names? | F1 delta |
| **Format Ablation** | Is failure due to prose parsing or reasoning? | Format variance |
| **CoT Mechanistic** | Does CoT cause prior hallucinations? | Hallucination rate |
| **Cognitive Refusal** | Is model blind to context or over-cautious? | Forcing effect |

### Running Hypothesis Tests

```bash
# Analyze existing benchmark results
python run_hypothesis_tests.py --results results/benchmark_results.json

# Run specific tests
python run_hypothesis_tests.py --tests frequency_confound alien_cell

# Estimate cost before running live tests
python run_hypothesis_tests.py --estimate-cost --tests alien_cell format_ablation

# Run tests
pytest tests/test_hypothesis_tests.py -v
```

### Key Findings Template

When presenting results, use this framework:

> "Is this a failure of reasoning or just a lack of training data frequency?
>
> To rule out the frequency confound, I designed an 'Alien Cell' ablation: I provide
> the model with valid marker logic (e.g., CD3+ CD4+) but label the target population
> with a nonsense word like 'Glorp Cells'.
>
> If the model relies on retrieval, it will fail to gate 'Glorp Cells' because it has
> no prior association with that token. If it relies on first-principles logic, it
> should solve it perfectly."

### Hypothesis Test Interpretation

| R² | Interpretation |
|----|----------------|
| > 0.8 | Frequency explains performance (memorization) |
| 0.5-0.8 | Mixed - both frequency and reasoning |
| < 0.5 | Reasoning deficit (not just frequency) |

| Alien Cell Delta | Interpretation |
|------------------|----------------|
| < 0.05 | Model reasons from markers |
| 0.05-0.20 | Mixed evidence |
| > 0.20 | Model relies on population name tokens |

## Related Projects

- **flow_panel_optimizer**: Tests spectral calculations (complements this project)
- Uses shared `libs/mcp_tester` for ablation framework
- Uses shared `libs/hypothesis_pipeline` for statistical analysis

---

## Related Resources

- HIPC Gating Standards: https://www.nature.com/articles/srep20686
- OMIP Papers: Cytometry Part A journal
- FlowRepository: https://flowrepository.org

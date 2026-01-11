# Flow Gating Benchmark

An evaluation framework for testing LLM capabilities in predicting flow cytometry gating strategies from panel information and experimental context.

## Research Question

> Can LLMs predict flow cytometry gating hierarchies given marker panels?

## Latest Results

| Model | Hierarchy F1 | Structure Acc | Critical Recall | Task Failure |
|-------|-------------|---------------|-----------------|--------------|
| **Sonnet 4** | **0.384** | 0.574 | **0.839** | ~5% |
| Opus 4 | 0.318 | **0.610** | 0.795 | ~8% |

**Key Finding:** Rich context with direct prompting yields best results (F1=0.467).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiment
PYTHONPATH=src python scripts/run_experiment.py --model sonnet --dry-run

# Generate manual review report
PYTHONPATH=src python scripts/generate_review_report.py results/experiment_results_*.json --level outliers
```

## Project Structure

```
flow_gating_benchmark/
├── src/
│   ├── curation/           # Data extraction from OMIP papers
│   │   ├── schemas.py      # TestCase, Panel, GatingHierarchy models
│   │   ├── paper_parser.py # XML/PDF content extraction
│   │   ├── marker_logic.py # Marker table → hierarchy conversion
│   │   ├── omip_extractor.py
│   │   └── auto_extractor.py
│   ├── evaluation/         # Scoring and metrics
│   │   ├── metrics.py      # F1, structure, hallucination metrics
│   │   ├── normalization.py # Gate name normalization (200+ synonyms)
│   │   ├── hierarchy.py    # Tree operations (extract, depth, paths)
│   │   ├── task_failure.py # Meta-question/refusal detection
│   │   ├── response_parser.py # LLM output parsing (JSON, markdown, indented)
│   │   └── scorer.py       # Combined scoring interface
│   ├── experiments/        # Experiment runner
│   │   ├── runner.py       # ExperimentRunner with checkpointing
│   │   ├── llm_client.py   # Unified Anthropic/OpenAI/Ollama client
│   │   ├── conditions.py   # Experimental conditions
│   │   └── prompts.py      # Prompt templates
│   └── analysis/           # Results analysis
│       ├── manual_review_report.py  # Side-by-side comparisons
│       ├── failure_analysis.py
│       └── visualization.py
├── data/
│   ├── ground_truth/       # 8 curated OMIP test cases
│   ├── raw/                # Downloaded papers and .wsp files
│   └── papers/             # OMIP paper PDFs/XMLs
├── scripts/
│   ├── run_experiment.py   # Main experiment CLI
│   └── generate_review_report.py  # Report generation CLI
├── results/                # Experiment outputs
└── tests/                  # Unit tests
```

## Core Modules

### Evaluation (`src/evaluation/`)

| Module | Purpose |
|--------|---------|
| `metrics.py` | Hierarchy F1, structure accuracy, hallucination rate |
| `normalization.py` | Gate name normalization with 200+ cell type synonyms |
| `hierarchy.py` | Tree operations: extract names, parent maps, depth |
| `task_failure.py` | Detect meta-questions, refusals, instructions |
| `response_parser.py` | Parse JSON, markdown, or indented text hierarchies |
| `scorer.py` | Combined scoring with task failure integration |

### Experiments (`src/experiments/`)

| Module | Purpose |
|--------|---------|
| `runner.py` | ExperimentRunner with checkpointing and multi-run support |
| `llm_client.py` | Unified interface for Anthropic, OpenAI, Ollama |
| `conditions.py` | Experimental conditions (context level, prompting strategy) |
| `prompts.py` | Prompt templates for different strategies |

### Analysis (`src/analysis/`)

| Module | Purpose |
|--------|---------|
| `manual_review_report.py` | Generate side-by-side comparison reports |
| `failure_analysis.py` | Categorize and analyze prediction failures |

## Metrics

| Metric | Definition |
|--------|------------|
| **Hierarchy F1** | Precision/recall on gate names with fuzzy matching |
| **Structure Accuracy** | % of parent-child relationships correct |
| **Critical Gate Recall** | % of must-have gates (live, singlets) present |
| **Hallucination Rate** | Gates referencing markers not in panel |
| **Task Failure Rate** | Meta-questions, refusals, or instructions instead of predictions |

### Task Failure Detection

The benchmark detects when models fail to complete the task:

```python
from evaluation.task_failure import detect_task_failure, TaskFailureType

result = detect_task_failure("What markers are you using?")
# result.is_failure = True
# result.failure_type = TaskFailureType.META_QUESTIONS
```

| Failure Type | Example |
|--------------|---------|
| META_QUESTIONS | "What markers are you using?" |
| REFUSAL | "I cannot predict without more information" |
| INSTRUCTIONS | "Here's how you would create a hierarchy..." |
| EMPTY | No response content |

## LLM Judge

The benchmark includes an LLM-based judge (Gemini 2.5 Pro) for qualitative assessment beyond automated metrics. The judge scores predictions on:

| Dimension | Description |
|-----------|-------------|
| **Completeness** | Are all expected gates present? |
| **Accuracy** | Are gate names and relationships correct? |
| **Scientific** | Is the reasoning biologically sound? |
| **Overall** | Holistic quality score |

### Hierarchy Flattening for Judge Prompts

Ground truth hierarchies are stored as nested JSON:

```json
{"root": {"name": "All Events", "children": [
  {"name": "Singlets", "children": [
    {"name": "Live", "children": [...]}
  ]}
]}}
```

For judge prompts, hierarchies are **flattened to arrow notation**:

```
All Events > Singlets > Live > CD45+ > T cells
All Events > Singlets > Live > CD45+ > B cells
```

**Why flatten?**

| Benefit | Rationale |
|---------|-----------|
| **Token efficiency** | Nested JSON is 10-20x more verbose; flattening reduces prompt size significantly |
| **Reduced parsing errors** | LLMs can misinterpret nested structure; flat paths are unambiguous |
| **Semantic clarity** | Judge needs to compare *relationships*, not parse syntax |
| **Garbage detection** | Truncated JSON produces artifacts (`"name"`, `"children"` as gate names); flattening exposes these |

**Tradeoff:** Gate metadata (`marker_logic`, `gate_type`, `is_critical`) is discarded. This is acceptable because the judge evaluates structural correctness, not gate definitions. For benchmarks requiring marker logic evaluation, preserve the full structure.

## Gate Name Normalization

The benchmark uses fuzzy matching with 200+ cell type synonyms:

```python
from evaluation.normalization import are_gates_equivalent

are_gates_equivalent("CD3+ T cells", "T cells")  # True
are_gates_equivalent("NK cells", "natural killer cells")  # True
are_gates_equivalent("classical monocytes", "classical monos")  # True
```

## Test Cases

8 OMIP papers with validated ground truth:

| OMIP | Focus | Colors | Difficulty |
|------|-------|--------|------------|
| OMIP-077 | Leukocyte populations | 14 | Easy |
| OMIP-101 | Fixed whole blood | 27 | Easy |
| OMIP-022 | gamma-delta T-cells | 15 | Medium |
| OMIP-076 | Murine T/B/ASC | 19 | Medium |
| OMIP-074 | B-cell subsets | 19 | Hard |
| OMIP-083 | Monocyte phenotyping | 21 | Hard |
| OMIP-064 | PBMC general | - | Variable |
| OMIP-095 | Spectral PBMC | - | Variable |

## Running Experiments

### Basic Usage

```bash
# Single model run
PYTHONPATH=src python scripts/run_experiment.py --model sonnet

# Dry run (mock API calls)
PYTHONPATH=src python scripts/run_experiment.py --model sonnet --dry-run

# With manual review report
PYTHONPATH=src python scripts/run_experiment.py --model sonnet --report outliers
```

### Report Generation

```bash
# From existing results
PYTHONPATH=src python scripts/generate_review_report.py \
  results/experiment_results_*.json \
  --level outliers \
  --output results/review.md
```

Report levels:
- `summary` - Overall metrics table only
- `outliers` - Details for outlier results
- `full` - All results with side-by-side comparisons

## Programmatic Usage

```python
from curation.omip_extractor import load_test_case
from evaluation.scorer import GatingScorer

# Load test case
test_case = load_test_case("data/ground_truth/omip_077.json")

# Score a prediction
scorer = GatingScorer()
result = scorer.score(
    response=llm_response,
    test_case=test_case,
    model="claude-sonnet-4",
    condition="rich_direct"
)

print(f"F1: {result.hierarchy_f1:.2f}")
print(f"Task failure: {result.is_task_failure}")
```

## Limitations

### Ground Truth
- Multiple valid strategies may exist for any panel
- Paper-based extraction introduces curation error
- Limited .wsp validation availability

### Evaluation
- Fuzzy matching may over/under-estimate performance
- Structure accuracy is strict (any mismatch = error)
- Hallucination detection is heuristic-based

### Coverage
- Biased toward PBMC samples
- Limited rare populations (ILCs, MAITs)
- Single-timepoint only

## Related Projects

- **[Flow Panel Optimizer](../flow_panel_optimizer/)** - MCP tools for spectral analysis
- Tests complementary capability: panel design vs. gating strategy

## License

MIT

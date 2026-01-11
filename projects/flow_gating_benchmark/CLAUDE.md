# Flow Gating Benchmark - Claude Code Instructions

## Project Goal

Evaluate whether LLMs can predict flow cytometry gating strategies from panel information and experimental context. Uses OMIP (Optimized Multicolor Immunofluorescence Panel) papers as ground truth.

## Research Question

Given a flow cytometry panel (markers, fluorophores, sample type), can LLMs predict the appropriate gating hierarchy?

## Project Structure

```
flow_gating_benchmark/
├── src/
│   ├── curation/              # Data preparation
│   │   ├── omip_extractor.py  # Extract panels from OMIP papers
│   │   ├── schemas.py         # GatingHierarchy, Gate models
│   │   └── wsp_cross_validator.py  # Validate against .wsp files
│   ├── evaluation/            # Scoring
│   │   ├── metrics.py         # Hierarchy F1, structure accuracy
│   │   ├── response_parser.py # Parse LLM gating predictions
│   │   └── scorer.py          # Compare predicted vs ground truth
│   ├── experiments/           # Experiment runner
│   │   ├── conditions.py      # Experimental conditions
│   │   ├── prompts.py         # Prompt templates
│   │   └── runner.py          # Main experiment loop
│   ├── analysis/              # Results analysis
│   │   ├── report_generator.py
│   │   ├── failure_analysis.py
│   │   └── visualization.py
│   └── validation/            # Phase 0 feasibility
│       ├── flowkit_validator.py
│       └── manual_llm_test.py
├── data/
│   ├── ground_truth/          # Curated test cases (JSON)
│   ├── raw/                   # Downloaded .wsp/.fcs files
│   └── extracted/             # Parsed hierarchies
├── results/                   # Experiment outputs
│   ├── benchmark_results_*.json
│   └── reports/
└── docs/
    ├── TEST_PLAN.md
    └── gating_workflow_guide.md
```

## Key Concepts

### Gating Hierarchy

A tree structure representing how cells are sequentially filtered:

```
Live cells
├── Singlets
│   ├── Lymphocytes
│   │   ├── CD3+ T cells
│   │   │   ├── CD4+ T cells
│   │   │   └── CD8+ T cells
│   │   └── CD19+ B cells
│   └── CD14+ Monocytes
```

### Evaluation Metrics

| Metric | What It Measures |
|--------|------------------|
| Hierarchy F1 | Gate name precision/recall |
| Structure Accuracy | Parent-child relationships |
| Critical Gate Recall | Must-have gates (live/dead, singlets) |
| Hallucination Rate | Gates that don't match panel markers |

### Experimental Conditions

| Factor | Levels |
|--------|--------|
| Prompting | Simple, Zero-shot, Chain-of-thought, Weight-of-thought |
| RAG | None, HIPC, OMIP, Search, Both |
| Model | Claude Sonnet, Claude Opus, GPT-4o |

### Ground Truth Standards

| Standard | Description |
|----------|-------------|
| HIPC (2016) | Expert-validated standardized definitions |
| OMIP | Paper-specific gating strategies |

Reference: https://www.nature.com/articles/srep20686

## Debugging Guidelines

**IMPORTANT**: When debugging experiments, always start with 1-2 conditions before running full test suites.

```bash
# Debug with single condition
python scripts/run_ab_test.py --conditions baseline_zero_shot --test-cases OMIP-023 --dry-run

# Debug with 2 conditions on 1 test case
python scripts/run_ab_test.py --conditions baseline_zero_shot cot_hipc_rag --test-cases OMIP-023

# Check parsed output
python -c "
import json
with open('results/ab_tests/LATEST_RESULTS.json') as f:
    data = json.load(f)
run = data['runs'][0]
print('Parsed gates:', [g['name'] for g in run['predicted_hierarchy'].get('children', [])])
print('HIPC matches:', run['ab_result']['hipc_matches'])
"
```

### Common Issues

1. **HIPC scores all same** - Check name matching in `find_hipc_match()`
2. **Empty parsed hierarchy** - Check response parser regex patterns
3. **Low OMIP scores** - Ground truth gate names may not match LLM output format

## Key Commands

```bash
# Install
cd projects/flow_gating_benchmark
pip install -r requirements.txt

# Run validation (Phase 0)
python -m src.validation.run_validation

# Run benchmark
python run_benchmark.py

# Generate reports
python -m src.analysis.report_generator results/benchmark_results_*.json
```

## Ground Truth Format

Test cases in `data/ground_truth/omip_XXX.json`:

```json
{
  "omip_id": "OMIP-069",
  "title": "40-color spectral flow cytometry panel",
  "sample_type": "PBMC",
  "species": "human",
  "panel": [
    {"marker": "CD3", "fluorophore": "BUV395", "clone": "UCHT1"},
    {"marker": "CD4", "fluorophore": "BUV496", "clone": "SK3"}
  ],
  "gating_hierarchy": {
    "name": "All Events",
    "children": [
      {
        "name": "Live cells",
        "marker_logic": "Zombie NIR-",
        "children": [...]
      }
    ]
  },
  "critical_gates": ["Live cells", "Singlets", "Lymphocytes"]
}
```

## Common Tasks

### Adding a New OMIP Test Case

1. Create JSON file in `data/ground_truth/omip_XXX.json`
2. Extract panel from OMIP paper
3. Define gating hierarchy from paper figures
4. Mark critical gates
5. Run validation: `python -m src.validation.run_validation`

### Running a Single Model Test

```python
from src.experiments.runner import run_single_test
from src.experiments.conditions import ExperimentCondition

result = run_single_test(
    test_case="omip_069",
    model="claude-sonnet-4-20250514",
    condition=ExperimentCondition.FULL_CONTEXT
)
```

### Analyzing Failures

```python
from src.analysis.failure_analysis import analyze_failures

failures = analyze_failures(results, threshold=0.5)
# Returns common failure patterns:
# - Missing critical gates
# - Incorrect parent assignments
# - Hallucinated populations
```

## OMIP Test Cases

| OMIP | Colors | Sample | Focus |
|------|--------|--------|-------|
| OMIP-069 | 40 | PBMC | Full spectrum |
| OMIP-058 | 30 | PBMC | T/NK/iNKT |
| OMIP-044 | 28 | PBMC | Dendritic cells |
| OMIP-023 | 10 | Blood | Basic leukocyte |

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

## Environment Variables

```bash
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

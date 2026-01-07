# DrugDevBench

**Benchmark for evaluating LLM interpretation of drug development figures**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

DrugDevBench evaluates how well large language models interpret scientific figures from drug development research. Unlike general benchmarks, it tests domain-specific conventions that practitioners rely on daily.

### Key Features

- **Multi-model support** via LiteLLM (Claude, Gemini, GPT)
- **Domain expert personas** (Immunologist, Pharmacologist, Bioanalytical Scientist, etc.)
- **Figure-type skills** (Western blot, dose-response, PK curves, flow cytometry)
- **Auto-generated questions** from paper figure legends
- **Cost-conscious design** with response caching
- **Ablation study framework** for systematic evaluation

## Architecture

```
┌─────────────────────────────────────────────┐
│              PERSONA LAYER                  │
│  (Immunologist, Pharmacologist, etc.)       │
└─────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│         BASE SCIENTIFIC REASONING           │
└─────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│            FIGURE-TYPE SKILL                │
│  (Western blot, dose-response, etc.)        │
└─────────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/jbalibalos/drugdevbench.git
cd drugdevbench

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Quick Start

### 1. Set up API keys

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run evaluation

```python
from drugdevbench.models import DrugDevBenchEvaluator, EvaluatorConfig
from drugdevbench.prompts import build_system_prompt
from drugdevbench.data import FigureType, PromptCondition

# Use cheap model for development
config = EvaluatorConfig(default_model="claude-haiku", use_cache=True)
evaluator = DrugDevBenchEvaluator(config)

# Build prompt
system_prompt = build_system_prompt(
    condition=PromptCondition.FULL_STACK,
    figure_type=FigureType.WESTERN_BLOT,
)

# Run evaluation
response = evaluator.evaluate(
    figure_id="fig_001",
    question_id="q_001",
    image_path="data/figures/western_blots/example.png",
    question="Is a loading control shown?",
    system_prompt=system_prompt,
    condition=PromptCondition.FULL_STACK,
)

print(response.response_text)
```

### 3. Generate questions from paper legend

```python
from drugdevbench.questions import generate_questions
from drugdevbench.data import FigureType

legend = """
Figure 2. Dose-response curve for Compound X.
IC50 = 2.3 μM. Data represent mean ± SEM, n=3.
"""

questions = generate_questions(
    figure_id="fig_001",
    figure_type=FigureType.DOSE_RESPONSE,
    legend_text=legend,
)

for q in questions:
    print(f"Q: {q.question_text}")
    print(f"A: {q.gold_answer}")
```

### 4. Run ablation study

```python
from drugdevbench.data import load_annotations
from drugdevbench.evaluation import AblationConfig, run_ablation_study

# Load your annotations
annotations = load_annotations("data/annotations/annotations.jsonl")

# Configure ablation
config = AblationConfig(
    models=["claude-haiku", "gemini-flash"],
    conditions=[
        PromptCondition.VANILLA,
        PromptCondition.BASE_ONLY,
        PromptCondition.FULL_STACK,
    ],
    max_figures=10,  # Limit for testing
)

# Run ablation study
results = run_ablation_study(annotations, config)
```

## CLI Usage

```bash
# List supported models
drugdevbench list-models

# List ablation conditions
drugdevbench list-conditions

# List figure types
drugdevbench list-figure-types

# Estimate costs
drugdevbench estimate-cost --figures 50 --questions 4 --conditions 5

# Run benchmark
drugdevbench run data/annotations/annotations.jsonl -m claude-haiku -o results/
```

## Supported Models

| Model             | Key             | Cost Tier | Notes               |
|-------------------|-----------------|-----------|---------------------|
| Claude 3.5 Sonnet | `claude-sonnet` | $$        | Primary evaluation  |
| Claude 3 Haiku    | `claude-haiku`  | $         | Development/testing |
| Gemini 1.5 Pro    | `gemini-pro`    | $$        | Comparison          |
| Gemini 1.5 Flash  | `gemini-flash`  | $         | Development/testing |
| GPT-4o            | `gpt-4o`        | $$        | Comparison          |
| GPT-4o-mini       | `gpt-4o-mini`   | $         | Development/testing |

## Figure Types

| Category         | Figure Types                | Persona                 |
|------------------|-----------------------------|-----------------------  |
| Protein Analysis | Western blot, Coomassie gel | Molecular Biologist     |
| Binding Assays   | ELISA, dose-response        | Bioanalytical Scientist |
| Pharmacokinetics | PK curves, AUC              | Pharmacologist          |
| Flow Cytometry   | Biaxial, histogram, gating  | Immunologist            |
| Genomics         | Heatmap, volcano plot       | Computational Biologist |
| Cell Assays      | Viability, proliferation    | Cell Biologist          |

## Ablation Conditions

| Condition         | Persona | Base | Skill          |
|-------------------|---------|------|----------------|
| `vanilla`         | ❌       | ❌    | ❌              |
| `base_only`       | ❌       | ✅    | ❌              |
| `persona_only`    | ✅       | ❌    | ❌              |
| `base_plus_skill` | ❌       | ✅    | ✅              |
| `full_stack`      | ✅       | ✅    | ✅              |
| `wrong_skill`     | ❌       | ✅    | ❌ (mismatched) |

## Cost Management

DrugDevBench includes several features to minimize API costs:

1. **Response caching**: Identical queries return cached results
2. **Cheap model defaults**: Development uses Haiku/Flash by default
3. **Cost estimation**: Preview costs before running

```python
from drugdevbench.models import estimate_benchmark_cost

estimate = estimate_benchmark_cost(
    n_figures=50,
    n_questions_per_figure=4,
    n_conditions=5,
    models=["claude-haiku", "gemini-flash"],
)
print(f"Estimated cost: ${estimate['total_cost_usd']:.2f}")
```

## Project Structure

```
drugdevbench/
├── src/drugdevbench/
│   ├── data/           # Schemas, loaders, sources
│   ├── models/         # LiteLLM wrapper, caching
│   ├── prompts/        # Base, personas, skills
│   ├── questions/      # Auto-generation
│   └── evaluation/     # Scoring, metrics, ablation
├── data/
│   ├── figures/        # Image files by type
│   ├── annotations/    # Ground truth JSONL
│   ├── papers/         # Source paper metadata
│   └── cache/          # Response cache
├── notebooks/          # Analysis notebooks
├── results/            # Ablation results
└── tests/              # Unit tests
```

## Running Tests

```bash
pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License

## Citation

```bibtex
@software{drugdevbench2025,
  author = {Balibalos, John},
  title = {DrugDevBench: Evaluating LLM Interpretation of Drug Development Figures},
  year = {2025},
  url = {https://github.com/jbalibalos/drugdevbench}
}
```

# DrugDevBench - Claude Code Instructions

## Project Goal

Evaluate LLM interpretation of scientific figures from drug development research. Tests whether domain-specific prompting (personas, figure-type skills) improves accuracy on figure analysis tasks.

## Research Question

Do structured prompts with domain expertise improve LLM figure interpretation compared to vanilla prompting?

## Project Structure

```
drugdevbench/
├── src/drugdevbench/
│   ├── data/              # Schemas and data loading
│   │   ├── schemas.py     # FigureAnnotation, Question models
│   │   ├── loader.py      # JSONL annotation loading
│   │   └── sources/       # PubMed, bioRxiv, OpenPMC clients
│   ├── models/            # LLM interface
│   │   ├── litellm_wrapper.py  # Multi-model support
│   │   ├── cache.py       # Response caching
│   │   └── mock_evaluator.py   # Testing without API calls
│   ├── prompts/           # Prompt engineering
│   │   ├── base.py        # Base scientific reasoning prompt
│   │   ├── dispatcher.py  # Route figure types to skills
│   │   ├── personas/      # Domain expert personas
│   │   │   ├── immunologist.py
│   │   │   ├── pharmacologist.py
│   │   │   └── ...
│   │   └── skills/        # Figure-type specific skills
│   │       ├── western_blot.py
│   │       ├── dose_response.py
│   │       └── ...
│   ├── questions/         # Question generation from legends
│   │   └── generator.py
│   └── evaluation/        # Scoring and analysis
│       ├── metrics.py     # Accuracy, F1, etc.
│       ├── rubric.py      # Scoring rubrics
│       ├── ablation.py    # Ablation study runner
│       └── reporting.py   # Results formatting
├── data/
│   ├── figures/           # Images organized by type
│   ├── annotations/       # Ground truth JSONL
│   └── cache/             # Response cache
├── results/               # Experiment outputs
└── tests/
```

## Key Concepts

### Ablation Conditions

Tests different prompt configurations:

| Condition | Persona | Base | Skill | Purpose |
|-----------|---------|------|-------|---------|
| `vanilla` | - | - | - | Baseline |
| `base_only` | - | Yes | - | Test base prompt alone |
| `persona_only` | Yes | - | - | Test persona alone |
| `base_plus_skill` | - | Yes | Yes | Test skill routing |
| `full_stack` | Yes | Yes | Yes | Full system |
| `wrong_skill` | - | Yes | Wrong | Test skill specificity |

### Figure Types

- Western blot, Coomassie gel
- ELISA, dose-response curves
- PK curves, AUC plots
- Flow cytometry (biaxial, histogram)
- Heatmaps, volcano plots

### Personas

Domain experts that frame how to approach figures:
- Immunologist
- Pharmacologist
- Molecular Biologist
- Bioanalytical Scientist
- Computational Biologist
- Cell Biologist

## Key Commands

```bash
# Install
cd projects/drugdevbench
pip install -e .

# Run tests
pytest tests/ -v

# CLI commands
drugdevbench list-models
drugdevbench list-conditions
drugdevbench estimate-cost --figures 50

# Run evaluation
drugdevbench run data/annotations/annotations.jsonl -m claude-haiku -o results/
```

## Common Tasks

### Adding a New Figure Type

1. Create skill in `prompts/skills/new_figure_type.py`
2. Add to `FigureType` enum in `data/schemas.py`
3. Update dispatcher in `prompts/dispatcher.py`
4. Add test cases

### Adding a New Persona

1. Create persona in `prompts/personas/new_persona.py`
2. Add to persona registry
3. Map to relevant figure types

### Running Cost-Effective Tests

```python
# Use cheap models during development
config = EvaluatorConfig(default_model="claude-haiku", use_cache=True)
```

## Data Format

Annotations in JSONL format:

```json
{
  "figure_id": "fig_001",
  "image_path": "figures/western_blots/example.png",
  "figure_type": "western_blot",
  "questions": [
    {
      "question_id": "q_001",
      "question_text": "Is a loading control shown?",
      "gold_answer": "Yes, beta-actin loading control in bottom panel"
    }
  ]
}
```

## Environment Variables

```bash
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

## References

- LiteLLM documentation: https://docs.litellm.ai/
- Vision model capabilities vary by provider

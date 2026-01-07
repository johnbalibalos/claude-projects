# FlowRepository Gating Benchmark

An evaluation framework for testing LLM capabilities in predicting flow cytometry gating strategies from panel information and experimental context.

## Project Goal

Build a benchmark that measures whether LLMs can predict appropriate flow cytometry gating hierarchies given:
- Panel information (markers, fluorophores, clones)
- Experimental context (sample type, species, application)
- Optional reference protocols (OMIP papers)

## Why This Matters

- Demonstrates evaluation methodology design for scientific AI
- Tests domain-specific reasoning capabilities
- Provides quantifiable metrics for scientific task performance

## Project Structure

```
flow_gating_benchmark/
├── src/
│   ├── validation/       # Phase 0: Feasibility validation
│   ├── curation/         # Phase 1: Data curation tools
│   ├── evaluation/       # Phase 2: Metrics and scoring
│   ├── experiments/      # Phase 3: Experiment runner
│   └── analysis/         # Phase 4: Results analysis
├── data/
│   ├── ground_truth/     # Curated test cases with gold standard hierarchies
│   ├── raw/              # Downloaded .wsp and .fcs files
│   └── extracted/        # Parsed hierarchies from flowkit
├── results/              # Experiment outputs and logs
└── docs/                 # Additional documentation
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Phase 0 validation
python -m src.validation.run_validation

# Run experiments (after curation complete)
python -m src.experiments.runner
```

## Key Dependencies

- `flowkit` - .wsp parsing and FCS reading
- `anthropic`, `openai` - LLM APIs
- `pandas` - Data wrangling
- `matplotlib`, `seaborn` - Visualization

## Test Case Sources

### OMIP Papers (Primary)
| OMIP | Description | FlowRepository ID | Colors |
|------|-------------|-------------------|--------|
| OMIP-069 | 40-color spectral PBMC | FR-FCM-Z7YM | 40 |
| OMIP-058 | 30-color T/NK/iNKT | FR-FCM-ZYRN | 30 |
| OMIP-044 | 28-color DC compartment | FR-FCM-ZYC2 | 28 |
| OMIP-043 | Antibody secreting cells | FR-FCM-ZYBP | 25 |
| OMIP-023 | 10-color leukocyte | FR-FCM-ZZ74 | 10 |

## Metrics

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| Hierarchy F1 | Precision/recall on gate names | Did LLM identify correct populations? |
| Structure Accuracy | % of parent-child relationships correct | Does logical flow match? |
| Critical Gate Recall | % of "must-have" gates present | Missed live/dead = fundamental error |
| Hallucination Rate | Gates predicted that don't match markers | Identifies confabulation |

## Related Projects

This benchmark complements the [Flow Panel Optimizer](../flow_panel_optimizer/) MCP project:
- **Gating Benchmark**: Tests "Can LLMs predict gating strategies?" (structure)
- **Panel Optimizer**: Tests "Can LLMs optimize fluorophore selection?" (panel design)

## License

MIT

# LLM Biology Research - Claude Code Instructions

## Repository Overview

Exploratory research projects investigating LLM capabilities in biological and biomedical domains. Each project tests specific hypotheses about AI performance on scientific tasks.

## Repository Structure

```
/
├── libs/                              # Shared libraries
│   ├── checkpoint/                    # Resumable workflow checkpointing
│   ├── mcp_tester/                    # MCP/tool ablation framework
│   └── hypothesis_pipeline/           # Modular hypothesis testing pipeline
└── projects/
    ├── flow_panel_optimizer/          # MCP tool for spectral analysis
    ├── drugdevbench/                  # Figure interpretation benchmark
    └── flow_gating_benchmark/         # Gating strategy prediction benchmark
```

## Projects

### Flow Panel Optimizer
**Path:** `projects/flow_panel_optimizer/`
**Type:** MCP Tool Server
**Question:** Does tool access improve LLM accuracy on spectral calculations?

Provides flow cytometry spectral analysis tools (similarity, complexity, spreading). Used to test whether MCP tools improve scientific reasoning vs. parametric knowledge alone.

### DrugDevBench
**Path:** `projects/drugdevbench/`
**Type:** Evaluation Benchmark
**Question:** Do domain-specific prompts improve figure interpretation?

Evaluates LLM interpretation of drug development figures (Western blots, dose-response, PK curves). Tests ablation of personas and figure-type skills.

### Flow Gating Benchmark
**Path:** `projects/flow_gating_benchmark/`
**Type:** Evaluation Benchmark
**Question:** Can LLMs predict gating strategies from panel information?

Tests whether LLMs can predict flow cytometry gating hierarchies given marker panels. Uses OMIP papers as ground truth.

## Shared Libraries

### Checkpoint (`libs/checkpoint/`)

Resumable workflows with automatic checkpointing:

```python
from checkpoint import CheckpointedRunner

runner = CheckpointedRunner("experiment")
for item in runner.iterate(items, key_fn=lambda x: x['id']):
    result = process(item)
    runner.save_result(item['id'], result)
```

### MCP Tester (`libs/mcp_tester/`)

Generic framework for tool ablation studies:

```python
from mcp_tester import AblationStudy, TestCase, Condition

study = AblationStudy(
    name="test",
    test_cases=[...],
    conditions=[
        Condition(name="baseline", tools_enabled=False),
        Condition(name="with_tools", tools_enabled=True, tools=[...])
    ]
)
results = study.run()
```

### Hypothesis Pipeline (`libs/hypothesis_pipeline/`)

Modular framework for hypothesis testing with independent dimensions:

```python
from hypothesis_pipeline import (
    HypothesisPipeline,
    PipelineConfig,
    ReasoningType,
    ContextLevel,
    RAGMode,
)

config = PipelineConfig(
    name="experiment",
    models=["claude-sonnet-4-20250514"],
    reasoning_types=[ReasoningType.DIRECT, ReasoningType.COT],
    context_levels=[ContextLevel.MINIMAL, ContextLevel.STANDARD],
    rag_modes=[RAGMode.NONE, RAGMode.VECTOR],
    tool_configs=[[], ["my_tool"]],  # With and without tools
)

pipeline = HypothesisPipeline(config, evaluator, trial_inputs)
results = pipeline.run()
```

**Pluggable components:**
- `PromptStrategy`: CoT, WoT, direct, few-shot, ReAct
- `RAGProvider`: none, vector, hybrid, oracle, negative
- `ContextBuilder`: minimal, standard, rich, oracle
- `ToolRegistry`: dynamic tool/MCP registration

## Working in This Repository

### Project-Specific Work

Each project has its own `CLAUDE.md` with detailed instructions:

```bash
# When working on a specific project, read its CLAUDE.md first
cat projects/flow_panel_optimizer/CLAUDE.md
cat projects/drugdevbench/CLAUDE.md
cat projects/flow_gating_benchmark/CLAUDE.md
```

### Running Tests

```bash
# Each project has its own test suite
cd projects/flow_panel_optimizer && pytest
cd projects/drugdevbench && pytest tests/
cd projects/flow_gating_benchmark && pytest tests/
```

### Environment Setup

API keys stored in root `.env`:

```bash
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

## Adding New Projects

1. Create directory under `projects/`
2. Add `README.md` (user-facing documentation)
3. Add `CLAUDE.md` (Claude Code instructions)
4. Include `pyproject.toml` or `requirements.txt`
5. Update this file and root `README.md`

## Code Style

- Python 3.11+
- Type hints encouraged
- Each project is self-contained
- Shared utilities go in `libs/`

## Running Experiments

### Cost Estimation

All experiments require user confirmation before running. The cost estimator shows:
- Number of API calls
- Estimated token usage
- Estimated cost in USD

```python
from hypothesis_pipeline.cost import confirm_experiment_cost, estimate_experiment_cost

# Get estimate without running
estimate = estimate_experiment_cost(config, n_test_cases=10)
print(estimate.format_summary())

# Interactive confirmation before running
if confirm_experiment_cost(config, n_test_cases=10):
    pipeline.run()
```

### IMPORTANT: Do Not Use --force Flag

Experiment scripts have a `--force` flag that skips cost confirmation.

**Claude Code should NEVER use the `--force` flag.** Always allow the user to review costs and confirm before running experiments. This prevents accidental expensive API calls.

```bash
# CORRECT: Let user confirm costs
python run_synthetic_ablation.py

# WRONG: Do not use --force
python run_synthetic_ablation.py --force  # DO NOT USE
```

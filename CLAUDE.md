# LLM Biology Research

Exploratory research testing LLM capabilities in biological/biomedical domains.

## Tech Stack

- Python 3.11+
- APIs: Anthropic, OpenAI, Google (keys in `.env`)
- Testing: pytest
- CI: GitHub Actions

## CRITICAL Rules

IMPORTANT: You MUST follow these rules:

1. **NEVER use `--force` on experiment scripts** - always confirm costs with user first
2. **NEVER run experiments without cost confirmation** - can cost $10-100+
3. **ALWAYS read project-specific CLAUDE.md** when working in a subdirectory

## Commands

```bash
# Environment setup
source .env

# Run tests (ALWAYS run before committing)
pytest projects/flow_gating_benchmark/tests/
pytest projects/flow_panel_optimizer/
pytest projects/drugdevbench/tests/

# GitHub
gh issue list
gh pr create

# Paper downloads
python -m paper_download
```

## Project Structure

```
libs/                            # Shared libraries
├── checkpoint/                  # Resumable workflow checkpointing
├── mcp_tester/                  # MCP/tool ablation framework
├── hypothesis_pipeline/         # Modular hypothesis testing
└── paper_download/              # PMC paper downloader

projects/
├── flow_panel_optimizer/        # MCP tool for spectral analysis
├── drugdevbench/                # Figure interpretation benchmark
└── flow_gating_benchmark/       # Gating strategy prediction
```

## Code Style

- Type hints encouraged
- Each project is self-contained
- Shared utilities go in `libs/`
- Use descriptive commit messages with `Co-Authored-By: Claude` when applicable

## Workflow

1. Read the relevant project's CLAUDE.md first
2. Check existing patterns in codebase
3. For experiments: estimate and confirm costs
4. Run tests before committing

### Cost Confirmation Pattern

```python
from hypothesis_pipeline.cost import confirm_experiment_cost
if confirm_experiment_cost(config, n_test_cases=10):
    pipeline.run()
```

## Project-Specific Instructions

Each project is standalone. When working in a project directory, read its CLAUDE.md:

- `projects/flow_panel_optimizer/CLAUDE.md` - Spectral similarity metrics for flow cytometry
- `projects/drugdevbench/CLAUDE.md` - LLM figure interpretation benchmark
- `projects/flow_gating_benchmark/CLAUDE.md` - Gating strategy prediction (~750 lines, comprehensive)

## Adding New Projects

1. Create directory under `projects/`
2. Add `README.md` and `CLAUDE.md`
3. Include `pyproject.toml` or `requirements.txt`
4. Update this file

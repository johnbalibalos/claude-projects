# LLM Biology Research

Exploratory research projects testing LLM capabilities in biological/biomedical domains.

## Quick Reference

```bash
# Environment
source .env  # API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY

# Tests
pytest projects/flow_gating_benchmark/tests/
pytest projects/flow_panel_optimizer/
pytest projects/drugdevbench/tests/

# Common tools
gh issue list                    # GitHub issues
gh pr create                     # Create PR
python -m paper_download         # Download PMC papers
```

## Repository Structure

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

## IMPORTANT Rules

1. **NEVER use `--force` flag** on experiment scripts. Always let user confirm costs.
2. **Read project CLAUDE.md first** when working in a specific project directory.
3. **Confirm before expensive API calls** - experiments can cost $10-100+.

## Code Style

- Python 3.11+
- Type hints encouraged
- Each project is self-contained
- Shared utilities go in `libs/`

## Workflow

### Before coding
- Read relevant project's `CLAUDE.md`
- Check existing patterns in codebase
- For experiments: estimate costs first

### When running experiments
```python
from hypothesis_pipeline.cost import confirm_experiment_cost
if confirm_experiment_cost(config, n_test_cases=10):
    pipeline.run()
```

### Committing
- Use descriptive commit messages
- Include `Co-Authored-By: Claude` when applicable
- Run tests before committing

## Project-Specific Instructions

Each project has its own CLAUDE.md with detailed guidance:
- `projects/flow_panel_optimizer/CLAUDE.md`
- `projects/drugdevbench/CLAUDE.md`
- `projects/flow_gating_benchmark/CLAUDE.md`

## Adding New Projects

1. Create directory under `projects/`
2. Add `README.md` and `CLAUDE.md`
3. Include `pyproject.toml` or `requirements.txt`
4. Update this file

# LLM Biology Research

Exploratory projects investigating large language model capabilities in biological and biomedical domains.

## Overview

This repository contains experimental tools and benchmarks for evaluating how LLMs perform on specialized scientific tasks. These projects are **exploratory research** - they test hypotheses about AI capabilities in biology rather than production-ready solutions.

### Research Questions

- Can LLMs effectively use domain-specific tools (MCP servers) for scientific calculations?
- How well do LLMs interpret specialized scientific figures from drug development?
- Can LLMs predict appropriate experimental protocols from panel information?

## Projects

| Project | Domain | Type | Status |
|---------|--------|------|--------|
| [flow_gating_benchmark](projects/flow_gating_benchmark/) | Flow Cytometry | Benchmark | Active |
| [flow_panel_optimizer](projects/flow_panel_optimizer/) | Flow Cytometry | MCP Tool | Experimental |
| [drugdevbench](projects/drugdevbench/) | Drug Development | Benchmark | Experimental |

### Flow Gating Benchmark

**Question:** Can LLMs predict flow cytometry gating strategies from panel information?

An evaluation framework testing whether LLMs can predict appropriate gating hierarchies given marker panels and experimental context. Uses OMIP papers as ground truth.

**Latest Results (Jan 2026):**
- Sonnet 4: F1=0.384, Critical Gate Recall=83.9%
- Task failure detection: Identifies when models ask questions vs. predicting

```
projects/flow_gating_benchmark/
├── src/
│   ├── curation/      # OMIP data extraction and paper parsing
│   ├── evaluation/    # Metrics, scoring, task failure detection
│   ├── experiments/   # Experiment runner with multi-model support
│   └── analysis/      # Manual review reports and visualization
├── data/ground_truth/ # 8 curated OMIP gating hierarchies
└── scripts/           # CLI tools for running experiments
```

### Flow Panel Optimizer

**Question:** Can an MCP server improve LLM accuracy on spectral similarity calculations?

A tool server providing flow cytometry spectral analysis capabilities. Tests whether giving LLMs access to real calculation tools improves their performance on panel design tasks.

```
projects/flow_panel_optimizer/
├── src/flow_panel_optimizer/
│   ├── spectral/      # Similarity, complexity, spreading calculations
│   ├── mcp/           # MCP server implementation
│   └── validation/    # OMIP panel validation
└── tests/mcp_effectiveness/  # Ablation studies
```

### DrugDevBench

**Question:** How well do LLMs interpret domain-specific scientific figures?

A benchmark for evaluating LLM interpretation of drug development figures (Western blots, dose-response curves, PK plots). Tests whether domain-specific prompting strategies improve accuracy.

```
projects/drugdevbench/
├── src/drugdevbench/
│   ├── prompts/       # Persona and skill-based prompts
│   ├── evaluation/    # Scoring and ablation framework
│   └── models/        # Multi-model support via LiteLLM
└── data/              # Figures and annotations
```

## Shared Libraries

Reusable utilities extracted from individual projects:

| Library | Purpose |
|---------|---------|
| `libs/checkpoint` | Resumable workflows with automatic checkpointing |
| `libs/mcp_tester` | Generic framework for MCP/tool ablation studies |
| `libs/paper_download` | PMC paper search and download client |
| `libs/results_processor` | Export experiment results to CSV and summaries |

## Getting Started

Each project is self-contained with its own dependencies:

```bash
# Example: Set up flow_gating_benchmark
cd projects/flow_gating_benchmark
pip install -r requirements.txt

# Run an experiment
PYTHONPATH=src python scripts/run_experiment.py --model sonnet --dry-run
```

See individual project READMEs for specific instructions.

## Repository Structure

```
/
├── README.md              # This file
├── CLAUDE.md              # Claude Code instructions
├── TODO.md                # Cross-project task tracking
├── libs/                  # Shared libraries
│   ├── checkpoint/        # Resumable workflow runner
│   ├── mcp_tester/        # Tool ablation framework
│   ├── paper_download/    # PMC paper downloader
│   └── results_processor/ # Results export tools
└── projects/
    ├── flow_gating_benchmark/  # Gating prediction benchmark
    ├── flow_panel_optimizer/   # Spectral analysis MCP server
    └── drugdevbench/           # Figure interpretation benchmark
```

## Recent Updates (Jan 2026)

### Flow Gating Benchmark
- Added **task failure detection** - identifies when models ask questions instead of predicting
- Refactored evaluation into focused modules (normalization, hierarchy, task_failure)
- Added **manual review report generator** with outlier detection
- Multi-model LLM client supporting Anthropic, OpenAI, and Ollama
- 8 OMIP test cases with ground truth hierarchies

### Flow Panel Optimizer
- MCP ablation studies comparing tool vs. no-tool performance
- Spectral similarity and spreading matrix calculations

## Research Context

These projects explore the intersection of:

- **Tool-augmented LLMs**: Testing whether domain-specific tools improve scientific reasoning
- **Scientific figure interpretation**: Evaluating multimodal capabilities on real research data
- **Protocol prediction**: Testing if LLMs can infer experimental procedures from context

All projects are experimental and intended for research purposes.

## License

MIT

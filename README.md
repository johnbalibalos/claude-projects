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
| [flow_panel_optimizer](projects/flow_panel_optimizer/) | Flow Cytometry | MCP Tool | Experimental |
| [drugdevbench](projects/drugdevbench/) | Drug Development | Benchmark | Experimental |
| [flow_gating_benchmark](projects/flow_gating_benchmark/) | Flow Cytometry | Benchmark | Experimental |

### Flow Panel Optimizer

**Question:** Can an MCP server improve LLM accuracy on spectral similarity calculations?

A tool server that provides flow cytometry spectral analysis capabilities. Tests whether giving LLMs access to real calculation tools improves their performance on panel design tasks compared to relying on parametric knowledge alone.

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

A benchmark for evaluating LLM interpretation of drug development figures (Western blots, dose-response curves, PK plots, flow cytometry). Tests whether domain-specific prompting strategies (personas, skills) improve accuracy.

```
projects/drugdevbench/
├── src/drugdevbench/
│   ├── prompts/       # Persona and skill-based prompts
│   ├── evaluation/    # Scoring and ablation framework
│   └── models/        # Multi-model support via LiteLLM
└── data/              # Figures and annotations
```

### Flow Gating Benchmark

**Question:** Can LLMs predict flow cytometry gating strategies from panel information?

An evaluation framework testing whether LLMs can predict appropriate gating hierarchies given marker panels and experimental context. Uses OMIP (Optimized Multicolor Immunofluorescence Panel) papers as ground truth.

```
projects/flow_gating_benchmark/
├── src/
│   ├── curation/      # OMIP data extraction
│   ├── evaluation/    # Metrics and scoring
│   └── experiments/   # Experiment runner
└── data/ground_truth/ # Curated OMIP gating hierarchies
```

## Shared Libraries

Reusable utilities extracted from individual projects:

| Library | Purpose |
|---------|---------|
| `libs/checkpoint` | Resumable workflows with automatic checkpointing |
| `libs/mcp_tester` | Generic framework for MCP/tool ablation studies |

## Getting Started

Each project is self-contained with its own dependencies:

```bash
# Example: Set up flow_panel_optimizer
cd projects/flow_panel_optimizer
pip install -e ".[dev]"
pytest
```

See individual project READMEs for specific instructions.

## Repository Structure

```
/
├── README.md              # This file
├── CLAUDE.md              # Claude Code instructions
├── libs/                  # Shared libraries
│   ├── checkpoint/
│   └── mcp_tester/
└── projects/
    ├── flow_panel_optimizer/
    ├── drugdevbench/
    └── flow_gating_benchmark/
```

## Research Context

These projects explore the intersection of:

- **Tool-augmented LLMs**: Testing whether domain-specific tools improve scientific reasoning
- **Scientific figure interpretation**: Evaluating multimodal capabilities on real research data
- **Protocol prediction**: Testing if LLMs can infer experimental procedures from context

All projects are experimental and intended for research purposes.

## License

MIT

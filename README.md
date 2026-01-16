# LLM Biology Research

Can LLMs reason about biology, or just pattern-match from training data?

We're testing this with flow cytometry gating prediction. Early finding: **gemini-2.5-pro leads at 0.36 F1**, but there's only weak correlation (r≈0.15) between string-matching metrics and LLM judge scores—suggesting current evals may be measuring the wrong thing.

## Projects

| Project | Question | Status |
|---------|----------|--------|
| [flow_gating_benchmark](projects/flow_gating_benchmark/) | Can LLMs predict gating strategies from panel info? | [Results](projects/flow_gating_benchmark/results/BENCHMARK_RESULTS_SUMMARY.md) |
| [flow_panel_optimizer](projects/flow_panel_optimizer/) | Does tool access improve spectral calculations? | 88.6% improvement with MCP tools |
| [drugdevbench](projects/drugdevbench/) | How well do LLMs interpret drug dev figures? | Framework built, eval pending |

## Quick Start

```bash
cd projects/flow_gating_benchmark
pip install -r requirements.txt
PYTHONPATH=src python scripts/run_experiment.py --model sonnet --dry-run
```

See individual project READMEs for details.

## Key Findings (Jan 2026)

### Flow Gating Benchmark

| Model | Hierarchy F1 | Judge Quality |
|-------|--------------|---------------|
| gemini-2.5-pro | 0.361 | 0.592 |
| claude-opus-4 | 0.330 | 0.523 |
| gemini-2.0-flash | 0.340 | 0.408 |

**Caveat:** Rich context included OMIP paper references, possibly allowing retrieval from training data. Re-evaluation pending.

See [STUDY_SUMMARY.md](projects/flow_gating_benchmark/STUDY_SUMMARY.md) for methodology.

### Flow Panel Optimizer

MCP tool access improves panel complexity accuracy by 88.6% vs baseline. Models without tools hallucinate spectral overlap values.

## Shared Libraries

| Library | Purpose |
|---------|---------|
| `libs/checkpoint` | Resumable workflows with checkpointing |
| `libs/mcp_tester` | MCP/tool ablation framework |
| `libs/paper_download` | PMC paper downloader |
| `libs/results_processor` | Export results to CSV |

## Structure

```
libs/                  # Shared utilities
projects/
├── flow_gating_benchmark/  # Gating prediction benchmark
├── flow_panel_optimizer/   # Spectral analysis MCP server
└── drugdevbench/           # Figure interpretation benchmark
```

## License

MIT

# Results Processor

Export experiment results to CSV and generate summary reports.

## Features

- JSON to CSV conversion with configurable columns
- Automatic column detection for different experiment types
- Summary report generation
- Aggregation by condition

## Installation

```bash
cd libs/results_processor
pip install -e .
```

## Usage

### Command Line

```bash
# Export to CSV
python -m results_processor export results/experiment.json

# Generate summary
python -m results_processor summarize results/experiment.json

# Custom output path
python -m results_processor export results/experiment.json -o results/data.csv
```

### Python API

```python
from results_processor import ResultsExporter, export_to_csv, generate_summary

# Quick export
csv_path = export_to_csv("results/experiment.json")

# Quick summary
summary_path = generate_summary("results/experiment.json")
```

### Custom Columns

```python
from results_processor import ResultsExporter

exporter = ResultsExporter(columns=[
    "test_case_id",
    "condition",
    "hierarchy_f1",
    "structure_accuracy",
])

exporter.export_to_csv("results/experiment.json", "results/custom.csv")
```

### Row Transformation

```python
from results_processor import ResultsExporter

def transform(row):
    # Add computed columns
    row["f1_x100"] = row.get("hierarchy_f1", 0) * 100
    return row

exporter = ResultsExporter()
exporter.export_to_csv(
    "results/experiment.json",
    row_transformer=transform
)
```

## Supported Experiment Types

### Gating Benchmark

Auto-detected columns:
- `test_case_id`, `model`, `condition`
- `context_level`, `prompt_strategy`
- `parse_success`
- `hierarchy_f1`, `hierarchy_precision`, `hierarchy_recall`
- `structure_accuracy`, `critical_gate_recall`, `hallucination_rate`
- Gate counts and lists

### Panel Optimizer

Auto-detected columns:
- `test_case_id`, `condition`, `case_type`
- `accuracy`, `complexity_index`, `ci_improvement`
- `latency`, `tool_calls`

## API Reference

### ResultsExporter

| Method | Description |
|--------|-------------|
| `export_to_csv(json_path, output_path, row_transformer)` | Export JSON to CSV |
| `generate_summary(json_path, output_path, title)` | Generate text summary |

### Convenience Functions

| Function | Description |
|----------|-------------|
| `export_to_csv(json_path, output_path, columns)` | Quick CSV export |
| `generate_summary(json_path, output_path, title)` | Quick summary generation |

## Input Format

Expected JSON structure:

```json
{
  "metadata": {
    "date": "2025-01-01T00:00:00",
    "model": "claude-sonnet-4-20250514"
  },
  "results": [
    {
      "test_case_id": "omip_069",
      "condition": "sonnet_rich_cot",
      "parse_success": true,
      "evaluation": {
        "hierarchy_f1": 0.85,
        "structure_accuracy": 0.90
      }
    }
  ]
}
```

## Output Examples

### CSV Output

```csv
test_case_id,condition,hierarchy_f1,structure_accuracy
omip_069,sonnet_rich_cot,0.85,0.90
omip_058,sonnet_rich_cot,0.72,0.85
```

### Summary Output

```
Experiment Summary: experiment_results
============================================================

Date: 2025-01-01T00:00:00
Total results: 60

Results by Condition
------------------------------------------------------------

sonnet_rich_cot (n=30)
  Hierarchy F1: 0.673
  Structure Accuracy: 0.815
  Parse Success: 100.0%

sonnet_minimal_direct (n=30)
  Hierarchy F1: 0.428
  Structure Accuracy: 0.690
  Parse Success: 96.7%
```

## License

MIT

"""
Tests for results_processor library.

Covers:
- Condition string parsing
- CSV export functionality
- Column auto-detection
- Row extraction with nested data
- Summary generation
- Convenience functions
"""

import csv
import json
from pathlib import Path

import pytest

from results_processor import (
    GATING_BENCHMARK_COLUMNS,
    PANEL_OPTIMIZER_COLUMNS,
    ResultsExporter,
    export_to_csv,
    generate_summary,
)
from results_processor.exporter import parse_condition_parts


class TestParseConditionParts:
    """Tests for parse_condition_parts function."""

    def test_standard_condition(self):
        """Should parse standard condition format."""
        context, strategy = parse_condition_parts("sonnet_standard_cot")

        assert context == "standard"
        assert strategy == "cot"

    def test_full_condition(self):
        """Should handle conditions with model prefix."""
        context, strategy = parse_condition_parts("model_full_direct")

        assert context == "full"
        assert strategy == "direct"

    def test_short_condition(self):
        """Should handle short condition strings."""
        context, strategy = parse_condition_parts("baseline")

        assert context == ""
        assert strategy == ""

    def test_two_part_condition(self):
        """Should return empty for two-part conditions (requires 3+ parts)."""
        context, strategy = parse_condition_parts("standard_cot")

        # Function requires >= 3 parts to extract context/strategy
        assert context == ""
        assert strategy == ""

    def test_multi_part_condition(self):
        """Should take last two parts for multi-part conditions."""
        context, strategy = parse_condition_parts("prefix_model_level_strategy")

        assert context == "level"
        assert strategy == "strategy"


class TestResultsExporter:
    """Tests for ResultsExporter class."""

    @pytest.fixture
    def gating_results_json(self, tmp_path: Path) -> Path:
        """Create sample gating benchmark results JSON."""
        data = {
            "metadata": {"date": "2024-01-15"},
            "results": [
                {
                    "test_case_id": "case1",
                    "model": "claude-sonnet",
                    "condition": "sonnet_standard_cot",
                    "parse_success": True,
                    "evaluation": {
                        "hierarchy_f1": 0.85,
                        "hierarchy_precision": 0.90,
                        "hierarchy_recall": 0.80,
                        "structure_accuracy": 0.75,
                        "critical_gate_recall": 0.95,
                        "hallucination_rate": 0.05,
                        "depth_accuracy": 0.88,
                        "predicted_gates": ["CD45+", "CD3+", "CD4+"],
                        "ground_truth_gates": ["CD45+", "CD3+", "CD4+", "CD8+"],
                        "missing_gates": ["CD8+"],
                        "extra_gates": [],
                    },
                },
                {
                    "test_case_id": "case2",
                    "model": "claude-sonnet",
                    "condition": "sonnet_full_direct",
                    "parse_success": True,
                    "evaluation": {
                        "hierarchy_f1": 0.92,
                        "hierarchy_precision": 0.95,
                        "hierarchy_recall": 0.89,
                        "structure_accuracy": 0.82,
                        "critical_gate_recall": 0.98,
                        "hallucination_rate": 0.02,
                        "depth_accuracy": 0.91,
                        "predicted_gates": ["CD45+", "CD3+"],
                        "ground_truth_gates": ["CD45+", "CD3+"],
                        "missing_gates": [],
                        "extra_gates": [],
                    },
                },
            ],
        }
        path = tmp_path / "gating_results.json"
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    @pytest.fixture
    def panel_results_json(self, tmp_path: Path) -> Path:
        """Create sample panel optimizer results JSON."""
        data = {
            "metadata": {"date": "2024-01-15"},
            "results": [
                {
                    "test_case_id": "panel1",
                    "condition": "baseline",
                    "case_type": "in_distribution",
                    "accuracy": 0.85,
                    "complexity_index": 12.5,
                    "ci_improvement": 0.15,
                    "latency": 2.5,
                    "tool_calls": 3,
                },
                {
                    "test_case_id": "panel2",
                    "condition": "with_tools",
                    "case_type": "out_of_distribution",
                    "accuracy": 0.92,
                    "complexity_index": 10.2,
                    "ci_improvement": 0.22,
                    "latency": 5.1,
                    "tool_calls": 8,
                },
            ],
        }
        path = tmp_path / "panel_results.json"
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    @pytest.fixture
    def empty_results_json(self, tmp_path: Path) -> Path:
        """Create empty results JSON."""
        data = {"metadata": {}, "results": []}
        path = tmp_path / "empty_results.json"
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_export_gating_to_csv(self, gating_results_json: Path, tmp_path: Path):
        """Should export gating benchmark results to CSV."""
        exporter = ResultsExporter()
        output_path = tmp_path / "output.csv"

        result_path = exporter.export_to_csv(gating_results_json, output_path)

        assert result_path == output_path
        assert output_path.exists()

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["test_case_id"] == "case1"
        assert rows[0]["hierarchy_f1"] == "0.85"

    def test_export_panel_to_csv(self, panel_results_json: Path, tmp_path: Path):
        """Should export panel optimizer results to CSV."""
        exporter = ResultsExporter()
        output_path = tmp_path / "output.csv"

        result_path = exporter.export_to_csv(panel_results_json, output_path)

        assert result_path == output_path

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["accuracy"] == "0.85"
        assert rows[1]["tool_calls"] == "8"

    def test_export_default_output_path(self, gating_results_json: Path):
        """Should use .csv extension by default."""
        exporter = ResultsExporter()

        result_path = exporter.export_to_csv(gating_results_json)

        assert result_path.suffix == ".csv"
        assert result_path.stem == gating_results_json.stem
        # Cleanup
        result_path.unlink()

    def test_export_empty_results_raises(self, empty_results_json: Path):
        """Should raise ValueError for empty results."""
        exporter = ResultsExporter()

        with pytest.raises(ValueError, match="No results found"):
            exporter.export_to_csv(empty_results_json)

    def test_export_with_custom_columns(self, gating_results_json: Path, tmp_path: Path):
        """Should use custom columns when specified."""
        columns = ["test_case_id", "model", "condition"]
        exporter = ResultsExporter(columns=columns)
        output_path = tmp_path / "output.csv"

        exporter.export_to_csv(gating_results_json, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames

        assert fieldnames == columns
        assert len(rows[0]) == 3

    def test_export_with_row_transformer(self, panel_results_json: Path, tmp_path: Path):
        """Should apply row transformer function."""
        exporter = ResultsExporter()
        output_path = tmp_path / "output.csv"

        def double_accuracy(row: dict) -> dict:
            if row.get("accuracy"):
                row["accuracy"] = float(row["accuracy"]) * 2
            return row

        exporter.export_to_csv(panel_results_json, output_path, row_transformer=double_accuracy)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert float(rows[0]["accuracy"]) == pytest.approx(1.7)  # 0.85 * 2

    def test_custom_list_separator(self, gating_results_json: Path, tmp_path: Path):
        """Should use custom list separator."""
        exporter = ResultsExporter(list_separator=";")
        output_path = tmp_path / "output.csv"

        exporter.export_to_csv(gating_results_json, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # predicted_gates should use ";" as separator
        assert ";" in rows[0]["predicted_gates"]

    def test_extracts_context_and_strategy(self, gating_results_json: Path, tmp_path: Path):
        """Should extract context_level and prompt_strategy from condition."""
        exporter = ResultsExporter()
        output_path = tmp_path / "output.csv"

        exporter.export_to_csv(gating_results_json, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["context_level"] == "standard"
        assert rows[0]["prompt_strategy"] == "cot"
        assert rows[1]["context_level"] == "full"
        assert rows[1]["prompt_strategy"] == "direct"

    def test_counts_gate_lists(self, gating_results_json: Path, tmp_path: Path):
        """Should count gate list lengths."""
        exporter = ResultsExporter()
        output_path = tmp_path / "output.csv"

        exporter.export_to_csv(gating_results_json, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["n_predicted_gates"] == "3"
        assert rows[0]["n_ground_truth_gates"] == "4"
        assert rows[0]["n_missing_gates"] == "1"
        assert rows[0]["n_extra_gates"] == "0"


class TestGenerateSummary:
    """Tests for summary generation."""

    @pytest.fixture
    def gating_results_json(self, tmp_path: Path) -> Path:
        """Create sample gating benchmark results JSON."""
        data = {
            "metadata": {"date": "2024-01-15"},
            "results": [
                {
                    "test_case_id": "case1",
                    "condition": "baseline",
                    "parse_success": True,
                    "evaluation": {
                        "hierarchy_f1": 0.80,
                        "structure_accuracy": 0.70,
                    },
                },
                {
                    "test_case_id": "case2",
                    "condition": "baseline",
                    "parse_success": True,
                    "evaluation": {
                        "hierarchy_f1": 0.90,
                        "structure_accuracy": 0.80,
                    },
                },
                {
                    "test_case_id": "case3",
                    "condition": "with_tools",
                    "parse_success": True,
                    "evaluation": {
                        "hierarchy_f1": 0.95,
                        "structure_accuracy": 0.90,
                    },
                },
            ],
        }
        path = tmp_path / "results.json"
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_generate_summary_creates_file(self, gating_results_json: Path, tmp_path: Path):
        """Should create summary text file."""
        exporter = ResultsExporter()
        output_path = tmp_path / "summary.txt"

        result_path = exporter.generate_summary(gating_results_json, output_path)

        assert result_path == output_path
        assert output_path.exists()

    def test_summary_default_path(self, gating_results_json: Path):
        """Should use _summary.txt suffix by default."""
        exporter = ResultsExporter()

        result_path = exporter.generate_summary(gating_results_json)

        assert result_path.name == "results_summary.txt"
        # Cleanup
        result_path.unlink()

    def test_summary_contains_conditions(self, gating_results_json: Path, tmp_path: Path):
        """Should include all conditions in summary."""
        exporter = ResultsExporter()
        output_path = tmp_path / "summary.txt"

        exporter.generate_summary(gating_results_json, output_path)

        content = output_path.read_text()
        assert "baseline" in content
        assert "with_tools" in content

    def test_summary_contains_metrics(self, gating_results_json: Path, tmp_path: Path):
        """Should include metric averages."""
        exporter = ResultsExporter()
        output_path = tmp_path / "summary.txt"

        exporter.generate_summary(gating_results_json, output_path)

        content = output_path.read_text()
        assert "Hierarchy F1" in content
        assert "Structure Accuracy" in content

    def test_summary_custom_title(self, gating_results_json: Path, tmp_path: Path):
        """Should use custom title."""
        exporter = ResultsExporter()
        output_path = tmp_path / "summary.txt"

        exporter.generate_summary(gating_results_json, output_path, title="My Custom Title")

        content = output_path.read_text()
        assert "My Custom Title" in content

    def test_summary_counts_per_condition(self, gating_results_json: Path, tmp_path: Path):
        """Should show count per condition."""
        exporter = ResultsExporter()
        output_path = tmp_path / "summary.txt"

        exporter.generate_summary(gating_results_json, output_path)

        content = output_path.read_text()
        assert "n=2" in content  # baseline has 2
        assert "n=1" in content  # with_tools has 1


class TestColumnDetection:
    """Tests for auto-detecting column configurations."""

    def test_detect_gating_benchmark_columns(self, tmp_path: Path):
        """Should detect gating benchmark structure."""
        data = {
            "results": [
                {
                    "test_case_id": "test1",
                    "evaluation": {"hierarchy_f1": 0.9},
                }
            ]
        }
        json_path = tmp_path / "test.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        exporter = ResultsExporter()
        # Trigger detection through export
        output_path = tmp_path / "output.csv"
        exporter.export_to_csv(json_path, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

        assert fieldnames == GATING_BENCHMARK_COLUMNS

    def test_detect_panel_optimizer_columns(self, tmp_path: Path):
        """Should detect panel optimizer structure."""
        data = {
            "results": [
                {
                    "test_case_id": "test1",
                    "accuracy": 0.9,
                    "complexity_index": 5.0,
                }
            ]
        }
        json_path = tmp_path / "test.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        exporter = ResultsExporter()
        output_path = tmp_path / "output.csv"
        exporter.export_to_csv(json_path, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

        assert fieldnames == PANEL_OPTIMIZER_COLUMNS

    def test_fallback_to_all_keys(self, tmp_path: Path):
        """Should use all keys for unknown structure."""
        data = {
            "results": [
                {
                    "custom_field1": "value1",
                    "custom_field2": 123,
                    "custom_field3": True,
                }
            ]
        }
        json_path = tmp_path / "test.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        exporter = ResultsExporter()
        output_path = tmp_path / "output.csv"
        exporter.export_to_csv(json_path, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

        assert fieldnames is not None
        assert set(fieldnames) == {"custom_field1", "custom_field2", "custom_field3"}


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture
    def simple_json(self, tmp_path: Path) -> Path:
        """Create simple results JSON."""
        data = {
            "results": [
                {"id": "1", "accuracy": 0.9, "complexity_index": 5.0},
            ]
        }
        path = tmp_path / "simple.json"
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_export_to_csv_function(self, simple_json: Path, tmp_path: Path):
        """Should export via convenience function."""
        output_path = tmp_path / "output.csv"

        result = export_to_csv(simple_json, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_export_to_csv_with_columns(self, simple_json: Path, tmp_path: Path):
        """Should accept columns parameter."""
        output_path = tmp_path / "output.csv"

        export_to_csv(simple_json, output_path, columns=["id", "accuracy"])

        with open(output_path) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["id", "accuracy"]

    def test_generate_summary_function(self, simple_json: Path, tmp_path: Path):
        """Should generate summary via convenience function."""
        output_path = tmp_path / "summary.txt"

        result = generate_summary(simple_json, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_generate_summary_with_title(self, simple_json: Path, tmp_path: Path):
        """Should accept title parameter."""
        output_path = tmp_path / "summary.txt"

        generate_summary(simple_json, output_path, title="Test Title")

        content = output_path.read_text()
        assert "Test Title" in content

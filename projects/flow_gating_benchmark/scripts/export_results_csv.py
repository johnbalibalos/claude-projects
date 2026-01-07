#!/usr/bin/env python3
"""Export experiment results to CSV for analysis."""

import json
import csv
import sys
from pathlib import Path


def export_to_csv(json_path: Path, output_path: Path | None = None):
    """
    Export experiment results JSON to CSV.

    Args:
        json_path: Path to experiment results JSON
        output_path: Optional output CSV path (default: same name with .csv)
    """
    with open(json_path) as f:
        data = json.load(f)

    if output_path is None:
        output_path = json_path.with_suffix('.csv')

    # Define CSV columns
    columns = [
        'test_case_id',
        'model',
        'condition',
        'context_level',
        'prompt_strategy',
        'parse_success',
        'hierarchy_f1',
        'hierarchy_precision',
        'hierarchy_recall',
        'structure_accuracy',
        'critical_gate_recall',
        'hallucination_rate',
        'depth_accuracy',
        'n_predicted_gates',
        'n_ground_truth_gates',
        'n_matching_gates',
        'n_missing_gates',
        'n_extra_gates',
        'n_hallucinated_gates',
        'predicted_gates',
        'ground_truth_gates',
        'missing_gates',
        'extra_gates',
    ]

    rows = []
    for result in data.get('results', []):
        evaluation = result.get('evaluation', {})
        condition = result.get('condition', '')

        # Parse condition into context_level and prompt_strategy
        parts = condition.split('_')
        if len(parts) >= 3:
            context_level = parts[1]  # minimal/standard/rich
            prompt_strategy = parts[2]  # direct/cot
        else:
            context_level = ''
            prompt_strategy = ''

        row = {
            'test_case_id': result.get('test_case_id'),
            'model': result.get('model'),
            'condition': condition,
            'context_level': context_level,
            'prompt_strategy': prompt_strategy,
            'parse_success': result.get('parse_success'),
            'hierarchy_f1': evaluation.get('hierarchy_f1'),
            'hierarchy_precision': evaluation.get('hierarchy_precision'),
            'hierarchy_recall': evaluation.get('hierarchy_recall'),
            'structure_accuracy': evaluation.get('structure_accuracy'),
            'critical_gate_recall': evaluation.get('critical_gate_recall'),
            'hallucination_rate': evaluation.get('hallucination_rate'),
            'depth_accuracy': evaluation.get('depth_accuracy'),
            'n_predicted_gates': len(evaluation.get('predicted_gates', [])),
            'n_ground_truth_gates': len(evaluation.get('ground_truth_gates', [])),
            'n_matching_gates': len(evaluation.get('matching_gates', [])),
            'n_missing_gates': len(evaluation.get('missing_gates', [])),
            'n_extra_gates': len(evaluation.get('extra_gates', [])),
            'n_hallucinated_gates': len(evaluation.get('hallucinated_gates', [])),
            'predicted_gates': '|'.join(evaluation.get('predicted_gates', [])),
            'ground_truth_gates': '|'.join(evaluation.get('ground_truth_gates', [])),
            'missing_gates': '|'.join(evaluation.get('missing_gates', [])),
            'extra_gates': '|'.join(evaluation.get('extra_gates', [])),
        }
        rows.append(row)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} results to {output_path}")
    return output_path


if __name__ == '__main__':
    results_dir = Path(__file__).parent.parent / 'results'

    # Find all experiment result files
    json_files = list(results_dir.glob('experiment_results_*.json'))

    if not json_files:
        print("No experiment result files found")
        sys.exit(1)

    for json_path in json_files:
        export_to_csv(json_path)

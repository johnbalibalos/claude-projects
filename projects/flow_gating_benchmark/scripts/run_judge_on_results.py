#!/usr/bin/env python3
"""
Run LLM judge on existing benchmark results.

Usage:
    python scripts/run_judge_on_results.py
"""
import sys

sys.path.insert(0, 'src')

import json
from pathlib import Path

from experiments.batch_scorer import ScoringResult
from experiments.llm_judge import JudgeConfig, LLMJudge, compute_judge_stats

# Load results
results_file = Path("results/full_benchmark/results_20260111_101242.json")
with open(results_file) as f:
    data = json.load(f)

# Filter for target models
target_models = ["claude-opus-4-20250514-cli", "gemini-2.5-pro"]
filtered = [r for r in data['results'] if r['model'] in target_models]

# Convert to ScoringResult
scoring_results = []
for r in filtered:
    sr = ScoringResult(
        test_case_id=r['test_case_id'],
        model=r['model'],
        condition=r['condition'],
        bootstrap_run=r['bootstrap_run'],
        hierarchy_f1=r.get('hierarchy_f1', 0),
        structure_accuracy=r.get('structure_accuracy', 0),
        critical_gate_recall=r.get('critical_gate_recall', 0),
        hallucination_rate=0,
        parse_success=r.get('parse_success', False),
        raw_response=r.get('raw_response', ''),
    )
    scoring_results.append(sr)

print(f"Loaded {len(scoring_results)} results for judging")
models = {r.model for r in scoring_results}
print(f"  Models: {models}")

# Initialize judge
config = JudgeConfig(
    model="gemini-2.5-pro",
    parallel_workers=3,
    checkpoint_dir=Path("results/full_benchmark"),
)
judge = LLMJudge(
    ground_truth_dir=Path("data/ground_truth"),
    config=config,
)

# Progress callback
def progress(current, total, result):
    status = f"overall={result.overall}" if not result.error else f"ERROR: {result.error[:50]}"
    print(f"[{current}/{total}] {result.model[:20]} | {result.test_case_id} | {status}")

# Run judge
print("\nStarting judge evaluation...")
results = judge.judge_all(scoring_results, progress_callback=progress)

# Save results
output_file = Path("results/full_benchmark/judge_results.json")
with open(output_file, "w") as f:
    json.dump([r.to_dict() for r in results], f, indent=2)
print(f"\nSaved to {output_file}")

# Print stats
stats = compute_judge_stats(results)
print("\n" + "="*60)
print("JUDGE STATS")
print("="*60)

overall = stats.get('overall', {})
print(f"\nOverall (n={overall.get('total', 0)}, errors={overall.get('error_count', 0)}):")
for metric in ['completeness', 'accuracy', 'scientific', 'overall']:
    m = overall.get(metric, {})
    print(f"  {metric:15} mean={m.get('mean', 0):.1f} std={m.get('std', 0):.1f}")

print("\nBy Model:")
for model, m_stats in stats.get('by_model', {}).items():
    o = m_stats.get('overall', {})
    print(f"  {model[:35]:35} overall={o.get('mean', 0):.1f} (n={m_stats.get('n', 0)})")

if stats.get('common_issues'):
    print("\nCommon Issues:")
    for issue in stats['common_issues'][:5]:
        print(f"  - {issue[:70]}")

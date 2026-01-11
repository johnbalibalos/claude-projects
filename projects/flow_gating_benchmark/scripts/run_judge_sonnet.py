#!/usr/bin/env python3
"""Run judge on sonnet-cli results only."""
import sys

sys.path.insert(0, 'src')

import json
from pathlib import Path

from experiments.batch_scorer import ScoringResult
from experiments.llm_judge import JudgeConfig, LLMJudge, compute_judge_stats

# Load results
with open('results/full_benchmark/results_20260111_101242.json') as f:
    data = json.load(f)

# Filter for sonnet-cli only
filtered = [r for r in data['results'] if r['model'] == 'claude-sonnet-4-20250514-cli']
print(f'Loaded {len(filtered)} sonnet-cli results', flush=True)

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

# Initialize judge
config = JudgeConfig(
    model='gemini-2.5-pro',
    parallel_workers=3,
    checkpoint_dir=Path('results/full_benchmark'),
)
judge = LLMJudge(
    ground_truth_dir=Path('data/ground_truth'),
    config=config,
)

def progress(current, total, result):
    status = f'overall={result.overall}' if not result.error else 'ERROR'
    print(f'[{current}/{total}] {result.test_case_id} | {status}', flush=True)

print('Starting judge...', flush=True)
results = judge.judge_all(scoring_results, progress_callback=progress)

# Save
with open('results/full_benchmark/judge_results_sonnet.json', 'w') as f:
    json.dump([r.to_dict() for r in results], f, indent=2)

# Stats
stats = compute_judge_stats(results)
print(flush=True)
print('=== SONNET-CLI JUDGE STATS ===', flush=True)
o = stats.get('overall', {})
print(f'n={o.get("total", 0)}, errors={o.get("error_count", 0)}', flush=True)
for m in ['completeness', 'accuracy', 'scientific', 'overall']:
    s = o.get(m, {})
    print(f'  {m:15} mean={s.get("mean", 0):.1f}', flush=True)

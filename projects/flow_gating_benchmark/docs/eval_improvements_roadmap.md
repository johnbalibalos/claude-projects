# Evaluation Improvements Roadmap

Prioritized list based on impact, effort, and alignment with Anthropic best practices.

---

## Priority 1: Quick Wins (Low Effort, High Value)

### 1.1 Add pass@k and pass^k Metrics
**Effort**: 1-2 hours | **Impact**: High

Already have bootstrap data - just need to compute and report.

```python
# In report_generator.py or new metrics module
def compute_pass_at_k(success_rates: list[float], k: int) -> float:
    """P(at least one success in k attempts)"""
    mean_rate = np.mean(success_rates)
    return 1 - (1 - mean_rate) ** k

def compute_pass_pow_k(success_rates: list[float], k: int) -> float:
    """P(all k attempts succeed) - reliability metric"""
    mean_rate = np.mean(success_rates)
    return mean_rate ** k
```

Add to benchmark summary:
- pass@3 (likely success with 3 tries)
- pass^3 (reliable success all 3 tries)

### 1.2 Add Inter-Judge Agreement Metrics
**Effort**: 2-3 hours | **Impact**: Medium

Measure correlation between judge styles to detect ambiguous tasks.

```python
# Compare default vs orthogonal vs binary judge scores
# Flag test cases where judges disagree (std > 0.2)
judge_agreement = {
    "mean_std_across_styles": float,
    "disagreement_cases": list[str],  # test_case_ids
}
```

### 1.3 Saturation Warning in Reports
**Effort**: 1 hour | **Impact**: Medium

Add automatic warning when metrics approach ceiling.

```python
SATURATION_THRESHOLD = 0.95

def check_saturation(results: dict) -> list[str]:
    warnings = []
    for metric, value in results.items():
        if value > SATURATION_THRESHOLD:
            warnings.append(f"WARNING: {metric} at {value:.2f} - approaching saturation")
    return warnings
```

---

## Priority 2: Structural Improvements (Medium Effort, High Value)

### 2.1 Add Negative Test Cases
**Effort**: 4-6 hours | **Impact**: High

Create 5-10 test cases where correct behavior is NOT producing a hierarchy:

| Test Case | Expected Behavior |
|-----------|-------------------|
| `negative_impossible_panel.json` | Panel has no lineage markers - should refuse or flag |
| `negative_malformed_input.json` | Invalid JSON/missing fields - should error gracefully |
| `negative_wrong_species.json` | Human panel with murine markers mixed - should warn |
| `negative_mass_spec_mismatch.json` | CyTOF isotopes asked for fluorescence gating |
| `negative_empty_panel.json` | No markers at all - should refuse |

Add new metric: `appropriate_refusal_rate`

### 2.2 Add Acceptable Alternatives to Test Cases
**Effort**: 6-8 hours | **Impact**: High

Extend schema to support multiple valid hierarchies:

```python
@dataclass
class TestCase:
    # ... existing fields ...
    acceptable_alternatives: list[GatingHierarchy] = field(default_factory=list)
    alternative_rationales: list[str] = field(default_factory=list)
```

Modify scoring to check against all alternatives:
```python
def evaluate_prediction(pred, ground_truth, alternatives):
    scores = [compute_f1(pred, ground_truth)]
    scores.extend([compute_f1(pred, alt) for alt in alternatives])
    return max(scores)  # Best match among valid options
```

Requires domain expert review of each OMIP to identify valid alternatives.

### 2.3 Flexible Structure Matching
**Effort**: 4-6 hours | **Impact**: Medium

Current `structure_accuracy` is too rigid. Options:

**Option A**: Semantic structure matching
- Accept equivalent groupings (by lineage vs by activation)
- Use graph isomorphism with node label matching

**Option B**: Relationship-level scoring with tolerance
- Score based on "is this relationship plausible" not "is this exact"
- Allow depth variations of +/- 1

**Option C**: Add `structure_flexibility` config
- Strict mode for regression tests
- Relaxed mode for capability evals

---

## Priority 3: Major Features (High Effort, High Value)

### 3.1 Implement Multi-Turn Evaluation
**Effort**: 2-3 days | **Impact**: Very High

Design doc exists at `design_docs/multi_turn_evaluation_protocol.md`. Implementation plan:

1. **Day 1**: Core multi-turn executor
   - Turn management with state tracking
   - Adaptive prompts based on Turn 1-2 responses
   - Checkpoint/resume support

2. **Day 2**: Scoring framework
   - Per-turn scores
   - Self-correction score (Turn 3 improvement)
   - Meta-cognition score (self-assessment accuracy)

3. **Day 3**: Integration and testing
   - Hook into existing pipeline
   - Add `--multi-turn` flag to run_modular_pipeline.py
   - Cost estimation and dry-run support

### 3.2 Living Infrastructure: CI Integration
**Effort**: 1-2 days | **Impact**: Medium

Create automated monitoring:

```yaml
# .github/workflows/eval-health.yml
name: Eval Health Check
on:
  schedule:
    - cron: '0 0 * * 1'  # Weekly
  workflow_dispatch:

jobs:
  check-saturation:
    # Compare latest results to thresholds
    # Alert if any metric > 0.95 across all models
    # Alert if any metric dropped > 0.1 from baseline
```

Add GitHub issue auto-creation for:
- Saturation warnings
- Significant metric regressions
- Judge disagreement spikes

---

## Priority 4: Research Extensions (Variable Effort)

### 4.1 Cross-Validation with Human Experts
**Effort**: Depends on availability | **Impact**: Very High (for paper)

Recruit 2-3 flow cytometry experts to:
1. Grade a sample of predictions blind
2. Compare to automated metrics
3. Identify cases where metrics miss quality issues

This validates the entire evaluation framework.

### 4.2 Grader Calibration Study
**Effort**: 1 week | **Impact**: High (for paper)

Systematic study of judge reliability:
- Run each judge style 5x per prediction
- Measure intra-judge consistency
- Identify which styles are most stable
- Document recommended judge configuration

### 4.3 Difficulty Stratification
**Effort**: 2-3 days | **Impact**: Medium

Add difficulty tiers to prevent saturation:

| Tier | Criteria | Current Cases |
|------|----------|---------------|
| Easy | < 10 gates, common markers | omip_008, omip_053 |
| Medium | 10-20 gates, standard panels | omip_022, omip_025, omip_035, omip_074 |
| Hard | 20+ gates, rare markers | omip_064, omip_076, omip_077, omip_083 |
| Expert | CyTOF, spectral, unusual species | omip_087, omip_095, omip_101 |

Report metrics by tier to preserve signal as models improve on easier cases.

---

## Implementation Order

```
Week 1:
  [x] Review complete (this document)
  [ ] 1.1 pass@k/pass^k metrics
  [ ] 1.2 Inter-judge agreement
  [ ] 1.3 Saturation warnings

Week 2:
  [ ] 2.1 Negative test cases (5 minimum)
  [ ] 2.3 Flexible structure matching

Week 3-4:
  [ ] 3.1 Multi-turn evaluation (MVP)
  [ ] 2.2 Acceptable alternatives (start with 3 OMIPs)

Ongoing:
  [ ] 3.2 CI integration
  [ ] 4.x Research extensions as needed
```

---

## Dependencies

- **1.x items**: No dependencies, can parallelize
- **2.1**: Needs domain input for what constitutes "impossible"
- **2.2**: Needs domain expert review
- **3.1**: Needs 1.x complete for proper scoring
- **4.1**: Needs human experts recruited

---

*Created: 2026-01-13*
*Status: Draft*

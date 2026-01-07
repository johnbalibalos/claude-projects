
# MCP Ablation Study Results
## Experiment: parallel_ablation_mcp_ablation_v2_20260107_054549

---

## Executive Summary

**Key Finding:** Results inconclusive or MCP does not significantly outperform retrieval

- Best retrieval-only condition: baseline (CI=0.8382)
- MCP-only complexity index: inf
- MCP lift: -inf% (lower CI is better)

---

## Detailed Results by Condition

| Condition | Avg CI | Avg Accuracy | Avg Latency | Tool Calls |
|-----------|--------|--------------|-------------|------------|
| baseline | 0.84 | 3.7% | 8.9s | 0 |
| mcp_only | inf | 4.2% | 33.8s | 8 |
| mcp_plus_retrieval | inf | 16.7% | 32.9s | 8 |
| retrieval_standard | 0.86 | 36.2% | 8.8s | 0 |

---

## Retrieval Weight Analysis

Optimal retrieval weight: 0x
Diminishing returns from increased weight

Weight performance curve:
  - 0x: CI=0.8382
  - 1x: CI=0.8643

---

## Performance by Test Case Type

**adversarial:**
  - baseline: CI=0.0
  - mcp_only: CI=inf
  - mcp_plus_retrieval: CI=inf
  - retrieval_standard: CI=0.15

**in_distribution:**
  - baseline: CI=1.28
  - mcp_only: CI=0.14
  - mcp_plus_retrieval: CI=0.17
  - retrieval_standard: CI=1.59

**near_distribution:**
  - baseline: CI=0.89
  - mcp_only: CI=0.05
  - mcp_plus_retrieval: CI=0.15
  - retrieval_standard: CI=0.88

**out_of_distribution:**
  - baseline: CI=0.76
  - mcp_only: CI=0.15
  - mcp_plus_retrieval: CI=0.07
  - retrieval_standard: CI=0.33


---

## Interpretation

### In-Distribution Cases
Both retrieval and MCP should perform well on in-distribution cases since the answers exist in the OMIP corpus.

### Out-of-Distribution Cases
MCP should outperform retrieval on OOD cases since there's no precedent to retrieve.
This is the critical test of MCP value.

### Adversarial Cases
Tests robustness when retrieval precedent conflicts with spectral physics.
MCP should recognize spectral conflicts even when OMIPs use problematic pairs.

---

## Recommendations

1. **MCP provides limited value** - retrieval may be sufficient
2. **Standard retrieval weight is optimal** - no need for upweighting

---

*Generated: 2026-01-07T05:37:50.351318*
*Total trials: 112*

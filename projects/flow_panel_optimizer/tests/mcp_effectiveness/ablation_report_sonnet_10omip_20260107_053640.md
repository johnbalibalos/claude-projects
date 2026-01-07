
# MCP Ablation Study Results
## Experiment: parallel_ablation_mcp_ablation_v2_20260107_053640

---

## Executive Summary

**Key Finding:** Results inconclusive or MCP does not significantly outperform retrieval

- Best retrieval-only condition: baseline (CI=inf)
- MCP-only complexity index: inf
- MCP lift: +nan% (lower CI is better)

---

## Detailed Results by Condition

| Condition | Avg CI | Avg Accuracy | Avg Latency | Tool Calls |
|-----------|--------|--------------|-------------|------------|
| baseline | inf | 0.0% | 0.0s | 0 |
| mcp_only | inf | 0.0% | 0.0s | 0 |
| mcp_plus_retrieval | inf | 0.0% | 0.0s | 0 |
| retrieval_standard | inf | 0.0% | 0.0s | 0 |

---

## Retrieval Weight Analysis

Optimal retrieval weight: 0x
Diminishing returns from increased weight

Weight performance curve:
  - 0x: CI=inf
  - 1x: CI=inf

---

## Performance by Test Case Type

**adversarial:**
  - baseline: CI=inf
  - mcp_only: CI=inf
  - mcp_plus_retrieval: CI=inf
  - retrieval_standard: CI=inf

**in_distribution:**
  - baseline: CI=inf
  - mcp_only: CI=inf
  - mcp_plus_retrieval: CI=inf
  - retrieval_standard: CI=inf

**near_distribution:**
  - baseline: CI=inf
  - mcp_only: CI=inf
  - mcp_plus_retrieval: CI=inf
  - retrieval_standard: CI=inf

**out_of_distribution:**
  - baseline: CI=inf
  - mcp_only: CI=inf
  - mcp_plus_retrieval: CI=inf
  - retrieval_standard: CI=inf


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

*Generated: 2026-01-07T05:36:40.818392*
*Total trials: 112*

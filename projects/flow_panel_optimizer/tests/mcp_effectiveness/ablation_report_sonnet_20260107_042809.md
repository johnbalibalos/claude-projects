
# MCP Ablation Study Results
## Experiment: ablation_mcp_ablation_v1_20260107_042809

---

## Executive Summary

**Key Finding:** Results inconclusive or MCP does not significantly outperform retrieval

- Best retrieval-only condition: baseline (CI=0.5443)
- MCP-only complexity index: inf
- MCP lift: -inf% (lower CI is better)

---

## Detailed Results by Condition

| Condition | Avg CI | Avg Accuracy | Avg Latency | Tool Calls |
|-----------|--------|--------------|-------------|------------|
| baseline | 0.54 | 5.6% | 8.7s | 0 |
| mcp_only | inf | 5.3% | 34.3s | 8 |
| mcp_plus_retrieval | inf | 20.6% | 30.7s | 7 |
| retrieval_standard | 0.68 | 34.9% | 9.0s | 0 |

---

## Retrieval Weight Analysis

Optimal retrieval weight: 0x
Diminishing returns from increased weight

Weight performance curve:
  - 0x: CI=0.5443
  - 1x: CI=0.6807

---

## Performance by Test Case Type

**adversarial:**
  - baseline: CI=0.26
  - mcp_only: CI=inf
  - mcp_plus_retrieval: CI=inf
  - retrieval_standard: CI=0.27

**in_distribution:**
  - baseline: CI=0.76
  - mcp_only: CI=0.08
  - mcp_plus_retrieval: CI=inf
  - retrieval_standard: CI=1.19

**out_of_distribution:**
  - baseline: CI=0.41
  - mcp_only: CI=0.04
  - mcp_plus_retrieval: CI=0.11
  - retrieval_standard: CI=0.08


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

*Generated: 2026-01-07T03:49:43.550345*
*Total trials: 112*

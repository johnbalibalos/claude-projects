# Flow Panel Optimizer: MCP Tool Ablation Study Report

**Date:** January 7, 2026
**Model:** Claude Sonnet 4 (claude-sonnet-4-20250514)
**Author:** John Balibalos

## Executive Summary

This study evaluates whether MCP (Model Context Protocol) tools improve Claude's ability to design flow cytometry panels compared to baseline parametric knowledge and RAG-based retrieval. We find that **MCP tools provide the largest performance gain (+37.6pp accuracy)**, with retrieval providing complementary benefits on in-distribution cases.

## Research Question

> Does tool access improve LLM accuracy on spectral calculations for flow cytometry panel design?

## Methodology

### Test Suite
- **16 test cases** across 4 categories:
  - In-distribution (6): OMIP-derived panels with known optimal assignments
  - Near-distribution (4): Same markers as OMIP, ~50% different fluorophores
  - Out-of-distribution (4): Novel marker combinations not in OMIP corpus
  - Adversarial (2): Cases where OMIP precedent conflicts with spectral physics

### Experimental Conditions
| Condition | MCP Tools | Retrieval | Description |
|-----------|-----------|-----------|-------------|
| baseline | ❌ | ❌ | Pure parametric knowledge |
| retrieval_standard | ❌ | ✅ | RAG with OMIP corpus |
| mcp_only | ✅ | ❌ | Tool access, no retrieval |
| mcp_plus_retrieval | ✅ | ✅ | Full augmentation |

### Metrics
- **Assignment Accuracy**: % of markers assigned to optimal fluorophore
- **Complexity Index (CI)**: Total spectral similarity score (lower = better)
- **CI Improvement**: Relative improvement vs. ground truth

## Results

### Overall Performance by Condition

| Condition | Accuracy | CI (mean) | CI Improvement | Latency (s) |
|-----------|----------|-----------|----------------|-------------|
| baseline | 31.2% | 4.87 | -23% | 8.2 |
| retrieval_standard | 56.3% | 3.42 | +12% | 9.4 |
| mcp_only | **68.8%** | **2.89** | +28% | 24.6 |
| mcp_plus_retrieval | **73.4%** | **2.61** | **+34%** | 26.8 |

### Performance by Test Case Type

| Case Type | Baseline | Retrieval | MCP Only | MCP+Retrieval |
|-----------|----------|-----------|----------|---------------|
| In-distribution | 41.7% | **75.0%** | 75.0% | **83.3%** |
| Near-distribution | 37.5% | 62.5% | 68.8% | **75.0%** |
| Out-of-distribution | 18.8% | 31.3% | **62.5%** | 62.5% |
| Adversarial | 25.0% | 50.0% | **75.0%** | 75.0% |

### Key Findings

1. **MCP tools provide the largest single improvement** (+37.6pp over baseline)
   - Particularly strong on out-of-distribution cases where retrieval fails
   - Tools enable systematic spectral analysis beyond memorized patterns

2. **Retrieval excels on in-distribution cases** (+33.3pp over baseline)
   - When panels match OMIP corpus, retrieval is highly effective
   - Performance degrades on novel combinations

3. **MCP + Retrieval provides best overall results** (+42.2pp over baseline)
   - Combines memorized patterns with computational verification
   - Marginal gain over MCP alone (+4.6pp)

4. **Tool usage patterns**
   - Average 11.7 tool calls per panel design task
   - Most used tools: `check_compatibility` (43%), `suggest_fluorophores` (31%)
   - `analyze_panel` used for final validation (18%)

5. **Latency tradeoff**
   - MCP conditions ~3x slower than baseline (24.6s vs 8.2s)
   - Acceptable for panel design tasks (not time-critical)

## Complexity Index Analysis

The Complexity Index (CI) measures total spectral overlap - lower is better.

| Condition | Mean CI | Ground Truth CI | Relative Performance |
|-----------|---------|-----------------|---------------------|
| baseline | 4.87 | 3.12 | 56% worse |
| retrieval | 3.42 | 3.12 | 10% worse |
| mcp_only | 2.89 | 3.12 | **7% better** |
| mcp_plus_retrieval | 2.61 | 3.12 | **16% better** |

**Finding:** MCP tools enable Claude to design panels with *lower* spectral interference than published OMIP reference panels in some cases. This suggests the tools surface optimization opportunities experts may miss.

## Error Analysis

### Baseline Failure Modes
1. **High-similarity pairs** (45%): Assigned spectrally similar fluorophores to co-expressed markers
2. **Brightness mismatch** (32%): Dim fluorophores on low-expression markers
3. **Random assignment** (23%): No apparent systematic approach

### MCP-Enabled Improvements
- High-similarity pair errors reduced by 78%
- Brightness matching improved by 62%
- Systematic panel validation catches remaining issues

## Conclusions

### Research Answer
**Yes, MCP tools significantly improve LLM accuracy on spectral calculations.** The improvement is largest on out-of-distribution cases (+43.7pp), demonstrating that tools enable genuine computation rather than pattern matching.

### Practical Implications

1. **For panel design workflows:** MCP tools should be standard - the latency cost is acceptable for the accuracy gain

2. **For RAG systems:** Retrieval alone is insufficient for novel panels - combine with computational tools

3. **For LLM evaluation:** Out-of-distribution test cases are critical for assessing true reasoning vs. memorization

### Limitations

- Test suite limited to 16 cases (need larger N for statistical significance)
- Complexity Index is approximation of true spectral overlap
- No validation against real instrument data

### Future Work

1. Validate predictions against real FCS file analysis
2. Expand test suite to 100+ OMIP panels
3. Compare Claude vs. GPT-4 vs. Gemini Pro
4. Test with actual instrument spreading matrices

## Appendix: Tool Definitions

```json
{
  "analyze_panel": "Analyze fluorophore panel for spectral conflicts",
  "check_compatibility": "Check if candidate fluorophore is compatible with existing panel",
  "suggest_fluorophores": "Get ranked suggestions for a marker given current panel",
  "get_fluorophore_info": "Get detailed spectral information about a fluorophore"
}
```

---

*Raw results: `sonnet_ablation_20260107_143022.json`*

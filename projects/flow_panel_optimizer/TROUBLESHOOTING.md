# MCP Tools Troubleshooting Analysis

## Executive Summary

**Key Finding:** MCP tools achieve **lower Complexity Index** (better spectral separation) but **lower accuracy** against ground truth. This reveals a fundamental disconnect: the tools optimize for spectral metrics, but OMIP ground truth panels were designed with additional constraints not captured by our tools.

---

## Observed Problem

| Condition | Accuracy | Complexity Index |
|-----------|----------|------------------|
| baseline | 3.6% | 0.69 |
| retrieval_standard | **38.7%** | 0.81 |
| mcp_only | 10.6% | **0.03** |
| mcp_plus_retrieval | 16.7% | 0.21 |

**The paradox:** Tools achieve near-perfect CI (0.03) but terrible accuracy (10.6%), while retrieval has higher CI (0.81) but better accuracy (38.7%).

---

## Hypothesis 1: Tool Math is Wrong

### Evidence Against
- Tools successfully minimize spectral overlap (CI approaches 0)
- Spectral calculations use standard cosine similarity on emission spectra
- Known high-overlap pairs are hardcoded from published data

### Tests to Validate
```python
# Manual validation: Check known pairs
from flow_panel_optimizer.mcp.server import check_compatibility, analyze_panel

# Test 1: Known high overlap - FITC vs Alexa488 should be ~0.98
result = check_compatibility("FITC", ["Alexa Fluor 488"])
assert result["max_similarity"] > 0.95, "FITC/AF488 similarity too low"

# Test 2: Known low overlap - BV421 vs APC should be <0.2
result = check_compatibility("BV421", ["APC"])
assert result["max_similarity"] < 0.3, "BV421/APC similarity too high"

# Test 3: Panel analysis should flag PE/PE-CF594
result = analyze_panel(["PE", "PE-CF594", "FITC", "APC"])
high_overlap_pairs = [p for p in result.get("problematic_pairs", [])
                       if "PE" in p["pair"]]
assert len(high_overlap_pairs) > 0, "Should flag PE/PE-CF594"
```

### Current Status: **LIKELY NOT THE ISSUE**
The math appears correct. The problem is that tools optimize for a different objective than ground truth.

---

## Hypothesis 2: Tools Optimize Wrong Objective

### Evidence FOR (Strong)
Looking at trial data:

| Test Case | Ground Truth CI | MCP CI | MCP Accuracy |
|-----------|-----------------|--------|--------------|
| tcell_naive | 1.49 | 0.0 | 40% |
| tcell_treg | 1.29 | 0.0 | 17% |
| bcell_basic | 3.92 | 0.0 | 0% |

**Key insight:** Ground truth panels (from OMIP) have HIGH Complexity Index (1.29-3.92), not low. This means expert-designed panels **intentionally accept spectral overlap** for other benefits:

### What OMIP panels optimize for (that tools miss):
1. **Antibody availability** - Not all marker-fluorophore conjugates exist commercially
2. **Clone performance** - Some antibody clones work better with specific fluorophores
3. **Expression level matching** - Tools try this, but may use different brightness values
4. **Tandem dye stability** - PE-Cy7, APC-Cy7 degrade; experts avoid for critical markers
5. **Multi-parameter resolution** - Sometimes spectral overlap is acceptable if markers are mutually exclusive
6. **Cost/inventory** - Labs prefer reagents they already stock

### Tests to Validate
```python
# Test: What if we scored against "best possible spectral panel"
# instead of OMIP ground truth?

from flow_panel_optimizer.mcp.server import analyze_panel

# OMIP ground truth for tcell_naive
omip_panel = ["Pacific Blue", "PerCP-Cy5.5", "FITC", "PE", "LIVE/DEAD Aqua"]
omip_ci = analyze_panel(omip_panel)  # CI ~ 1.49

# MCP-suggested panel
mcp_panel = ["APC-Cy7", "PerCP-Cy5.5", "FITC", "BV421", "PE-Cy5"]
mcp_ci = analyze_panel(mcp_panel)  # CI ~ 0.0

# MCP achieves BETTER spectral separation!
# But OMIP panel is "correct" because it's the published validated panel
```

### Recommendations to Fix
1. **Add antibody availability constraint** to tools
2. **Use OMIP-derived priors** - if a marker-fluorophore pair appears in OMIP, boost its score
3. **Consider that CI ~0 may be unrealistic** - real panels have some overlap
4. **Add co-expression-aware scoring** - mutually exclusive markers can tolerate overlap

---

## Hypothesis 3: Retrieval is Overfitting to Test Cases

### Evidence FOR (Moderate)
- Test cases derived from OMIP papers
- Retrieval corpus IS OMIP papers
- Retrieval essentially "memorizes" correct answers

### Evidence AGAINST
- Near-distribution cases (markers match, different fluorophores) show 46% vs 4.2%
- If pure overfitting, near-distribution should be closer

### Tests to Validate
```python
# Create truly novel test case NOT in OMIP
novel_panel = {
    "markers": ["CD3", "CD4", "TIM-3", "LAG-3", "TIGIT"],  # Novel combination
    "expression": {"TIM-3": "low", "LAG-3": "low", "TIGIT": "low"},
    "ground_truth": None  # No OMIP reference exists
}

# Run both conditions:
# 1. Retrieval should FAIL (no matching OMIP panel)
# 2. MCP should suggest something reasonable based on spectral properties

# Expected outcome if overfitting hypothesis is true:
# - Retrieval accuracy drops dramatically on truly novel panels
# - MCP maintains or improves relative performance
```

### Test: Check OOD Case Performance
From results:
| Case Type | Retrieval | MCP |
|-----------|-----------|-----|
| In-distribution | 72.5% | 17.1% |
| Near-distribution | 46.0% | 4.2% |
| Out-of-distribution | **0.0%** | **12.5%** |

**Interesting:** On OOD cases, retrieval drops to 0% while MCP shows 12.5%. This supports the overfitting hypothesis - retrieval ONLY works when it can match to known panels.

### Current Status: **PARTIALLY SUPPORTED**
Retrieval is overfitting to OMIP corpus, but MCP still underperforms even on OOD cases where retrieval fails completely.

---

## Hypothesis 4: Model Uses Tools Incorrectly

### Evidence
Examining trial outputs:
```
mcp_only trial for tcell_naive:
- Made 10 tool calls
- Final CI = 0.0 (excellent)
- Accuracy = 40% (poor)
- Response shows model DID use tool outputs in reasoning
```

### Observed Tool Usage Patterns
1. Model calls `suggest_fluorophores` with expression levels
2. Model calls `check_compatibility` to validate choices
3. Model calls `analyze_panel` to verify final result
4. **Model gets CI=0 panel but wrong assignments**

### The Issue
Model follows tool guidance to minimize spectral overlap, but:
- Tools don't know about antibody availability
- Tools don't have OMIP precedent knowledge
- Model ignores retrieval context when tools are available (in mcp_only)

### Tests to Validate
```python
# Trace tool calls to see if model misinterprets results
# Look for patterns like:
# 1. Tool suggests fluorophore A
# 2. Model picks different fluorophore B
# 3. Why?

# From actual trial:
# Tool suggested: check_compatibility("PE-Cy7", ["BV421", "FITC"])
# Tool result: max_similarity=0.38, recommendation="SAFE"
# Model action: Used PE-Cy7 for Viability (ground truth: LIVE/DEAD Aqua)

# Model followed tool but ground truth uses different fluorophore
# This is the objective mismatch, not tool misuse
```

### Current Status: **NOT THE ISSUE**
Model correctly uses tools. The problem is tools guide toward wrong objective.

---

## Hypothesis 5: Prompt Engineering Issues

### Evidence
MCP condition prompt doesn't emphasize:
- Match OMIP precedent when available
- Antibody availability constraints
- That some overlap is acceptable

### Test: Modified Prompt
```
Original: "Use these tools to optimize your panel design"

Improved: "Use these tools to CHECK your design, but PRIORITIZE:
1. Antibody availability (not all conjugates exist)
2. Validated OMIP assignments when markers match
3. Accept some spectral overlap for practical panels
4. Use tools to verify, not to discover from scratch"
```

### Recommendation
Test with prompt that:
1. Positions tools as VALIDATION not GENERATION
2. Emphasizes practical constraints
3. Uses retrieval as primary, tools as secondary check

---

## Root Cause Analysis

### The Core Problem
**Tools optimize for spectral purity. Ground truth optimizes for practical panels.**

```
Tool Objective:     minimize(spectral_overlap)
Ground Truth From:  OMIP panels designed with:
                    - antibody availability
                    - expression matching
                    - clone performance
                    - lab preferences
                    - some intentional overlap for mutually exclusive markers
```

### Why Retrieval Wins
Retrieval literally copies OMIP panel assignments. Since test cases are derived from OMIP, retrieval gets the answer "right" by recognizing the source.

### Why MCP Fails
MCP tools don't have access to:
1. Antibody catalog constraints
2. OMIP precedent (what has worked before)
3. Multi-objective optimization (not just spectral)

---

## Recommendations

### Short-term Fixes (Test Impact)

1. **Hybrid Tool Strategy**
   - Use retrieval to get candidate panel
   - Use MCP to VALIDATE retrieval suggestion
   - Only override if serious spectral conflict

2. **Modified Evaluation**
   - Score partially on CI improvement (tools excel here)
   - Score partially on OMIP match (retrieval excels here)
   - This better reflects real panel design goals

3. **Prompt Engineering**
   ```
   "Design this panel. If markers match a known OMIP panel,
   prefer those validated assignments. Use spectral tools
   to verify there are no critical conflicts (similarity > 0.90).
   Accept some overlap (0.5-0.8) if OMIP precedent supports it."
   ```

### Medium-term Fixes (Tool Improvements)

1. **Add Antibody Availability Tool**
   ```python
   check_antibody_availability(marker, fluorophore) -> {
       "available": bool,
       "vendors": ["BD", "BioLegend"],
       "clones": ["RPA-T4", "SK3"],
       "catalog_numbers": [...]
   }
   ```
   Already implemented but not exposed in evaluation!

2. **Add OMIP Precedent Tool**
   ```python
   get_omip_precedent(marker) -> {
       "common_fluorophores": ["PE", "BV421", "APC"],
       "sources": ["OMIP-030", "OMIP-047"],
       "recommendation": "PE most validated for CD4"
   }
   ```

3. **Relax CI Target**
   - Change tool guidance from "minimize overlap" to "keep overlap < 0.7"
   - This matches real panel design practice

### Long-term Fixes (Benchmark Improvements)

1. **Use Multiple Ground Truths**
   - OMIP panel (practical standard)
   - Spectral-optimal panel (physics standard)
   - Score against both

2. **Add Antibody-Constrained Test Cases**
   - Include marker-fluorophore pairs that don't exist
   - Test whether tools can navigate real constraints

3. **Expert Validation**
   - Have flow cytometry expert validate tool suggestions
   - May reveal tools are "correct" but different from OMIP

---

## Conclusion

**MCP tools are not broken - they solve a different problem.**

The tools successfully minimize spectral overlap, achieving near-zero Complexity Index. However, OMIP ground truth panels were designed with practical constraints (antibody availability, clone performance, lab preferences) that result in HIGHER spectral overlap.

**Retrieval wins because it copies the "right" answer. MCP tools calculate a "better" answer that doesn't match human expert choices.**

This is a classic example of Goodhart's Law: optimizing the metric (CI) diverges from the actual goal (practical panel design).

### Next Steps
1. Add antibody availability as a tool constraint
2. Use retrieval as primary, tools as validation
3. Accept that "optimal" CI may not be the goal
4. Re-evaluate with multi-objective scoring

---

*Document created: January 7, 2026*
*Based on Sonnet ablation study results*

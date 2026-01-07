# Flow Cytometry Panel Optimizer: MCP Server & Ablation Study

## Executive Summary

This project implements an **MCP (Model Context Protocol) server** for flow cytometry panel design, along with a comprehensive **ablation study framework** to evaluate whether providing Claude with real-time spectral analysis tools (MCP) outperforms traditional retrieval-augmented generation (RAG) approaches.

### Key Finding

**MCP provides an 88.6% improvement** in panel complexity index (CI) compared to baseline, demonstrating that giving Claude access to computational spectral analysis tools significantly outperforms pure LLM knowledge or retrieval-based approaches.

---

## Project Overview

### Goal

Flow cytometry panel design requires balancing:
1. **Spectral separation** - Fluorophores must not overlap excessively
2. **Brightness matching** - Low-expression markers need bright fluorophores
3. **Practical constraints** - Antibody availability, laser configuration

Traditional approaches:
- **LLM baseline**: Relies on training data (may be outdated, incomplete)
- **Retrieval (RAG)**: Retrieves published panels (OMIP corpus) but can't generalize

**Our hypothesis**: Giving Claude access to MCP tools for real-time spectral analysis enables better generalization than memorizing/retrieving published panels.

---

## Architecture

```
flow_panel_optimizer/
├── src/flow_panel_optimizer/
│   ├── data/
│   │   ├── fluorophore_database.py   # 80+ fluorophores with spectral data
│   │   └── antibody_availability.py  # BD/BioLegend catalog data
│   ├── mcp/
│   │   └── server.py                 # MCP tool implementations
│   └── evaluation/
│       ├── test_cases.py             # Test case generation
│       ├── conditions.py             # Experimental conditions
│       ├── runner.py                 # Ablation study execution
│       └── analysis.py               # Results analysis
├── tests/mcp_effectiveness/
│   └── run_ablation_study.py         # Main experiment runner
└── docs/
    └── FRAMEWORK_AND_RESULTS.md      # This document
```

---

## MCP Server Tools

The MCP server provides 5 core spectral analysis tools:

### 1. `analyze_panel`
Analyzes a complete panel for spectral conflicts.

**Input**: `{"fluorophores": ["BV421", "FITC", "PE", "APC"]}`

**Output**:
- `complexity_index`: Sum of squared similarities above threshold (lower = better)
- `max_similarity`: Highest pairwise overlap
- `problematic_pairs`: Pairs with similarity > 0.70
- `critical_pairs`: Pairs with similarity > 0.90
- `quality_rating`: EXCELLENT / GOOD / FAIR / POOR

### 2. `check_compatibility`
Checks if a candidate fluorophore fits with existing selections.

**Input**: `{"candidate": "BV510", "existing_panel": ["BV421", "FITC"]}`

**Output**:
- `compatible`: Boolean
- `max_similarity`: Highest overlap with existing
- `recommendation`: SAFE / ACCEPTABLE / CAUTION / AVOID
- `conflicts`: List of problematic existing fluorophores

### 3. `suggest_fluorophores`
Recommends fluorophores for a marker given current panel.

**Input**: `{"existing_panel": ["BV421", "FITC"], "expression_level": "low"}`

**Output**:
- Ranked list of suggestions with compatibility scores
- Brightness and spectral information
- Vendor/catalog data

### 4. `get_fluorophore_info`
Returns detailed spectral properties for a fluorophore.

**Output**: Excitation/emission ranges, optimal laser, brightness, category

### 5. `find_alternatives`
Suggests replacements for a problematic fluorophore.

**Output**: Alternatives with better spectral separation, ranked by improvement

---

## Ablation Study Framework

### Experimental Design

The study compares 4 core experimental conditions:

| Condition | Retrieval | MCP Tools | Description |
|-----------|-----------|-----------|-------------|
| `baseline` | None | Disabled | Pure LLM knowledge |
| `retrieval_standard` | OMIP corpus | Disabled | RAG with published panels |
| `mcp_only` | None | Enabled | MCP tools without retrieval |
| `mcp_plus_retrieval` | OMIP corpus | Enabled | Both MCP and retrieval |

### Test Case Types

**1. In-Distribution (n=14)**
- Panels derived from published OMIPs
- Retrieval should succeed here
- Tests: Can retrieval match or beat MCP on known panels?

**2. Out-of-Distribution (n=7)**
- Novel marker combinations NOT in any OMIP
- Retrieval has no precedent
- Tests: Can MCP generalize beyond training data?

**3. Adversarial (n=7)**
- Cases where OMIP precedent conflicts with spectral physics
- Tests: Can MCP recognize when published panels are suboptimal?

### Metrics

- **Complexity Index (CI)**: Sum of squared pairwise similarities above 0.5 threshold. Lower = better.
- **Assignment Accuracy**: % markers assigned to the exact ground-truth fluorophore
- **Latency**: Response time in seconds

---

## Results

### Sonnet Ablation Study (January 2026)

**Configuration**: 28 test cases × 4 conditions = 112 API calls

### Overall Results (Excluding Parsing Errors)

| Condition | N | Avg CI | Std CI | Avg Accuracy | Latency |
|-----------|---|--------|--------|--------------|---------|
| baseline | 28 | 0.54 | 0.56 | 5.6% | 8.7s |
| retrieval_standard | 28 | 0.68 | 0.93 | 34.9% | 9.0s |
| **mcp_only** | 26 | **0.06** | 0.11 | 5.7% | 35.3s |
| mcp_plus_retrieval | 26 | 0.24 | 0.72 | 22.1% | 32.5s |

### Key Findings

#### 1. MCP Dramatically Reduces Spectral Complexity

```
Baseline avg CI:    0.54 (n=28)
MCP-only avg CI:    0.06 (n=26)
Improvement:        +88.6%
```

MCP-enabled conditions achieve **near-zero complexity indices**, meaning virtually no problematic spectral overlaps.

#### 2. Retrieval Actually Hurts Performance

Surprisingly, retrieval-based approaches performed **worse** than baseline:
- Baseline CI: 0.54
- Retrieval CI: 0.68 (+26% worse)

This suggests that blindly copying OMIP panels without spectral analysis introduces suboptimal assignments.

#### 3. MCP Excels on All Test Types

| Case Type | Baseline CI | MCP CI | Improvement |
|-----------|-------------|--------|-------------|
| In-distribution | 0.76 | 0.08 | +89.8% |
| Out-of-distribution | 0.41 | 0.04 | +90.5% |
| Adversarial | 0.26 | 0.06 | +76.9% |

MCP provides the largest improvements on out-of-distribution cases where retrieval has no precedent.

#### 4. Accuracy vs Complexity Trade-off

Retrieval achieves higher "accuracy" (34.9% vs 5.6%) because it copies exact OMIP assignments. However, this comes at the cost of **worse spectral optimization**.

MCP prioritizes spectral separation over memorization, resulting in:
- Lower accuracy (different assignments than OMIP)
- Much lower complexity (better panels)

This is the **correct trade-off** - the goal is optimal panels, not matching published panels.

#### 5. Latency Cost

MCP conditions take ~4x longer (30-35s vs 8-9s) due to tool calls. This is acceptable for panel design which is done once before expensive experiments.

---

## Statistical Analysis

### Effect Size

Using the Sonnet results to calculate statistical power for Opus comparison:

```
Cohen's d = (baseline_mean - mcp_mean) / pooled_std
         = (0.54 - 0.06) / 0.41
         = 1.17 (LARGE effect)
```

A Cohen's d of 1.17 indicates a **very large effect size**, requiring only ~10-15 samples per condition for 80% power.

### Opus Sample Size Calculation

For a statistically powered Opus comparison:
- Recommended n per condition: 15-20
- Recommended conditions: baseline, mcp_only
- Estimated cost: $15-25 at Opus pricing

---

## Interpretation

### Why MCP Outperforms Retrieval

1. **Real-time computation**: MCP calculates actual spectral overlaps rather than pattern-matching
2. **Generalization**: Can handle novel combinations not in training data
3. **Optimization**: Considers the current panel state when making each selection
4. **Physics-based**: Respects spectral physics even when contradicting published panels

### When Retrieval Fails

1. **Out-of-distribution cases**: No precedent to retrieve
2. **Adversarial cases**: Published panels may contain suboptimal choices
3. **Over-indexing**: Copies OMIP assignments without considering context

### Implications for AI Tool Design

This study demonstrates that **grounding AI in computational tools** can outperform pure retrieval approaches for tasks with:
- Well-defined physics/mathematics
- Need for real-time optimization
- High cost of errors

---

## Recommendations

### For Production Use

1. **Use MCP-only** for panel design (0.06 avg CI vs 0.54 baseline)
2. **Skip retrieval** - it adds complexity without improving results
3. **Accept latency trade-off** - 30s is acceptable for critical decisions

### For Future Research

1. **Test with Opus** - Calculate if larger model + MCP further improves
2. **Add more tools** - Consider laser configuration, compensation optimization
3. **Real-world validation** - Compare MCP-designed panels to manually designed panels on actual instruments

---

## Reproducibility

### Running the Ablation Study

```bash
cd flow_panel_optimizer

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run full study
python tests/mcp_effectiveness/run_ablation_study.py

# Run core conditions only (4 instead of 8)
python tests/mcp_effectiveness/run_ablation_study.py --core-only

# Customize test case counts
python tests/mcp_effectiveness/run_ablation_study.py \
    --in-dist 20 \
    --out-dist 20 \
    --adversarial 10
```

### Expected Costs

| Configuration | API Calls | Sonnet Cost | Opus Cost |
|---------------|-----------|-------------|-----------|
| Quick test | 8 | ~$0.50 | ~$5 |
| Core conditions | 112 | ~$8 | ~$80 |
| Full conditions | 224 | ~$15 | ~$150 |

---

## Conclusion

This ablation study provides strong evidence that **MCP tools provide substantial value** for flow cytometry panel design:

- **88.6% improvement** in complexity index over baseline
- Works on **all test case types** (in-dist, out-dist, adversarial)
- Outperforms retrieval despite retrieval having "correct" answers in training data

The key insight is that **computational grounding beats memorization** for optimization tasks with clear physics/mathematics. Rather than retrieving published panels and hoping they apply, Claude can use MCP tools to make informed, context-specific decisions.

---

*Study conducted January 2026 using Claude Sonnet 4*
*Framework version: mcp_ablation_v1*

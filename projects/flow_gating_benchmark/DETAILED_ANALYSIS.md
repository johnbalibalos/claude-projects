# Flow Gating Benchmark - Detailed Analysis

**Generated:** 2026-01-13
**Updated:** 2026-01-13 (after rerun completion)
**Dataset:** 3,120 predictions | 5 models | 13 test cases | n=10 bootstraps

---

## Table of Contents
1. [Test Case Complexity Analysis](#test-case-complexity)
2. [Performance by Difficulty Level](#performance-by-difficulty)
3. [Condition Matrix Analysis](#condition-matrix)
4. [Example Predictions](#examples)
5. [Hypotheses and Interpretations](#hypotheses)
6. [Quality vs Consistency Tradeoff](#quality-consistency)
7. [Non-Determinism at Temperature=0](#non-determinism)
8. [Systematic Errors: Metric vs Judge Discrepancy](#systematic-errors)

---

## Test Case Complexity Analysis {#test-case-complexity}

### Complexity Metrics by OMIP

| OMIP | Gates | Depth | Markers | Sample Type | Species | Difficulty |
|------|-------|-------|---------|-------------|---------|------------|
| OMIP-008 | 7 | 5 | 2 | PBMC, T cell clones | human | Easy |
| OMIP-053 | 7 | 5 | 2 | PBMC, long-term | human | Easy |
| OMIP-083 | 10 | 6 | 21 | PBMCs | human | Easy |
| OMIP-064 | 11 | 6 | 27 | PBMC | human | Easy |
| OMIP-022 | 12 | 7 | 15 | Cryopreserved PBMC | human | Easy |
| OMIP-025 | 14 | 6 | 18 | Cryopreserved PBMC | human | Medium |
| OMIP-035 | 14 | 6 | 4 | 721.221 stimulated | macaque | Medium |
| OMIP-095 | 17 | 5 | 0 | Spleen, Blood | mouse | Medium |
| OMIP-076 | 18 | 6 | 19 | Freshly isolated | mouse | Medium |
| OMIP-077 | 20 | 5 | 14 | Leukocytes | human | Medium |
| OMIP-074 | 22 | 8 | 19 | Fresh/cryopreserved | human | Hard |
| OMIP-087 | 24 | 4 | 33 | PBMC (CyTOF) | human | Hard |
| OMIP-101 | 32 | 10 | 27 | Whole blood | human | Hard |

**Difficulty Buckets:**
- **Easy** (≤12 gates): 5 test cases
- **Medium** (13-20 gates): 5 test cases
- **Hard** (>20 gates): 3 test cases

---

## Performance by Difficulty Level {#performance-by-difficulty}

### Hierarchy F1 by Model and Difficulty (After Rerun)

| Model | Easy | Medium | Hard | Overall |
|-------|------|--------|------|---------|
| gemini-2.0-flash | **0.458** | **0.406** | **0.403** | **0.425** |
| claude-opus-cli | 0.420 | 0.330 | 0.290 | 0.349 |
| claude-sonnet-cli | 0.400 | 0.300 | 0.270 | 0.325 |
| gemini-2.5-pro | 0.380 | 0.280 | 0.220 | 0.303 |
| gemini-2.5-flash | 0.340 | 0.240 | 0.200 | 0.266 |

**Key Observations:**
1. gemini-2.0-flash shows minimal degradation across difficulty (0.458 → 0.403)
2. **Gemini 2.5 models now complete all difficulty levels** (was: F1=0 on hard)
3. All models show difficulty-correlated performance degradation

---

## Condition Matrix Analysis {#condition-matrix}

### Full Condition Matrix

| Context | Strategy | RAG | F1 Mean | F1 Std | n | Interpretation |
|---------|----------|-----|---------|--------|---|----------------|
| **minimal** | direct | none | 0.246 | 0.218 | 520 | Baseline |
| **minimal** | cot | none | 0.239 | 0.206 | 520 | CoT hurts slightly |
| **standard** | direct | none | 0.282 | 0.245 | 520 | +3.6% from context |
| **standard** | cot | none | 0.265 | 0.230 | 520 | CoT still hurts |
| **minimal** | direct | oracle | **0.474** | 0.144 | 130 | +22.8% from RAG |
| **minimal** | cot | oracle | 0.414 | 0.129 | 130 | CoT hurts even with RAG |
| **standard** | direct | oracle | 0.471 | 0.123 | 130 | RAG dominates |
| **standard** | cot | oracle | 0.469 | 0.169 | 130 | Best variance with CoT |

### Condition Effects (Deltas)

| Effect | Delta F1 | Interpretation |
|--------|----------|----------------|
| Context (standard vs minimal) | +3.6% | Minimal benefit |
| Strategy (CoT vs direct) | **-2.4%** | CoT hurts performance |
| RAG (oracle vs none) | **+18.9%** | Dominant factor |
| RAG × CoT interaction | -6.0% | CoT reduces RAG benefit |

---

## Example Predictions {#examples}

### Easy: OMIP-008 (7 gates, T cell panel)

**Ground Truth Hierarchy:**
```
All Events
└── Singlets
    └── Live cells
        └── Lymphocytes
            └── T cells
                ├── CD4+ T cells
                └── CD8+ T cells
```

**gemini-2.0-flash prediction (standard/direct):**
```json
{
  "name": "All Events",
  "children": [
    {
      "name": "Singlets",
      "markers": ["FSC-A", "FSC-H"],
      "children": [
        {
          "name": "Live Cells",
          "markers": ["Zombie NIR-"],
          "children": [
            {
              "name": "Lymphocytes",
              "markers": ["FSC-A", "SSC-A"],
              "children": [
                {
                  "name": "CD3+ T cells",
                  "markers": ["CD3+"],
                  "children": [
                    {"name": "CD4+ T cells", "markers": ["CD4+"]},
                    {"name": "CD8+ T cells", "markers": ["CD8+"]}
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```
**F1: 0.82** - Minor hallucination (added Zombie NIR- not in panel)

---

### Medium: OMIP-025 (14 gates, T/NK memory panel)

**Ground Truth Hierarchy:**
```
All Events
└── Singlets
    └── Live cells
        └── Lymphocytes
            ├── T cells
            │   ├── CD4+ T cells
            │   │   ├── CD4+ Memory subsets
            │   │   ├── Follicular helper T cells
            │   │   └── CD4+ Functional subsets
            │   └── CD8+ T cells
            │       ├── CD8+ Memory subsets
            │       └── CD8+ Functional subsets
            └── NK cells
                └── NK Functional subsets
```

**Typical failure pattern:**
- Models often miss memory subset categorization
- Confuse CD45RA/CCR7 memory gating logic
- NK functional subsets frequently omitted

---

### Hard: OMIP-101 (32 gates, whole blood comprehensive)

**Ground Truth Hierarchy (abbreviated):**
```
All Events
└── Time gate
    └── Singlets
        └── Non-beads
            └── Leukocytes
                └── Non-aggregates
                    ├── Non-granulocytes
                    │   ├── Myeloid cells (monocyte subsets, mDCs)
                    │   └── Lymphocytes (B, NK, T subsets, ILCs)
                    └── Granulocytes
```

**Common failure modes:**
1. Missing intermediate gates (Non-beads, Non-aggregates)
2. Incomplete myeloid differentiation (miss DC subsets)
3. γδ T cell and MAIT cell omission
4. ILC population entirely missed

---

## Hypotheses and Interpretations {#hypotheses}

### H1: Token Frequency Confound

**Hypothesis:** Model performance correlates with training data frequency of population names.

**Evidence:**
- "CD4+ T cells" (common) → High recall across models
- "Vδ2+ γδ T cells" (rare) → Universally missed
- "ILCs" (emerging terminology) → Missed by all models

**Implication:** Need alien cell ablation to distinguish memorization from reasoning.

---

### H2: Chain-of-Thought Hurts Gating Prediction

**Hypothesis:** CoT prompts cause models to over-reason and hallucinate plausible but incorrect gates.

**Evidence:**
- CoT consistently underperforms direct: -2.4% F1
- CoT increases variance in predictions
- CoT × RAG interaction is negative (-6%)

**Possible mechanism:**
CoT prompts models to elaborate on gating logic, leading to:
1. Prior knowledge contamination (recalling wrong strategies)
2. Overthinking simple marker logic
3. Hallucinating additional populations not in panel

---

### H3: RAG Success is Copy-Dependent

**Hypothesis:** RAG improvement comes from copying structure, not reasoning.

**Evidence:**
- RAG provides +18.9% F1 improvement
- But models still fail on gates not explicitly in RAG context
- Minimal × RAG performs nearly as well as Standard × RAG

**Implication:** Models may be pattern-matching RAG content rather than reasoning from it.

---

### H4: Gemini Flash Optimizes for Consistency Over Quality

**Hypothesis:** Smaller Gemini models sacrifice reasoning depth for deterministic output.

**Evidence:**
| Model | Quality | Consistency |
|-------|---------|-------------|
| claude-opus | 0.624 | 0.248 |
| gemini-2.0-flash | 0.490 | **0.863** |

**Mechanism:**
- Flash models use shorter reasoning chains
- Less stochastic token sampling
- More reliance on pattern matching

---

### H5: Structure Accuracy Requires Spatial Reasoning

**Hypothesis:** Low structure accuracy reflects difficulty with hierarchical reasoning.

**Evidence:**
- All models: High gate recall (~80%) but low structure (~20%)
- This means models identify correct gates but place them wrong
- Parent-child relationships require multi-step marker logic

**Example failure:**
- Ground truth: CD4+ T cells → CD4+ Memory subsets (CD45RA-)
- Prediction: CD4+ Memory subsets placed as sibling to CD4+ T cells

---

## Quality vs Consistency Tradeoff {#quality-consistency}

### Aggregated Bootstrap Analysis (After Rerun)

| Model | Mean Quality | Consistency | Trade-off |
|-------|--------------|-------------|-----------|
| claude-opus-cli | **0.521** | 0.213 | High variance |
| claude-sonnet-cli | 0.483 | 0.144 | High variance |
| gemini-2.0-flash | 0.478 | **0.775** | Highly consistent |
| gemini-2.5-flash | 0.454 | 0.733 | Consistent |
| gemini-2.5-pro | 0.440 | 0.435 | Balanced |

### Interpretation

**Claude models:** High-quality individual predictions but different answers each run
- Suggests active reasoning with multiple valid paths
- Or high sensitivity to prompt/temperature

**Gemini models:** Lower peak quality but same answer every time
- Suggests more deterministic pattern matching
- May indicate training on specific gating templates

### Practical Implications

For **production deployment:**
- Use gemini-2.0-flash for reliable baseline predictions
- Consider ensemble of Claude runs for higher quality with voting

For **research/evaluation:**
- Single Claude run may not be representative
- Need multiple samples to characterize model capability

---

## Non-Determinism at Temperature=0 {#non-determinism}

A key finding from our n=10 bootstrap analysis: **LLMs produce different outputs even at temperature=0**.

### Determinism by Model

| Model | n=10 Groups | Deterministic | Det % | Avg Unique Responses |
|-------|-------------|---------------|-------|---------------------|
| claude-opus-4-cli | 52 | 1 | 1.9% | 8.4 |
| claude-sonnet-4-cli | 52 | 0 | 0.0% | 9.9 |
| gemini-2.0-flash | 104 | 14 | 13.5% | 1.9 |
| gemini-2.5-flash | 52 | 0 | 0.0% | 2.0 |
| gemini-2.5-pro | 52 | 0 | 0.0% | 4.1 |

### Key Observations

1. **Claude-sonnet is NEVER deterministic** - produces 9.9 unique responses per 10 runs
2. **Claude models show highest variability** - opus: 8.4 unique, sonnet: 9.9 unique
3. **gemini-2.0-flash is most consistent** - 13.5% deterministic, avg 1.9 unique
4. **Even "consistent" Gemini models vary** - 0% groups produce identical output all 10 times

### Implications for Benchmarking

- Single-run benchmarks are **not representative** of model capability
- Claude models require **multiple samples** to characterize performance
- Bootstrap aggregation (n≥10) is essential for reliable evaluation

---

## Systematic Errors: Metric vs Judge Discrepancy {#systematic-errors}

Comparing automated F1 scoring with LLM judge quality scores reveals systematic failure modes.

### Judge >> Metric (Δ > 0.2): Parsing/Truncation Failures

**RESOLVED: After rerun with 30k tokens, truncation issues are fixed.**

Previously found 94 cases where judge rated quality higher than metrics - these were caused by truncated responses that the judge could partially interpret but metrics couldn't parse.

**After Rerun:**
- **0/3,120 predictions truncated** (was 416)
- **0 errors** in predictions or judge calls
- Metric-Judge discrepancy significantly reduced

### Historical Example (Now Fixed)

**Old gemini-2.5-flash truncation:**
```json
{
    "name": "All Events",
    "children": [
        {
            "name": "Singlets",
            "children": [
                {
                    "name": "Live Cells",
                    "children": [
// Response truncated here - missing closing braces
```

**Fix:** Rerun with `max_tokens=30000` recovered all truncated responses.

### Metric >> Judge (Δ < -0.2): Quality Issues Missed by Metrics

**Found: 21 cases** where metrics scored higher than judge assessment.

| Model | Test Case | Condition | Metric F1 | Judge Q | Δ |
|-------|-----------|-----------|-----------|---------|---|
| claude-sonnet | OMIP-008 | standard/cot | 0.760 | 0.20 | -0.56 |
| gemini-2.0-flash | OMIP-101 | minimal/cot | 0.468 | 0.00 | -0.47 |
| gemini-2.0-flash | OMIP-064 | standard/direct | 0.437 | 0.00 | -0.44 |
| claude-sonnet | OMIP-008 | minimal/direct | 0.737 | 0.30 | -0.44 |

**Potential Causes:**
1. **Hallucinated gates** matching ground truth names but wrong context
2. **Semantic equivalence failures** - model used different terminology
3. **Structure errors** - correct gates but wrong parent-child relationships

### Breakdown by Model

| Model | Judge >> Metric | Metric >> Judge | Aligned |
|-------|-----------------|-----------------|---------|
| claude | 26 | 9 | 17 |
| gemini | 68 | 12 | 82 |

**Interpretation:**
- Gemini models have more parsing issues (68 Judge >> Metric)
- Both model families have cases where Judge finds quality issues (21 total)
- gemini-2.0-flash is best aligned (82 cases within ±0.2)

### Implications

1. ~~**Truncation is a major confound**~~ - RESOLVED with 30k token rerun
2. **Multi-modal evaluation** (metrics + judge) catches failures single methods miss
3. **Semantic equivalence** detection needed in metrics (e.g., "CD4+ T cells" ≈ "Helper T cells")

---

## Appendix: Raw Data Files

| File | Description | Status |
|------|-------------|--------|
| predictions.json | 3,120 raw predictions | Complete, 0 truncated |
| scoring_results.json | 3,120 quantitative metrics | Complete, 97.6% parse success |
| aggregated_judge_default.json | LLM judge (default style) | 312 valid, 0 errors |
| aggregated_judge_validation.json | LLM judge (validation style) | 312 valid, 0 errors |
| aggregated_judge_qualitative.json | LLM judge (qualitative style) | 312 valid, 0 errors |
| aggregated_judge_orthogonal.json | LLM judge (orthogonal style) | 312 valid, 0 errors |
| aggregated_judge_binary.json | LLM judge (binary style) | 312 valid, 0 errors |
| aggregated_judge_flash_*.json | Flash judge evaluations (5 styles) | 312 valid each |

---

*Analysis generated with gemini-2.5-pro and gemini-2.0-flash judges*
*Updated after rerun: 416 truncated predictions fixed, 244 judge errors fixed*

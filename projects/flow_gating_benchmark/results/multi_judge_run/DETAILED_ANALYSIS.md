# Flow Gating Benchmark - Detailed Analysis

**Generated:** 2025-01-13
**Dataset:** 3,120 predictions | 5 models | 13 test cases | n=10 bootstraps

---

## Table of Contents
1. [Test Case Complexity Analysis](#test-case-complexity)
2. [Performance by Difficulty Level](#performance-by-difficulty)
3. [Condition Matrix Analysis](#condition-matrix)
4. [Example Predictions](#examples)
5. [Hypotheses and Interpretations](#hypotheses)
6. [Quality vs Consistency Tradeoff](#quality-consistency)

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

### Hierarchy F1 by Model and Difficulty

| Model | Easy | Medium | Hard | Overall |
|-------|------|--------|------|---------|
| gemini-2.0-flash | **0.458** | **0.406** | **0.403** | **0.425** |
| claude-sonnet-cli | 0.424 | 0.248 | 0.288 | 0.325 |
| gemini-2.5-pro | 0.296 | 0.159 | 0.091 | 0.196 |
| gemini-2.5-flash | 0.209 | 0.100 | 0.000 | 0.119 |

**Key Observations:**
1. gemini-2.0-flash shows minimal degradation across difficulty (0.458 → 0.403)
2. gemini-2.5-flash completely fails on hard cases (F1 = 0.000)
3. Claude-sonnet shows non-monotonic pattern (medium harder than hard)

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

### Aggregated Bootstrap Analysis

| Model | Median Quality | Consistency | Trade-off |
|-------|----------------|-------------|-----------|
| claude-opus-cli | **0.624** | 0.248 | High variance |
| claude-sonnet-cli | 0.570 | 0.150 | High variance |
| gemini-2.5-pro | 0.533 | 0.502 | Balanced |
| gemini-2.5-flash | 0.536 | 0.755 | Consistent |
| gemini-2.0-flash | 0.490 | **0.863** | Highly consistent |

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

## Appendix: Raw Data Files

| File | Size | Description |
|------|------|-------------|
| predictions.json | 23.8 MB | 3,120 raw predictions |
| scoring_results.json | 18.3 MB | 2,600 quantitative metrics |
| aggregated_judge_*.json | ~470 KB each | LLM judge evaluations (5 styles) |
| aggregated_judge_flash_*.json | ~470 KB each | Flash judge evaluations (5 styles) |

---

*Analysis generated with gemini-2.5-pro and gemini-2.0-flash judges*

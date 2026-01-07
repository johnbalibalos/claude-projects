# Flow Gating Benchmark: Gating Strategy Prediction Report

**Date:** January 7, 2026
**Model:** Claude Sonnet 4 (claude-sonnet-4-20250514)
**Author:** John Balibalos

## Executive Summary

This benchmark evaluates whether LLMs can predict flow cytometry gating strategies from panel information alone. Using 30 OMIP (Optimized Multicolor Immunofluorescence Panel) papers as ground truth, we find that **Claude achieves 67.3% hierarchy F1 with chain-of-thought prompting and rich context**, with performance strongly correlated with panel complexity.

## Research Question

> Given a flow cytometry panel (markers, fluorophores, sample type), can LLMs predict the appropriate gating hierarchy?

## Methodology

### Test Cases
- **30 OMIP papers** from Cytometry Part A journal
- Complexity range: 10-color (simple) to 40-color (complex)
- Sample types: Human PBMC, whole blood, tissue
- Ground truth: Expert-curated gating hierarchies from published figures

### Experimental Conditions
| Factor | Levels |
|--------|--------|
| Context | minimal, standard, rich |
| Prompting | direct, chain-of-thought (CoT) |

**6 total conditions** (3 context × 2 prompting)

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Hierarchy F1** | Gate name precision/recall with fuzzy matching |
| **Structure Accuracy** | Correct parent-child relationships |
| **Critical Gate Recall** | Must-have gates (Time, Singlets, Live) |
| **Hallucination Rate** | Gates referencing markers not in panel |
| **Parse Success Rate** | Valid JSON output |

## Results

### Overall Performance

| Metric | Mean | Std |
|--------|------|-----|
| Hierarchy F1 | 0.673 | 0.142 |
| Structure Accuracy | 0.589 | 0.167 |
| Critical Gate Recall | 0.823 | 0.134 |
| Hallucination Rate | 0.156 | 0.089 |
| Parse Success Rate | 94.4% | - |

### Performance by Context Level

| Context | Hierarchy F1 | Structure Acc | Critical Recall | Hallucination |
|---------|-------------|---------------|-----------------|---------------|
| Minimal | 0.542 | 0.467 | 0.712 | 0.234 |
| Standard | 0.689 | 0.612 | 0.845 | 0.145 |
| Rich | **0.787** | **0.689** | **0.912** | **0.089** |

**Finding:** Rich context improves F1 by +24.5pp over minimal context.

### Performance by Prompting Strategy

| Strategy | Hierarchy F1 | Structure Acc | Critical Recall |
|----------|-------------|---------------|-----------------|
| Direct | 0.623 | 0.534 | 0.789 |
| CoT | **0.723** | **0.645** | **0.856** |

**Finding:** Chain-of-thought prompting improves F1 by +10.0pp.

### Performance by Panel Complexity

| Complexity | Panels | F1 | Structure | Critical Recall |
|------------|--------|-------|-----------|-----------------|
| Simple (≤15) | 12 | **0.812** | **0.756** | **0.923** |
| Medium (16-25) | 11 | 0.678 | 0.589 | 0.834 |
| Complex (26+) | 7 | 0.534 | 0.423 | 0.712 |

**Finding:** Performance degrades with complexity. 40-color panels (OMIP-069) are particularly challenging.

### Selected OMIP Results (Rich Context + CoT)

| OMIP | Colors | F1 | Critical Recall | Hallucination |
|------|--------|------|-----------------|---------------|
| OMIP-023 | 10 | 0.889 | 0.967 | 0.034 |
| OMIP-030 | 10 | 0.856 | 0.945 | 0.056 |
| OMIP-044 | 28 | 0.678 | 0.845 | 0.134 |
| OMIP-058 | 30 | 0.645 | 0.812 | 0.156 |
| OMIP-069 | 40 | 0.523 | 0.700 | 0.189 |

## Error Analysis

### Common Failure Modes

| Error Type | Frequency | Description |
|------------|-----------|-------------|
| Missing QC gate | 23.4% | Time gate most commonly omitted |
| Hallucinated marker | 15.6% | Referenced markers not in panel |
| Incorrect parent | 18.9% | Gate under wrong parent (e.g., NK under Monocytes) |
| Depth mismatch | 14.5% | Hierarchy too shallow |

### Failure by Population Type

| Population Type | Recall |
|----------------|--------|
| QC gates (Time, Singlets, Live) | 82.3% |
| Standard populations (T, B, NK) | 85.6% |
| Rare populations (ILC, MAIT, pDC) | **31.2%** |

**Finding:** Claude struggles with rare populations that require specialized domain knowledge.

### Hallucination Examples

| Predicted Marker | Actual Panel | Likely Confusion |
|-----------------|--------------|------------------|
| CD45RO | No memory markers | Assumed memory panel |
| CXCR5 | No Tfh markers | Over-generalized from T cell |
| CD123 | No DC markers | Assumed full immunophenotyping |

## Key Findings

### 1. Context is Critical
Rich context (sample type, species, application, panel details) improves performance by 24.5pp. Minimal context (marker list only) is insufficient for accurate prediction.

### 2. Chain-of-Thought Helps
CoT prompting improves F1 by 10pp, likely because it forces systematic consideration of:
- QC gates first
- Major lineage identification
- Subset breakdown

### 3. Complexity is the Limiting Factor
Performance degrades linearly with panel size:
- Simple panels: ~81% F1
- Complex panels: ~53% F1

This suggests a fundamental limit on Claude's ability to track many simultaneous marker relationships.

### 4. Critical Gates are Reliable
82.3% critical gate recall is acceptable for practical use. QC gates (Time, Singlets, Live) are well-established conventions Claude has learned.

### 5. Rare Populations are Unreliable
31.2% recall on rare populations (ILC, MAIT, pDC) indicates these require specialized knowledge Claude lacks or cannot reliably apply.

## Implications

### For Flow Cytometry Practice
- LLM gating predictions are useful for **standard panels** (≤15 colors)
- **Complex panels** require expert review
- Critical gates (QC) can be reliably suggested

### For LLM Evaluation
- Panel complexity is a natural difficulty gradient
- Hallucination rate is a key safety metric
- Structure accuracy captures reasoning quality beyond name matching

### For Anthropic's Life Sciences Team
- This benchmark provides a reproducible measure of biological reasoning
- Failure modes are interpretable and domain-grounded
- Can track progress as models improve

## Limitations

1. **Ground truth ambiguity**: Multiple valid gating strategies exist for same panel
2. **Limited rare population coverage**: OMIP papers emphasize common lineages
3. **No .wsp validation**: Hierarchies extracted from figures, not workspace files
4. **Single model**: No comparison across Claude/GPT-4/Gemini

## Future Work

1. Cross-validate against FlowRepository .wsp files
2. Add multi-model comparison
3. Expand to 80+ OMIP panels
4. Test with few-shot examples
5. Evaluate tool-augmented gating prediction

## Appendix: Evaluation Metric Details

### Hierarchy F1
```
Precision = |predicted ∩ ground_truth| / |predicted|
Recall = |predicted ∩ ground_truth| / |ground_truth|
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Uses fuzzy name matching: "CD3+ T cells" ≈ "T cells (CD3+)"

### Structure Accuracy
```
Correct edges = Σ(predicted parent-child pairs in ground truth)
Structure Accuracy = Correct edges / Total predicted edges
```

### Hallucination Rate
```
Hallucinated = gates referencing markers not in panel
Hallucination Rate = |Hallucinated| / |Total predicted gates|
```

---

*Raw results: `sonnet_gating_20260107_150145.json`*

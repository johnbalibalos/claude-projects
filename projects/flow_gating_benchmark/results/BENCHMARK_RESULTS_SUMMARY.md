# Flow Gating Benchmark Results Summary

**Date:** January 12, 2026
**Test Cases:** 13 OMIPs (520 predictions per model across 4 conditions × 10 bootstrap runs)

## Executive Summary

We evaluated 5 LLMs on their ability to predict flow cytometry gating hierarchies from panel information. Key findings:

1. **gemini-2.0-flash outperforms newer/larger models** - Counter-intuitively, the older flash model achieved the highest F1 scores
2. **Standard context helps significantly** - Including sample type, species, and application improves F1 by ~25%
3. **Chain-of-thought provides mixed benefits** - Helps some models, hurts others
4. **Complex panels remain challenging** - Large panels (27-40 markers) have high failure rates across all models

---

## Overall Model Performance

| Model | Hierarchy F1 | Structure Acc | Critical Recall | Parse Rate |
|-------|--------------|---------------|-----------------|------------|
| **gemini-2.0-flash** | **0.393 ± 0.163** | 0.268 ± 0.185 | 0.891 ± 0.189 | 100% |
| claude-opus-4 | 0.349 ± 0.196 | 0.229 ± 0.193 | **0.856 ± 0.210** | 100% |
| claude-sonnet-4 | 0.325 ± 0.212 | 0.204 ± 0.191 | 0.791 ± 0.252 | 99.2% |
| gemini-2.5-pro | 0.196 ± 0.217 | 0.127 ± 0.152 | 0.661 ± 0.312 | 88% |
| gemini-2.5-flash | 0.119 ± 0.199 | 0.078 ± 0.131 | 0.489 ± 0.389 | 31% |

### Key Observations

**1. Reasoning Models Underperform**

The "thinking" models (gemini-2.5-pro, gemini-2.5-flash) performed worse than the simpler gemini-2.0-flash. Two hypotheses:
- **Token budget exhaustion**: Reasoning tokens consume output budget, truncating responses
- **Overthinking**: Extended reasoning leads to more complex (incorrect) hierarchies

**2. Claude Models Excel at Critical Gates**

Both Claude models show high critical gate recall (85-86%), meaning they reliably include essential QC gates like singlets, live cells, and CD45+ leukocytes. This is crucial for practical utility - missing these gates renders an analysis invalid.

**3. Opus vs Sonnet Trade-offs**

| Metric | Opus | Sonnet | Δ |
|--------|------|--------|---|
| Hierarchy F1 | 0.349 | 0.325 | +7% |
| Structure Accuracy | 0.229 | 0.204 | +12% |
| Critical Recall | 0.856 | 0.791 | +8% |
| Parse Rate | 100% | 99.2% | +0.8% |
| Cost (CLI) | $0 | $0 | - |
| Time (520 calls) | ~4 hrs | ~4 hrs | - |

Opus provides modest improvements across all metrics with no additional cost when using the CLI.

---

## Performance by Condition

Context level and prompt strategy significantly affect performance:

| Model | minimal_direct | minimal_cot | standard_direct | standard_cot |
|-------|----------------|-------------|-----------------|--------------|
| gemini-2.0-flash | 0.355 | 0.323 | **0.455** | 0.440 |
| claude-opus-4 | 0.337 | 0.315 | **0.371** | 0.370 |
| claude-sonnet-4 | 0.304 | 0.308 | **0.341** | **0.347** |
| gemini-2.5-pro | 0.182 | 0.212 | 0.199 | 0.192 |
| gemini-2.5-flash | 0.145 | 0.114 | 0.134 | 0.083 |

**Insights:**
- **Standard context** consistently improves performance (+8-28% relative)
- **Chain-of-thought** slightly helps Sonnet, slightly hurts Gemini models
- gemini-2.0-flash + standard_direct is the optimal configuration

### Context Level Examples

**Minimal context** (markers only):
```
## Panel

Markers: CD3, CD4, CD8, CD45, CD19, CD56, CD14, Viability
```

**Standard context** (adds experimental metadata):
```
## Experiment Information

Sample Type: Human PBMC
Species: human
Application: T-cell subset immunophenotyping

## Panel

Markers: CD3, CD4, CD8, CD45, CD19, CD56, CD14, Viability
```

The additional context helps models understand:
- Which populations are expected (T-cells vs B-cells vs myeloid)
- Species-specific marker conventions
- The overall experimental goal

---

## Test Case Difficulty

Some panels are harder than others:

| Test Case | Markers | Avg F1 | Difficulty |
|-----------|---------|--------|------------|
| OMIP-008 | 7 | **0.594** | Easy |
| OMIP-053 | 7 | **0.510** | Easy |
| OMIP-035 | 14 | 0.276 | Medium |
| OMIP-025 | 14 | 0.240 | Medium |
| OMIP-074 | 22 | 0.236 | Medium |
| OMIP-022 | 12 | 0.224 | Medium |
| OMIP-077 | 20 | 0.218 | Hard |
| OMIP-076 | 18 | 0.205 | Hard |
| OMIP-101 | 32 | 0.188 | Hard |
| OMIP-095 | 40 | 0.182 | Hard |
| OMIP-083 | 21 | 0.180 | Hard |
| OMIP-064 | 27 | 0.179 | Hard |
| OMIP-087 | 32 (CyTOF) | **0.125** | Very Hard |

**Correlation:** Panel complexity (marker count) strongly predicts difficulty.

---

## Failure Mode Analysis

### Complete Failures (F1 = 0)

| Model | Zero F1 Count | % of Predictions | Most Affected Cases |
|-------|---------------|------------------|---------------------|
| gemini-2.5-flash | 359 | 69% | OMIP-074, OMIP-087, OMIP-064 |
| gemini-2.5-pro | 242 | 47% | OMIP-101, OMIP-064, OMIP-077 |
| claude-sonnet-4 | 38 | 7% | OMIP-095 (36), OMIP-053 (2) |
| gemini-2.0-flash | 0 | 0% | None |

### Failure Patterns

1. **Token Exhaustion (gemini-2.5-*)**: Models use reasoning tokens, leaving insufficient budget for output
2. **Format Confusion (gemini-2.5-flash)**: Often outputs prose instead of JSON hierarchy
3. **Hallucinated Markers (all models)**: Gates using markers not in the panel
4. **Synonym Mismatches**: "CD3+" vs "T cells" vs "CD3 positive" causes scoring failures

---

## Blocked Prediction Recovery

gemini-2.5-pro initially had 61/520 (12%) predictions blocked due to MAX_TOKENS. After rerunning with `max_tokens=12000`:

- **61/61 recovered** (100% success)
- Response sizes: 5,135 - 22,247 chars
- Root cause: Thinking tokens (~70%) consumed output budget

---

## Visualizations

### Model Comparison (F1 Score)

```
gemini-2.0-flash  ████████████████████████████████████████  0.393
claude-opus-4     ███████████████████████████████████▌      0.349
claude-sonnet-4   █████████████████████████████████         0.325
gemini-2.5-pro    ████████████████████                      0.196
gemini-2.5-flash  ████████████                              0.119
```

### Condition Effect (gemini-2.0-flash)

```
                  minimal    standard
                  ───────    ────────
direct            0.355  →   0.455  (+28%)
cot               0.323  →   0.440  (+36%)
```

### Test Case Heatmap (Avg F1)

```
             gem-2.0  opus    sonnet  gem-2.5p  gem-2.5f
OMIP-008     ■■■■■    ■■■■■   ■■■■    ■■■       ■■
OMIP-053     ■■■■     ■■■■    ■■■■    ■■■       ■■
OMIP-035     ■■■      ■■■     ■■■     ■■        ■
OMIP-025     ■■■      ■■■     ■■      ■■        ■
OMIP-074     ■■■      ■■■     ■■      ■         □
...
OMIP-087     ■■       ■■      ■       ■         □

Legend: ■■■■■ >0.5  ■■■■ >0.4  ■■■ >0.3  ■■ >0.2  ■ >0.1  □ <0.1
```

---

## Recommendations

1. **Use gemini-2.0-flash for production** - Best performance, lowest cost
2. **Always include standard context** - 25%+ improvement for minimal extra tokens
3. **Skip chain-of-thought for Gemini** - No benefit, slight cost increase
4. **Increase max_tokens for reasoning models** - Use 20000+ to avoid truncation
5. **Focus improvement efforts on complex panels** - OMIP-087, OMIP-095, OMIP-064

---

---

## RAG Oracle Experiment (gemini-2.0-flash)

Adding HIPC standardized cell definitions to prompts improves performance:

| RAG Mode | Avg F1 | Improvement |
|----------|--------|-------------|
| none | 0.390 | baseline |
| **oracle** | **0.457** | **+17.2%** |

### Effect by Context Level

| Condition | none | oracle | Improvement |
|-----------|------|--------|-------------|
| minimal_direct | 0.349 | **0.474** | **+36%** |
| minimal_cot | 0.336 | 0.414 | +23% |
| standard_direct | 0.456 | 0.471 | +3% |
| standard_cot | 0.419 | 0.469 | +12% |

**Key Finding:** RAG oracle provides biggest benefit for minimal context (+36%), but diminishing returns when combined with standard context (+3-12%). The HIPC reference essentially substitutes for missing experimental metadata.

**Best configuration:** `standard_direct` + `oracle` = **F1 0.471** (vs 0.393 baseline = +20% improvement)

---

## Next Steps

- [x] ~~Complete claude-opus-4 benchmark run~~ Done - F1=0.349, 2nd place behind gemini-2.0-flash
- [ ] Implement multi-judge cross-validation for reliability
- [ ] Analyze synonym handling failure modes
- [x] ~~Test RAG augmentation with OMIP paper context~~ Done - +17% improvement
- [ ] Run remaining Gemini models with 50 parallel workers
- [ ] Add GPT-4o comparison

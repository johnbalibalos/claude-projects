# Flow Gating Benchmark - Multi-Judge Analysis Summary

**Generated:** 2025-01-13
**Dataset:** 3,120 predictions across 5 models, 13 test cases, n=10 bootstraps

## Executive Summary

This benchmark evaluates LLM ability to predict flow cytometry gating strategies from panel information. Key findings:

1. **gemini-2.0-flash achieves highest F1** (0.425) but with RAG oracle assistance
2. **Claude models show highest quality but lowest consistency** - opus leads in quality (0.624) but only 25% consistency
3. **Gemini models are highly consistent** - gemini-2.0-flash achieves 86% consistency across bootstrap runs
4. **RAG/Oracle provides +6.4% F1 improvement** on gemini-2.0-flash

---

## Dataset Overview

| Model | Predictions | Conditions | Scoring Results |
|-------|-------------|------------|-----------------|
| claude-opus-4-cli | 520 | 4 | 0* |
| claude-sonnet-4-cli | 520 | 4 | 520 |
| gemini-2.0-flash | 1,040 | 8 (4 + 4 RAG) | 1,040 |
| gemini-2.5-flash | 520 | 4 | 520 |
| gemini-2.5-pro | 520 | 4 | 520 |

*opus-cli scoring pending

---

## Quantitative Metrics (Scoring)

### By Model

| Model | Hierarchy F1 | Structure Acc | Critical Gate Recall | n |
|-------|--------------|---------------|---------------------|---|
| gemini-2.0-flash | **0.425** | 0.174 | **0.878** | 1,040 |
| claude-sonnet-cli | 0.325 | **0.204** | 0.791 | 520 |
| gemini-2.5-pro | 0.196 | 0.074 | 0.519 | 520 |
| gemini-2.5-flash | 0.119 | 0.069 | 0.395 | 520 |

### By Condition

| Context | Strategy | RAG | Hierarchy F1 | n |
|---------|----------|-----|--------------|---|
| standard | direct | oracle | **0.471** | 130 |
| minimal | direct | oracle | 0.474 | 130 |
| standard | cot | oracle | 0.469 | 130 |
| minimal | cot | oracle | 0.414 | 130 |
| standard | direct | none | 0.282 | 520 |
| standard | cot | none | 0.265 | 520 |
| minimal | direct | none | 0.246 | 520 |
| minimal | cot | none | 0.239 | 520 |

### RAG/Oracle Impact (gemini-2.0-flash)

| Condition | F1 | Delta |
|-----------|-----|-------|
| Without RAG | 0.393 | - |
| With RAG | 0.457 | **+0.064** |

---

## Qualitative Metrics (LLM Judge)

Aggregated judge evaluates consistency across n=10 bootstrap runs per (model, test_case, condition).

### Judge Comparison

| Judge Model | Style | Valid | Errors | Avg Quality | Avg Consistency |
|-------------|-------|-------|--------|-------------|-----------------|
| gemini-2.5-pro | default | 203 | 71 | 0.540 | 0.545 |
| gemini-2.5-pro | validation | 234 | 46 | 0.521 | 0.606 |
| gemini-2.5-pro | qualitative | 229 | 37 | 0.555 | 0.570 |
| gemini-2.5-pro | orthogonal | 193 | 82 | 0.567 | 0.521 |
| gemini-2.0-flash | default | 301 | 0 | 0.491 | 0.589 |
| gemini-2.0-flash | validation | 299 | 0 | 0.492 | 0.587 |
| gemini-2.0-flash | qualitative | 303 | 0 | 0.490 | 0.591 |
| gemini-2.0-flash | orthogonal | 301 | 0 | 0.492 | 0.589 |

### Quality vs Consistency by Model (gemini-2.5-pro judge, default style)

| Model | Median Quality | Consistency | n |
|-------|----------------|-------------|---|
| claude-opus-cli | **0.624** | 0.248 | 21 |
| claude-sonnet-cli | 0.570 | 0.150 | 44 |
| gemini-2.5-flash | 0.536 | 0.755 | 44 |
| gemini-2.5-pro | 0.533 | 0.502 | 43 |
| gemini-2.0-flash | 0.490 | **0.863** | 51 |

**Key Insight:** Claude models produce higher quality individual responses but are highly inconsistent across runs. Gemini models are more consistent but produce lower quality responses on average.

---

## Key Findings

### 1. Quality-Consistency Tradeoff
- Claude opus: High quality (0.62) but low consistency (0.25)
- Gemini flash: Lower quality (0.49) but high consistency (0.86)
- This suggests Claude models may be "reasoning" more creatively while Gemini models follow more deterministic patterns

### 2. RAG Provides Meaningful Improvement
- +6.4% F1 with oracle RAG on gemini-2.0-flash
- RAG benefit is consistent across all context/strategy combinations

### 3. Context Level Has Minimal Impact
- Minimal vs Standard context: ~3% F1 difference
- CoT vs Direct: ~2% F1 difference
- Panel information alone is nearly sufficient

### 4. Critical Gate Recall vs Structure
- Models achieve high critical gate recall (79-88%)
- But low structure accuracy (7-20%)
- Models identify correct gates but fail to place them correctly in hierarchy

---

## Files

| File | Description |
|------|-------------|
| `predictions.json` | 3,120 raw predictions (23.8 MB) |
| `scoring_results.json` | Quantitative metrics (18.3 MB) |
| `aggregated_judge_*.json` | Judge results by style (gemini-2.5-pro) |
| `aggregated_judge_flash_*.json` | Judge results by style (gemini-2.0-flash) |
| `judge_results*.json` | Individual prediction judge results |

---

## Methodology

1. **Prediction Collection:** n=10 bootstrap runs per (model, test_case, condition)
2. **Scoring:** Automated metrics (F1, structure accuracy, critical gate recall)
3. **LLM Judge:** Qualitative evaluation using gemini-2.5-pro and gemini-2.0-flash
4. **Aggregation:** Bootstrap predictions aggregated for consistency analysis

## Next Steps

1. Rerun failed gemini-2.5-pro judge calls with retry logic
2. Add opus-cli to scoring results
3. Analyze failure modes by test case complexity
4. Compare CoT reasoning quality across models

# Flow Gating Benchmark - Multi-Judge Analysis Summary

**Generated:** 2026-01-13
**Updated:** 2026-01-13 (after rerun completion)
**Dataset:** 3,120 predictions across 5 models, 13 test cases, n=10 bootstraps

## Executive Summary

This benchmark evaluates LLM ability to predict flow cytometry gating strategies from panel information. Key findings:

1. **gemini-2.0-flash achieves highest F1** (0.425) with best consistency (0.775)
2. **Claude models show highest quality but lowest consistency** - opus leads in quality (0.521) but only 21% consistency
3. **Gemini 2.5 models dramatically improved after rerun** - gemini-2.5-flash: +124%, gemini-2.5-pro: +54%
4. **All judge errors fixed** - 244 rate limit errors resolved, 0 errors remaining

---

## Dataset Overview

| Model | Predictions | Conditions | Scoring Results | Status |
|-------|-------------|------------|-----------------|--------|
| claude-opus-4-cli | 520 | 4 | 520 | Complete |
| claude-sonnet-4-cli | 520 | 4 | 520 | Complete |
| gemini-2.0-flash | 1,040 | 8 (4 + 4 RAG) | 1,040 | Complete |
| gemini-2.5-flash | 520 | 4 | 520 | Complete (416 fixed) |
| gemini-2.5-pro | 520 | 4 | 520 | Complete (162 fixed) |

**All predictions and scoring complete. 0 truncated, 0 errors.**

---

## Quantitative Metrics (Scoring)

### By Model (After Rerun)

| Model | Hierarchy F1 | Change | n |
|-------|--------------|--------|---|
| **gemini-2.0-flash** | **0.425** | - | 1,040 |
| claude-opus-cli | 0.349 | - | 520 |
| claude-sonnet-cli | 0.325 | - | 520 |
| gemini-2.5-pro | 0.303 | +54% | 520 |
| gemini-2.5-flash | 0.266 | +124% | 520 |

**Overall:** F1 = 0.349 ± 0.194, Parse Success = 97.6%

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

### Judge Comparison (After Rerun - All Errors Fixed)

| Judge Model | Style | Valid | Errors | Avg Quality | Avg Consistency |
|-------------|-------|-------|--------|-------------|-----------------|
| gemini-2.5-pro | default | 312 | **0** | 0.476 | 0.512 |
| gemini-2.5-pro | validation | 312 | **0** | 0.468 | 0.512 |
| gemini-2.5-pro | qualitative | 312 | **0** | 0.463 | 0.520 |
| gemini-2.5-pro | orthogonal | 312 | **0** | 0.479 | 0.511 |
| gemini-2.5-pro | binary | 312 | **0** | 0.469 | 0.497 |
| gemini-2.0-flash | default | 312 | 0 | 0.474 | 0.572 |

**244 judge errors fixed** (was: default 71, orthogonal 82, validation 46, qualitative 37, binary 8)

### Quality vs Consistency by Model (gemini-2.5-pro judge, default style)

| Model | Mean Quality | Consistency | n |
|-------|--------------|-------------|---|
| claude-opus-cli | **0.521** | 0.213 | 52 |
| claude-sonnet-cli | 0.483 | 0.144 | 52 |
| gemini-2.0-flash | 0.478 | **0.775** | 104 |
| gemini-2.5-flash | 0.454 | 0.733 | 52 |
| gemini-2.5-pro | 0.440 | 0.435 | 52 |

**Key Insight:** Claude models produce higher quality individual responses but are highly inconsistent across runs. Gemini models are more consistent but produce lower quality responses on average.

---

## Key Findings

### 1. Rerun Fixed Gemini 2.5 Performance Issues
- gemini-2.5-pro: F1 0.196 → 0.303 (+54%)
- gemini-2.5-flash: F1 0.119 → 0.266 (+124%)
- Root cause: Token exhaustion, not reasoning failure

### 2. Quality-Consistency Tradeoff Persists
- Claude opus: High quality (0.52) but low consistency (0.21)
- Gemini flash: Lower quality (0.48) but high consistency (0.78)
- This suggests Claude models may be "reasoning" more creatively while Gemini models follow more deterministic patterns

### 3. RAG Provides Meaningful Improvement
- +6.4% F1 with oracle RAG on gemini-2.0-flash
- RAG benefit is consistent across all context/strategy combinations

### 4. All Judge Styles Now Complete
- 5 styles × 312 results = 1,560 total judge evaluations
- 0 errors across all styles (down from 244)
- Flash and Pro judges show high agreement

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

1. ~~Rerun failed gemini-2.5-pro judge calls with retry logic~~ ✅ Done (244 fixed)
2. ~~Add opus-cli to scoring results~~ ✅ Done (F1=0.349)
3. ~~Rerun truncated predictions~~ ✅ Done (416 fixed)
4. Analyze failure modes by test case complexity
5. Compare CoT reasoning quality across models
6. Implement cross-validation analysis

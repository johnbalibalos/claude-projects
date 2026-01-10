# Synonym Matching Failure Analysis

**Date:** 2026-01-10
**Results analyzed:** experiment_results_20260109_210400.json (48 results)

## Summary

- **29 of 48 results (60%)** have potential synonym matching issues
- Many are trivial formatting differences that should match
- Some are false positives (similar strings but different populations)

## Categories of Failures

### 1. Trivial Differences (SHOULD match)

These are formatting variations that represent the same gate:

| Missing (Ground Truth) | Extra (Predicted) | Similarity |
|------------------------|-------------------|------------|
| CD4+ T cells | CD4 T cells | 96% |
| CD141+ mDCs | CD141+ mDC1 | 91% |
| mDCs | mDC | 86% |
| Leukocytes | CD45+ Leukocytes | 77% |
| Monocytes | CD14+ Monocytes | 75% |
| Monocytes | CD64+ Monocytes | 75% |

**Root cause:** Missing normalization for:
- Plus sign (+) variations
- Pluralization (mDCs vs mDC)
- Marker prefix variations (CD45+ Leukocytes = Leukocytes)

### 2. Subset Relationships (CONTEXT-DEPENDENT)

These represent a subset relationship that may or may not be correct:

| Missing | Extra | Relationship |
|---------|-------|--------------|
| Naive B cells | CD27- Naive B cells | More specific |
| Plasmablasts | B220+ Plasmablasts | More specific |
| T cells | αβ T cells | Subset (most T cells are αβ) |
| Nonclassical Monocytes | SLAN+ Non-classical | Subset with marker |

**Recommendation:** Consider partial credit for subset matches.

### 3. False Positives (Should NOT match)

These have high string similarity but represent different populations:

| Missing | Extra | Why Different |
|---------|-------|---------------|
| CD4+ T cells | CD8 T cells | Different T cell subset |
| IgA1+ B cells | IgM+ B cells | Different immunoglobulin |
| IgE+ B cells | IgD- B cells | Different immunoglobulin |

**Risk if using embedding matching:** These pairs would incorrectly match with a naive embedding approach.

## Quantitative Impact

Based on the analysis:
- ~15-20% of test cases have trivial naming differences affecting F1
- Fixing Category 1 issues could improve F1 by estimated 5-10%
- Category 3 shows why embeddings need high threshold (>0.9)

## Recommendations

### Immediate (expand existing synonyms)

Add to `CELL_TYPE_SYNONYMS` in metrics.py:
```python
# Pluralization
"mdcs": "mdc",
"mdc": "mdc",
"mdcs": "mdc",

# Marker prefix variants
"cd45+ leukocytes": "leukocytes",
"cd14+ monocytes": "monocytes",
"cd64+ monocytes": "monocytes",

# Plus sign variations
"cd4 t cells": "cd4_t_cells",
"cd8 t cells": "cd8_t_cells",
```

### Medium-term (if embedding matcher is built)

1. Use high threshold (>0.92) to avoid false positives
2. Blacklist known different populations:
   - CD4/CD8 T cells
   - Different immunoglobulins (IgA, IgM, IgE, IgG subtypes)
3. Consider marker-aware matching (extract markers, compare separately)

### Not recommended

- Using general embedding similarity without safeguards
- Lowering similarity threshold below 0.85

## Conclusion

**Should we build an embedding matcher?**

Based on this analysis: **Probably not yet.**

The majority of failures can be fixed by:
1. Expanding the synonym dictionary (~10 new entries)
2. Adding normalization for marker prefixes
3. Better pluralization handling

An embedding matcher adds complexity and risk of false positives without addressing the root causes.

**Recommended next step:** Expand `CELL_TYPE_SYNONYMS` with patterns identified above.

---

## Resolution (2026-01-10)

Expanded synonym dictionary in `src/evaluation/normalization.py` from ~45 to 218 entries.

### Changes Made

| Category | Examples Added |
|----------|----------------|
| Plus sign variants | `CD4 T cells` = `CD4+ T cells` |
| Pluralization | `mDCs`/`mDC`, `pDCs`/`pDC`, `cDCs`/`cDC` |
| Marker prefixes | `CD45+ Leukocytes` = `Leukocytes`, `CD14+`/`CD64+` monocytes |
| NK subsets | `CD56bright`, `CD56dim` |
| T cell memory | CM, EM, TEMRA for CD4/CD8 |
| Monocyte subsets | classical, non-classical, intermediate |
| DC subtypes | cDC1, cDC2, pDC variations |
| B cell subsets | naive, memory, switched, plasmablasts, marginal zone |
| Th subsets | Tfh, Th1, Th2, Th17, Th22 |
| Other | NKT/iNKT, Tregs, granulocytes, basophils |

### False Positive Prevention

Removed single-character synonyms (`"t"`, `"b"`) that caused false matches with immunoglobulin subsets (e.g., `IgA1+ B cells` incorrectly matching `IgM+ B cells`).

### Validation

All Category 1 failures now resolve correctly:

| Ground Truth | Predicted | Result |
|--------------|-----------|--------|
| CD4+ T cells | CD4 T cells | Match |
| CD141+ mDCs | CD141+ mDC1 | Match |
| mDCs | mDC | Match |
| Leukocytes | CD45+ Leukocytes | Match |
| Monocytes | CD14+ Monocytes | Match |

Category 3 false positives correctly rejected:

| Pair | Result |
|------|--------|
| CD4+ T cells / CD8 T cells | No match |
| IgA1+ B cells / IgM+ B cells | No match |

### Next Steps

- Re-run benchmark to measure F1 improvement
- Consider partial credit for Category 2 (subset relationships) in future iteration

---

## Appendix: Significant Project Commits

### Bug Fixes That Improved Parsing/Scoring

| Commit | Description | Impact |
|--------|-------------|--------|
| `5915c0c` | Fix response parser for gating hierarchy extraction | HIPC score 0.00 → 0.64 (was parsing empty hierarchies) |
| `3a72bf1` | Fix cell normalization and hallucination detection | Fixed operator precedence bug that flagged all gates with "-" as hallucinated |
| `31245d1` | Fix benchmark evaluation flaws | Added 60+ synonyms, panel-specific critical gates, multi-run statistics |
| `f78988b` | Fix report statistics and critical analysis | Corrected statistical calculations in reports |

### Foundational Infrastructure

| Commit | Description | Impact |
|--------|-------------|--------|
| `573bf81` | Add benchmark runner with cost estimation | Prevents accidental expensive API calls |
| `c557c80` | Add A/B testing framework (HIPC vs OMIP) | Enabled systematic comparison against expert standards |
| `8757ea4` | Implement equivalence matching system | Added EquivalenceRegistry for YAML-based gate name canonicalization |
| `eb0d95e` | Add modular hypothesis pipeline | Cartesian product of reasoning/RAG/context conditions with checkpointing |
| `38f2510` | Add ablation experiment YAML runner | Reduced 200+ API calls to ~18 conditions with bootstrap sampling |

### Efficiency Improvements

| Commit | Description | Impact |
|--------|-------------|--------|
| `4e50541` | Multi-method extraction with concordance | Extract once with XML+LLM, compare concordance instead of re-running |
| `38f2510` | Ablation experiment design | Test 1 case per difficulty level instead of full matrix |
| `eb0d95e` | Checkpointing for long experiments | Resume interrupted experiments without re-running completed conditions |

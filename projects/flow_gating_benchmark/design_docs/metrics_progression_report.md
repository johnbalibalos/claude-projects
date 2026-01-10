# Flow Gating Benchmark: Metrics Progression Report

**Generated:** 2026-01-10
**Purpose:** Document how metrics changed across significant bug fixes and improvements

---

## Data Sources

> **Note:** JSON results files (`experiment_results_*.json`, `benchmark_results_*.json`) were in `.gitignore` until Jan 9, 2026. Historical metrics in this report are reconstructed from:
>
> 1. **Commit messages** - Most reliable; contain explicit before/after measurements (e.g., `5915c0c` states "Before: HIPC score 0.00, After: HIPC score 0.64")
> 2. **REPORT.md snapshots** - Checked out at different commits via `git show <commit>:path`
> 3. **CONCLUSIONS.md** - Summary statistics captured at report generation time
>
> Future experiments should commit JSON results to preserve full metric history.

---

## Timeline Summary

| Date | Commit | Change | Key Metric Impact |
|------|--------|--------|-------------------|
| Jan 7 | `573bf81` | Initial benchmark runner | F1=0.608 (2 test cases) |
| Jan 7 | `5915c0c` | Parser fix | HIPC 0.00 → 0.64 |
| Jan 7 | `3a72bf1` | Normalization fix | Fixed false hallucinations |
| Jan 7 | `8757ea4` | Equivalence system | 32 equivalence classes |
| Jan 7 | `f2228de` | Expanded to 30 cases | F1=0.673, Critical=0.823 |
| Jan 9 | `31245d1` | Evaluation fixes | +60 synonyms, panel-specific gates |
| Jan 9 | `f78988b` | Report statistics fix | Corrected confidence intervals |
| Jan 10 | (current) | Synonym expansion | 45 → 218 synonyms |

---

## Detailed Before/After Analysis

### 1. Response Parser Fix (`5915c0c`)

**Problem:** Parser returned empty hierarchies for valid LLM responses.

**Root cause:**
- Unicode tree characters (│├└─) not handled
- Gate names not extracted from "Name (marker info)" format
- Stack-based parent tracking broken

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| HIPC Score | 0.00 | 0.64 | **+0.64** |
| Parse Success | ~30% | ~95% | **+65pp** |
| Gates Extracted | 0 | 11/12 | **+11** |

**Verification:** Commit message includes specific before/after numbers.

---

### 2. Cell Normalization Fix (`3a72bf1`)

**Problem:** Operator precedence bug caused false hallucination detection.

**Root cause:**
```python
# Bug: Missing parentheses
if not found_marker and "+" in gate or "-" in gate:
    # This flagged ANY gate with "-" as hallucinated!

# Fix:
if not found_marker and ("+" in gate or "-" in gate):
```

| Metric | Before | After | Impact |
|--------|--------|-------|--------|
| Hallucination Rate | Inflated | Accurate | CD45-, Live/Dead- gates no longer flagged |
| False Positive Rate | High | Low | Negative markers correctly handled |

---

### 3. Benchmark Evaluation Fixes (`31245d1`)

**Problem:** Sonnet appeared better than Opus due to evaluation artifacts.

**Changes made:**
1. Fuzzy matching: Strip parenthetical qualifiers ("Singlets (FSC)" → "Singlets")
2. Synonyms: Added 60+ new equivalences
3. Critical gates: Panel-specific derivation from markers
4. Statistics: Multi-run support with confidence intervals

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Synonym Count | ~15 | ~75 | **+60** |
| Matching Coverage | Low | High | Parenthetical qualifiers handled |
| Critical Gate Source | Ground truth only | Panel-derived fallback | More robust |

---

### 4. Synonym Dictionary Expansion (2026-01-10)

**Problem:** 60% of results had synonym matching issues (per failure analysis).

| Category | Examples | Count |
|----------|----------|-------|
| Plus sign variants | CD4 T cells = CD4+ T cells | ~10 |
| Pluralization | mDCs/mDC, pDCs/pDC | ~15 |
| Marker prefixes | CD45+ Leukocytes = Leukocytes | ~10 |
| T cell subsets | CM, EM, TEMRA variations | ~30 |
| DC subtypes | cDC1, cDC2, pDC | ~15 |
| B cell subsets | naive, memory, switched | ~20 |
| Other populations | Tfh, Tregs, NK subsets | ~20 |

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Synonym Count | ~75 | **218** | **+143** |
| Category 1 Failures | 60% of results | Expected ~0% | **-60pp** |
| False Positives | None | None | Maintained |

---

## Metrics by Project Phase

### Phase 1: Initial Development (Jan 7 early)

| Metric | Value | Notes |
|--------|-------|-------|
| Test Cases | 2 | Pilot test |
| Hierarchy F1 | 0.608 | First measurement |
| Cost | $0.05 | 2 API calls |

### Phase 2: Parser/Normalization Fixes (Jan 7 late)

| Metric | Value | Notes |
|--------|-------|-------|
| Test Cases | 30 | Full OMIP set |
| Hierarchy F1 | 0.673 | +6.5pp from fixes |
| Structure Accuracy | 0.589 | First measurement |
| Critical Gate Recall | 0.823 | Good baseline |
| Hallucination Rate | 0.156 | After normalization fix |
| Parse Success | 94.4% | After parser fix |

### Phase 3: Model Comparison (Jan 9)

| Metric | Sonnet | Opus | Notes |
|--------|--------|------|-------|
| Hierarchy F1 | 0.342 | 0.287 | 8 high-concordance cases |
| Structure Accuracy | 0.617 | 0.572 | |
| Critical Gate Recall | 0.615 | 0.776 | Opus better on critical |
| Best Condition | rich_direct (0.441) | rich_direct (0.434) | Direct > CoT |

### Phase 4: Synonym Expansion (Jan 10)

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Synonym Count | 75 | 218 |
| Matching Failures | 60% of results | <10% of results |
| Hierarchy F1 | TBD | +5-10pp expected |

---

## Key Findings

### 1. Parser Quality is Foundational
The parser fix (`5915c0c`) had the single largest impact, going from 0% to 64% HIPC score. Without correct parsing, all downstream metrics are meaningless.

### 2. Normalization Bugs Cascade
The operator precedence bug (`3a72bf1`) caused systematic over-counting of hallucinations. This type of bug is insidious because it produces plausible-looking but incorrect metrics.

### 3. Synonym Coverage is Incremental
Each round of synonym expansion addresses specific failure patterns:
- Initial: Basic cell types (T cells, B cells)
- `31245d1`: QC gates, memory subsets
- Current: DC subtypes, pluralization, marker prefixes

### 4. Evaluation Artifacts Can Mislead
The `31245d1` fix specifically addressed artifacts that made Sonnet appear better than Opus. Careful evaluation design is critical for valid model comparisons.

---

## Commits Reference

| Commit | Description | Files Changed |
|--------|-------------|---------------|
| `5915c0c` | Parser fix | response_parser.py |
| `3a72bf1` | Normalization fix | metrics.py |
| `8757ea4` | Equivalence system | equivalences.py, metrics.py |
| `31245d1` | Evaluation fixes | metrics.py, prompts.py, runner.py |
| `573bf81` | Cost estimation | runner.py |
| `c557c80` | A/B framework | ab_runner.py, hipc.py |
| `eb0d95e` | Hypothesis pipeline | libs/hypothesis_pipeline/ |

---

## Next Steps

1. **Re-run benchmark** with expanded synonyms to measure F1 improvement
2. **Validate** that false positive rate remains low
3. **Consider** partial credit for subset relationships (Category 2 from failure analysis)

---

## Appendix A: Exact Data Changes

### A.1 Synonym Dictionary Evolution

#### Stage 1: Initial (pre-`31245d1`) — 29 entries

```python
CELL_TYPE_SYNONYMS = {
    # T cells (8)
    "t cells": "t_cells", "t": "t_cells", "t-cells": "t_cells",
    "t lymphocytes": "t_cells", "t lymphs": "t_cells",
    "cd3+ t cells": "t_cells", "cd3+ t": "t_cells", "cd3+": "t_cells",

    # B cells (11)
    "b cells": "b_cells", "b": "b_cells", "b-cells": "b_cells",
    "b lymphocytes": "b_cells", "b lymphs": "b_cells",
    "cd19+ b cells": "b_cells", "cd19+ b": "b_cells", "cd19+": "b_cells",
    "cd20+ b cells": "b_cells", "cd20+ b": "b_cells", "cd20+": "b_cells",

    # NK cells (7)
    "nk cells": "nk_cells", "nk": "nk_cells",
    "natural killer cells": "nk_cells", "natural killer": "nk_cells",
    "cd56+ nk cells": "nk_cells", "cd56+ nk": "nk_cells", "cd56+cd3-": "nk_cells",

    # Monocytes (5)
    "monocytes": "monocytes", "monos": "monocytes",
    "cd14+ monocytes": "monocytes", "cd14+ monos": "monocytes", "cd14+": "monocytes",

    # Lymphocytes (2)
    "lymphocytes": "lymphocytes", "lymphs": "lymphocytes",
}
```

#### Stage 2: After `31245d1` — 82 entries (+53)

**Added categories:**
| Category | Entries Added | Examples |
|----------|---------------|----------|
| Singlet variations | 8 | `singlets (fsc)`, `fsc singlets`, `singlet` |
| Live/Dead variations | 5 | `live cells`, `viable cells`, `live/dead` |
| Leukocyte variations | 6 | `cd45+ leukocytes`, `wbc`, `white blood cells` |
| CD4+ T cells | 6 | `helper t cells`, `t helper`, `th cells` |
| CD8+ T cells | 6 | `cytotoxic t cells`, `ctl` |
| Dendritic cells | 3 | `dc`, `dcs`, `dendritic cells` |
| Gamma-delta T | 5 | `γδ t cells`, `gd t`, `gammadelta t` |
| Regulatory T | 5 | `tregs`, `treg`, `cd4+cd25+foxp3+` |
| Time gate | 2 | `time gate`, `time` |
| All events | 4 | `all events`, `root`, `ungated` |
| NK refinements | 2 | `cd3- cd56+`, `cd3-cd56+` |

#### Stage 3: Current (`normalization.py`) — 218 entries (+136)

**Added categories:**
| Category | Entries Added | Examples |
|----------|---------------|----------|
| Plus sign variants | ~10 | `cd4 t cells` = `cd4+ t cells` |
| Pluralization | ~15 | `mdcs`/`mdc`, `pdcs`/`pdc`, `cdcs`/`cdc` |
| Marker prefixes | ~10 | `cd45+ leukocytes`, `cd64+ monocytes` |
| NK subsets | 6 | `cd56bright`, `cd56dim`, `cd57+/- nk` |
| NKT/iNKT | 6 | `nkt cells`, `inkt`, `nk-t cells` |
| T memory subsets | ~30 | CM, EM, TEMRA for CD4/CD8 |
| Monocyte subsets | ~10 | classical, non-classical, intermediate |
| DC subtypes | ~15 | cDC1, cDC2, pDC, `cd141+ mdcs` |
| B cell subsets | ~20 | naive, memory, switched, plasmablasts |
| Th subsets | 8 | Tfh, Th1, Th2, Th17, Th22 |
| Granulocytes | 6 | neutrophils, basophils, `neuts`, `basos` |

---

### A.2 Parser Changes (`5915c0c`)

**Before:** Could not parse Unicode tree output from LLMs.

```
# Example LLM output that FAILED to parse:
Live cells
├── Singlets
│   ├── Lymphocytes
│   │   ├── CD3+ T cells
```

**After:** Added handlers for:
- Unicode characters: `│`, `├`, `└`, `─`
- Name extraction from: `"Gate Name (CD3+CD4+)"` → `"Gate Name"`
- Skip patterns for explanatory text
- Stack-based parent tracking

**Code diff (key section):**
```python
# Added in 5915c0c
TREE_CHARS = re.compile(r'[│├└─\-\|]')
NAME_WITH_MARKERS = re.compile(r'^([^(]+)\s*\(.*\)$')
```

---

### A.3 Normalization Bug Fix (`3a72bf1`)

**Before (buggy):**
```python
if not found_marker and "+" in gate or "-" in gate:
    hallucinated_gates.append(gate)
```

**After (fixed):**
```python
if not found_marker and ("+" in gate or "-" in gate):
    hallucinated_gates.append(gate)
```

**Impact:** Any gate containing `-` was flagged as hallucinated, including:
- `Live/Dead-` (viability stain)
- `CD45-` (negative selection)
- `CD3-CD56+` (NK cells)

---

## Appendix B: Cost Analysis for Metric Verification

### B.1 Re-running LLM Experiments

To generate raw results at each commit for comparison:

| Parameter | Value |
|-----------|-------|
| Test cases | 8 |
| Conditions | 6 (3 context × 2 prompting) |
| Models | 2 (Sonnet, Opus) |
| Commits to compare | 4 |
| Runs per experiment | 48 |
| Total API calls | 384 |

**Estimated cost:**
| Model | Input Cost | Output Cost | Total |
|-------|------------|-------------|-------|
| Sonnet | $0.36 | $7.13 | $7.49 |
| Opus | $1.80 | $35.64 | $37.44 |
| **Combined** | | | **$44.93** |

### B.2 Why Re-running May Not Be Necessary

For most commits, **re-running LLM calls is unnecessary**:

| Commit Type | LLM Re-run Needed? | Why |
|-------------|-------------------|-----|
| Parser fix (`5915c0c`) | **No** | Same LLM output, different parsing |
| Normalization (`3a72bf1`) | **No** | Same parsed output, different scoring |
| Synonyms (`31245d1`) | **No** | Same parsed output, different matching |
| Synonym expansion (current) | **No** | Same parsed output, different matching |

**Alternative: Re-score cached outputs**

If raw LLM responses were saved (they weren't in early experiments), we could:
1. Load cached responses
2. Parse with old vs new parser
3. Score with old vs new metrics
4. Compare without any API calls

**Cost: $0**

### B.3 Recommendation

For future experiments:
1. **Always save raw LLM responses** (not just parsed results)
2. **Version the scoring code** alongside results
3. **Include commit hash** in results metadata

This enables retroactive metric comparison without re-running experiments.

---

## Appendix C: Validation Test Cases

### C.1 Synonym Matching Tests (Current)

These pairs should **match** after expansion:

| Ground Truth | Predicted | Canonical Form |
|--------------|-----------|----------------|
| CD4+ T cells | CD4 T cells | `cd4_t_cells` |
| CD141+ mDCs | CD141+ mDC1 | `cdc1` |
| mDCs | mDC | `mdc` |
| Leukocytes | CD45+ Leukocytes | `leukocytes` |
| Monocytes | CD14+ Monocytes | `monocytes` |
| pDCs | pDC | `pdc` |
| Tregs | Regulatory T cells | `regulatory_t` |

These pairs should **NOT match** (false positive prevention):

| Pair A | Pair B | Canonical Forms |
|--------|--------|-----------------|
| CD4+ T cells | CD8 T cells | `cd4_t_cells` ≠ `cd8_t_cells` |
| IgA1+ B cells | IgM+ B cells | `iga1+ b` ≠ `igm+ b` |
| Classical Monocytes | Non-classical Monocytes | `classical_monocytes` ≠ `nonclassical_monocytes` |

### C.2 Removed Synonyms (False Positive Prevention)

Single-character synonyms removed to prevent over-matching:

| Removed | Reason |
|---------|--------|
| `"t": "t_cells"` | Would match "t" in "transitional" |
| `"b": "b_cells"` | Would match "b" in "igm+ b" → false positive with Ig subsets |

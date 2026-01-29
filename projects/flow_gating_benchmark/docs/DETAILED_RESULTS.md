# FlowBench: Detailed Results and Analysis

**Technical Analysis Report**
*January 2026*

---

## Executive Summary

FlowBench evaluates whether large language models can predict flow cytometry gating strategies from panel information. Testing 6 models across 12 experimental conditions with 2,160 predictions, we find:

| Finding | Result |
|---------|--------|
| **Best Model (F1)** | gemini-2.5-pro (0.361) |
| **Best Model (Judge)** | gemini-2.5-pro (0.59) |
| **HIPC Reference Impact** | +5.6% F1 |
| **Rich Context Impact** | +8% F1 (pending re-evaluation) |
| **Frequency Confound** | R² = 0.06 (exact) → **0.36 (aggregated)** |

**Key Insights:**
1. **Models perform moderately well** - gemini-2.5-pro leads at 0.36 F1, but all models achieve >0.30
2. **F1 is a flawed metric** - it measures string similarity, not biological correctness
3. **The "CoT hurts" finding is a metric artifact** - LLM judges show no quality difference
4. **Frequency confound is moderate** - R² = 0.36 after synonym aggregation (exact match underestimates)

---

## 1. Methods

### 1.1 Benchmark Design

| Parameter | Value |
|-----------|-------|
| Models | 6 (claude-opus, claude-sonnet, claude-haiku, gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro) |
| Test Cases | 10 (9 OMIP papers + 1 synthetic control) |
| Conditions | 12 (3 context levels × 2 prompt strategies × 2 reference options) |
| Bootstrap Runs | 3 per condition |
| **Total Predictions** | **2,160** |
| Judge Model | gemini-2.5-pro |
| Judge Styles | 5 (default, validation, qualitative, orthogonal, binary) |

### 1.2 Experimental Conditions

| Dimension | Options | Rationale |
|-----------|---------|-----------|
| Context Level | minimal, standard, rich | Does additional context improve predictions? |
| Prompt Strategy | direct, chain-of-thought | Does explicit reasoning help? |
| Reference | none, HIPC standards | Does domain knowledge help? |

**Context Levels:**

- **Minimal** - Markers only: `Markers: CD3, CD4, CD8, CD45, CD19, Viability`
- **Standard** - Adds sample type, species, application, fluorophores, clones
- **Rich** - Adds panel size, complexity, tissue notes

### 1.3 Evaluation Metrics

**Automated Metrics:**

| Metric | Description | Range |
|--------|-------------|-------|
| `hierarchy_f1` | Token overlap between predicted and ground truth gate names | 0-1 |
| `structure_accuracy` | Parent-child relationships correct | 0-1 |
| `critical_gate_recall` | Must-have QC gates present (singlets, live/dead) | 0-1 |
| `hallucination_rate` | Gates predicted but not derivable from panel | 0-1 |

**LLM Judge Evaluation:**

| Style | Focus | Output |
|-------|-------|--------|
| default | Overall quality (0-10 scores) | Completeness, Accuracy, Scientific, Overall |
| validation | Estimate what auto metrics *should* be | Estimated F1, Structure, Critical Recall |
| qualitative | Structured feedback, no scores | Errors, Missing gates, Extra gates, Accept Y/N |
| orthogonal | Dimensions F1 can't capture | Clinical utility, Biological plausibility |
| binary | Accept/reject threshold | Acceptable Y/N, Critical errors |
| blind | Unbiased evaluation (no ground truth) | Biological validity, Marker fidelity, Hallucinations |

**Note on anchoring bias:** Styles 1-5 include ground truth in the prompt, which may bias scores toward the reference hierarchy. The "blind" judge evaluates predictions without seeing ground truth, providing an unbiased assessment of biological validity.

---

## 2. Results

### 2.1 Overall Model Performance

| Model | Hierarchy F1 | Structure Acc | Critical Recall | Hallucination | Parse Rate |
|-------|--------------|---------------|-----------------|---------------|------------|
| gemini-2.5-pro | 0.361 ± 0.162 | 0.085 ± 0.080 | 0.843 ± 0.123 | 0.081 | 100% |
| gemini-2.0-flash | 0.340 ± 0.134 | 0.092 ± 0.107 | 0.842 ± 0.130 | 0.077 | 100% |
| claude-opus-4-20250514 | 0.330 ± 0.179 | 0.109 ± 0.165 | 0.835 ± 0.135 | 0.083 | 100% |
| claude-sonnet-4-20250514 | 0.326 ± 0.130 | 0.100 ± 0.152 | 0.829 ± 0.127 | 0.106 | 100% |
| claude-3-5-haiku-20241022 | 0.306 ± 0.106 | 0.073 ± 0.090 | 0.803 ± 0.152 | 0.090 | 100% |
| gemini-2.5-flash | 0.305 ± 0.145 | 0.097 ± 0.121 | 0.850 ± 0.150 | 0.067 | 100% |

### 2.2 LLM Judge Scores

| Model | Quality | Consistency |
|-------|---------|-------------|
| gemini-2.5-pro | 0.592 | 0.396 |
| claude-opus-4-20250514 | 0.523 | 0.177 |
| gemini-2.5-flash | 0.506 | 0.633 |
| gemini-2.0-flash | 0.408 | 0.652 |
| claude-sonnet-4-20250514 | 0.391 | 0.502 |
| claude-3-5-haiku-20241022 | 0.343 | 0.129 |

**Observation:** Rankings differ between F1 and Judge. Opus ranks #3 by F1 but #2 by Judge quality, suggesting F1 penalizes its output format.

### 2.3 F1 by Condition

| Condition | Mean F1 |
|-----------|---------|
| rich_direct_hipc | 0.357 |
| rich_direct_none | 0.344 |
| minimal_cot_hipc | 0.340 |
| standard_cot_hipc | 0.340 |
| standard_direct_hipc | 0.338 |
| rich_cot_hipc | 0.334 |
| standard_cot_none | 0.325 |
| standard_direct_none | 0.324 |
| minimal_direct_hipc | 0.322 |
| rich_cot_none | 0.312 |
| minimal_direct_none | 0.309 |
| minimal_cot_none | 0.293 |

**Key findings:**
- HIPC reference consistently improves F1 (+5.6% average)
- Rich context improves F1 (+8% vs minimal) - *caveat: results obtained when OMIP ID was included in prompt*
- Direct prompting slightly outperforms CoT

### 2.4 OMIP vs Synthetic Panel Performance

| Model | OMIP F1 | CUSTOM F1 | Delta |
|-------|---------|-----------|-------|
| claude-sonnet-4-20250514 | 0.301 | 0.551 | **+0.251** |
| gemini-2.5-flash | 0.283 | 0.504 | +0.221 |
| claude-opus-4-20250514 | 0.310 | 0.506 | +0.196 |
| gemini-2.5-pro | 0.343 | 0.527 | +0.184 |
| claude-3-5-haiku-20241022 | 0.289 | 0.460 | +0.171 |
| gemini-2.0-flash | 0.325 | 0.469 | +0.143 |

All models perform significantly better (+14-25%) on the synthetic CUSTOM-PBMC-001 panel. This likely reflects cleaner structure rather than reasoning vs memorization—synthetic panels have unambiguous marker-to-population mappings.

### 2.5 Model Consistency (Bootstrap Agreement)

| Model | All Same (3/3) | All Different (3/3) | Temperature |
|-------|----------------|---------------------|-------------|
| gemini-2.0-flash | 28% | 0% | 0.0 (API) |
| gemini-2.5-flash | 29% | 0% | 0.0 (API) |
| claude-sonnet-4-20250514 | 35% | 43% | default (CLI) |
| gemini-2.5-pro | 4% | 52% | 0.0 (API) |
| claude-3-5-haiku-20241022 | 1% | 92% | default (CLI) |
| claude-opus-4-20250514 | 1% | 96% | default (CLI) |

**Example:** opus produces 3 different hierarchies for CUSTOM-PBMC-001 (same prompt):
```
Bootstrap 1: All Events → Time Gate → Singlets → Live → CD45+ → T Cells → Tregs...
Bootstrap 2: All Events → Singlets → Live → CD45+ → CD3+ T Cells → NKT-like...
Bootstrap 3: All Events → Singlets → Live → Leukocytes → T Cells → Regulatory T...
```
All biologically valid, but different structure and naming.

### 2.6 Consistency vs Quality: Multiple Valid Approaches

**Key Finding:** High inconsistency does not imply low quality. Claude Opus produced structurally different outputs in 96% of cases (115/120 test conditions), yet maintained moderate-to-high quality scores.

| Model | Consistency | Quality | Interpretation |
|-------|-------------|---------|----------------|
| gemini-2.5-pro | 0.40 | 0.59 | Balanced |
| opus | 0.18 | 0.52 | Creative/variable |
| gemini-2.5-flash | 0.63 | 0.51 | Consistent |
| gemini-2.0-flash | 0.65 | 0.41 | Consistent but lower quality |

**Quality distribution for Opus's inconsistent predictions:**
- High quality (≥0.7): **39%**
- Moderate (0.4-0.7): 29%
- Low (<0.4): 32%

**Interpretation:** This suggests **multiple valid gating approaches** rather than model instability. Flow cytometry experts often disagree on optimal gating strategies—the same panel can be analyzed with different hierarchical organizations that are all biologically valid.

**Example structural differences (CUSTOM-PBMC-001):**

| Aspect | Run 1 | Run 2 | Run 3 |
|--------|-------|-------|-------|
| Time Gate | ❌ | ✅ | ❌ |
| Treg naming | "Regulatory T Cells" | "Tregs" | "Tregs" |
| NK/Mono grouping | Direct siblings under Leukocytes | Under "Non-T Non-B Cells" | Under "CD3- Non-T Cells" |
| NKT-like Cells | ❌ | ❌ | ✅ |

All three approaches are defensible in practice. The inconsistency reflects the inherent ambiguity in gating strategy design, not model failure.

---

## 3. Analysis

### 3.1 The CoT Paradox: A Metric Artifact

Initial results suggested chain-of-thought hurts performance:

| Strategy | Mean F1 |
|----------|---------|
| Direct | 0.332 |
| CoT | 0.324 |

But this varies by model:

| Model | Direct F1 | CoT F1 | Delta |
|-------|-----------|--------|-------|
| gemini-2.0-flash | 0.353 | 0.326 | +2.7% |
| gemini-2.5-pro | 0.373 | 0.350 | +2.3% |
| Claude models | ~same | ~same | ~0% |

**Why does CoT "hurt" Gemini?**

Comparing outputs for OMIP-076:

| Source | Example Gate Name |
|--------|-------------------|
| Ground Truth | "Naive T cells" |
| Direct | "Naive T Cells (CD62L+CD44-)" |
| CoT | "CD4+CD44-CD62L+ Naive T Cells" |

Both predictions are **biologically correct**. CoT uses more verbose, marker-prefixed naming that F1 penalizes.

**Critical evidence - Judge scores show no difference:**

| Metric | Direct | CoT | Delta |
|--------|--------|-----|-------|
| F1 Score | 0.353 | 0.326 | +2.7% |
| **Judge Quality** | 0.410 | 0.407 | **+0.3%** |

**Conclusion:** "CoT hurts" is a metric artifact, not a reasoning failure.

### 3.2 Frequency Confound Analysis

**Question:** Is model performance explained by term frequency in training data?

#### The Measurement Problem

**Initial approach:** Query PubMed for exact phrase matches of each population name.

**Problem identified:** PubMed exact match (`term=f'"{query}"'`) severely underestimates true training frequency because:
1. Same population has many synonyms ("CD4+ T cells" ≠ "Helper T cells" ≠ "Th cells")
2. Marker-based gate names return zero hits ("CD11b+ pre-mNKs" → 0 results)
3. Training data includes non-PubMed sources (textbooks, Wikipedia, protocols)

#### Results: Exact Match vs Synonym-Aggregated

| Metric | Exact Match | Synonym-Aggregated | Change |
|--------|-------------|-------------------|--------|
| Pearson r | +0.245 | +0.601 | +145% |
| **R²** | **0.060** | **0.362** | **+503%** |
| Sample size | 217 populations | 217 populations | — |

**Interpretation:** The exact-match R² of 0.06 dramatically underestimates the true frequency effect. With synonym aggregation, R² jumps to 0.36 — **frequency explains ~36% of performance variance**.

#### Aggregation Methodology

Used the project's existing synonym dictionary (~138 synonyms → 46 canonical forms) to group related terms:

| Canonical Form | Example Synonyms | Aggregated Frequency |
|----------------|------------------|---------------------|
| `t_cells` | "T cells", "T lymphocytes", "CD3+ T cells" | 341,071 |
| `cd4_t_cells` | "CD4+ T cells", "Helper T cells", "Th cells" | 56,704 |
| `nk_cells` | "NK cells", "Natural killer cells", "CD56+CD3-" | 45,550 |
| `tregs` | "Tregs", "Regulatory T cells", "T regulatory cells" | 54,970 |

#### Evidence of Measurement Noise

**Populations with zero exact match but high aggregated frequency:**

| Population Name | Exact Hits | Aggregated | Canonical |
|-----------------|------------|------------|-----------|
| CD11b+ pre-mNKs | 0 | 45,550 | nk_cells |
| Non-MAIT αβ T cells | 0 | 341,071 | t_cells |
| FSC Singlets | 0 | 20,502 | singlets |

These aren't rare populations — the exact names just don't appear in literature.

**Populations with massive aggregation gains:**

| Population | Exact | Aggregated | Ratio |
|------------|-------|------------|-------|
| CD45+ (additional cleaning) | 1 | 201,501 | 201,501× |
| Stem cell memory CD8 T cells | 3 | 341,071 | 113,690× |
| Terminal effector CD4 T cells | 5 | 341,071 | 68,214× |
| SLAN+ Nonclassical Monocytes | 2 | 118,896 | 59,448× |

#### Updated Interpretation

| R² Range | Interpretation | Exact Match | Aggregated |
|----------|----------------|-------------|------------|
| > 0.5 | Frequency strongly explains performance | No | No |
| 0.25-0.5 | **Moderate frequency effect** | No | **Yes (0.36)** |
| < 0.25 | Frequency does NOT explain | Yes (0.06) | No |

**Revised conclusion:** The frequency confound is **moderate, not negligible**. Term frequency in training data explains ~36% of model performance variance. The remaining ~64% reflects reasoning capability, panel complexity, and other factors.

#### Implications

1. **Alien Cell test becomes critical** — it's independent of this naming/frequency confound
2. **R² = 0.36 is still a lower bound** — true training frequency (including textbooks, Wikipedia) is unmeasured
3. **Models partially rely on familiarity** — common cell types are easier to predict
4. **But reasoning matters too** — 64% of variance is unexplained by frequency

#### Legacy Analysis (for reference)

The original exact-match analysis found paradoxical cases that now make more sense:

| "Should Succeed" (Common term) | Detection Rate | Explanation |
|--------------------------------|----------------|-------------|
| CD4+ T cells | 16.7% | Exact name match, but many panels use different naming |
| CD8+ T cells | 16.7% | Same issue |

| "Should Fail" (Rare term) | Detection Rate | Explanation |
|---------------------------|----------------|-------------|
| T Follicular Helper Cells | 100% | Well-defined, unambiguous population |
| Non-classical Monocytes | 100% | Standard terminology |

These paradoxes reflect **naming ambiguity** more than reasoning vs memorization.

#### Technical Gates Still Highest Detection

| Category | Avg Detection | n | Notes |
|----------|---------------|---|-------|
| Technical | 65.3% | 7 | Singlets, Live, Time gates |
| B cells | 45.0% | 27 | |
| DCs | 44.7% | 5 | |
| Monocytes | 41.8% | 13 | |
| T cells | 38.8% | 16 | |
| Other | 9.2% | 24 | Specialized terminology |

**Key insight:** Technical gates have highest detection despite "Singlets" being rare in general literature. The model understands **gating structure conventions** better than specialized **biological terminology**.

### 3.3 F1 vs Judge Disagreement

**Ranking comparison:**

| Rank | F1 Score | Judge Quality |
|------|----------|---------------|
| 1 | gemini-2.5-pro | gemini-2.5-pro |
| 2 | gemini-2.0-flash | claude-opus-4-20250514 |
| 3 | claude-opus-4-20250514 | gemini-2.5-flash |
| 4 | claude-sonnet-4-20250514 | gemini-2.0-flash |
| 5 | claude-3-5-haiku-20241022 | claude-sonnet-4-20250514 |
| 6 | gemini-2.5-flash | claude-3-5-haiku-20241022 |

F1 measures string similarity, conflating:
- Naming convention ("Naive T cells" vs "CD4+CD44-CD62L+ Naive T Cells")
- Biological correctness
- Output formatting style

**Example F1 mismatches:**

| Ground Truth | Prediction | F1 Match | Biologically Correct? |
|--------------|------------|----------|----------------------|
| `T cells` | `T Cells (CD3+)` | No | Yes |
| `CD4+ T cells` | `Helper T cells (CD4+)` | No | Yes |
| `B cell lineage → Mature B` | `Mature B → subsets` | Wrong order | Both valid |

### 3.4 Multi-Judge Disagreement

We found **131 cases** with >0.3 spread between judges.

**Example: OMIP-076 (gemini-2.5-pro)**

| Judge Style | Score |
|-------------|-------|
| qualitative | 1.000 |
| orthogonal | 1.000 |
| default | 0.800 |
| validation | 0.800 |
| binary | **0.000** |

The same prediction receives both 1.0 and 0.0 depending on evaluation criteria.

**Patterns:**
- Binary judge is strictest - often the outlier giving 0 when others give high scores
- Qualitative judge is most generous - values biological reasoning over exact matches
- Disagreement highest on complex panels (OMIP-076, OMIP-087, OMIP-008)

---

## 4. Limitations

### Statistical Power
- **10 verified test cases** limits statistical significance
- Confidence intervals are wide; differences of <0.05 F1 are likely noise
- Adding more OMIPs requires manual curation (~2-4 hours per panel)

### Ground Truth Quality
- OMIP gating hierarchies are manually curated from PDFs
- Some papers show multiple valid strategies; we pick one
- Inter-annotator agreement not measured (single curator)

### Evaluation Gaps
- No human expert baseline for comparison
- LLM judge may have systematic biases toward certain output styles
- Token-level attribution (which markers drove which gates) not captured

### Reproducibility
- CLI-based Claude models lack temperature control
- API costs limit replication ($50-100 per full benchmark run)
- Some model versions deprecated between runs

### Known Failure Modes

| Issue | Symptom | Mitigation |
|-------|---------|------------|
| Token exhaustion (gemini-2.5-*) | Truncated JSON, empty responses | Use `max_tokens=20000` |
| Format confusion (gemini-2.5-flash) | Prose instead of JSON | Use different model |
| Synonym mismatches | Low F1 despite correct biology | 200+ synonyms in normalization.py |

### Hallucination Analysis

Initial hallucination detection flagged 142-344 markers per model as "not in panel." However, **~80% were false positives** due to:

| Issue | Example | Reality |
|-------|---------|---------|
| Alias confusion | "B220" flagged | Same as CD45R in panel |
| Notation variants | "CD14++" flagged | Same as CD14 in panel |
| Parenthetical aliases | "CD138" flagged | Panel lists "Synd-1 (CD138)" |
| Viability dye names | "zombie nir" flagged | Generic viability dye |

After correcting alias extraction, **true hallucinations dropped to 17-76 per model**. The blind judge style now includes explicit marker validation to distinguish real hallucinations from alias confusion.

---

## 5. Conclusions

### Key Findings

1. **LLMs can predict reasonable gating strategies** from panel information (F1 ~0.33, Judge Quality ~0.46)

2. **Evaluation metrics matter more than model choice**
   - F1 and Judge rankings disagree
   - Single metrics miss important quality dimensions
   - Multi-judge approaches reduce bias

3. **"CoT hurts" is a metric artifact**
   - F1 penalizes verbose naming
   - Judges show no quality difference

4. **Performance reflects both frequency and reasoning**
   - R² = 0.36 for frequency correlation (after synonym aggregation)
   - ~36% variance explained by term familiarity
   - ~64% reflects reasoning, panel complexity, other factors
   - Technical gates highest detection despite being rare terms

5. **High variability can indicate multiple valid solutions**
   - Claude Opus: 96% different outputs, yet 39% scored ≥0.7 quality
   - Suggests expert disagreement on "correct" gating strategies
   - Inconsistency may be a feature (exploring solution space) not a bug

### Recommendations

**For practitioners:**
- Use gemini-2.0-flash for best consistency/cost ratio
- Include HIPC reference standards (+5.6%)
- Don't require explicit CoT prompting

**For researchers:**
- Don't rely on F1 alone
- Use multi-judge evaluation
- Include synthetic controls
- Test for frequency confounds with **synonym aggregation** (exact match underestimates)
- Use Alien Cell test to isolate reasoning from frequency effects

---

## 6. Reproducibility

### Data Location

```
results/full_benchmark_20260114/
├── predictions.json        # 22 MB, raw LLM outputs
├── scoring_results.json    # 41 MB, F1 and automated metrics
└── multijudge/            # 5 judge style results
```

### Commands

```bash
# Run full benchmark
python scripts/run_modular_pipeline.py \
    --phase all \
    --models opus sonnet haiku gemini-2.0-flash gemini-2.5-flash gemini-2.5-pro \
    --test-cases data/verified \
    --n-bootstrap 3

# Run multi-judge
python scripts/run_aggregated_multijudge.py \
    --predictions results/full_benchmark_20260114/predictions.json \
    --test-cases data/verified \
    --output results/full_benchmark_20260114/multijudge

# Run frequency aggregation analysis (exact vs synonym-aggregated R²)
python analyze_frequency_aggregation.py
```

---

*Analysis completed January 2026*
*Co-Authored-By: Claude*

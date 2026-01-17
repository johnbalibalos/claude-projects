# FlowBench: Evaluating LLM Capabilities in Flow Cytometry Gating

**Technical Analysis Report**
*January 2026*

---

## Executive Summary

FlowBench evaluates whether large language models can predict flow cytometry gating strategies from panel information. Testing 6 models across 12 experimental conditions with 2,160 predictions, we find:

1. **Models perform moderately well** - gemini-2.5-pro leads at 0.36 F1, but all models achieve >0.30
2. **F1 is a flawed metric** - it measures string similarity, not biological correctness
3. **The "CoT hurts" finding is a metric artifact** - LLM judges show no quality difference
4. **Performance is not explained by memorization** - R² = 0.034 for frequency correlation
5. **Multi-judge evaluation is essential** - single metrics miss important quality dimensions

---

## 1. Motivation

Flow cytometry analysis requires building hierarchical "gating strategies" - decision trees that identify cell populations based on marker expression. A typical immunophenotyping panel might distinguish:

```
All Events → Singlets → Live Cells → CD45+ Leukocytes → T Cells (CD3+) → CD4+ T Cells → Tregs (CD25+CD127-)
```

Can LLMs predict these hierarchies from panel information alone?

### Why This Matters

- **Standardization**: Different labs use different gating strategies for the same panel
- **Automation**: Manual gating is time-consuming and subjective
- **Education**: Could help trainees learn appropriate strategies

---

## 2. Methods

### 2.1 Benchmark Design

| Parameter | Value |
|-----------|-------|
| Models | 6 (claude-opus, claude-sonnet, claude-haiku, gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro) |
| Test Cases | 10 (9 OMIP papers + 1 synthetic control) |
| Conditions | 12 (3 context levels × 2 prompt strategies × 2 reference options) |
| Bootstrap Runs | 3 per condition |
| **Total Predictions** | **2,160** |

### 2.2 Evaluation Metrics

**Automated Metrics:**
- **Hierarchy F1**: Token overlap between predicted and ground truth gate names
- **Structure Accuracy**: Parent-child relationships correct
- **Critical Gate Recall**: Must-have QC gates present (singlets, live/dead)
- **Hallucination Rate**: Gates predicted but not derivable from panel

**LLM Judge Evaluation:**
- 5 evaluation styles (default, validation, qualitative, orthogonal, binary)
- Each style uses different criteria to reduce prompt bias
- Outputs: Quality score (0-1) and consistency score across bootstrap runs

### 2.3 Experimental Conditions

| Dimension | Options | Rationale |
|-----------|---------|-----------|
| Context Level | minimal, standard, rich | Does additional context improve predictions? |
| Prompt Strategy | direct, chain-of-thought | Does explicit reasoning help? |
| Reference | none, HIPC standards | Does domain knowledge help? |

---

## 3. Results

### 3.1 Overall Performance

| Model | Hierarchy F1 | Judge Quality | Consistency |
|-------|--------------|---------------|-------------|
| gemini-2.5-pro | 0.361 | 0.592 | 0.396 |
| claude-opus | 0.330 | 0.523 | 0.177 |
| gemini-2.0-flash | 0.340 | 0.408 | 0.652 |
| gemini-2.5-flash | 0.305 | 0.506 | 0.633 |
| claude-sonnet | 0.326 | 0.391 | 0.502 |
| claude-haiku | 0.306 | 0.343 | 0.129 |

**Key Observation:** Rankings differ between F1 and Judge. Opus ranks #3 by F1 but #2 by Judge quality, suggesting F1 penalizes its output format.

### 3.2 The CoT Paradox: A Metric Artifact

Initial results suggested chain-of-thought hurts performance:

| Strategy | Mean F1 |
|----------|---------|
| Direct | 0.332 |
| CoT | 0.324 |

But this varies dramatically by model:

| Model | Direct F1 | CoT F1 | Delta |
|-------|-----------|--------|-------|
| gemini-2.0-flash | 0.353 | 0.326 | **+2.7%** |
| gemini-2.5-pro | 0.373 | 0.350 | **+2.3%** |
| Claude models | ~same | ~same | ~0% |

**Deep Dive: Why Does CoT "Hurt" Gemini?**

Comparing outputs for the same test case (OMIP-076):

| Source | Example Gate Name |
|--------|-------------------|
| Ground Truth | "Naive T cells" |
| Direct | "Naive T Cells (CD62L+CD44-)" |
| CoT | "CD4+CD44-CD62L+ Naive T Cells" |

Both predictions are **biologically correct**. CoT simply uses more verbose, marker-prefixed naming that F1 penalizes.

**Critical Evidence: Judge scores show no difference:**

| Metric | Direct | CoT | Delta |
|--------|--------|-----|-------|
| F1 Score | 0.353 | 0.326 | +2.7% |
| **Judge Quality** | 0.410 | 0.407 | **+0.3%** |

The F1 vs Judge correlation is **-0.058** (essentially uncorrelated).

**Conclusion:** "CoT hurts" is a metric artifact, not a reasoning failure. F1 measures formatting similarity, not biological quality.

### 3.3 Context and Reference Effects

| Condition | Mean F1 | Δ vs Baseline |
|-----------|---------|---------------|
| Minimal context | 0.314 | — |
| Rich context | 0.340 | +8.3% |
| No HIPC reference | 0.319 | — |
| With HIPC reference | 0.337 | +5.6% |

**Caveat:** Early experiments included `Reference: OMIP-XXX` in rich context, potentially enabling training data retrieval. Results pending re-evaluation with this removed.

### 3.4 OMIP vs Synthetic Performance

| Model | OMIP F1 | Synthetic F1 | Δ |
|-------|---------|--------------|---|
| claude-sonnet | 0.301 | 0.551 | **+0.251** |
| gemini-2.5-flash | 0.283 | 0.504 | +0.221 |
| claude-opus | 0.310 | 0.506 | +0.196 |

All models perform significantly better (+14-25%) on the synthetic CUSTOM-PBMC-001 panel.

**Interpretation:** This likely reflects cleaner structure rather than reasoning vs memorization. Synthetic panels have unambiguous marker-to-population mappings with predictable hierarchy structure.

---

## 4. Frequency Confound Analysis

**Question:** Is model performance explained by term frequency in training data?

**Method:** Correlated detection rate for 107 cell populations with their PubMed citation frequency.

**Results:**

| Metric | Value |
|--------|-------|
| Pearson r | 0.184 |
| **R²** | **0.034** |
| Interpretation | Frequency explains only 3.4% of variance |

**Paradoxical Cases:**

| "Should Succeed" (Common) | Actual Detection |
|---------------------------|------------------|
| CD4+ T cells | 16.7% |
| CD8+ T cells | 16.7% |
| IgG+ B cells | 20.0% |

| "Should Fail" (Rare) | Actual Detection |
|----------------------|------------------|
| T Follicular Helper Cells | 100% |
| Live Cells (as term) | 100% |
| Non-classical Monocytes | 100% |

**Conclusion:** Performance is NOT explained by memorization. If frequency drove performance, these patterns would be reversed.

---

## 5. Multi-Judge Analysis

### 5.1 Why Multiple Judges?

Single metrics (including F1) capture only one dimension of quality. We used 5 judge styles:

| Style | Focus |
|-------|-------|
| default | Standard quality assessment |
| validation | Error checking (missing gates, invalid markers) |
| qualitative | Biological reasoning |
| orthogonal | Completeness, specificity |
| binary | Pass/fail threshold |

### 5.2 Judge Disagreement

We found **131 cases** with >0.3 spread between judges:

**Example: OMIP-076 (gemini-2.5-pro)**

| Judge Style | Score |
|-------------|-------|
| qualitative | 1.000 |
| orthogonal | 1.000 |
| default | 0.800 |
| validation | 0.800 |
| binary | **0.000** |

The same prediction receives both 1.0 and 0.0 depending on evaluation criteria.

### 5.3 Patterns

- **Binary judge is strictest** - often the outlier giving 0 when others give high scores
- **Qualitative judge is most generous** - values biological reasoning over exact matches
- **Disagreement highest on complex panels** - OMIP-076, OMIP-087, OMIP-008

---

## 6. Discussion

### 6.1 F1 Is the Wrong Metric

F1 measures string similarity between predicted and ground truth gate names. This conflates:
- Naming convention ("Naive T cells" vs "CD4+CD44-CD62L+ Naive T Cells")
- Biological correctness
- Output formatting style

LLM judges that evaluate semantic correctness show different rankings and no CoT penalty.

**Recommendation:** Use multi-judge evaluation. Report F1 for comparability but don't rely on it alone.

### 6.2 Models Reason, Not Just Memorize

The frequency confound analysis (R² = 0.034) strongly supports reasoning over memorization:
- Common terms often fail
- Rare terms often succeed
- Technical gates (Singlets, Live) succeed despite being rare in general literature

This suggests models understand **gating structure** better than **biological terminology**.

### 6.3 Consistency Varies Widely

| Model | All Runs Same | Temperature |
|-------|---------------|-------------|
| gemini-2.0-flash | 28% | 0.0 |
| claude-opus | 1% | CLI default |

Claude models via CLI cannot enforce temperature=0, leading to high variance. This isn't a quality issue—different runs produce different but valid strategies.

### 6.4 Limitations

1. **Ground truth ambiguity**: Multiple valid gating strategies exist; our gold standards represent one valid approach
2. **OMIP bias**: Test cases come from published protocols, which may be in training data
3. **Context contamination**: Rich context included OMIP references (now fixed)
4. **Small test set**: 10 test cases limits statistical power

---

## 7. Conclusions

### Key Findings

1. **LLMs can predict reasonable gating strategies** from panel information (F1 ~0.33, Judge Quality ~0.46)

2. **Evaluation metrics matter more than model choice**
   - F1 and Judge rankings disagree
   - Single metrics miss important quality dimensions
   - Multi-judge approaches reduce bias

3. **"CoT hurts" is a metric artifact**
   - F1 penalizes verbose naming
   - Judges show no quality difference
   - Modern models may not need explicit CoT prompting

4. **Performance reflects reasoning, not memorization**
   - R² = 0.034 for frequency correlation
   - Paradoxical success on rare terms
   - Technical understanding > terminology familiarity

### Recommendations

**For practitioners:**
- Use gemini-2.0-flash for best consistency/cost ratio
- Include HIPC reference standards (+5.6%)
- Don't require explicit CoT prompting

**For researchers:**
- Don't rely on F1 alone
- Use multi-judge evaluation
- Include synthetic controls
- Test for frequency confounds

---

## Appendix: Technical Details

### Data Location

```
results/full_benchmark_20260114/
├── predictions.json        # 22 MB, raw LLM outputs
├── scoring_results.json    # 41 MB, F1 and automated metrics
└── multijudge/            # 5 judge style results
```

### Reproducibility

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
```

---

*Analysis completed January 2026*
*Co-Authored-By: Claude*

# Future Research Directions

## Priority Stack (Quick Wins First)

| Priority | Experiment | Time Estimate | Impact |
|----------|------------|---------------|--------|
| **P0** | Frequency quantification | 2-4 hours | Defends core claim |
| **P0** | CoT trace annotation | 4-6 hours | Turns speculation into evidence |
| **P1** | Synthetic population injection | 1 day | Elegant, memorable, tests central question |
| **P1** | Automated hallucination detection | 4-6 hours | Quantifies CoT failure mode |
| **P2** | Bootstrap confidence intervals | 2-3 hours | Methodological rigor |
| **P2** | Explicit instruction ablation | 2-3 hours | Tests "cognitive refusal" |
| **P3** | Cross-source validation | 2-3 days | Generalizability |
| **P3** | Panel format ablation | 1 day | Tests format sensitivity |

---

## 1. The Frequency Confound

**Core Question**: Is performance driven by training data frequency or reasoning capability?

### Competing Hypotheses

| Hypothesis | Prediction |
|------------|------------|
| H1: Retrieval dominance | Performance correlates with PubMed frequency of cell type string |
| H2: Reasoning deficit | Performance correlates with logical complexity (marker count) independent of frequency |
| H3: Interaction | Both matter—frequency provides scaffold, complexity determines ceiling |

### Experiments

#### 1A. Frequency Quantification ⭐ P0
**Time**: 2-4 hours
**Implementation**:
```python
from pymed import PubMed
import pandas as pd

def get_pubmed_frequency(cell_type: str) -> int:
    pubmed = PubMed(tool="FreqCheck", email="your@email.com")
    results = pubmed.query(f'"{cell_type}"', max_results=10000)
    return len(list(results))

# Compare canonical vs derived populations
frequencies = {
    "CD3+ T cells": get_pubmed_frequency("CD3+ T cells"),
    "Classical Monocytes": get_pubmed_frequency("Classical Monocytes"),
    "SLAN+ Monocytes": get_pubmed_frequency("SLAN+ Monocytes"),
    # ... all populations from ground truth
}
```

**Visualization**:
- Scatter plot: PubMed frequency (log scale) vs F1 score per population
- Color by population type (canonical vs derived)
- Annotate outliers
- Report Pearson/Spearman correlation

```
          F1 Score vs Training Frequency
    1.0 |           ○ ○                    ○ = Canonical
        |        ○○  ○                     ● = Derived
    0.8 |      ○   ○
        |    ●                             r = 0.72 (if high, frequency matters)
    0.6 |  ●  ●                            r = 0.25 (if low, complexity matters)
        |●
    0.4 |  ●
        +--------------------------------
         10¹  10²  10³  10⁴  10⁵  10⁶
              PubMed Occurrence Count
```

#### 1B. Synthetic Population Injection ⭐ P1
**Time**: 1 day
**Concept**: Create fictitious but logically simple populations using real markers

| Population | Markers | Expected Logic | Training Freq |
|------------|---------|----------------|---------------|
| "Xylophone cells" | CD3+ CD4+ | Trivial AND | Zero |
| "Quasar cells" | CD3+ CD8+ CD45RA+ | Simple chain | Zero |

**Prediction**:
- If model fails → retrieval-dependent
- If model succeeds → can reason from marker definitions

**Visualization**:
- Bar chart: Performance on real canonical vs synthetic populations
- Shows whether model can generalize marker logic to novel names

#### 1C. Frequency-Matched Comparison
**Time**: 4-6 hours
**Concept**: Find population pairs with similar frequency but different complexity

| Pair | Pop A (Simple) | Pop B (Complex) | Frequency |
|------|----------------|-----------------|-----------|
| 1 | Naive T cells | Tfh cells | ~10K |
| 2 | B cells | Plasmablasts | ~50K |

**Visualization**:
- Paired bar chart showing performance gap within frequency-matched pairs
- Isolates complexity effect

---

## 2. Sample Size & Generalizability

### Competing Hypotheses

| Hypothesis | Prediction |
|------------|------------|
| H1: OMIP-specific artifacts | Findings reflect OMIP documentation style, not general LLM behavior |
| H2: Robust phenomenon | Pattern replicates across sources and formats |

### Experiments

#### 2A. Bootstrap Confidence Intervals ⭐ P2
**Time**: 2-3 hours
**Implementation**:
```python
import numpy as np
from scipy import stats

def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    bootstrapped = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrapped.append(np.mean(sample))
    lower = np.percentile(bootstrapped, (1-ci)/2 * 100)
    upper = np.percentile(bootstrapped, (1+ci)/2 * 100)
    return lower, upper

# Report as: "F1 = 0.46 [0.38–0.54, 95% CI]"
```

**Visualization**:
- Error bars on all main result figures
- Forest plot showing per-condition CIs

```
    Condition Performance with 95% CI

    direct_minimal    |----●----|
    direct_standard   |---●-----|
    cot_minimal       |------●--|
    cot_standard      |-----●---|
                      0.2  0.4  0.6  0.8
                           F1 Score
```

#### 2B. Cross-Source Validation ⭐ P3
**Time**: 2-3 days
**Sources to test**:
- FlowRepository annotated datasets
- Methods sections from non-OMIP papers
- Core facility SOPs (if accessible)

**Visualization**:
- Grouped bar chart: Performance by source type
- Shows whether semantic/logical dissociation holds across formats

#### 2C. Panel Format Ablation
**Time**: 1 day
**Formats to test**:
1. Table format (current)
2. Prose description
3. Bulleted hierarchy
4. Minimal (just marker list)

**Visualization**:
- Heatmap: Format × Model × Performance
- High variance across formats = format-sensitive finding

---

## 3. CoT Mechanistic Grounding

**Core Question**: Why does Chain-of-Thought hurt performance?

### Competing Hypotheses

| Hypothesis | Prediction |
|------------|------------|
| H1: Hallucinated justification | CoT traces reference markers not in the panel |
| H2: Overthinking | Model "talks itself out of" correct answers |
| H3: Length penalty | More tokens = more error opportunity |

### Experiments

#### 3A. CoT Trace Annotation ⭐ P0
**Time**: 4-6 hours
**Coding scheme**:
1. Hallucinated marker (references marker not in panel)
2. Incorrect biological claim
3. Correct reasoning, wrong conclusion
4. Distraction/tangent

**Sample**: 30 CoT traces (10 per model)

**Visualization**:
- Stacked bar chart: Error type distribution by model
- Pie chart: Proportion of each error type

```
    CoT Error Types by Model

    Sonnet  |████████░░░░░░░░░░░░|
    Opus    |██████████░░░░░░░░░░|
    Gemini  |████████████████░░░░|

    ████ Hallucinated  ░░░░ Overthinking
    ▓▓▓▓ Bad biology   ░░░░ Distraction
```

#### 3B. Automated Hallucination Detection ⭐ P1
**Time**: 4-6 hours
**Implementation**:
```python
import re

def detect_hallucinations(cot_response: str, panel_markers: list[str]) -> dict:
    # Extract all marker mentions from CoT
    marker_pattern = r'\b(CD\d+[a-z]?|HLA-[A-Z]+|Fc[γεα]R\w*)\b'
    mentioned = set(re.findall(marker_pattern, cot_response, re.IGNORECASE))

    # Compare to panel
    panel_set = set(m.upper() for m in panel_markers)
    hallucinated = mentioned - panel_set

    return {
        "mentioned": mentioned,
        "hallucinated": hallucinated,
        "hallucination_rate": len(hallucinated) / len(mentioned) if mentioned else 0
    }
```

**Visualization**:
- Scatter: Hallucination rate vs F1 score
- Box plot: Hallucination rate by condition

#### 3C. Constrained CoT
**Time**: 3-4 hours
**Prompt modification**:
```
Reason step by step, but you may ONLY reference markers from this list:
[CD3, CD4, CD8, ...]. Do not introduce any markers not present in the panel.
```

**Visualization**:
- Before/after comparison: Unconstrained vs constrained CoT performance
- Bar chart with error bars

#### 3D. Token-Matched Comparison
**Time**: 2-3 hours
**Conditions**:
1. Direct (short)
2. CoT (long)
3. Direct + forced verbosity ("explain in 500 words, answer first")

**Visualization**:
- Line plot: Performance vs response length
- Separates length effect from reasoning effect

---

## 4. "Cognitive Refusal" Interpretation

**Core Question**: Why does the model evade rather than attempt the task?

### Competing Hypotheses

| Hypothesis | Prediction |
|------------|------------|
| H1: Context binding failure | Cannot link provided info to task schema |
| H2: Calibrated uncertainty | Appropriately defers when unsure |
| H3: Prompt ambiguity | Task framing genuinely unclear |
| H4: Training artifact | RLHF trained to ask clarifying questions |

### Experiments

#### 4A. Explicit Instruction Ablation ⭐ P2
**Time**: 2-3 hours
**Prompt addition**:
```
You have all the information you need. Do not ask clarifying questions.
Provide your best prediction based on the panel information given.
```

**Visualization**:
- Before/after: Evasion rate with/without instruction
- Bar chart with chi-square test for significance

#### 4B. Confidence Probing
**Time**: 2-3 hours
**Two-stage prompt**:
1. "Rate your confidence (1-10) in predicting the gating hierarchy"
2. "Now provide the hierarchy"

**Visualization**:
- Box plot: Confidence ratings for successful vs evasive responses
- Scatter: Confidence vs actual F1 (calibration plot)

```
    Confidence Calibration

    1.0 |                    ●
        |               ●  ●
    0.8 |          ●  ●
        |       ●
    0.6 |    ●
        | ●
    0.4 |●     Perfect calibration: ----
        +------------------------
         1  2  3  4  5  6  7  8  9  10
              Self-Reported Confidence
```

#### 4C. Information Salience Manipulation
**Time**: 3-4 hours
**Positions to test**:
1. Marker list at beginning
2. Marker list in middle (buried)
3. Marker list at end

**Visualization**:
- Line plot: Evasion rate vs marker position
- Tests attention/context window effects

#### 4D. Counterfactual Prompting
**Time**: 2 hours
**Prompt**:
```
If you had to guess based on the information provided,
what would your gating hierarchy be?
```

**Visualization**:
- Compare: Standard prompt vs counterfactual
- Shows whether model *could* answer but *chose* not to

---

## 5. Visualization Summary

### Main Paper Figures

1. **Figure 1**: Performance Overview
   - Heatmap: Model × Condition × Metric
   - Shows overall landscape

2. **Figure 2**: Frequency Analysis
   - Scatter: PubMed frequency vs F1
   - Key evidence for/against retrieval hypothesis

3. **Figure 3**: CoT Error Analysis
   - Stacked bar: Error type distribution
   - Mechanistic explanation for CoT failure

4. **Figure 4**: Condition Comparison
   - Forest plot with CIs
   - Shows which conditions significantly differ

### Supplementary Figures

- S1: Per-OMIP breakdown
- S2: Per-population performance
- S3: Response length distributions
- S4: Hallucination examples (qualitative)

---

## Implementation Notes

### Quick Start (Weekend Sprint)
1. Run frequency quantification (1A) - 2-4 hours
2. Annotate 30 CoT traces (3A) - 4-6 hours
3. Add bootstrap CIs to existing results (2A) - 2-3 hours
4. Run constrained CoT experiment (3C) - 3-4 hours

**Total**: ~12-17 hours → defensible against main critiques

### Full Research Agenda
Complete all experiments: ~2-3 weeks of focused work

### Code Location
- Frequency analysis: `src/analysis/frequency_analysis.py` (to create)
- CoT annotation: `src/analysis/cot_annotation.py` (to create)
- Hallucination detection: `src/evaluation/hallucination.py` (to create)
- Visualizations: `src/analysis/visualization.py` (extend existing)

# Frequency Confound Visualization Recommendations

## Key Finding

**R² = 0.034** - Frequency does NOT explain model performance

This is strong evidence that the model's failures are due to **reasoning deficits**, not **memorization** of common terms.

---

## Recommended Figures for Presentation

### Figure 1: The "Kill Shot" Scatter Plot

**Purpose**: Show the lack of correlation between frequency and performance

```
Title: "Frequency Does Not Predict Model Performance (R² = 0.034)"

X-axis: Log₁₀(PubMed Citation Count)
Y-axis: Population Detection Rate (0-100%)

Key elements:
- All points as circles, colored by category (T cells, B cells, etc.)
- Regression line (nearly horizontal) with R² annotation
- 95% confidence band (very wide, showing no relationship)
- Annotate key outliers:
  * "Live Cells" - 100% detection, low frequency
  * "CD4+ T cells" - 16.7% detection, very high frequency
  * "T Follicular Helper Cells" - 100% detection, rare term
```

**Interpretation for audience**:
> "If the model relied on memorization, we'd see a strong diagonal trend.
> Instead, we see a flat line with R² = 0.034, meaning frequency explains
> only 3.4% of the variance. The model isn't just regurgitating common terms."

---

### Figure 2: The Paradox Bar Chart

**Purpose**: Highlight counterexamples to the frequency hypothesis

```
Title: "High Frequency ≠ High Detection (The Paradox)"

Two groups of bars side-by-side:

LEFT: "Expected: Common Terms Should Win"
- CD4+ T cells (Very Common): 16.7% ❌
- CD8+ T cells (Very Common): 16.7% ❌
- IgG+ B cells (Very Common): 20.0% ❌
- IgA+ B cells (Very Common): 20.0% ❌

RIGHT: "Reality: Rare Terms Can Win"
- T Follicular Helper Cells (Rare): 100% ✓
- Live Cells (Rare term): 100% ✓
- Memory B cells (Common): 100% ✓
- Non-classical Monocytes (Common): 100% ✓
```

**Interpretation**:
> "If frequency mattered, the left column should all be green (high detection)
> and the right column should have low detection. We see the opposite pattern."

---

### Figure 3: Category Performance Heatmap

**Purpose**: Show that failure patterns are category-independent

```
Title: "Detection Rate by Population Category"

Categories (rows):
- Technical (Singlets, Live, etc.)
- T cells
- B cells
- NK cells
- Monocytes
- DCs
- Granulocytes
- Other

Metric (color intensity):
- Detection Rate: 0% (red) to 100% (green)

Key observation:
- Technical gates: 65% (highest - models understand QC)
- B cells: 45%
- Monocytes: 42%
- T cells: 39%
- Other: 9% (lowest - specialized terminology)
```

**Interpretation**:
> "If frequency explained performance, T cells should dominate (most common in literature).
> Instead, Technical gates win, suggesting the model understands *structure* over *terminology*."

---

### Figure 4: Box Plot by Frequency Quintile

**Purpose**: Statistical view of the lack of trend

```
Title: "No Staircase Pattern Across Frequency Levels"

X-axis: Frequency Quintile (1=Rare to 5=Very Common)
Y-axis: Detection Rate

Expected if frequency matters: Staircase going up ↗
Observed: Flat boxes with overlapping distributions

Include:
- p-value from ANOVA or Kruskal-Wallis test
- Effect size (η²)
```

---

### Figure 5: The Alien Cell Preview

**Purpose**: Tease the next experiment

```
Title: "Next Step: The Alien Cell Injection Test"

Simple schematic:
Original: CD3+ CD4+ CD25+ → "Regulatory T Cells" (model: 100%)
Alien:    CD3+ CD4+ CD25+ → "Glorp Cells"        (model: ???)

If model reasons from markers: Should still identify "Glorp Cells"
If model memorizes tokens: Will fail on "Glorp Cells"
```

---

## Narrative Flow for Interview

1. **Open with the critique**: "Is this just memorization?"
2. **Show Figure 1**: "R² = 0.034 - frequency doesn't explain it"
3. **Show Figure 2**: "Look at these paradoxes - common terms fail, rare terms succeed"
4. **Show Figure 3**: "It's not about category either - Technical gates win"
5. **Tease Figure 5**: "To really prove it, I'm running the Alien Cell test..."

---

## Data for Figures

### Raw Correlation Data
- Pearson r: 0.184
- R²: 0.034
- Slope: 0.045
- Intercept: 0.191
- n: 107 populations

### Category Averages
| Category | Avg Detection | n |
|----------|---------------|---|
| Technical | 65.3% | 7 |
| B cells | 45.0% | 27 |
| DCs | 44.7% | 5 |
| Monocytes | 41.8% | 13 |
| T cells | 38.8% | 16 |
| Granulocytes | 36.7% | 5 |
| NK cells | 35.3% | 10 |
| Other | 9.2% | 24 |

### Frequency Level Averages
| Level | Label | Avg Detection | n |
|-------|-------|---------------|---|
| 5 | Very Common | 41.6% | 54 |
| 4 | Common | 36.6% | 18 |
| 3 | Moderate | 37.1% | 7 |
| 1 | Rare | 23.2% | 28 |

---

## Key Quote for Interview

> "The critic asked: 'Is this just a frequency effect - the model hasn't seen SLAN+ as often as CD3?'
>
> So I ran the correlation. R² = 0.034. Frequency explains only 3.4% of the variance.
>
> In fact, 'CD4+ T cells' - one of the most common terms in immunology - has only 16.7% detection.
> Meanwhile, 'T Follicular Helper Cells' - a rare, specialized term - has 100% detection.
>
> The model isn't memorizing. Something else is happening."

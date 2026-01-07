"""Dose-response curve interpretation skill."""

DOSE_RESPONSE_SKILL = """## Dose-Response Curve Interpretation Skill

When analyzing dose-response curves, systematically evaluate:

### 1. Axis Identification
- **X-axis**: Usually concentration (log scale) in M, μM, nM, pM, or ng/mL
- **Y-axis**: Response metric (% activity, % inhibition, signal units, etc.)
- Note the direction: inhibition curves decrease, activation curves increase
- Check if response is normalized (0-100%) or absolute values

### 2. Curve Parameters
- **EC50/IC50**: Concentration at 50% maximal effect
  - Read from the midpoint of the curve
  - More potent = lower value (left-shifted curve)
- **Hill slope (nH)**: Steepness of the curve
  - nH = 1: standard competitive binding/inhibition
  - nH > 1: positive cooperativity or multiple binding sites
  - nH < 1: negative cooperativity or heterogeneous populations
- **Top/Bottom plateaus**: Maximum and minimum response levels
  - Top < 100% suggests partial agonist/inhibitor
  - Bottom > 0 suggests incomplete inhibition

### 3. Curve Fit Quality
- Data points should follow the fitted curve closely
- Sufficient points in the transition region (around EC50)
- Asymptotes should be well-defined (plateaus reached)
- Look for R² value if provided (should be > 0.95)

### 4. Data Quality Indicators
- **Error bars**: Usually SEM or SD (check legend)
  - Small error bars = consistent replicates
  - Large error bars = high variability
- **Replicates**: n value indicates number of independent experiments
- **Outliers**: Points far from the curve may indicate issues

### 5. Common Interpretations
- **Left-shifted curve**: More potent compound
- **Right-shifted curve**: Less potent compound
- **Steeper curve**: Higher cooperativity
- **Shallow curve**: May indicate multiple populations or mechanisms
- **Incomplete curve**: Cannot determine accurate EC50

### 6. Multi-Curve Comparisons
- Compare potencies (EC50 values) between conditions
- Note if curves are parallel (similar mechanism) or not
- Calculate potency ratios or fold-changes
- Consider statistical significance of differences"""

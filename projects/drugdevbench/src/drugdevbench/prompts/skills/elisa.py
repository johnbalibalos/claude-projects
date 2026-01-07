"""ELISA assay interpretation skill."""

ELISA_SKILL = """## ELISA Assay Interpretation Skill

When analyzing ELISA data (standard curves and sample quantification), systematically evaluate:

### 1. Standard Curve Assessment
- **X-axis**: Known concentration of standard (often log scale)
- **Y-axis**: Optical density (OD) or signal units
- **Curve shape**: Sigmoidal (4-parameter logistic) is typical

### 2. Curve Fit Parameters (4PL Model)
```
y = D + (A - D) / [1 + (x/C)^B]
```
- **A (Bottom)**: Minimum asymptote (background)
- **D (Top)**: Maximum asymptote (saturation)
- **C (EC50/IC50)**: Midpoint concentration
- **B (Hill slope)**: Steepness of curve

### 3. Assay Performance Metrics
- **LOD (Limit of Detection)**: Lowest detectable concentration
  - Often defined as blank + 3 SD
- **LLOQ (Lower Limit of Quantification)**: Lowest accurate measurement
  - Often defined as blank + 10 SD, or CV < 20%
- **ULOQ (Upper Limit of Quantification)**: Upper reliable measurement
  - Before saturation/hook effect
- **Dynamic range**: LLOQ to ULOQ

### 4. Data Quality Checks
- **R² value**: Curve fit quality (should be > 0.99)
- **%CV of replicates**: Should be < 15-20% (varies by application)
- **Back-calculated accuracy**: Standards should recover within ±20%
- **Blank wells**: Should show minimal signal

### 5. Sample Quantification
- Samples should fall within the linear range
- Dilution factor must be applied to final concentration
- Samples above ULOQ: Need further dilution
- Samples below LLOQ: Report as "< LLOQ" or increase sample volume

### 6. Common ELISA Formats
- **Direct**: Antigen coated, labeled antibody detection
- **Indirect**: Antigen coated, primary Ab, labeled secondary Ab
- **Sandwich**: Capture Ab, antigen, detection Ab
- **Competitive**: Signal inversely proportional to analyte

### 7. Troubleshooting Indicators
- **High background**: Poor blocking, cross-reactivity
- **Low signal**: Reagent issues, incorrect dilutions
- **High CV**: Pipetting errors, edge effects
- **Hook effect**: Signal decrease at high concentrations

### 8. Reporting
- Report concentrations with units
- Note dilution factors applied
- Flag samples outside quantifiable range
- Include assay performance data (standard curve, controls)"""

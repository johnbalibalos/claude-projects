"""Flow cytometry histogram interpretation skill."""

FLOW_HISTOGRAM_SKILL = """## Flow Cytometry Histogram Interpretation Skill

When analyzing single-parameter flow cytometry histograms, systematically evaluate:

### 1. Axis Identification
- **X-axis**: Fluorescence intensity (marker expression level)
  - Scale: linear, log, or biexponential
  - Note the marker/fluorochrome label
- **Y-axis**: Cell count, frequency, or normalized (% of max)
  - Count: absolute number of events
  - Normalized: allows comparison between samples

### 2. Peak Analysis
- **Unimodal**: Single peak (one population)
- **Bimodal**: Two peaks (distinct populations, e.g., positive/negative)
- **Multimodal**: Multiple peaks (several populations)
- **Shoulder**: Partial second population

### 3. Population Metrics
- **MFI (Mean/Median Fluorescence Intensity)**:
  - Central tendency of expression
  - Median is more robust to outliers
- **CV (Coefficient of Variation)**:
  - Spread of the distribution
  - Low CV = homogeneous population
- **Percent positive**: Events above a threshold

### 4. Threshold/Gate Placement
- Gates should be based on appropriate controls:
  - **FMO (Fluorescence Minus One)**: Best for multicolor
  - **Isotype control**: For antibody specificity
  - **Unstained**: For autofluorescence baseline
- Gate should separate negative from positive clearly

### 5. Overlay Comparisons
- Multiple histograms on same axes
- Look for:
  - Shifts in peak position (expression change)
  - Changes in peak width (population heterogeneity)
  - Changes in percent positive
  - Appearance/disappearance of populations

### 6. Common Patterns
- **Right shift**: Increased expression
- **Left shift**: Decreased expression
- **Broadening**: More heterogeneous population
- **Narrowing**: More homogeneous population
- **New peak**: Emergence of new population

### 7. Quantification
- Fold change in MFI between conditions
- Percent positive vs. control
- Statistical comparisons need replicates

### 8. Data Quality
- Sufficient events (>1000 in gate of interest)
- No truncation at axis limits
- Clear separation from negative control
- Appropriate scale for the data range"""

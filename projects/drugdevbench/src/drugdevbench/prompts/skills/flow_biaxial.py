"""Flow cytometry biaxial plot interpretation skill."""

FLOW_BIAXIAL_SKILL = """## Flow Cytometry Biaxial Plot Interpretation Skill

When analyzing biaxial (two-parameter) flow cytometry plots, systematically evaluate:

### 1. Axis Identification
- Each axis represents fluorescence intensity for one marker
- Identify the markers (e.g., CD4 vs CD8, FSC vs SSC)
- Note the scale: linear, log, or biexponential (logicle)
- Check axis labels for fluorochrome (e.g., CD4-FITC, CD8-APC)

### 2. Plot Types
- **Dot plot**: Each dot is one cell
- **Density plot/Contour**: Color or contours show cell density
- **Zebra plot**: Combination of dots and contours

### 3. Quadrant Analysis
- Four quadrants based on positive/negative for each marker:
  - Q1 (upper left): X-negative, Y-positive
  - Q2 (upper right): X-positive, Y-positive (double positive)
  - Q3 (lower left): X-negative, Y-negative (double negative)
  - Q4 (lower right): X-positive, Y-negative
- Percentages should sum to ~100%

### 4. Gating Assessment
- Gates should separate populations clearly
- Gate placement affects quantification
- Look for:
  - Rectangular gates (quadrants)
  - Polygonal gates (irregular shapes)
  - Boolean gates (combinations)

### 5. Population Identification
- **Distinct populations**: Clearly separated clusters
- **Continuous distribution**: No clear separation
- **Outliers**: Events far from main populations
- Consider expected biology for the markers shown

### 6. Data Quality Indicators
- **Compensation**: No diagonal streaks from one axis
- **Doublets**: May appear as diagonal populations
- **Dead cells**: Often autofluorescent (high in multiple channels)
- **Cell number**: Sufficient events for statistics (>100 per gate)

### 7. Common Marker Combinations
- FSC vs SSC: Cell size and granularity
- CD4 vs CD8: T cell subsets
- CD45 vs SSC: Leukocyte identification
- Viability dye vs FSC: Dead cell exclusion

### 8. Quantification
- Report percentages of parent population
- Note the parent gate (e.g., "% of CD3+ cells")
- Consider absolute counts if provided
- Statistical comparisons need appropriate controls"""

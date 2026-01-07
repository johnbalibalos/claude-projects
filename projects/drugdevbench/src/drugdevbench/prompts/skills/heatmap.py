"""Expression heatmap interpretation skill."""

HEATMAP_SKILL = """## Expression Heatmap Interpretation Skill

When analyzing expression heatmaps (RNA-seq, microarray, proteomics), systematically evaluate:

### 1. Axis Identification
- **Rows**: Usually genes/proteins/features
- **Columns**: Usually samples/conditions
- Check for clustering dendrograms on either axis
- Note row/column labels if visible

### 2. Color Scale
- **Diverging scale** (e.g., blue-white-red):
  - Center (white): No change / baseline
  - One color: Downregulated / low expression
  - Other color: Upregulated / high expression
- **Sequential scale** (e.g., white-to-red):
  - Intensity indicates magnitude
- **Note the range**: Is it centered at 0? Symmetric?

### 3. Data Normalization
- **Z-score**: Row-normalized, mean=0, SD=1
  - Values typically range -3 to +3
  - Compares relative expression across samples
- **Log2 fold change**: Relative to control
  - +1 = 2-fold increase, -1 = 2-fold decrease
- **Raw values**: Absolute expression levels

### 4. Clustering Analysis
- **Hierarchical clustering**: Tree structure (dendrogram)
  - Height indicates similarity
  - Adjacent items are most similar
- **K-means**: Predefined number of clusters
- **No clustering**: Ordered by other criteria

### 5. Pattern Recognition
- **Sample clustering**: Do biological replicates cluster together?
- **Gene clusters**: Groups of co-regulated genes
- **Outlier samples**: One sample very different from others
- **Batch effects**: Technical grouping instead of biological

### 6. Statistical Significance
- Are shown genes pre-filtered by significance?
- Look for p-value or FDR cutoffs in legend/caption
- Top N genes by specific criteria?

### 7. Biological Interpretation
- Do clusters correspond to known pathways?
- Are there expected markers in the data?
- Do conditions separate as expected biologically?

### 8. Quality Assessment
- Sufficient contrast to see patterns
- Appropriate color scale for the data
- Missing values indicated (usually gray/white)
- Sample and gene numbers reasonable for visualization"""

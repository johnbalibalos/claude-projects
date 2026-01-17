## Reference: HIPC 2016 Standardized Cell Definitions
Source: https://www.nature.com/articles/srep20686

### Quality Control Gates (Required)
- **Time Gate**: Exclude acquisition artifacts (if applicable)
- **Singlets**: Doublet exclusion (FSC-A vs FSC-H for flow cytometry; event length for mass cytometry)
- **Live cells**: Viability dye negative (e.g., Zombie, Live/Dead, cisplatin for CyTOF)

### Major Lineage Definitions
| Population | Markers | Parent |
|------------|---------|--------|
| T cells | CD3+ CD19- | Lymphocytes |
| CD4+ T cells | CD3+ CD4+ CD8- | T cells |
| CD8+ T cells | CD3+ CD4- CD8+ | T cells |
| B cells | CD3- CD19+ (or CD20+) | Lymphocytes |
| NK cells | CD3- CD56+ | Lymphocytes |
| Monocytes | CD14+ | Leukocytes |

### T Cell Memory Subsets (if CD45RA/CCR7 in panel)
| Subset | Phenotype |
|--------|-----------|
| Naive | CD45RA+ CCR7+ |
| Central Memory (CM) | CD45RA- CCR7+ |
| Effector Memory (EM) | CD45RA- CCR7- |
| TEMRA | CD45RA+ CCR7- |

### B Cell Subsets (if CD27/IgD in panel)
| Subset | Phenotype |
|--------|-----------|
| Naive B | CD19+ IgD+ CD27- |
| Memory B | CD19+ CD27+ |
| Transitional B | CD19+ CD24hi CD38hi |
| Plasmablasts | CD19+ CD27++ CD38++ |

### NK Cell Subsets (if CD16 in panel)
| Subset | Phenotype |
|--------|-----------|
| CD56bright NK | CD3- CD56bright CD16dim/- |
| CD56dim NK | CD3- CD56dim CD16+ |

### Monocyte Subsets (if CD16 in panel)
| Subset | Phenotype |
|--------|-----------|
| Classical | CD14++ CD16- |
| Intermediate | CD14++ CD16+ |
| Non-classical | CD14dim CD16++ |

**Note**: HIPC recommends gating directly on lineage markers rather than scatter-based lymphocyte gates to reduce variability. For mass cytometry (CyTOF), scatter parameters are not available - use CD45 or other lineage markers instead.

# Flow Cytometry Gating Workflow Guide

A reference document for understanding standard immunologist gating strategies, intended to improve LLM performance on flow cytometry analysis tasks.

---

## Overview

Flow cytometry gating is a sequential process of identifying cell populations by their physical and fluorescent properties. Gates are applied hierarchically, starting with quality control (QC) steps to remove debris and artifacts, followed by lineage identification to isolate populations of interest.

---

## Standard Gating Hierarchy

```
All Events
├── Time Gate (QC)
│   └── FSC-A vs SSC-A (Cells)
│       └── FSC-A vs FSC-H (Singlets)
│           └── Live/Dead negative (Live Cells)
│               └── CD45+ (Immune Cells) ──────────────┐
│                   ├── Lymphocytes (FSC-A/SSC-A low)  │
│                   │   ├── T Cells (CD3+)             │
│                   │   ├── B Cells (CD19+ or CD20+)   │
│                   │   └── NK Cells (CD56+CD3-)       │
│                   ├── Monocytes (CD14+)              │
│                   └── Granulocytes (SSC-A high)      │
│                                                      │
│               └── CD45- (Non-immune)                 │
│                   ├── EpCAM+ (Tumor/Epithelial)      │
│                   └── CD31+ (Endothelial)            │
```

---

## Phase 1: Quality Control Gates

### 1.1 Time Gate
**Purpose:** Remove acquisition artifacts (air bubbles, clogging, laser fluctuations)

**Parameters:** Time vs. any fluorescence channel (typically FSC-A or a stable marker)

**Strategy:**
- Plot Time on X-axis, fluorescence on Y-axis
- Draw rectangle gate excluding regions with sudden drops, spikes, or instability
- Should retain >95% of events if acquisition was clean

**Common names:** `Time`, `Time Gate`, `Stable Events`

---

### 1.2 Cell Gate (FSC vs SSC)
**Purpose:** Exclude debris, dead cells, and identify intact cells

**Parameters:** FSC-A (Forward Scatter Area) vs SSC-A (Side Scatter Area)

**Strategy:**
- FSC correlates with cell size
- SSC correlates with granularity/internal complexity
- Draw polygon excluding:
  - Low FSC/Low SSC corner (debris)
  - Very high SSC (large aggregates)

**Common names:** `Cells`, `Intact Cells`, `Non-Debris`, `FSC-SSC Gate`

**Typical populations by FSC/SSC:**
| Population | FSC | SSC |
|------------|-----|-----|
| Lymphocytes | Low-Medium | Low |
| Monocytes | Medium-High | Medium |
| Granulocytes | Medium-High | High |
| Debris | Very Low | Very Low |
| Dead cells | Low | Variable |

---

### 1.3 Singlet Gate (Doublet Exclusion)
**Purpose:** Remove cell doublets/aggregates that can cause false positives

**Parameters:** FSC-A vs FSC-H (or FSC-A vs FSC-W, or SSC-A vs SSC-H)

**Strategy:**
- Single cells have proportional Area and Height
- Doublets have higher Area relative to Height
- Draw diagonal gate along the main population
- Can apply sequential FSC and SSC singlet gates for stringency

**Common names:** `Singlets`, `Single Cells`, `Singlets (FSC)`, `Singlets (SSC)`

**Note:** This is one of the most commonly missed gates by LLMs. Always include singlet discrimination.

---

### 1.4 Live/Dead Gate
**Purpose:** Exclude dead cells that non-specifically bind antibodies

**Parameters:** Live/Dead dye (e.g., Zombie, LIVE/DEAD Fixable, 7-AAD, PI, DAPI)

**Strategy:**
- Dead cells have compromised membranes, take up viability dye
- Gate on dye-negative population
- Must be done before fixation for amine-reactive dyes

**Common names:** `Live`, `Live Cells`, `Viable`, `Live/Dead-`, `Zombie-`, `7-AAD-`

**Common viability dyes:**
| Dye | Excitation | Use Case |
|-----|------------|----------|
| 7-AAD | 488nm | Unfixed cells only |
| DAPI | 405nm | Unfixed cells only |
| PI (Propidium Iodide) | 488nm | Unfixed cells only |
| Zombie Aqua/UV/NIR | Various | Fixed or unfixed |
| LIVE/DEAD Fixable | Various | Fixed or unfixed |

---

## Phase 2: Lineage Identification

### 2.1 CD45 Gate (Pan-Leukocyte)
**Purpose:** Separate immune cells from non-immune cells

**Parameters:** CD45 vs SSC-A (or CD45 vs FSC-A)

**Strategy:**
- CD45+ = All leukocytes (immune cells)
- CD45- = Non-immune (epithelial, endothelial, stromal, tumor)
- CD45dim populations may include specific subsets (e.g., granulocytes, some tumor-infiltrating lymphocytes)

**Common names:** `CD45+`, `Leukocytes`, `Immune Cells`, `CD45+ Cells`

---

### 2.2 Lymphocyte Gate
**Purpose:** Identify lymphocyte population within CD45+ cells

**Parameters:** FSC-A vs SSC-A (within CD45+)

**Strategy:**
- Lymphocytes are FSC-low, SSC-low
- Distinct cluster in lower-left quadrant
- Can also use CD45 vs SSC-A (lymphocytes are CD45-bright, SSC-low)

**Common names:** `Lymphocytes`, `Lymphs`, `Lymphocyte Gate`

---

### 2.3 Red Blood Cell Exclusion
**Purpose:** Remove erythrocytes if not lysed during sample prep

**Parameters:** CD235a (Glycophorin A) negative gate

**Strategy:**
- CD235a is specific to erythrocytes
- Gate on CD235a- population
- Usually not needed if RBC lysis was performed

**Common names:** `CD235a-`, `RBC-excluded`, `Non-RBC`

---

### 2.4 Tumor/Epithelial Cell Identification
**Purpose:** Identify epithelial/tumor cells in tissue samples

**Parameters:** EpCAM (CD326) vs CD45

**Strategy:**
- EpCAM+/CD45- = Epithelial/tumor cells
- EpCAM-/CD45+ = Immune cells
- EpCAM-/CD45- = Stromal, endothelial, other

**Common names:** `EpCAM+`, `Tumor Cells`, `Epithelial Cells`, `CD326+`

---

## Phase 3: Subset Identification

### 3.1 T Cell Subsets
```
CD3+ (T Cells)
├── CD4+ (Helper T Cells)
│   ├── CD45RA+CCR7+ (Naive)
│   ├── CD45RA-CCR7+ (Central Memory)
│   ├── CD45RA-CCR7- (Effector Memory)
│   └── CD45RA+CCR7- (TEMRA)
│
├── CD8+ (Cytotoxic T Cells)
│   └── [Same memory subsets as CD4]
│
├── CD4+CD25+CD127low (Tregs)
│
└── CD4-CD8- (Double Negative)
    └── TCRgd+ (Gamma-Delta T Cells)
```

### 3.2 B Cell Subsets
```
CD19+ or CD20+ (B Cells)
├── IgD+CD27- (Naive)
├── IgD+CD27+ (Non-switched Memory)
├── IgD-CD27+ (Switched Memory)
├── CD38++CD27++ (Plasmablasts)
└── CD24++CD38++ (Transitional)
```

### 3.3 NK Cell Subsets
```
CD3-CD56+ (NK Cells)
├── CD56bright CD16- (Cytokine-producing)
└── CD56dim CD16+ (Cytotoxic)
```

### 3.4 Myeloid Subsets
```
CD45+ SSC-high or CD14+/CD11b+
├── CD14++CD16- (Classical Monocytes)
├── CD14++CD16+ (Intermediate Monocytes)
├── CD14+CD16++ (Non-classical Monocytes)
├── CD11c+HLA-DR+ (Dendritic Cells)
│   ├── CD1c+ (cDC2)
│   ├── CD141+ (cDC1)
│   └── CD123+CD303+ (pDC)
└── CD66b+CD15+ (Neutrophils)
```

---

## Critical Gates Checklist

When predicting a gating hierarchy, **always include these QC gates**:

| Gate | Priority | Reason |
|------|----------|--------|
| Time | High | Removes acquisition artifacts |
| FSC/SSC (Cells) | Critical | Removes debris |
| Singlets | Critical | Removes doublets - major source of false positives |
| Live/Dead | Critical | Dead cells bind antibodies non-specifically |
| CD45 | High | Separates immune from non-immune (for immunophenotyping) |

---

## Common Gating Errors

### Errors to Avoid
1. **Missing singlet gate** - Most common QC omission
2. **Wrong gate order** - Live/Dead before singlets won't remove dead doublets
3. **Over-gating** - Too stringent gates lose real populations
4. **Under-gating** - Including debris/doublets causes artifacts
5. **Assuming markers** - Not all panels include CD45 or viability dyes

### Gate Naming Conventions
- Use consistent naming: `CD3+ T Cells` not `T Cells (CD3+)` or `CD3+`
- Include parent context when ambiguous
- Standard abbreviations: `Lymphs`, `Monos`, `Tregs`, `DCs`

---

## Species-Specific Considerations

### Human vs Mouse Markers
| Population | Human | Mouse |
|------------|-------|-------|
| Pan-T | CD3 | CD3 |
| Helper T | CD4 | CD4 |
| Cytotoxic T | CD8 | CD8 |
| B Cells | CD19, CD20 | CD19, B220 |
| NK Cells | CD56, CD16 | NK1.1, NKp46 |
| Monocytes | CD14 | CD11b, Ly6C |
| Granulocytes | CD66b | Ly6G, Gr-1 |
| Pan-leukocyte | CD45 | CD45 |
| Viability | 7-AAD, Zombie | Same |

---

## Application-Specific Workflows

### Tumor Immunology (TIL Analysis)
```
All Events → Time → Cells → Singlets → Live
└── CD45+ (TILs)
    ├── CD3+ T Cells → CD4/CD8 subsets
    ├── CD19+ B Cells
    └── CD14+ TAMs
└── CD45-
    ├── EpCAM+ (Tumor)
    └── CD31+ (Endothelial)
```

### PBMC Immunophenotyping
```
All Events → Time → Cells → Singlets → Live → CD45+
├── Lymphocytes (FSC/SSC)
│   ├── CD3+ T Cells
│   ├── CD19+ B Cells
│   └── CD3-CD56+ NK Cells
├── CD14+ Monocytes
└── HLA-DR+CD11c+ DCs
```

### Bone Marrow Analysis
```
All Events → Time → Cells → Singlets → Live
└── CD45 vs SSC
    ├── Lymphocytes (CD45bright, SSC-low)
    ├── Monocytes (CD45bright, SSC-medium)
    ├── Granulocytes (CD45dim, SSC-high)
    └── Blasts (CD45dim, SSC-low)
```

---

## References

1. Cossarizza A, et al. Guidelines for the use of flow cytometry and cell sorting in immunological studies. Eur J Immunol. 2019;49(10):1457-1973.
2. Maecker HT, et al. Standardizing immunophenotyping for the Human Immunology Project. Nat Rev Immunol. 2012;12(3):191-200.
3. FlowJo Documentation: https://docs.flowjo.com/flowjo/
4. OMIP Series: Cytometry Part A (various issues)

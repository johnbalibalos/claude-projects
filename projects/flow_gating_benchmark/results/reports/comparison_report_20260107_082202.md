# Manual Review Report: LLM vs Ground Truth Comparison

**Generated:** 2026-01-07 08:22:02
**Model:** claude-sonnet-4-20250514

This report shows the LLM-predicted gating hierarchy alongside the ground truth (OMIP)
hierarchy for manual comparison and review.

---

## OMIP-013

**Panel Markers:** CD3, CD4, CD8, CD45RA, CXCR3, CCR6, CXCR5, CCR4, CD45, CCR10... (+4 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 43.8% |
| Structure Accuracy | 20.0% |
| Critical Gate Recall | 33.3% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ Lymphocytes [FSC-A, SSC-A]
        ├─ CD3+ T cells [CD3]
          ├─ CD4+ T cells [CD4]
            ├─ Memory CD4 [CD45RA]
              ├─ Th1 [CXCR3, CCR6]
              ├─ Th2 [CCR4, CXCR3]
              ├─ Th17 [CCR6, CD161]
              ├─ Tfh [CXCR5]
              ├─ Th22 [CCR10, CCR6]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-H, FSC-A]
      ├─ Singlets (SSC) [SSC-H, SSC-A]
        ├─ Live Cells [Zombie NIR]
          ├─ Leukocytes [CD45]
            ├─ T Cells [CD3]
              ├─ CD4+ T Cells [CD4]
                ├─ Regulatory T Cells [CD25, CD127]
                ├─ Conventional CD4+ T Cells [CD25, CD127]
                  ├─ Naive CD4+ T Cells [CD45RA]
                  ├─ Memory CD4+ T Cells [CD45RA]
                    ├─ Th1 [CXCR3, CCR6]
                    ├─ Th2 [CCR4, CXCR3, CCR6]
                    ├─ Th17 [CCR6, CCR4]
                    ├─ Th1/17 [CXCR3, CCR6]
                    ├─ Tfh [CXCR5]
                    ├─ Skin-homing T Cells [CCR10]
                    ├─ CD161+ T Cells [CD161]
              ├─ CD8+ T Cells [CD8]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (7) | All Events, Live Cells, CD4+ T Cells, Th17, Th1, Th2, Tfh |
| ✗ **Missing** (5) | Memory CD4, Th22, Singlets, CD3+ T cells, Lymphocytes |
| ⚠ **Extra** (13) | Memory CD4+ T Cells, CD161+ T Cells, Time Gate, Leukocytes, Regulatory T Cells, Singlets (FSC), Singlets (SSC), Th1/17... |
| 🚨 **Missing Critical** (2) | Singlets, Lymphocytes |

### Structure Errors

- Gate 'Th17': predicted parent='Memory CD4+ T Cells', expected parent='Memory CD4'
- Gate 'Tfh': predicted parent='Memory CD4+ T Cells', expected parent='Memory CD4'
- Gate 'Th2': predicted parent='Memory CD4+ T Cells', expected parent='Memory CD4'
- Gate 'Th1': predicted parent='Memory CD4+ T Cells', expected parent='Memory CD4'

---

## OMIP-032

**Panel Markers:** CD11b, CD11c, Ly6C, Ly6G, F4/80, CD45, MHCII, CD115, CD64, SiglecF... (+2 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 12.5% |
| Structure Accuracy | 100.0% |
| Critical Gate Recall | 33.3% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ CD45+ [CD45]
        ├─ CD11b+ [CD11b]
          ├─ Neutrophils [Ly6G]
          ├─ Monocytes [Ly6C, Ly6G]
            ├─ Ly6Chi Monocytes [Ly6C]
            ├─ Ly6Clo Monocytes [Ly6C]
          ├─ Macrophages [F4/80, CD64]
          ├─ Eosinophils [SiglecF]
        ├─ Dendritic cells [CD11c, MHCII]
          ├─ cDC1 [CD103]
          ├─ cDC2 [CD11b]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-A, FSC-H]
      ├─ Singlets (SSC) [SSC-A, SSC-H]
        ├─ Live Cells [Zombie NIR]
          ├─ CD45+ Leukocytes [CD45]
            ├─ CD11b+ Myeloid [CD11b]
              ├─ Ly6G+ Neutrophils [Ly6G]
              ├─ Ly6G- Non-neutrophil Myeloid [Ly6G]
                ├─ Ly6C High Monocytes [Ly6C]
                ├─ Ly6C Low Monocytes [Ly6C]
                ├─ F4/80+ CD64+ Macrophages [F4/80, CD64]
                  ├─ SiglecF+ Eosinophils [SiglecF]
                  ├─ SiglecF- Macrophages [SiglecF]
            ├─ CD11c+ Dendritic Cells [CD11c]
              ├─ MHCII+ CD11c+ DCs [MHCII]
                ├─ CD103+ DCs [CD103]
                ├─ CD103- CD11b+ DCs [CD103, CD11b]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (2) | Live Cells, All Events |
| ✗ **Missing** (12) | Ly6Clo Monocytes, Monocytes, Singlets, Neutrophils, CD45+, cDC1, cDC2, Ly6Chi Monocytes... |
| ⚠ **Extra** (16) | CD103- CD11b+ DCs, Ly6G+ Neutrophils, SiglecF- Macrophages, F4/80+ CD64+ Macrophages, Singlets (SSC), SiglecF+ Eosinophils, Ly6C Low Monocytes, Time Gate... |
| 🚨 **Missing Critical** (2) | Singlets, CD45+ |

---

## OMIP-007

**Panel Markers:** CD14, CD16, HLA-DR, CD45, CD3, CD19, CD56, Live/Dead

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 31.6% |
| Structure Accuracy | 50.0% |
| Critical Gate Recall | 66.7% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [7-AAD]
      ├─ CD45+ [CD45]
        ├─ Lineage- [CD3, CD19, CD56]
          ├─ Monocytes [CD14, HLA-DR]
            ├─ Classical [CD14, CD16]
            ├─ Intermediate [CD14, CD16]
            ├─ Non-classical [CD14, CD16]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets [FSC-A, FSC-H]
      ├─ Live Cells [7-AAD]
        ├─ Leukocytes [CD45]
          ├─ Non-Lymphocytes [CD3, CD19, CD56]
            ├─ Myeloid Cells [HLA-DR]
              ├─ Classical Monocytes [CD14, CD16]
              ├─ Intermediate Monocytes [CD14, CD16]
              ├─ Non-Classical Monocytes [CD14, CD16]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (3) | Live Cells, Singlets, All Events |
| ✗ **Missing** (6) | Monocytes, Non-classical, CD45+, Classical, Intermediate, Lineage- |
| ⚠ **Extra** (7) | Non-Lymphocytes, Time Gate, Leukocytes, Myeloid Cells, Non-Classical Monocytes, Classical Monocytes, Intermediate Monocytes |
| 🚨 **Missing Critical** (1) | CD45+ |

### Structure Errors

- Gate 'Singlets': predicted parent='Time Gate', expected parent='All Events'

---

## OMIP-030

**Panel Markers:** CD3, CD4, CD8a, CD44, CD62L, CD45, CD25, FoxP3, TCRb, Live/Dead

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 33.3% |
| Structure Accuracy | 100.0% |
| Critical Gate Recall | 33.3% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie Aqua]
      ├─ CD45+ [CD45]
        ├─ T cells [CD3, TCRb]
          ├─ CD4+ T cells [CD4]
            ├─ Tregs [CD25, FoxP3]
            ├─ Naive CD4 [CD44, CD62L]
            ├─ Memory CD4 [CD44, CD62L]
          ├─ CD8+ T cells [CD8a]
            ├─ Naive CD8 [CD44, CD62L]
            ├─ Effector CD8 [CD44, CD62L]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-H, FSC-A]
      ├─ Singlets (SSC) [SSC-H, SSC-A]
        ├─ Live Cells [Zombie Aqua]
          ├─ Leukocytes [CD45]
            ├─ T Cells [CD3, TCRb]
              ├─ CD4+ T Cells [CD4]
                ├─ CD4+ Naive [CD44, CD62L]
                ├─ CD4+ Central Memory [CD44, CD62L]
                ├─ CD4+ Effector Memory [CD44, CD62L]
                ├─ CD4+ Effector [CD44, CD62L]
                ├─ Regulatory T Cells [CD25, FoxP3]
              ├─ CD8+ T Cells [CD8a]
                ├─ CD8+ Naive [CD44, CD62L]
                ├─ CD8+ Central Memory [CD44, CD62L]
                ├─ CD8+ Effector Memory [CD44, CD62L]
                ├─ CD8+ Effector [CD44, CD62L]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (5) | All Events, CD4+ T Cells, T Cells, Live Cells, CD8+ T Cells |
| ✗ **Missing** (7) | Naive CD4, Memory CD4, Singlets, CD45+, Tregs, Naive CD8, Effector CD8 |
| ⚠ **Extra** (13) | CD4+ Effector Memory, CD8+ Effector, CD4+ Effector, Leukocytes, CD4+ Central Memory, Time Gate, CD8+ Naive, CD4+ Naive... |
| 🚨 **Missing Critical** (2) | Singlets, CD45+ |

---

## OMIP-062

**Panel Markers:** CD45, CD3, CD4, CD8a, B220, CD11b, CD11c, F4/80, Ly6G, Ly6C... (+18 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 19.2% |
| Structure Accuracy | 100.0% |
| Critical Gate Recall | 33.3% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ CD45+ TILs [CD45]
        ├─ T cells [CD3]
          ├─ CD8+ TIL [CD8a]
            ├─ Exhausted CD8 [PD-1, TIM-3, LAG-3]
            ├─ Effector CD8 [Granzyme B]
          ├─ CD4+ TIL [CD4]
            ├─ Tregs [CD25, FoxP3]
        ├─ NK cells [NK1.1]
        ├─ Myeloid [CD11b]
          ├─ TAMs [F4/80]
          ├─ MDSCs [Ly6C, Ly6G]
          ├─ DCs [CD11c, MHCII]
      ├─ Tumor cells [CD45]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-H, FSC-A]
      ├─ Singlets (SSC) [SSC-H, SSC-A]
        ├─ Live Cells [Zombie NIR]
          ├─ CD45+ Immune Cells [CD45]
            ├─ T Cells [CD3]
              ├─ CD4+ T Cells [CD4]
                ├─ Conventional CD4+ T Cells [FoxP3]
                  ├─ CD4+ Naive [CD44, CD62L]
                  ├─ CD4+ Effector Memory [CD44, CD62L]
                  ├─ CD4+ Central Memory [CD44, CD62L]
                ├─ Regulatory T Cells (Tregs) [FoxP3]
                  ├─ Activated Tregs [CD25]
              ├─ CD8+ T Cells [CD8a]
                ├─ CD8+ Naive [CD44, CD62L]
                ├─ CD8+ Effector Memory [CD44, CD62L]
                ├─ CD8+ Central Memory [CD44, CD62L]
                ├─ Tissue Resident CD8+ [CD103, CD69]
                ├─ Exhausted CD8+ [PD-1, TIM-3]
            ├─ Non-T Cells [CD3]
              ├─ B Cells [B220]
                ├─ Activated B Cells [CD44, MHCII]
              ├─ Non-B Cells [B220]
                ├─ NK Cells [NK1.1]
                  ├─ Activated NK Cells [CD69]
                ├─ Myeloid Cells [CD11b]
                  ├─ Neutrophils [Ly6G]
                  ├─ Non-Neutrophil Myeloid [Ly6G]
                    ├─ Macrophages [F4/80, CD11c]
                      ├─ M2-like Macrophages [MHCII]
                    ├─ Dendritic Cells [CD11c, F4/80]
                      ├─ Mature DCs [MHCII]
                    ├─ Monocytes [Ly6C, F4/80, CD11c]
                      ├─ Classical Monocytes [Ly6C]
                      ├─ Non-Classical Monocytes [Ly6C]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (5) | All Events, NK Cells, Myeloid Cells, T Cells, Live Cells |
| ✗ **Missing** (11) | CD45+ TILs, Singlets, CD8+ TIL, DCs, CD4+ TIL, MDSCs, Exhausted CD8, Tregs... |
| ⚠ **Extra** (31) | Activated B Cells, CD4+ Naive, CD8+ Naive, Exhausted CD8+, Dendritic Cells, Classical Monocytes, Time Gate, CD4+ Central Memory... |
| 🚨 **Missing Critical** (2) | Singlets, CD45+ TILs |

---

## OMIP-066

**Panel Markers:** CD45, CD3, CD4, CD8a, B220, CD19, IgM, IgD, GL7, CD95... (+16 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 43.9% |
| Structure Accuracy | 100.0% |
| Critical Gate Recall | 33.3% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ CD45+ [CD45]
        ├─ T cells [CD3, TCRb]
          ├─ CD4+ T [CD4]
            ├─ Tfh [CXCR5, PD-1]
            ├─ Tregs [CD25, FoxP3]
          ├─ CD8+ T [CD8a]
        ├─ B cells [B220, CD19]
          ├─ Follicular B [IgD, CD21, CD23]
          ├─ GC B [GL7, CD95]
          ├─ Plasma cells [CD138]
        ├─ NK cells [NK1.1]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-A, FSC-H]
      ├─ Singlets (SSC) [SSC-A, SSC-H]
        ├─ Live Cells [Zombie NIR]
          ├─ CD45+ Immune Cells [CD45]
            ├─ T Cells [CD3, TCRb]
              ├─ CD4+ T Cells [CD4]
                ├─ CD4+ Tregs [CD25, FoxP3]
                ├─ CD4+ Tfh [PD-1, CXCR5]
                ├─ CD4+ Naive [CD44, CD62L]
                ├─ CD4+ Memory [CD44, CD62L]
              ├─ CD8+ T Cells [CD8a]
                ├─ CD8+ Naive [CD44, CD62L]
                ├─ CD8+ Central Memory [CD44, CD62L]
                ├─ CD8+ Effector Memory [CD44, CD62L]
            ├─ B Cells [B220, CD19]
              ├─ Germinal Center B Cells [GL7, CD95]
                ├─ GC B Cells Bcl6+ [Bcl6]
              ├─ Non-GC B Cells [GL7, CD95]
                ├─ Follicular B Cells [CD21, CD23]
                ├─ Marginal Zone B Cells [CD21, CD23]
                ├─ Naive B Cells [IgM, IgD]
                ├─ Memory B Cells [CD44, IgM, IgD]
              ├─ Plasma Cells [CD138]
            ├─ NK Cells [NK1.1, CD3]
            ├─ Other CD45+ Cells [CD3, B220, NK1.1]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (9) | Plasma Cells, All Events, NK Cells, Follicular B Cells, CD4+ T Cells, T Cells, Live Cells, B Cells... |
| ✗ **Missing** (5) | Singlets, CD45+, Tregs, GC B, Tfh |
| ⚠ **Extra** (18) | Marginal Zone B Cells, Time Gate, CD4+ Naive, CD4+ Tregs, CD8+ Naive, Germinal Center B Cells, Naive B Cells, Singlets (FSC)... |
| 🚨 **Missing Critical** (2) | Singlets, CD45+ |

---

## OMIP-041

**Panel Markers:** CD45, CD3, CD4, CD8a, B220, CD19, CD44, CD62L, CD25, FoxP3... (+6 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 55.6% |
| Structure Accuracy | 100.0% |
| Critical Gate Recall | 66.7% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ CD45+ [CD45]
        ├─ T cells [CD3]
          ├─ CD4+ T [CD4]
            ├─ Tfh [CXCR5, PD-1, Bcl6]
            ├─ Th17 [IL-17A]
            ├─ Th1 [IFNg]
            ├─ Tregs [CD25, FoxP3]
          ├─ CD8+ T [CD8a]
        ├─ B cells [B220, CD19]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-A, FSC-H]
      ├─ Singlets (SSC) [SSC-A, SSC-H]
        ├─ Live Cells [Zombie NIR]
          ├─ Lymphocytes [FSC-A, SSC-A]
            ├─ CD45+ Cells [CD45]
              ├─ T Cells [CD3]
                ├─ CD4+ T Cells [CD4]
                  ├─ CD4+ Naive [CD44, CD62L]
                  ├─ CD4+ Memory/Effector [CD44]
                    ├─ Th1 Cells [IFNg]
                    ├─ Th17 Cells [IL-17A]
                  ├─ Regulatory T Cells [CD25, FoxP3]
                  ├─ Tfh Cells [PD-1, CXCR5]
                    ├─ Bcl6+ Tfh [Bcl6]
                ├─ CD8+ T Cells [CD8a]
                  ├─ CD8+ Naive [CD44, CD62L]
                  ├─ CD8+ Memory/Effector [CD44]
                    ├─ IFNg+ CD8+ [IFNg]
              ├─ B Cells [B220]
                ├─ CD19+ B Cells [CD19]
                  ├─ Naive B Cells [CD44, CD62L]
                  ├─ Activated B Cells [CD44]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (10) | All Events, CD45+ Cells, CD4+ T Cells, Th17 Cells, Th1 Cells, T Cells, Live Cells, B Cells... |
| ✗ **Missing** (2) | Tregs, Singlets |
| ⚠ **Extra** (14) | CD4+ Memory/Effector, Activated B Cells, CD8+ Memory/Effector, Time Gate, CD19+ B Cells, CD4+ Naive, Bcl6+ Tfh, CD8+ Naive... |
| 🚨 **Missing Critical** (1) | Singlets |

---

## OMIP-001

**Panel Markers:** CD3, CD4, CD8, CD45RA, CD45RO, CCR7, Live/Dead

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 51.9% |
| Structure Accuracy | 75.0% |
| Critical Gate Recall | 100.0% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [7-AAD]
      ├─ CD3+ T cells [CD3]
        ├─ CD4+ T cells [CD4]
          ├─ CD4+ Naive [CD45RA, CCR7]
          ├─ CD4+ CM [CD45RO, CCR7]
          ├─ CD4+ EM [CD45RO, CCR7]
        ├─ CD8+ T cells [CD8]
          ├─ CD8+ Naive [CD45RA, CCR7]
          ├─ CD8+ CM [CD45RO, CCR7]
          ├─ CD8+ EM [CD45RO, CCR7]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets [FSC-A, FSC-H]
      ├─ Live Cells [7-AAD]
        ├─ T Cells [CD3]
          ├─ CD4+ T Cells [CD4]
            ├─ CD4+ Naive [CD45RA, CD45RO, CCR7]
            ├─ CD4+ Central Memory [CD45RA, CD45RO, CCR7]
            ├─ CD4+ Effector Memory [CD45RA, CD45RO, CCR7]
            ├─ CD4+ TEMRA [CD45RA, CD45RO, CCR7]
          ├─ CD8+ T Cells [CD8]
            ├─ CD8+ Naive [CD45RA, CD45RO, CCR7]
            ├─ CD8+ Central Memory [CD45RA, CD45RO, CCR7]
            ├─ CD8+ Effector Memory [CD45RA, CD45RO, CCR7]
            ├─ CD8+ TEMRA [CD45RA, CD45RO, CCR7]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (7) | CD4+ Naive, All Events, CD8+ Naive, Singlets, CD4+ T Cells, Live Cells, CD8+ T Cells |
| ✗ **Missing** (5) | CD4+ EM, CD8+ CM, CD4+ CM, CD3+ T cells, CD8+ EM |
| ⚠ **Extra** (8) | CD4+ Effector Memory, Time Gate, CD4+ Central Memory, CD8+ Central Memory, CD8+ TEMRA, T Cells, CD4+ TEMRA, CD8+ Effector Memory |
| 🚨 **Missing Critical** (0) | None |

### Structure Errors

- Gate 'Singlets': predicted parent='Time Gate', expected parent='All Events'

---

## OMIP-003

**Panel Markers:** CD19, CD20, CD27, IgD, CD38, CD24, CD45, Live/Dead

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 63.6% |
| Structure Accuracy | 33.3% |
| Critical Gate Recall | 66.7% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [7-AAD]
      ├─ CD45+ [CD45]
        ├─ B cells [CD19, CD20]
          ├─ Naive B [IgD, CD27]
          ├─ Memory B [IgD, CD27]
          ├─ Plasmablasts [CD38, CD27]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets [FSC-A, FSC-H]
      ├─ Live Cells [7-AAD]
        ├─ Leukocytes [CD45]
          ├─ B Cells [CD19]
            ├─ Naive B Cells [CD27, IgD]
              ├─ Transitional B Cells [CD38, CD24]
              ├─ Mature Naive B Cells [CD38, CD24]
            ├─ Class-Switched Memory B Cells [CD27, IgD]
              ├─ Memory B Cells [CD38]
              ├─ Plasmablasts [CD38]
            ├─ Non-Class-Switched Memory B Cells [CD27, IgD]
            ├─ Double Negative B Cells [CD27, IgD]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (7) | B Cells, All Events, Singlets, Naive B Cells, Memory B Cells, Live Cells, Plasmablasts |
| ✗ **Missing** (1) | CD45+ |
| ⚠ **Extra** (7) | Time Gate, Leukocytes, Double Negative B Cells, Mature Naive B Cells, Transitional B Cells, Class-Switched Memory B Cells, Non-Class-Switched Memory B Cells |
| 🚨 **Missing Critical** (1) | CD45+ |

### Structure Errors

- Gate 'Plasmablasts': predicted parent='Class-Switched Memory B Cells', expected parent='B cells'
- Gate 'Singlets': predicted parent='Time Gate', expected parent='All Events'

---

## OMIP-035

**Panel Markers:** CD45, CD3, CD4, CD8a, B220, CD19, CD44, CD62L, CD127, KLRG1... (+10 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 31.8% |
| Structure Accuracy | 100.0% |
| Critical Gate Recall | 33.3% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ CD45+ [CD45]
        ├─ T cells [CD3, TCRb]
          ├─ CD4+ T [CD4]
            ├─ Naive CD4 [CD44, CD62L]
            ├─ Memory CD4 [CD44, CD62L]
            ├─ Tregs [CD25, FoxP3]
          ├─ CD8+ T [CD8a]
            ├─ Naive CD8 [CD44, CD62L]
            ├─ Memory CD8 [CD44, CD62L]
            ├─ SLEC [KLRG1, CD127]
            ├─ MPEC [KLRG1, CD127]
        ├─ B cells [B220, CD19]
        ├─ NK cells [NK1.1, CD49b]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-A, FSC-H]
      ├─ Singlets (SSC) [SSC-A, SSC-H]
        ├─ Live Cells [Zombie NIR]
          ├─ CD45+ Leukocytes [CD45]
            ├─ T Cells [CD3, TCRb]
              ├─ CD4+ T Cells [CD4]
                ├─ Regulatory T Cells [CD25, FoxP3]
                ├─ CD4+ Naive [CD44, CD62L]
                ├─ CD4+ Central Memory [CD44, CD62L]
                ├─ CD4+ Effector Memory [CD44, CD62L]
                  ├─ CD4+ KLRG1+ Senescent [KLRG1, CD127]
                ├─ CD4+ PD-1+ Exhausted [PD-1]
              ├─ CD8+ T Cells [CD8a]
                ├─ CD8+ Naive [CD44, CD62L]
                ├─ CD8+ Central Memory [CD44, CD62L]
                ├─ CD8+ Effector Memory [CD44, CD62L]
                  ├─ CD8+ KLRG1+ Senescent [KLRG1, CD127]
                ├─ CD8+ PD-1+ Exhausted [PD-1]
            ├─ Non-T Cells [CD3]
              ├─ B Cells [B220, CD19]
                ├─ B Cell Subsets [CD21, CD44]
              ├─ NK Cells [NK1.1, CD49b]
                ├─ Immature NK [CD11b]
                ├─ Mature NK [CD11b, KLRG1]
              ├─ Myeloid Cells [CD11b]
                ├─ Ly6C+ Monocytes [Ly6C]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (7) | All Events, NK Cells, CD4+ T Cells, T Cells, Live Cells, B Cells, CD8+ T Cells |
| ✗ **Missing** (9) | Memory CD4, Singlets, CD45+, Naive CD8, MPEC, Memory CD8, Tregs, Naive CD4... |
| ⚠ **Extra** (21) | CD4+ Naive, CD8+ Naive, Time Gate, CD4+ Central Memory, Singlets (FSC), Mature NK, CD8+ Central Memory, CD8+ KLRG1+ Senescent... |
| 🚨 **Missing Critical** (2) | Singlets, CD45+ |

---

## OMIP-044

**Panel Markers:** CD3, CD19, CD20, CD14, CD16, HLA-DR, CD11c, CD123, CD1c, CD141... (+2 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 44.4% |
| Structure Accuracy | 50.0% |
| Critical Gate Recall | 25.0% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Time [Time]
    ├─ Singlets [FSC-A, FSC-H]
      ├─ Live [Zombie NIR]
        ├─ CD45+ [CD45]
          ├─ Lineage- [CD3, CD19, CD20, CD14]
            ├─ HLA-DR+ [HLA-DR]
              ├─ pDC [CD123, CD11c]
              ├─ cDC [CD11c]
                ├─ cDC1 [CD141]
                ├─ cDC2 [CD1c]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-A, FSC-H]
      ├─ Singlets (SSC) [SSC-A, SSC-H]
        ├─ Live Cells [Zombie NIR]
          ├─ Leukocytes [CD45]
            ├─ Lineage Negative [CD3, CD19, CD20, CD14, CD16]
              ├─ HLA-DR+ [HLA-DR]
                ├─ Plasmacytoid DCs [CD123, CD11c]
                ├─ Conventional DCs [CD11c, CD123]
                  ├─ cDC1 [CD141]
                  ├─ cDC2 [CD1c]
            ├─ T Cells [CD3]
            ├─ B Cells [CD19, CD20]
            ├─ Monocytes [CD14]
            ├─ NK Cells [CD16, CD3]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (6) | All Events, cDC2, HLA-DR+, cDC1, Live Cells, Lineage Negative |
| ✗ **Missing** (5) | Singlets, CD45+, cDC, Time, pDC |
| ⚠ **Extra** (10) | Monocytes, Time Gate, Leukocytes, Conventional DCs, NK Cells, Plasmacytoid DCs, Singlets (FSC), Singlets (SSC)... |
| 🚨 **Missing Critical** (3) | Time, Singlets, CD45+ |

### Structure Errors

- Gate 'cDC1': predicted parent='Conventional DCs', expected parent='cDC'
- Gate 'cDC2': predicted parent='Conventional DCs', expected parent='cDC'

---

## OMIP-058

**Panel Markers:** CD3, CD4, CD8, CD45, CD45RA, CD45RO, CCR7, CD27, CD28, CD57... (+20 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 30.2% |
| Structure Accuracy | 100.0% |
| Critical Gate Recall | 50.0% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Time [Time]
    ├─ Singlets [FSC-A, FSC-H]
      ├─ Live [Zombie NIR]
        ├─ Lymphocytes [FSC-A, SSC-A]
          ├─ T cells [CD3]
            ├─ CD4+ T [CD4]
              ├─ CD4 Naive [CD45RA, CCR7]
              ├─ CD4 CM [CD45RO, CCR7]
              ├─ CD4 EM [CD45RO, CCR7]
              ├─ CD4 TEMRA [CD45RA, CCR7]
            ├─ CD8+ T [CD8]
              ├─ CD8 Naive [CD45RA, CCR7]
              ├─ CD8 CM [CD45RO, CCR7]
              ├─ CD8 EM [CD45RO, CCR7]
              ├─ CD8 TEMRA [CD45RA, CCR7]
            ├─ iNKT cells [Va24-Ja18]
          ├─ NK cells [CD3, CD56]
            ├─ CD56bright [CD56, CD16]
            ├─ CD56dim [CD56, CD16]
              ├─ CD57+ NK [CD57]
              ├─ Adaptive NK [NKG2C]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-H, FSC-A]
      ├─ Singlets (SSC) [SSC-H, SSC-A]
        ├─ Live Cells [Zombie NIR]
          ├─ Lymphocytes [FSC-A, SSC-A]
            ├─ CD45+ Leukocytes [CD45]
              ├─ T Cells [CD3]
                ├─ Conventional T Cells [Va24-Ja18]
                  ├─ CD4+ T Cells [CD4]
                    ├─ CD4+ Naive [CD45RA, CCR7]
                    ├─ CD4+ Central Memory [CD45RO, CCR7]
                    ├─ CD4+ Effector Memory [CD45RO, CCR7]
                    ├─ CD4+ TEMRA [CD45RA, CCR7]
                  ├─ CD8+ T Cells [CD8]
                    ├─ CD8+ Naive [CD45RA, CCR7]
                    ├─ CD8+ Central Memory [CD45RO, CCR7]
                    ├─ CD8+ Effector Memory [CD45RO, CCR7]
                    ├─ CD8+ TEMRA [CD45RA, CCR7]
                ├─ iNKT Cells [Va24-Ja18]
                  ├─ CD4+ iNKT [CD4]
                  ├─ CD8+ iNKT [CD8]
                  ├─ DN iNKT [CD4, CD8]
              ├─ NK Cells [CD3, CD56, CD16]
                ├─ CD56bright NK [CD56, CD16]
                  ├─ CD56bright CD57- [CD57]
                  ├─ CD56bright CD57+ [CD57]
                ├─ CD56dim NK [CD56, CD16]
                  ├─ CD56dim CD57- [CD57]
                  ├─ CD56dim CD57+ [CD57]
                ├─ CD56- CD16+ NK [CD56, CD16]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (8) | iNKT Cells, All Events, NK Cells, CD4+ T Cells, Lymphocytes, T Cells, Live Cells, CD8+ T Cells |
| ✗ **Missing** (14) | Singlets, CD4 CM, CD57+ NK, Adaptive NK, CD4 Naive, CD4 EM, CD56bright, CD56dim... |
| ⚠ **Extra** (23) | CD56bright CD57+, CD56dim NK, CD4+ Naive, CD8+ Naive, Time Gate, CD4+ Central Memory, DN iNKT, Singlets (FSC)... |
| 🚨 **Missing Critical** (2) | Time, Singlets |

---

## OMIP-064

**Panel Markers:** CD3, CD4, CD8, CD45, CD19, CD14, CD16, CD56, HLA-DR, CD38... (+22 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 42.6% |
| Structure Accuracy | 66.7% |
| Critical Gate Recall | 25.0% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Time [Time]
    ├─ Singlets [FSC-A, FSC-H]
      ├─ Live [Zombie NIR]
        ├─ CD45+ [CD45]
          ├─ T cells [CD3]
            ├─ CD4+ T [CD4]
              ├─ Tfh [CXCR5, PD-1]
              ├─ Tregs [CD25, FoxP3]
            ├─ CD8+ T [CD8]
              ├─ Activated CD8 [CD38, HLA-DR]
              ├─ Exhausted CD8 [PD-1, TIM-3]
          ├─ B cells [CD19]
            ├─ Plasmablasts [CD27, CD38]
          ├─ NK cells [CD56, CD3]
          ├─ Monocytes [CD14, CD16]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-A, FSC-H]
      ├─ Singlets (SSC) [SSC-A, SSC-H]
        ├─ Live Cells [Zombie NIR-]
          ├─ Lymphocytes [FSC-A, SSC-A]
            ├─ CD45+ Leukocytes [CD45+]
              ├─ T Cells [CD3+]
                ├─ CD4+ T Cells [CD4+, CD8-]
                  ├─ CD4+ Naive [CD45RA+, CCR7+]
                  ├─ CD4+ Central Memory [CD45RA-, CCR7+]
                  ├─ CD4+ Effector Memory [CD45RA-, CCR7-]
                  ├─ CD4+ TEMRA [CD45RA+, CCR7-]
                  ├─ Tfh Cells [CXCR5+, PD-1+]
                  ├─ Tregs [CD25+, FoxP3+, CD127-]
                ├─ CD8+ T Cells [CD8+, CD4-]
                  ├─ CD8+ Naive [CD45RA+, CCR7+]
                  ├─ CD8+ Central Memory [CD45RA-, CCR7+]
                  ├─ CD8+ Effector Memory [CD45RA-, CCR7-]
                  ├─ CD8+ TEMRA [CD45RA+, CCR7-]
                ├─ NKT Cells [CD56+]
              ├─ Non-T Cells [CD3-]
                ├─ B Cells [CD19+]
                  ├─ Naive B Cells [IgD+, CD27-]
                  ├─ Memory B Cells [IgD-, CD27+]
                  ├─ Marginal Zone B Cells [IgD+, CD27+]
                  ├─ Double Negative B Cells [IgD-, CD27-]
                ├─ NK Cells [CD56+, CD19-]
                  ├─ CD56bright NK [CD56bright, CD16-/dim]
                  ├─ CD56dim NK [CD56dim, CD16+]
                ├─ Monocytes [CD14+]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (10) | Monocytes, All Events, NK Cells, Live Cells, CD4+ T Cells, T Cells, Tregs, B Cells... |
| ✗ **Missing** (6) | Singlets, CD45+, Exhausted CD8, Time, Plasmablasts, Activated CD8 |
| ⚠ **Extra** (21) | CD56dim NK, CD4+ Naive, CD8+ Naive, Time Gate, CD4+ Central Memory, Naive B Cells, Singlets (FSC), CD8+ Central Memory... |
| 🚨 **Missing Critical** (3) | Time, Singlets, CD45+ |

### Structure Errors

- Gate 'Monocytes': predicted parent='Non-T Cells', expected parent='CD45+'

---

## OMIP-021

**Panel Markers:** CD3, CD4, CD8, TCRgd, Va7.2, CD161, Va24-Ja18, CD45, CD56, CD27... (+2 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 34.3% |
| Structure Accuracy | 40.0% |
| Critical Gate Recall | 66.7% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ Lymphocytes [FSC-A, SSC-A]
        ├─ CD3+ T cells [CD3]
          ├─ gd T cells [TCRgd]
          ├─ ab T cells [TCRgd]
            ├─ MAIT cells [Va7.2, CD161]
            ├─ iNKT cells [Va24-Ja18]
            ├─ Conventional T [Va7.2, Va24-Ja18]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-A, FSC-H]
      ├─ Singlets (SSC) [SSC-A, SSC-H]
        ├─ Live Cells [Zombie NIR]
          ├─ Lymphocytes [FSC-A, SSC-A]
            ├─ CD45+ Leukocytes [CD45]
              ├─ CD3+ T cells [CD3]
                ├─ MAIT cells [Va7.2, CD161]
                  ├─ MAIT CD4/CD8 subsets [CD4, CD8]
                  ├─ MAIT memory subsets [CD27, CD45RA]
                ├─ iNKT cells [Va24-Ja18, CD161]
                  ├─ iNKT CD4/CD8 subsets [CD4, CD8]
                  ├─ iNKT memory subsets [CD27, CD45RA]
                ├─ γδ T cells [TCRgd]
                  ├─ γδ T CD4/CD8 subsets [CD4, CD8]
                  ├─ γδ T memory subsets [CD27, CD45RA]
                ├─ Conventional αβ T cells [Va7.2, Va24-Ja18, TCRgd]
                  ├─ CD4+ T cells [CD4]
                    ├─ CD4+ memory subsets [CD27, CD45RA]
                  ├─ CD8+ T cells [CD8]
                    ├─ CD8+ memory subsets [CD27, CD45RA]
              ├─ CD3- cells [CD3]
                ├─ NK cells [CD56]
                  ├─ NK memory subsets [CD27, CD45RA]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (6) | MAIT cells, iNKT cells, All Events, CD3+ T cells, Lymphocytes, Live Cells |
| ✗ **Missing** (4) | Conventional T, Singlets, ab T cells, gd T cells |
| ⚠ **Extra** (19) | iNKT memory subsets, CD3- cells, MAIT CD4/CD8 subsets, NK memory subsets, MAIT memory subsets, CD4+ memory subsets, Time Gate, Singlets (FSC)... |
| 🚨 **Missing Critical** (1) | Singlets |

### Structure Errors

- Gate 'iNKT cells': predicted parent='CD3+ T cells', expected parent='ab T cells'
- Gate 'CD3+ T cells': predicted parent='CD45+ Leukocytes', expected parent='Lymphocytes'
- Gate 'MAIT cells': predicted parent='CD3+ T cells', expected parent='ab T cells'

---

## OMIP-025

**Panel Markers:** CD34, CD38, CD45RA, CD90, CD49f, CD10, CD45, CD7, CD123, CD135... (+12 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 58.1% |
| Structure Accuracy | 25.0% |
| Critical Gate Recall | 66.7% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ CD45+ [CD45]
        ├─ CD34+ HSPCs [CD34]
          ├─ HSC [CD38, CD90, CD45RA]
          ├─ MPP [CD38, CD90, CD45RA]
          ├─ CMP [CD38, CD123, CD45RA]
          ├─ GMP [CD38, CD123, CD45RA]
          ├─ MEP [CD38, CD123, CD45RA]
          ├─ CLP [CD38, CD10]
        ├─ Mature cells [CD34]
          ├─ Erythroid [CD235a, CD71]
          ├─ Megakaryocytes [CD41a]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-H, FSC-A]
      ├─ Singlets (SSC) [SSC-H, SSC-A]
        ├─ Live Cells [Zombie NIR]
          ├─ CD45+ [CD45]
            ├─ Lineage Negative [CD3, CD19, CD56, CD14, CD16, CD11b, CD15, CD235a, CD41a]
              ├─ CD34+ [CD34]
                ├─ CD38- [CD38]
                  ├─ HSC [CD90, CD45RA]
                  ├─ MPP [CD90, CD45RA]
                  ├─ LMPP [CD90, CD45RA]
                ├─ CD38+ [CD38]
                  ├─ CMP [CD123, CD45RA]
                  ├─ GMP [CD123, CD45RA]
                  ├─ MEP [CD123, CD45RA]
                  ├─ CLP [CD10, CD7]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (9) | MPP, All Events, HSC, CMP, CD45+, GMP, MEP, Live Cells... |
| ✗ **Missing** (5) | CD34+ HSPCs, Singlets, Erythroid, Megakaryocytes, Mature cells |
| ⚠ **Extra** (8) | Time Gate, LMPP, CD34+, Singlets (FSC), Singlets (SSC), CD38+, Lineage Negative, CD38- |
| 🚨 **Missing Critical** (1) | Singlets |

### Structure Errors

- Gate 'HSC': predicted parent='CD38-', expected parent='CD34+ HSPCs'
- Gate 'CMP': predicted parent='CD38+', expected parent='CD34+ HSPCs'
- Gate 'GMP': predicted parent='CD38+', expected parent='CD34+ HSPCs'
- Gate 'MEP': predicted parent='CD38+', expected parent='CD34+ HSPCs'
- Gate 'MPP': predicted parent='CD38-', expected parent='CD34+ HSPCs'

---

## OMIP-015

**Panel Markers:** CD3, CD19, CD14, CD127, CD117, CRTH2, NKp44, CD45, CD56, CD161... (+5 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 53.3% |
| Structure Accuracy | 40.0% |
| Critical Gate Recall | 66.7% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ Lymphocytes [FSC-A, SSC-A]
        ├─ Lineage- [CD3, CD19, CD14]
          ├─ CD127+ ILCs [CD127]
            ├─ ILC1 [CD117, CRTH2]
            ├─ ILC2 [CRTH2, CD294]
            ├─ ILC3 [CD117, NKp44]
          ├─ NK cells [CD56]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-H, FSC-A]
      ├─ Singlets (SSC) [SSC-H, SSC-A]
        ├─ Live Cells [Zombie NIR]
          ├─ Lymphocytes [FSC-A, SSC-A]
            ├─ CD45+ [CD45]
              ├─ Lineage Negative [CD3, CD19, CD14]
                ├─ ILCs [CD127]
                  ├─ ILC2 [CRTH2]
                  ├─ CRTH2- ILCs [CRTH2]
                    ├─ ILC1 [CD117]
                    ├─ ILC3 [CD117]
                      ├─ NKp44+ ILC3 [NKp44]
                      ├─ NKp44- ILC3 [NKp44]
                ├─ NK Cells [CD127, CD56]
                  ├─ CD56bright NK [CD56]
                  ├─ CD56dim NK [CD56]
                    ├─ NKG2A+ CD56dim [NKG2A]
                    ├─ NKG2A- CD56dim [NKG2A]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (8) | ILC3, All Events, NK Cells, ILC2, Lymphocytes, Live Cells, Lineage Negative, ILC1 |
| ✗ **Missing** (2) | Singlets, CD127+ ILCs |
| ⚠ **Extra** (12) | Time Gate, CD56dim NK, NKG2A+ CD56dim, CRTH2- ILCs, CD45+, Singlets (FSC), NKp44- ILC3, Singlets (SSC)... |
| 🚨 **Missing Critical** (1) | Singlets |

### Structure Errors

- Gate 'ILC2': predicted parent='ILCs', expected parent='CD127+ ILCs'
- Gate 'ILC3': predicted parent='CRTH2- ILCs', expected parent='CD127+ ILCs'
- Gate 'ILC1': predicted parent='CRTH2- ILCs', expected parent='CD127+ ILCs'

---

## OMIP-039

**Panel Markers:** CD3, CD4, CD8, CD45, CD19, CD14, CD16, HLA-DR, CD38, CD27... (+12 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 39.1% |
| Structure Accuracy | 100.0% |
| Critical Gate Recall | 33.3% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ CD45+ [CD45]
        ├─ T cells [CD3]
          ├─ CD4+ T [CD4]
            ├─ Th1 [CXCR3, CCR6]
            ├─ Th17 [CCR6, CD161]
            ├─ Tfh [CXCR5, PD-1]
            ├─ Tregs [CD25, FoxP3]
          ├─ CD8+ T [CD8]
        ├─ B cells [CD19]
        ├─ Monocytes [CD14, HLA-DR]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-H, FSC-A]
      ├─ Singlets (SSC) [SSC-H, SSC-A]
        ├─ Live Cells [Zombie NIR]
          ├─ Leukocytes [CD45]
            ├─ T Cells [CD3]
              ├─ CD4+ T Cells [CD4]
                ├─ Naive CD4+ [CD45RA, CD27]
                ├─ Central Memory CD4+ [CD45RA, CD27]
                ├─ Effector Memory CD4+ [CD45RA, CD27]
                ├─ Regulatory T Cells [CD25, FoxP3, CD127]
                ├─ Tfh Cells [CXCR5, PD-1]
                ├─ Th1 Cells [CXCR3, CCR6]
                ├─ Th17 Cells [CCR6, CXCR3]
                  ├─ IL-17A+ Th17 [IL-17A]
                ├─ Activated CD4+ [HLA-DR, CD38]
              ├─ CD8+ T Cells [CD8]
                ├─ Naive CD8+ [CD45RA, CD27]
                ├─ Central Memory CD8+ [CD45RA, CD27]
                ├─ Effector Memory CD8+ [CD45RA, CD27]
                ├─ Activated CD8+ [HLA-DR, CD38]
            ├─ B Cells [CD19]
              ├─ Naive B Cells [CD27, CD38]
              ├─ Memory B Cells [CD27, CD38]
              ├─ Plasmablasts [CD27, CD38]
              ├─ Activated B Cells [HLA-DR]
            ├─ Myeloid Cells [CD3, CD19]
              ├─ Classical Monocytes [CD14, CD16]
                ├─ Activated Classical Monocytes [HLA-DR]
              ├─ Intermediate Monocytes [CD14, CD16]
              ├─ Non-classical Monocytes [CD14, CD16]
              ├─ NK Cells [CD16, CD3]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (9) | All Events, CD4+ T Cells, Th17 Cells, Th1 Cells, T Cells, Live Cells, B Cells, Tfh Cells... |
| ✗ **Missing** (4) | Tregs, Singlets, CD45+, Monocytes |
| ⚠ **Extra** (24) | Activated B Cells, Central Memory CD4+, Effector Memory CD8+, IL-17A+ Th17, Classical Monocytes, Time Gate, Naive B Cells, Singlets (FSC)... |
| 🚨 **Missing Critical** (2) | Singlets, CD45+ |

---

## OMIP-043

**Panel Markers:** CD3, CD14, CD19, CD20, CD27, CD38, CD45, CD138, IgD, IgM... (+15 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 35.9% |
| Structure Accuracy | 25.0% |
| Critical Gate Recall | 33.3% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ CD45+ [CD45]
        ├─ B lineage [CD19]
          ├─ Naive B [IgD, CD27]
          ├─ Memory B [CD27, IgD]
            ├─ Switched Memory [IgG, IgA]
            ├─ Unswitched Memory [IgM]
          ├─ Plasmablasts [CD38, CD27]
          ├─ Plasma cells [CD138, CD38]
            ├─ IgG PC [IgG]
            ├─ IgA PC [IgA]
            ├─ IgM PC [IgM]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-H, FSC-A]
      ├─ Singlets (SSC) [SSC-H, SSC-A]
        ├─ Live Cells [Zombie NIR-]
          ├─ Lymphocytes [FSC-A, SSC-A]
            ├─ CD45+ Leukocytes [CD45+]
              ├─ Non-T Non-Myeloid [CD3-, CD14-]
                ├─ B Cells [CD19+]
                  ├─ Mature B Cells [CD20+]
                    ├─ Naive B Cells [IgD+, CD27-]
                    ├─ Unswitched Memory [IgD+, CD27+]
                    ├─ Switched Memory [IgD-, CD27+]
                      ├─ IgG+ Memory [IgG+]
                      ├─ IgA+ Memory [IgA+]
                    ├─ Double Negative Memory [IgD-, CD27-]
                  ├─ Antibody-Secreting Cells [CD27+, CD38++]
                    ├─ Plasmablasts [CD20low, CD138-]
                      ├─ IgG+ Plasmablasts [IgG+]
                      ├─ IgA+ Plasmablasts [IgA+]
                      ├─ IgM+ Plasmablasts [IgM+]
                    ├─ Plasma Cells [CD20-, CD138+]
                      ├─ IgG+ Plasma Cells [IgG+]
                      ├─ IgA+ Plasma Cells [IgA+]
                      ├─ IgM+ Plasma Cells [IgM+]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (7) | Plasma Cells, All Events, Naive B Cells, Live Cells, Plasmablasts, Switched Memory, Unswitched Memory |
| ✗ **Missing** (7) | B lineage, Singlets, CD45+, IgG PC, Memory B, IgA PC, IgM PC |
| ⚠ **Extra** (18) | Double Negative Memory, Time Gate, IgM+ Plasmablasts, IgG+ Plasmablasts, CD45+ Leukocytes, Singlets (FSC), Singlets (SSC), IgG+ Plasma Cells... |
| 🚨 **Missing Critical** (2) | Singlets, CD45+ |

### Structure Errors

- Gate 'Plasmablasts': predicted parent='Antibody-Secreting Cells', expected parent='B lineage'
- Gate 'Switched Memory': predicted parent='Mature B Cells', expected parent='Memory B'
- Gate 'Unswitched Memory': predicted parent='Mature B Cells', expected parent='Memory B'

---

## OMIP-060

**Panel Markers:** CD3, CD4, CD8, CD45, CD45RA, CD19, CD20, CD14, CD16, CD56... (+25 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 33.3% |
| Structure Accuracy | 50.0% |
| Critical Gate Recall | 25.0% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Time [Time]
    ├─ Singlets [FSC-A, FSC-H]
      ├─ Live [Zombie NIR]
        ├─ CD45+ [CD45]
          ├─ T cells [CD3]
            ├─ CD4+ T [CD4]
            ├─ CD8+ T [CD8]
            ├─ gd T [TCRgd]
            ├─ MAIT [Va7.2, CD161]
          ├─ B cells [CD19, CD20]
          ├─ NK cells [CD56, CD3]
          ├─ Monocytes [CD14]
          ├─ DCs [CD11c, HLA-DR]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-A, FSC-H]
      ├─ Singlets (SSC) [SSC-A, SSC-H]
        ├─ Live Cells [Zombie NIR-]
          ├─ Leukocytes [CD45+]
            ├─ T Cells [CD3+]
              ├─ CD4+ T Cells [CD4+, CD8-]
                ├─ CD4+ Naive [CD45RA+, CCR7+]
                ├─ CD4+ Central Memory [CD45RA-, CCR7+]
                ├─ CD4+ Effector Memory [CD45RA-, CCR7-]
                  ├─ Th1 [CXCR3+, CCR6-]
                  ├─ Th17 [CXCR3-, CCR6+]
                  ├─ Th1/17 [CXCR3+, CCR6+]
                ├─ CD4+ TEMRA [CD45RA+, CCR7-]
                ├─ Tregs [CD25+, FoxP3+, CD127-]
                ├─ Tfh [CXCR5+, PD-1+]
              ├─ CD8+ T Cells [CD8+, CD4-]
                ├─ CD8+ Naive [CD45RA+, CCR7+]
                ├─ CD8+ Central Memory [CD45RA-, CCR7+]
                ├─ CD8+ Effector Memory [CD45RA-, CCR7-]
                ├─ CD8+ TEMRA [CD45RA+, CCR7-]
              ├─ γδ T Cells [TCRgd+]
              ├─ MAIT Cells [Va7.2+, CD161+]
            ├─ B Cells [CD19+, CD3-]
              ├─ Naive B Cells [CD27-, CD38-]
              ├─ Memory B Cells [CD27+, CD38-]
              ├─ Plasmablasts [CD27+, CD38++]
            ├─ NK Cells [CD3-, CD56+]
              ├─ CD56bright NK [CD56++, CD16-]
              ├─ CD56dim NK [CD56+, CD16+]
                ├─ Mature NK [CD57+, NKG2A-]
                ├─ Immature NK [CD57-, NKG2A+]
            ├─ Monocytes [CD14+, CD3-, CD19-, CD56-]
              ├─ Classical Monocytes [CD14++, CD16-]
              ├─ Intermediate Monocytes [CD14+, CD16+]
              ├─ Non-classical Monocytes [CD14+, CD16++]
            ├─ Dendritic Cells [HLA-DR+, CD14-, CD3-, CD19-, CD56-]
              ├─ Conventional DCs [CD11c+, CD123-]
              ├─ Plasmacytoid DCs [CD123+, CD11c-]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (9) | Monocytes, MAIT Cells, All Events, NK Cells, CD4+ T Cells, T Cells, Live Cells, B Cells... |
| ✗ **Missing** (5) | Singlets, CD45+, DCs, gd T, Time |
| ⚠ **Extra** (31) | CD56dim NK, CD4+ Naive, CD8+ Naive, Th17, Th1/17, Dendritic Cells, Classical Monocytes, Time Gate... |
| 🚨 **Missing Critical** (3) | Time, Singlets, CD45+ |

### Structure Errors

- Gate 'Monocytes': predicted parent='Leukocytes', expected parent='CD45+'

---

## OMIP-072

**Panel Markers:** CD45, CD3, CD4, CD8a, B220, CD19, CD11b, CD11c, F4/80, Ly6G... (+20 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 37.2% |
| Structure Accuracy | 50.0% |
| Critical Gate Recall | 25.0% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Time [Time]
    ├─ Singlets [FSC-A, FSC-H]
      ├─ Live [Zombie NIR]
        ├─ CD45+ [CD45]
          ├─ T cells [CD3]
            ├─ CD4+ T [CD4]
            ├─ CD8+ T [CD8]
          ├─ B cells [CD19]
          ├─ NK cells [CD56]
          ├─ Monocytes [CD14]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-A, FSC-H]
      ├─ Singlets (SSC) [SSC-A, SSC-H]
        ├─ Live Cells [Zombie NIR]
          ├─ CD45+ Immune Cells [CD45]
            ├─ T Cells [CD3]
              ├─ Conventional T Cells [TCRb]
                ├─ CD4+ T Cells [CD4]
                  ├─ CD4+ Naive [CD44, CD62L]
                  ├─ CD4+ Memory [CD44, CD62L]
                  ├─ Tregs [CD25, FoxP3]
                  ├─ Tfh Cells [PD-1, CXCR5]
                ├─ CD8+ T Cells [CD8a]
                  ├─ CD8+ Naive [CD44, CD62L]
                  ├─ CD8+ Memory [CD44, CD62L]
                  ├─ CD8+ Effector [Granzyme B, IFNg]
              ├─ γδ T Cells [TCRgd]
            ├─ B Cells [B220, CD19]
              ├─ Naive B Cells [IgM, IgD]
              ├─ Memory B Cells [IgM, IgD, CD44]
              ├─ Germinal Center B Cells [GL7, CD95]
              ├─ Plasma Cells [CD138]
            ├─ NK Cells [NK1.1, CD3]
            ├─ Myeloid Cells [CD11b]
              ├─ Neutrophils [Ly6G]
              ├─ Monocytes [Ly6C, Ly6G]
                ├─ Classical Monocytes [Ly6C]
                ├─ Non-classical Monocytes [Ly6C]
              ├─ Macrophages [F4/80]
            ├─ Dendritic Cells [CD11c, MHCII]
              ├─ Conventional DC [CD11c, MHCII]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (8) | Monocytes, All Events, NK Cells, CD4+ T Cells, T Cells, Live Cells, B Cells, CD8+ T Cells |
| ✗ **Missing** (3) | Singlets, CD45+, Time |
| ⚠ **Extra** (24) | Plasma Cells, CD4+ Naive, CD8+ Naive, Dendritic Cells, Classical Monocytes, CD8+ Effector, Time Gate, Naive B Cells... |
| 🚨 **Missing Critical** (3) | Time, Singlets, CD45+ |

### Structure Errors

- Gate 'Monocytes': predicted parent='Myeloid Cells', expected parent='CD45+'

---

## OMIP-005

**Panel Markers:** CD3, CD56, CD16, NKG2D, NKp46, CD57, CD94, CD45, KIR, Live/Dead

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 19.4% |
| Structure Accuracy | 100.0% |
| Critical Gate Recall | 33.3% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ Lymphocytes [FSC-A, SSC-A]
        ├─ NK cells [CD3, CD56]
          ├─ CD56bright [CD56, CD16]
          ├─ CD56dim [CD56, CD16]
            ├─ CD57+ NK [CD57]
            ├─ CD57- NK [CD57]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-H, FSC-A]
      ├─ Singlets (SSC) [SSC-H, SSC-A]
        ├─ Live Cells [Zombie NIR]
          ├─ Leukocytes [CD45]
            ├─ T Cells [CD3, CD56]
            ├─ NKT Cells [CD3, CD56]
            ├─ NK Cells [CD3, CD56]
              ├─ CD56bright NK [CD56, CD16]
                ├─ CD56bright NKG2D+ [NKG2D]
                ├─ CD56bright NKp46+ [NKp46]
                ├─ CD56bright CD57+ [CD57]
                ├─ CD56bright CD94+ [CD94]
                ├─ CD56bright KIR+ [KIR]
              ├─ CD56dim NK [CD56, CD16]
                ├─ CD56dim NKG2D+ [NKG2D]
                ├─ CD56dim NKp46+ [NKp46]
                ├─ CD56dim CD57+ [CD57]
                ├─ CD56dim CD94+ [CD94]
                ├─ CD56dim KIR+ [KIR]
            ├─ Other Cells [CD3, CD56]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (3) | Live Cells, NK Cells, All Events |
| ✗ **Missing** (6) | CD57- NK, Singlets, CD57+ NK, CD56bright, Lymphocytes, CD56dim |
| ⚠ **Extra** (19) | CD56bright CD57+, CD56dim NK, Leukocytes, Other Cells, Singlets (SSC), CD56dim CD94+, CD56bright NK, CD56dim CD57+... |
| 🚨 **Missing Critical** (2) | Singlets, Lymphocytes |

---

## OMIP-023

**Panel Markers:** CD3, CD4, CD8, CD45, CD19, CD14, CD16, CD56, Live/Dead

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 53.3% |
| Structure Accuracy | 50.0% |
| Critical Gate Recall | 50.0% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [7-AAD]
      ├─ CD45+ [CD45]
        ├─ Lymphocytes [FSC-A, SSC-A]
          ├─ T cells [CD3]
            ├─ CD4+ T cells [CD4]
            ├─ CD8+ T cells [CD8]
          ├─ B cells [CD19]
          ├─ NK cells [CD56, CD16]
        ├─ Monocytes [CD14, FSC-A, SSC-A]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets [FSC-A, FSC-H]
      ├─ Live Cells [7-AAD]
        ├─ Leukocytes [CD45]
          ├─ T Cells [CD3]
            ├─ CD4+ T Cells [CD4]
            ├─ CD8+ T Cells [CD8]
            ├─ DN T Cells [CD4, CD8]
          ├─ Non-T Cells [CD3]
            ├─ B Cells [CD19]
            ├─ Non-T Non-B Cells [CD19]
              ├─ NK Cells [CD56]
                ├─ CD56bright NK [CD56, CD16]
                ├─ CD56dim NK [CD56, CD16]
              ├─ Myeloid Cells [CD56]
                ├─ Classical Monocytes [CD14, CD16]
                ├─ Intermediate Monocytes [CD14, CD16]
                ├─ Non-classical Monocytes [CD14, CD16]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (8) | All Events, NK Cells, Singlets, CD4+ T Cells, T Cells, Live Cells, B Cells, CD8+ T Cells |
| ✗ **Missing** (3) | CD45+, Monocytes, Lymphocytes |
| ⚠ **Extra** (11) | Time Gate, CD56dim NK, Leukocytes, Intermediate Monocytes, DN T Cells, Non-T Non-B Cells, Myeloid Cells, Non-classical Monocytes... |
| 🚨 **Missing Critical** (2) | CD45+, Lymphocytes |

### Structure Errors

- Gate 'Singlets': predicted parent='Time Gate', expected parent='All Events'

---

## OMIP-017

**Panel Markers:** CD3, CD4, CD8, PD-1, TIM-3, LAG-3, TIGIT, CD39, CD45, CD45RA... (+8 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 40.0% |
| Structure Accuracy | 80.0% |
| Critical Gate Recall | 66.7% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ Lymphocytes [FSC-A, SSC-A]
        ├─ CD3+ T cells [CD3]
          ├─ CD8+ T cells [CD8]
            ├─ Exhausted CD8 [PD-1, TIM-3, LAG-3]
              ├─ Terminal Tex [CD39, TOX]
              ├─ Progenitor Tex [TCF1, TOX]
            ├─ Activated CD8 [CD69, HLA-DR, Ki67]
          ├─ CD4+ T cells [CD4]
            ├─ Exhausted CD4 [PD-1, TIM-3]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-A, FSC-H]
      ├─ Singlets (SSC) [SSC-A, SSC-H]
        ├─ Live Cells [Zombie NIR]
          ├─ Lymphocytes [FSC-A, SSC-A]
            ├─ CD45+ Leukocytes [CD45]
              ├─ CD3+ T cells [CD3]
                ├─ CD4+ T cells [CD4]
                  ├─ CD4+ Naive [CD45RA, CD27]
                  ├─ CD4+ Central Memory [CD45RA, CD27]
                  ├─ CD4+ Effector Memory [CD45RA, CD27]
                  ├─ CD4+ TEMRA [CD45RA, CD27]
                ├─ CD8+ T cells [CD8]
                  ├─ CD8+ Naive [CD45RA, CD27]
                  ├─ CD8+ Central Memory [CD45RA, CD27]
                  ├─ CD8+ Effector Memory [CD45RA, CD27]
                  ├─ CD8+ TEMRA [CD45RA, CD27]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (6) | All Events, CD4+ T cells, CD3+ T cells, Lymphocytes, Live Cells, CD8+ T cells |
| ✗ **Missing** (6) | Exhausted CD4, Singlets, Exhausted CD8, Progenitor Tex, Activated CD8, Terminal Tex |
| ⚠ **Extra** (12) | CD4+ Effector Memory, Time Gate, CD4+ Central Memory, CD4+ Naive, CD8+ Naive, CD45+ Leukocytes, Singlets (FSC), Singlets (SSC)... |
| 🚨 **Missing Critical** (1) | Singlets |

### Structure Errors

- Gate 'CD3+ T cells': predicted parent='CD45+ Leukocytes', expected parent='Lymphocytes'

---

## OMIP-011

**Panel Markers:** B220, CD19, IgM, IgD, CD21, CD23, CD45, CD138, GL7, CD95... (+2 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 25.0% |
| Structure Accuracy | 100.0% |
| Critical Gate Recall | 33.3% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ CD45+ [CD45]
        ├─ B cells [B220, CD19]
          ├─ Transitional [IgM, CD21, CD23]
          ├─ Follicular [IgD, CD21, CD23]
          ├─ Marginal Zone [IgM, CD21, CD23]
          ├─ GC B cells [GL7, CD95]
          ├─ Plasma cells [CD138, B220]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-H, FSC-A]
      ├─ Singlets (SSC) [SSC-H, SSC-A]
        ├─ Live Cells [Zombie NIR]
          ├─ CD45+ Leukocytes [CD45]
            ├─ B Cells [B220, CD19]
              ├─ Plasma Cells [CD138]
              ├─ Non-Plasma B Cells [CD138]
                ├─ Immature B Cells [IgM, IgD]
                ├─ Transitional B Cells [IgM, IgD]
                  ├─ T1 B Cells [CD21, CD23]
                  ├─ T2 B Cells [CD21, CD23]
                ├─ Mature B Cells [IgM, IgD]
                  ├─ Follicular B Cells [CD21, CD23]
                    ├─ Naive Follicular B Cells [GL7, CD95]
                      ├─ CD38+ Memory-like [CD38]
                      ├─ CD38- Naive [CD38]
                    ├─ Germinal Center B Cells [GL7, CD95]
                  ├─ Marginal Zone B Cells [CD21, CD23]
                    ├─ CD38+ Activated MZ [CD38]
                    ├─ CD38- Resting MZ [CD38]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (4) | Live Cells, B Cells, Plasma Cells, All Events |
| ✗ **Missing** (6) | Marginal Zone, Follicular, Transitional, Singlets, CD45+, GC B cells |
| ⚠ **Extra** (18) | CD38+ Memory-like, T2 B Cells, Immature B Cells, Follicular B Cells, T1 B Cells, Singlets (SSC), CD38- Naive, Non-Plasma B Cells... |
| 🚨 **Missing Critical** (2) | Singlets, CD45+ |

---

## OMIP-027

**Panel Markers:** CD45, CD3, CD4, CD8, CD19, CD56, CD14, CD16, HLA-DR, CD11b... (+14 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 28.0% |
| Structure Accuracy | 50.0% |
| Critical Gate Recall | 33.3% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ CD45+ TILs [CD45]
        ├─ T cells [CD3]
          ├─ CD8+ TIL [CD8]
            ├─ Exhausted [PD-1, TIM-3, LAG-3]
            ├─ Resident [CD103, CD69]
            ├─ Cytotoxic [Granzyme B]
          ├─ CD4+ TIL [CD4]
            ├─ Tregs [CD25, FoxP3]
            ├─ Th1 [PD-1]
        ├─ NK cells [CD56, CD3]
        ├─ B cells [CD19]
        ├─ Myeloid [CD14, CD11b]
          ├─ TAMs [CD14, HLA-DR]
          ├─ MDSCs [CD11b, HLA-DR]
      ├─ Tumor cells [CD45]
        ├─ PD-L1+ tumor [PD-L1]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-A, FSC-H]
      ├─ Singlets (SSC) [SSC-A, SSC-H]
        ├─ Live Cells [Zombie NIR]
          ├─ Leukocytes [CD45]
            ├─ T Cells [CD3]
              ├─ CD4+ T Cells [CD4]
                ├─ Tregs [CD25, FoxP3]
                ├─ CD4+ Activated [CD69]
                ├─ CD4+ Proliferating [Ki67]
                ├─ CD4+ PD-1+ [PD-1]
                ├─ CD4+ Exhausted [PD-1, TIM-3, LAG-3]
              ├─ CD8+ T Cells [CD8]
                ├─ CD8+ Cytotoxic [Granzyme B]
                ├─ CD8+ Tissue Resident [CD103]
                ├─ CD8+ Activated [CD69]
                ├─ CD8+ PD-1+ [PD-1]
                ├─ CD8+ Exhausted [PD-1, TIM-3, LAG-3]
            ├─ Non-T Cells [CD3]
              ├─ B Cells [CD19]
                ├─ Activated B Cells [HLA-DR]
              ├─ NK Cells [CD56]
                ├─ Cytotoxic NK [CD16]
                ├─ NK Activated [CD69]
              ├─ Myeloid Cells [CD11b]
                ├─ Monocytes/Macrophages [CD14]
                  ├─ Activated Macrophages [HLA-DR]
                  ├─ PD-L1+ Macrophages [PD-L1]
                ├─ Dendritic Cells [CD11c, HLA-DR]
                  ├─ PD-L1+ DCs [PD-L1]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (7) | All Events, NK Cells, Tregs, Myeloid Cells, T Cells, Live Cells, B Cells |
| ✗ **Missing** (12) | Resident, CD45+ TILs, Singlets, CD8+ TIL, CD4+ TIL, Th1, MDSCs, TAMs... |
| ⚠ **Extra** (24) | Activated B Cells, CD8+ Cytotoxic, Monocytes/Macrophages, Dendritic Cells, CD4+ Activated, CD8+ Tissue Resident, Time Gate, Singlets (FSC)... |
| 🚨 **Missing Critical** (2) | Singlets, CD45+ TILs |

### Structure Errors

- Gate 'Tregs': predicted parent='CD4+ T Cells', expected parent='CD4+ TIL'

---

## OMIP-019

**Panel Markers:** CD45, CD3, CD4, CD8a, B220, CD19, CD11b, CD11c, F4/80, Ly6G... (+10 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 46.8% |
| Structure Accuracy | 50.0% |
| Critical Gate Recall | 33.3% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ CD45+ [CD45]
        ├─ T cells [CD3, TCRb]
          ├─ CD4+ T [CD4]
            ├─ Tregs [CD25, FoxP3]
          ├─ CD8+ T [CD8a]
          ├─ gd T cells [TCRgd]
        ├─ B cells [B220, CD19]
        ├─ NK cells [NK1.1, CD3]
        ├─ Myeloid [CD11b]
          ├─ Neutrophils [Ly6G]
          ├─ Monocytes [Ly6C]
          ├─ Macrophages [F4/80]
        ├─ DCs [CD11c, MHCII]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-H, FSC-A]
      ├─ Singlets (SSC) [SSC-H, SSC-A]
        ├─ Live Cells [Zombie NIR]
          ├─ CD45+ Leukocytes [CD45]
            ├─ T Cells [CD3]
              ├─ TCRb+ T Cells [TCRb]
                ├─ CD4+ T Cells [CD4]
                  ├─ CD4+ Conventional T Cells [FoxP3]
                    ├─ CD4+ Naive [CD44, CD62L]
                    ├─ CD4+ Central Memory [CD44, CD62L]
                    ├─ CD4+ Effector Memory [CD44, CD62L]
                  ├─ CD4+ Regulatory T Cells [FoxP3, CD25]
                ├─ CD8+ T Cells [CD8a]
                  ├─ CD8+ Naive [CD44, CD62L]
                  ├─ CD8+ Central Memory [CD44, CD62L]
                  ├─ CD8+ Effector Memory [CD44, CD62L]
              ├─ TCRgd+ T Cells [TCRgd]
            ├─ Non-T Cells [CD3]
              ├─ B Cells [B220]
                ├─ B220+ CD19+ B Cells [CD19]
                  ├─ Naive B Cells [CD44]
                  ├─ Activated B Cells [CD44, MHCII]
              ├─ NK Cells [NK1.1]
              ├─ Myeloid Cells [CD11b]
                ├─ Neutrophils [Ly6G]
                ├─ Ly6G- Myeloid [Ly6G]
                  ├─ Monocytes [Ly6C, F4/80]
                  ├─ Macrophages [F4/80]
                  ├─ Dendritic Cells [CD11c, MHCII]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (11) | Monocytes, All Events, NK Cells, Neutrophils, CD4+ T Cells, Macrophages, Myeloid Cells, T Cells... |
| ✗ **Missing** (5) | Singlets, CD45+, DCs, gd T cells, Tregs |
| ⚠ **Extra** (20) | Activated B Cells, Ly6G- Myeloid, CD4+ Naive, CD8+ Naive, B220+ CD19+ B Cells, Dendritic Cells, CD4+ Conventional T Cells, Time Gate... |
| 🚨 **Missing Critical** (2) | Singlets, CD45+ |

### Structure Errors

- Gate 'Monocytes': predicted parent='Ly6G- Myeloid', expected parent='Myeloid'
- Gate 'Macrophages': predicted parent='Ly6G- Myeloid', expected parent='Myeloid'

---

## OMIP-069

**Panel Markers:** CD3, CD4, CD8, CD45, CD45RA, CD19, CD14, CD16, CD56, CD127... (+1 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 68.2% |
| Structure Accuracy | 77.8% |
| Critical Gate Recall | 40.0% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Time [Time]
    ├─ Singlets [FSC-A, FSC-H]
      ├─ Live [Zombie NIR]
        ├─ CD45+ [CD45]
          ├─ Lymphocytes [FSC-A, SSC-A]
            ├─ T cells [CD3]
              ├─ CD4+ T cells [CD4]
                ├─ CD4+ Naive [CD45RA, CD127]
                ├─ CD4+ Memory [CD45RA]
              ├─ CD8+ T cells [CD8]
                ├─ CD8+ Naive [CD45RA, CD127]
                ├─ CD8+ Memory [CD45RA]
            ├─ B cells [CD19]
            ├─ NK cells [CD56, CD16]
              ├─ CD56bright NK [CD56]
              ├─ CD56dim NK [CD56, CD16]
          ├─ Monocytes [CD14, FSC-A, SSC-A]
            ├─ Classical Monocytes [CD14, CD16]
            ├─ Non-classical Monocytes [CD14, CD16]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-A, FSC-H]
      ├─ Singlets (SSC) [SSC-A, SSC-H]
        ├─ Live Cells [Zombie NIR]
          ├─ Lymphocytes [FSC-A, SSC-A]
            ├─ CD45+ Immune Cells [CD45]
              ├─ T Cells [CD3]
                ├─ CD4+ T Cells [CD4]
                  ├─ CD4+ Naive [CD45RA, CD127]
                  ├─ CD4+ Memory [CD45RA, CD127]
                ├─ CD8+ T Cells [CD8]
                  ├─ CD8+ Naive [CD45RA, CD127]
                  ├─ CD8+ Memory [CD45RA, CD127]
              ├─ Non-T Cells [CD3]
                ├─ B Cells [CD19]
                  ├─ Naive B Cells [CD45RA]
                  ├─ Memory B Cells [CD45RA]
                ├─ Non-B Cells [CD19]
                  ├─ Classical Monocytes [CD14, CD16]
                  ├─ Intermediate Monocytes [CD14, CD16]
                  ├─ NK Cells [CD56]
                    ├─ CD56bright NK [CD56, CD16]
                    ├─ CD56dim NK [CD56, CD16]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (15) | B Cells, CD56dim NK, CD4+ Naive, All Events, CD8+ Naive, NK Cells, CD4+ T Cells, CD8+ T Cells... |
| ✗ **Missing** (5) | Monocytes, Singlets, CD45+, Time, Non-classical Monocytes |
| ⚠ **Extra** (9) | Time Gate, Naive B Cells, Singlets (FSC), Singlets (SSC), CD45+ Immune Cells, Memory B Cells, Non-B Cells, Intermediate Monocytes... |
| 🚨 **Missing Critical** (3) | Time, Singlets, CD45+ |

### Structure Errors

- Gate 'Classical Monocytes': predicted parent='Non-B Cells', expected parent='Monocytes'
- Gate 'Lymphocytes': predicted parent='Live Cells', expected parent='CD45+'

---

## OMIP-070

**Panel Markers:** CD3, CD4, CD8, CD45, CD45RA, CD19, CD14, CD16, CD56, HLA-DR... (+28 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 32.7% |
| Structure Accuracy | 50.0% |
| Critical Gate Recall | 25.0% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Time [Time]
    ├─ Singlets [FSC-A, FSC-H]
      ├─ Live [Zombie NIR]
        ├─ CD45+ [CD45]
          ├─ T cells [CD3]
            ├─ CD4+ T [CD4]
            ├─ CD8+ T [CD8]
          ├─ B cells [CD19]
          ├─ NK cells [CD56]
          ├─ Monocytes [CD14]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-A, FSC-H]
      ├─ Singlets (SSC) [SSC-A, SSC-H]
        ├─ Live Cells [Zombie NIR]
          ├─ Leukocytes [CD45]
            ├─ T Cells [CD3]
              ├─ CD4+ T Cells [CD4]
                ├─ CD4+ Naive [CD45RA, CCR7]
                ├─ CD4+ Central Memory [CD45RA, CCR7]
                ├─ CD4+ Effector Memory [CD45RA, CCR7]
                  ├─ Th1 [CXCR3]
                  ├─ Th17 [CCR6]
                  ├─ Tfh [CXCR5]
                ├─ CD4+ TEMRA [CD45RA, CCR7]
                ├─ Tregs [CD25, FoxP3]
              ├─ CD8+ T Cells [CD8]
                ├─ CD8+ Naive [CD45RA, CCR7]
                ├─ CD8+ Central Memory [CD45RA, CCR7]
                ├─ CD8+ Effector Memory [CD45RA, CCR7]
                ├─ CD8+ TEMRA [CD45RA, CCR7]
              ├─ γδ T Cells [TCRgd]
              ├─ MAIT Cells [Va7.2, CD161]
            ├─ B Cells [CD19]
              ├─ Naive B Cells [CD27]
              ├─ Memory B Cells [CD27]
            ├─ NK Cells [CD3, CD56]
              ├─ CD56bright NK [CD56]
              ├─ CD56dim NK [CD56]
                ├─ CD57+ NK [CD57]
                ├─ NKG2A+ NK [NKG2A]
            ├─ Monocytes [CD14]
              ├─ Classical Monocytes [CD14, CD16]
              ├─ Intermediate Monocytes [CD14, CD16]
              ├─ Non-classical Monocytes [CD14, CD16]
            ├─ Dendritic Cells [HLA-DR]
              ├─ mDCs [CD11c]
              ├─ pDCs [CD123]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (8) | Monocytes, All Events, NK Cells, CD4+ T Cells, T Cells, Live Cells, B Cells, CD8+ T Cells |
| ✗ **Missing** (3) | Singlets, CD45+, Time |
| ⚠ **Extra** (30) | CD56dim NK, CD4+ Naive, CD8+ Naive, Th17, NKG2A+ NK, Dendritic Cells, Classical Monocytes, Time Gate... |
| 🚨 **Missing Critical** (3) | Time, Singlets, CD45+ |

### Structure Errors

- Gate 'Monocytes': predicted parent='Leukocytes', expected parent='CD45+'

---

## OMIP-009

**Panel Markers:** CD3, CD4, CD25, CD127, FoxP3, CD45RA, CTLA-4, CD45, Helios, Live/Dead

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 42.9% |
| Structure Accuracy | 60.0% |
| Critical Gate Recall | 66.7% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ Lymphocytes [FSC-A, SSC-A]
        ├─ CD3+ T cells [CD3]
          ├─ CD4+ T cells [CD4]
            ├─ Tregs [CD25, CD127]
              ├─ FoxP3+ Tregs [FoxP3]
              ├─ Naive Tregs [CD45RA]
              ├─ Memory Tregs [CD45RA]
            ├─ Tconv [CD25, CD127]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-A, FSC-H]
      ├─ Singlets (SSC) [SSC-A, SSC-H]
        ├─ Live Cells [Zombie NIR]
          ├─ Lymphocytes [FSC-A, SSC-A]
            ├─ CD45+ Leukocytes [CD45]
              ├─ CD3+ T cells [CD3]
                ├─ CD4+ T cells [CD4]
                  ├─ CD25+ CD127low/- Tregs [CD25, CD127]
                    ├─ FoxP3+ Tregs [FoxP3]
                      ├─ Naive Tregs (CD45RA+) [CD45RA]
                        ├─ CTLA-4+ Naive Tregs [CTLA-4]
                        ├─ Helios+ Naive Tregs [Helios]
                      ├─ Memory Tregs (CD45RA-) [CD45RA]
                        ├─ CTLA-4+ Memory Tregs [CTLA-4]
                        ├─ Helios+ Memory Tregs [Helios]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (6) | All Events, CD4+ T cells, CD3+ T cells, FoxP3+ Tregs, Lymphocytes, Live Cells |
| ✗ **Missing** (5) | Tconv, Naive Tregs, Singlets, Memory Tregs, Tregs |
| ⚠ **Extra** (11) | Helios+ Naive Tregs, Time Gate, CD45+ Leukocytes, CTLA-4+ Memory Tregs, Singlets (FSC), Singlets (SSC), CD25+ CD127low/- Tregs, Memory Tregs (CD45RA-)... |
| 🚨 **Missing Critical** (1) | Singlets |

### Structure Errors

- Gate 'FoxP3+ Tregs': predicted parent='CD25+ CD127low/- Tregs', expected parent='Tregs'
- Gate 'CD3+ T cells': predicted parent='CD45+ Leukocytes', expected parent='Lymphocytes'

---

## OMIP-037

**Panel Markers:** CD34, CD38, CD45RA, CD90, CD49f, CD45, CD10, CD7, CD123, CD135... (+8 more)

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | 13.8% |
| Structure Accuracy | 100.0% |
| Critical Gate Recall | 33.3% |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
├─ All Events
  ├─ Singlets [FSC-A, FSC-H]
    ├─ Live [Zombie NIR]
      ├─ CD45low [CD45]
        ├─ CD34+ [CD34]
          ├─ HSC [CD38, CD90, CD45RA]
          ├─ MPP [CD38, CD90, CD45RA]
          ├─ CMP [CD38, CD123, CD45RA]
          ├─ GMP [CD38, CD123, CD45RA]
          ├─ MEP [CD38, CD123, CD45RA]
          ├─ CLP [CD38, CD10, CD7]
```

</td>
<td>

```
├─ All Events
  ├─ Time Gate [Time]
    ├─ Singlets (FSC) [FSC-A, FSC-H]
      ├─ Singlets (SSC) [SSC-A, SSC-H]
        ├─ Live Cells [Zombie NIR]
          ├─ CD45+ Hematopoietic [CD45]
            ├─ Lineage Negative [CD3, CD19, CD33, CD71, CD235a, CD41a]
              ├─ CD34+ Stem/Progenitors [CD34]
                ├─ CD34+CD38- Primitive [CD38]
                  ├─ HSC (CD90+CD45RA-) [CD90, CD45RA]
                    ├─ LT-HSC (CD49f+) [CD49f]
                    ├─ ST-HSC (CD49f-) [CD49f]
                  ├─ MPP (CD90-CD45RA-) [CD90, CD45RA]
                  ├─ LMPP (CD90-CD45RA+) [CD90, CD45RA]
                ├─ CD34+CD38+ Committed Progenitors [CD38]
                  ├─ CMP (CD123-CD45RA-) [CD123, CD45RA]
                  ├─ GMP (CD123-CD45RA+) [CD123, CD45RA]
                  ├─ MEP (CD123+CD45RA-) [CD123, CD45RA]
```

</td>
</tr>
</table>

### Gate Analysis

| Category | Gates |
|----------|-------|
| ✓ **Matching** (2) | Live Cells, All Events |
| ✗ **Missing** (9) | CD45low, MPP, CD34+, HSC, Singlets, CMP, GMP, MEP... |
| ⚠ **Extra** (16) | HSC (CD90+CD45RA-), CD34+CD38+ Committed Progenitors, GMP (CD123-CD45RA+), Singlets (SSC), CMP (CD123-CD45RA-), LMPP (CD90-CD45RA+), CD34+CD38- Primitive, LT-HSC (CD49f+)... |
| 🚨 **Missing Critical** (2) | Singlets, CD45low |

---

## Legend

- **Matching Gates**: Gates correctly predicted (present in both)
- **Missing Gates**: Gates in ground truth but not predicted
- **Extra Gates**: Gates predicted but not in ground truth
- **Missing Critical**: Essential QC/lineage gates that were missed
- **Structure Errors**: Parent-child relationships that don't match

## Review Guidelines

When manually reviewing:
1. Check if "extra" gates are reasonable alternatives (may indicate ground truth gaps)
2. Evaluate if missing gates are truly missing or just named differently
3. Assess biological plausibility of the predicted hierarchy
4. Note any systematic patterns across test cases

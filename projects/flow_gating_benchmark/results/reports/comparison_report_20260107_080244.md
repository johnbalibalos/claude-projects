# Manual Review Report: LLM vs Ground Truth Comparison

**Generated:** 2026-01-07 08:02:44
**Model:** claude-sonnet-4-20250514

This report shows the LLM-predicted gating hierarchy alongside the ground truth (OMIP)
hierarchy for manual comparison and review.

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
| ✓ **Matching** (8) | NK Cells, Singlets, CD4+ T Cells, CD8+ T Cells, All Events, Live Cells, B Cells, T Cells |
| ✗ **Missing** (3) | Monocytes, CD45+, Lymphocytes |
| ⚠ **Extra** (11) | Leukocytes, CD56dim NK, Non-T Cells, DN T Cells, Non-T Non-B Cells, Myeloid Cells, Time Gate, Non-classical Monocytes... |
| 🚨 **Missing Critical** (2) | CD45+, Lymphocytes |

### Structure Errors

- Gate 'Singlets': predicted parent='Time Gate', expected parent='All Events'

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
| ✓ **Matching** (15) | CD8+ Naive, NK Cells, CD8+ T Cells, CD56dim NK, CD4+ T Cells, Lymphocytes, CD4+ Naive, All Events... |
| ✗ **Missing** (5) | Monocytes, Singlets, CD45+, Time, Non-classical Monocytes |
| ⚠ **Extra** (9) | Singlets (SSC), Non-B Cells, Naive B Cells, Non-T Cells, Memory B Cells, Time Gate, Singlets (FSC), Intermediate Monocytes... |
| 🚨 **Missing Critical** (3) | Time, Singlets, CD45+ |

### Structure Errors

- Gate 'Lymphocytes': predicted parent='Live Cells', expected parent='CD45+'
- Gate 'Classical Monocytes': predicted parent='Non-B Cells', expected parent='Monocytes'

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

# Flow Cytometry Gating Benchmark Test Plan

## Overview

This test plan evaluates LLM performance on predicting flow cytometry gating hierarchies, with ablation studies to determine if MCP tools, domain skills, or improved prompting strategies improve accuracy.

**Primary Research Questions:**
1. Why do LLMs hallucinate gates not present in panels?
2. Why do LLMs miss critical QC gates (Time, Singlets, Live/Dead)?
3. Does providing flowkit MCP tools improve structure accuracy?
4. Does injecting immunology domain knowledge (skills) improve F1?
5. What prompting strategies yield best results?

---

## Test Matrix

### Experimental Conditions

| Condition | Strategy | Context | Ablation | Description |
|-----------|----------|---------|----------|-------------|
| baseline | cot | standard | none | Chain-of-thought, standard context |
| direct_baseline | direct | standard | none | No reasoning, direct output |
| mcp_cot | cot | standard | mcp | With flowkit MCP tools |
| mcp_direct | direct | standard | mcp | MCP tools, no reasoning |
| skills_cot | cot | standard | skills | With immunology skills |
| skills_direct | direct | standard | skills | Skills, no reasoning |
| rich_context | cot | rich | none | Enhanced panel context |
| minimal_context | cot | minimal | none | Reduced context |
| few_shot | few_shot | standard | none | Example-based prompting |

### Test Cases

- **Total:** 30 OMIP test cases
- **Distribution:**
  - Simple (≤15 colors): 14 cases
  - Medium (16-25 colors): 9 cases
  - Complex (26+ colors): 7 cases
  - Human: 20 cases, Mouse: 10 cases

---

## Ablation Studies

### 1. MCP Ablation (FlowKit Integration)

**Hypothesis:** Providing MCP tools for reading workspace files will improve structure accuracy by giving the model access to validated gating hierarchies.

**MCP Tools Simulated:**
```
read_workspace(path) -> WorkspaceInfo
get_gating_hierarchy(workspace) -> GatingHierarchy
list_populations(workspace) -> list[Population]
get_gate_statistics(workspace, gate) -> GateStats
```

**Expected Improvements:**
- Higher structure accuracy (parent-child relationships)
- Better critical gate recall (standard QC workflow)
- Reduced hallucination rate

**Run Command:**
```bash
python run_benchmark.py --ablation mcp --strategy cot
python run_benchmark.py --ablation mcp --strategy direct
```

### 2. Skills Ablation (Domain Knowledge)

**Hypothesis:** Injecting immunologist domain knowledge as a "skill" will improve F1 by providing standard gating workflows and nomenclature.

**Skills Provided:**
- Standard QC workflow (Time → Cells → Singlets → Live)
- Lineage identification rules (CD45, EpCAM, CD235a)
- T/B/NK/Myeloid subset hierarchies
- Critical gate checklist
- Gate naming conventions

**Expected Improvements:**
- Higher critical gate recall (explicit QC instructions)
- Better hierarchy F1 (standard nomenclature)
- Reduced missing gates

**Run Command:**
```bash
python run_benchmark.py --ablation skills --strategy cot
python run_benchmark.py --ablation skills --strategy direct
```

### 3. Prompting Strategy Ablation

**Hypothesis:** Chain-of-thought (CoT) prompting yields better results than direct prompting by encouraging systematic analysis.

**Strategies:**
| Strategy | Description |
|----------|-------------|
| `cot` | "Think step by step about the gating workflow..." |
| `direct` | "Output the gating hierarchy as JSON" |
| `few_shot` | Provide 1-2 example hierarchies before query |

**Run Command:**
```bash
python run_benchmark.py --strategy cot
python run_benchmark.py --strategy direct
python run_benchmark.py --strategy few_shot
```

### 4. Context Level Ablation

**Hypothesis:** Richer context (sample type, species, application) improves accuracy by providing biological context.

**Context Levels:**
| Level | Information Provided |
|-------|---------------------|
| `minimal` | Panel markers only |
| `standard` | Markers + species + sample type |
| `rich` | Markers + species + sample + application + fluorochrome details |

**Run Command:**
```bash
python run_benchmark.py --context minimal
python run_benchmark.py --context standard
python run_benchmark.py --context rich
```

---

## Metrics

### Primary Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Hierarchy F1** | 2 × (P × R) / (P + R) | > 0.60 |
| **Structure Accuracy** | Correct parent-child / Total | > 0.80 |
| **Critical Gate Recall** | Found critical / Total critical | > 0.80 |
| **Hallucination Rate** | Non-panel gates / Total predicted | < 0.10 |

### Secondary Metrics

| Metric | Description |
|--------|-------------|
| Parse Success Rate | % of valid JSON outputs |
| Depth Accuracy | Match of hierarchy depth |
| Gate Precision | Predicted ∩ Ground Truth / Predicted |
| Gate Recall | Predicted ∩ Ground Truth / Ground Truth |

### Critical Gates

These gates MUST be present in any immunophenotyping panel:

| Gate | Priority | Reason |
|------|----------|--------|
| Time | High | QC - acquisition stability |
| Singlets | Critical | QC - doublet exclusion |
| Live/Dead | Critical | QC - viability |
| CD45+ | High | Immune cell identification |
| Lymphocytes | Medium | Major population gate |

---

## Execution Plan

### Phase 1: Baseline (Current)
```bash
# Already completed - baseline results
python run_benchmark.py --strategy cot --context standard
```
**Results:** F1=38.9%, Critical Recall=44.4%

### Phase 2: MCP Ablation
```bash
# Test MCP with CoT
python run_benchmark.py --ablation mcp --strategy cot

# Test MCP with direct
python run_benchmark.py --ablation mcp --strategy direct
```
**Expected Runtime:** ~15 min each, ~$0.90 each

### Phase 3: Skills Ablation
```bash
# Test Skills with CoT
python run_benchmark.py --ablation skills --strategy cot

# Test Skills with direct
python run_benchmark.py --ablation skills --strategy direct
```
**Expected Runtime:** ~15 min each, ~$0.90 each

### Phase 4: Prompting Strategy Comparison
```bash
# Already have CoT baseline
python run_benchmark.py --strategy direct --context standard
python run_benchmark.py --strategy few_shot --context standard
```

### Phase 5: Context Level Comparison
```bash
python run_benchmark.py --context minimal --strategy cot
python run_benchmark.py --context rich --strategy cot
```

### Phase 6: Combined Best Approach
```bash
# Combine best ablation with best strategy
python run_benchmark.py --ablation skills --strategy cot --context rich
```

---

## Analysis Plan

### 1. Per-Test Analysis
After each test case, generate detailed report showing:
- Side-by-side hierarchy comparison (LLM vs Gold Standard)
- Gate-by-gate analysis (matching, missing, extra)
- Critical gate checklist
- Recommendations for improvement

### 2. Strategy Comparison
Generate comparison table showing:
- F1 by strategy/ablation
- Critical gate recall improvement
- Hallucination rate changes
- Cost per condition

### 3. Error Analysis
Categorize failures:
1. **Missing QC Gates** - Time, Singlets, Live/Dead omitted
2. **Naming Mismatches** - "Lymphs" vs "Lymphocytes"
3. **Structure Errors** - Wrong parent-child relationships
4. **Hallucinations** - Gates using non-panel markers
5. **Over-generation** - Too many subset populations

### 4. Complexity Analysis
Compare performance by:
- Panel size (colors)
- Species (human vs mouse)
- Sample type (PBMC vs tissue)

---

## Success Criteria

### Minimum Viable Performance
- Parse success rate: > 95%
- Hierarchy F1: > 50%
- Critical gate recall: > 60%

### Target Performance
- Hierarchy F1: > 70%
- Structure accuracy: > 85%
- Critical gate recall: > 90%
- Hallucination rate: < 5%

### Ablation Success
An ablation is considered successful if it improves F1 by > 10% relative to baseline.

| Baseline F1 | Target with Ablation |
|-------------|---------------------|
| 38.9% | > 42.8% (+10% relative) |

---

## Quick Test Commands

```bash
# Quick test (3 cases) to verify pipeline
python run_benchmark.py --limit 3 --verbose

# Full baseline
python run_benchmark.py

# MCP ablation
python run_benchmark.py --ablation mcp

# Skills ablation
python run_benchmark.py --ablation skills

# Full comparison (all strategies)
./scripts/run_full_comparison.sh
```

---

## Output Structure

```
results/
├── benchmark_results_YYYYMMDD_HHMMSS.json   # Raw results
├── reports/
│   ├── YYYYMMDD_HHMMSS/
│   │   ├── per_test/
│   │   │   ├── omip_001_*.txt              # Per-test detailed reports
│   │   │   ├── omip_003_*.txt
│   │   │   └── ...
│   │   ├── comparison_table.md              # LLM vs Gold Standard table
│   │   └── strategy_comparison.md           # Cross-strategy analysis
│   ├── summary_report_*.md                  # Executive summary
│   └── comparison_report_*.md               # Side-by-side review
```

---

## Future Enhancements

### Real MCP Implementation
Replace simulated MCP with actual flowkit integration:
```python
# MCP server definition
@mcp.tool()
def read_workspace(path: str) -> dict:
    """Read FlowJo workspace file."""
    import flowkit as fk
    wsp = fk.Workspace(path)
    return wsp.get_gating_hierarchy()
```

### Additional Test Cases
- Add OMIP panels with workspace files (.wsp)
- Include real-world FlowRepository datasets
- Add edge cases (unusual panels, rare populations)

### Fuzzy Matching
Implement gate name normalization:
- "Lymphocytes" ≈ "Lymphs" ≈ "Lymphocyte Gate"
- "CD4+ T Cells" ≈ "CD4 T Cells" ≈ "Helper T Cells"

### Interactive Evaluation
Build UI for manual review of predictions with expert annotation.

---

## References

1. Gating Workflow Guide: `docs/gating_workflow_guide.md`
2. OMIP Series: Cytometry Part A
3. FlowKit Documentation: https://flowkit.readthedocs.io/
4. FlowRepository: https://flowrepository.org/

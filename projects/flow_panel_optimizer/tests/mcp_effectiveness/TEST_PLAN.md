# MCP Panel Designer Effectiveness Test Plan

## Objective

Measure whether Claude + MCP tools produces better flow cytometry panels than Claude alone, using quantitative spectral metrics.

## Hypothesis

Claude with access to MCP panel design tools will produce panels with:
- **Lower complexity index** (less overall interference)
- **Lower maximum pairwise similarity** (fewer critical conflicts)
- **Better brightness matching** (appropriate fluorophores for expression levels)

---

## Test Design

### A/B Comparison

| Condition | Description |
|-----------|-------------|
| **Control (A)** | Claude designs panel using only its training knowledge |
| **Treatment (B)** | Claude designs panel with MCP tools available |

### Metrics

1. **Complexity Index (CI)** - Lower is better
2. **Max Pairwise Similarity** - Lower is better
3. **Count of Critical Pairs (SI > 0.95)** - Zero is ideal
4. **Count of High-Risk Pairs (SI > 0.90)** - Fewer is better
5. **Brightness Match Score** - How well fluorophore brightness matches marker expression

---

## Test Cases

### Test Case 1: Basic T-Cell Panel (8 markers)
```yaml
name: "basic_tcell_panel"
description: "Simple T-cell immunophenotyping"
instrument: "4-laser (VBYR)"
markers:
  - name: "CD3"
    expression: "high"
    cell_type: "T cells"
  - name: "CD4"
    expression: "high"
    cell_type: "Helper T"
  - name: "CD8"
    expression: "high"
    cell_type: "Cytotoxic T"
  - name: "CD45RA"
    expression: "high"
    cell_type: "Naive T"
  - name: "CD45RO"
    expression: "medium"
    cell_type: "Memory T"
  - name: "CCR7"
    expression: "medium"
    cell_type: "Central memory"
  - name: "CD27"
    expression: "medium"
    cell_type: "Memory T"
  - name: "Viability"
    expression: "high"
    cell_type: "Dead cell exclusion"

expected_difficulty: "easy"
reference_panel: null
```

### Test Case 2: Memory T-Cell Subset Panel (12 markers)
```yaml
name: "memory_tcell_panel"
description: "Detailed memory T-cell characterization"
instrument: "4-laser (VBYR)"
markers:
  - {name: "CD3", expression: "high"}
  - {name: "CD4", expression: "high"}
  - {name: "CD8", expression: "high"}
  - {name: "CD45RA", expression: "high"}
  - {name: "CCR7", expression: "medium"}
  - {name: "CD27", expression: "medium"}
  - {name: "CD28", expression: "medium"}
  - {name: "CD57", expression: "low"}
  - {name: "CD127", expression: "low"}
  - {name: "CD25", expression: "low"}
  - {name: "PD-1", expression: "low"}
  - {name: "Viability", expression: "high"}

expected_difficulty: "medium"
coexpression_constraints:
  - ["CD4", "CD8"]  # Mutually exclusive
  - ["CD45RA", "CCR7"]  # Define memory subsets together
```

### Test Case 3: OMIP-030 Recreation (13 markers)
```yaml
name: "omip030_recreation"
description: "Recreate OMIP-030 T-cell panel"
instrument: "4-laser (VBYR)"
markers:
  - {name: "CD3", expression: "high"}
  - {name: "CD4", expression: "high"}
  - {name: "CD8", expression: "high"}
  - {name: "CD14", expression: "high"}
  - {name: "CD16", expression: "medium"}
  - {name: "CD19", expression: "high"}
  - {name: "CD25", expression: "low"}
  - {name: "CD27", expression: "medium"}
  - {name: "CD45RA", expression: "high"}
  - {name: "CD56", expression: "medium"}
  - {name: "CD127", expression: "low"}
  - {name: "CCR7", expression: "medium"}
  - {name: "Viability", expression: "high"}

expected_difficulty: "medium"
reference_panel: "OMIP-030"
reference_complexity_index: null  # Not published
```

### Test Case 4: Treg Panel (15 markers)
```yaml
name: "treg_panel"
description: "Regulatory T-cell identification and characterization"
instrument: "5-laser (UVBYR)"
markers:
  - {name: "CD3", expression: "high"}
  - {name: "CD4", expression: "high"}
  - {name: "CD8", expression: "high"}
  - {name: "CD25", expression: "medium"}  # High on Tregs
  - {name: "CD127", expression: "low"}     # Low on Tregs
  - {name: "FOXP3", expression: "medium", intracellular: true}
  - {name: "Helios", expression: "medium", intracellular: true}
  - {name: "CTLA-4", expression: "low", intracellular: true}
  - {name: "CD45RA", expression: "high"}
  - {name: "CCR7", expression: "medium"}
  - {name: "CD39", expression: "low"}
  - {name: "CD73", expression: "low"}
  - {name: "TIGIT", expression: "low"}
  - {name: "LAG-3", expression: "low"}
  - {name: "Viability", expression: "high"}

expected_difficulty: "hard"
notes: "Many dim markers requiring bright fluorophores"
```

### Test Case 5: Full PBMC Panel (20 markers)
```yaml
name: "full_pbmc_panel"
description: "Comprehensive PBMC immunophenotyping"
instrument: "5-laser (UVBYR)"
markers:
  # Lineage
  - {name: "CD3", expression: "high"}
  - {name: "CD4", expression: "high"}
  - {name: "CD8", expression: "high"}
  - {name: "CD14", expression: "high"}
  - {name: "CD16", expression: "medium"}
  - {name: "CD19", expression: "high"}
  - {name: "CD56", expression: "medium"}
  # T-cell memory
  - {name: "CD45RA", expression: "high"}
  - {name: "CCR7", expression: "medium"}
  - {name: "CD27", expression: "medium"}
  - {name: "CD28", expression: "medium"}
  # Activation
  - {name: "CD38", expression: "variable"}
  - {name: "HLA-DR", expression: "variable"}
  - {name: "CD69", expression: "low"}
  # Exhaustion
  - {name: "PD-1", expression: "low"}
  - {name: "TIM-3", expression: "low"}
  - {name: "LAG-3", expression: "low"}
  # Proliferation
  - {name: "Ki-67", expression: "low", intracellular: true}
  # Other
  - {name: "CD127", expression: "low"}
  - {name: "Viability", expression: "high"}

expected_difficulty: "very_hard"
notes: "Large panel with many dim markers - maximum challenge"
```

---

## Prompt Templates

### Control Prompt (No MCP)
```
You are an expert flow cytometry panel designer. Design a flow cytometry
panel for the following markers on a {instrument} cytometer.

Markers to include:
{marker_list}

For each marker, assign an appropriate fluorophore. Consider:
1. Spectral overlap between fluorophores
2. Matching fluorophore brightness to antigen expression level
3. Available laser lines and detector configuration

Provide your panel as a table with columns:
| Marker | Fluorophore | Rationale |

After the table, estimate the overall panel quality (excellent/good/fair/poor)
and list any potentially problematic fluorophore pairs.
```

### Treatment Prompt (With MCP)
```
You are an expert flow cytometry panel designer with access to spectral
analysis tools. Design a flow cytometry panel for the following markers
on a {instrument} cytometer.

Markers to include:
{marker_list}

You have access to these tools:
- analyze_panel: Check similarity matrix and complexity index
- check_compatibility: Verify if a fluorophore works with existing panel
- suggest_fluorophores: Get ranked suggestions for a marker

Use the tools iteratively to build an optimized panel. For each marker:
1. Use suggest_fluorophores to get candidates
2. Use check_compatibility to verify the best option
3. After adding all markers, use analyze_panel to verify overall quality

Provide your final panel as a table with the analysis results.
```

---

## Evaluation Pipeline

```python
class MCPEffectivenessTest:
    """Test framework for MCP panel designer effectiveness."""

    def __init__(self):
        self.test_cases = load_test_cases()
        self.results = []

    async def run_test(self, test_case: dict, n_trials: int = 3):
        """Run A/B test for a single test case."""

        control_results = []
        treatment_results = []

        for trial in range(n_trials):
            # Run control (no MCP)
            control_panel = await self.run_control(test_case)
            control_metrics = self.evaluate_panel(control_panel)
            control_results.append(control_metrics)

            # Run treatment (with MCP)
            treatment_panel = await self.run_treatment(test_case)
            treatment_metrics = self.evaluate_panel(treatment_panel)
            treatment_results.append(treatment_metrics)

        return {
            "test_case": test_case["name"],
            "control": aggregate_results(control_results),
            "treatment": aggregate_results(treatment_results),
            "improvement": calculate_improvement(control_results, treatment_results)
        }

    def evaluate_panel(self, panel: dict) -> dict:
        """Calculate metrics for a panel."""
        fluorophores = [m["fluorophore"] for m in panel["assignments"]]

        # Get spectra and calculate matrices
        spectra = get_spectra(fluorophores)
        names, sim_matrix = build_similarity_matrix(spectra)

        return {
            "complexity_index": complexity_index(sim_matrix),
            "max_similarity": np.max(sim_matrix[np.triu_indices(len(names), k=1)]),
            "critical_pairs": count_pairs_above(sim_matrix, names, 0.95),
            "high_risk_pairs": count_pairs_above(sim_matrix, names, 0.90),
            "valid_panel": all(f in KNOWN_FLUOROPHORES for f in fluorophores),
        }

    async def run_control(self, test_case: dict) -> dict:
        """Run Claude without MCP tools."""
        prompt = format_control_prompt(test_case)
        response = await claude_api.complete(prompt, tools=[])
        return parse_panel_response(response)

    async def run_treatment(self, test_case: dict) -> dict:
        """Run Claude with MCP tools."""
        prompt = format_treatment_prompt(test_case)
        response = await claude_api.complete(prompt, tools=MCP_TOOLS)
        return parse_panel_response(response)
```

---

## Statistical Analysis

### Primary Metrics
```python
def analyze_results(results: list[dict]):
    """Perform statistical analysis on A/B test results."""

    for metric in ["complexity_index", "max_similarity", "critical_pairs"]:
        control_values = [r["control"][metric] for r in results]
        treatment_values = [r["treatment"][metric] for r in results]

        # Paired t-test (same test cases)
        t_stat, p_value = scipy.stats.ttest_rel(control_values, treatment_values)

        # Effect size (Cohen's d)
        effect_size = (np.mean(control_values) - np.mean(treatment_values)) / np.std(control_values)

        print(f"{metric}:")
        print(f"  Control mean: {np.mean(control_values):.2f}")
        print(f"  Treatment mean: {np.mean(treatment_values):.2f}")
        print(f"  Improvement: {(1 - np.mean(treatment_values)/np.mean(control_values))*100:.1f}%")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Effect size (d): {effect_size:.2f}")
```

### Success Criteria
| Metric | Success Threshold |
|--------|------------------|
| Complexity Index | ≥15% reduction |
| Critical Pairs | 100% elimination |
| High-Risk Pairs | ≥50% reduction |
| p-value | < 0.05 |

---

## Expected Output

### Per Test Case
```
Test Case: memory_tcell_panel (12 markers)
============================================
                    Control    Treatment    Improvement
Complexity Index       45.2        28.7         36.5%
Max Similarity         0.94        0.87          7.4%
Critical Pairs            2           0        -100.0%
High-Risk Pairs           5           2         60.0%

Top improvements from MCP:
- Avoided PE-Cy5/PE-Cy5.5 conflict (SI=0.96) by using PE-Cy5 + APC-R700
- Used BV421 instead of Pacific Blue for dim CD127 (better brightness)
- Detected and corrected FITC/BB515 near-conflict
```

### Overall Summary
```
MCP Effectiveness Test Results (N=5 test cases, 3 trials each)
================================================================
Overall Improvements:
- Complexity Index: 32.4% reduction (p=0.003)
- Critical Pairs: 89% eliminated (p=0.001)
- High-Risk Pairs: 54% reduction (p=0.012)

Effect sizes:
- Complexity Index: d=1.8 (large effect)
- Max Similarity: d=0.9 (large effect)

Conclusion: MCP tools significantly improve panel design quality
```

---

## Implementation Files

```
tests/mcp_effectiveness/
├── TEST_PLAN.md              # This document
├── test_cases/
│   ├── basic_tcell.yaml
│   ├── memory_tcell.yaml
│   ├── omip030_recreation.yaml
│   ├── treg_panel.yaml
│   └── full_pbmc.yaml
├── prompts/
│   ├── control_prompt.txt
│   └── treatment_prompt.txt
├── conftest.py               # Pytest fixtures
├── test_mcp_effectiveness.py # Main test runner
├── evaluation.py             # Metrics calculation
├── analysis.py               # Statistical analysis
└── results/                  # Output directory
    ├── raw_results.json
    └── summary_report.md
```

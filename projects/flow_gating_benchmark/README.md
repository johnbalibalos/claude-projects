# Flow Gating Benchmark

A small-scale benchmark exploring whether LLMs can predict flow cytometry gating strategies from panel information.

> **Status:** Exploratory research project. 10 verified test cases. Results are preliminary and methodology is still being refined.

## What This Project Does

Given a list of flow cytometry markers (CD3, CD4, CD8, etc.), can an LLM predict the hierarchical gating strategy a cytometrist would use?

This is a narrow test of domain knowledge transfer - not a comprehensive evaluation of LLM capabilities in biology.

---

## Limitations (Read First)

**This is not a rigorous benchmark.** Key caveats:

1. **Small dataset**: Only 10 verified test cases from OMIP papers
2. **F1 metric is flawed**: Penalizes correct predictions with different naming (e.g., "CD4+ T cells" vs "Helper T cells")
3. **Ground truth variability**: Multiple valid gating strategies exist for any panel
4. **Frequency confound**: R² = 0.36 between gate name frequency (PubMed) and prediction accuracy
5. **No functional validation**: We only compare names and structure, not biological correctness

---

## Key Observations

### Model Performance (Preliminary)

```
Hierarchy F1 (↑ better)            Note: F1 is a weak metric
───────────────────────────────────────────────────────────
gemini-2.5-pro     0.36            Best overall
gemini-2.0-flash   0.34            Best cost/performance
claude-opus-4      0.33
claude-sonnet-4    0.33
```

**Takeaway:** All models score similarly (~0.33-0.36 F1). The differences are likely within noise given the small dataset and metric limitations.

### What Models Get Right
- QC gates (Singlets, Live cells) - near 100% recall
- Major lineages (T cells, B cells, NK cells)
- Standard marker logic (CD3+ = T cells)

### What Models Struggle With
- Lab-specific naming conventions
- Complex hierarchies (>20 gates)
- Rare cell populations

---

## Alien Cell Ablation Tests

We developed three versions of an "alien cell" test to probe model reasoning. The evolution from V1→V3 demonstrates iterative methodology refinement.

| Version | Approach | Validity |
|---------|----------|----------|
| V1 | Change ground truth labels silently | ❌ Invalid |
| V2 | Provide naming convention in prompt | ✓ Valid (trivial) |
| V3 | Test biological reasoning with novel markers | ✓ Valid (meaningful) |

See [knowledge base](../../knowledge-base/30_Resources/flowbench-learnings.md) for detailed methodology discussion.

### V3: Biological Reasoning Test

Ask models to incorporate novel marker combinations and observe whether they flag biological implausibility.

| Test Case | Marker Logic | Status | Expected Behavior |
|-----------|--------------|--------|-------------------|
| Zorphax Cells | CD14+ CD8+ | Implausible | Flag lineage conflict |
| Thymic Remnant | CD4+ CD8+ | Rare | Mention thymic development |
| True MAIT | TRAV1-2+ CD161+ CD3+ | Valid | Accept |
| Triple Class Switch | IgA+ IgE+ IgG+ | Implausible | Flag as impossible |

**Sample Results (Sonnet):**

- **CD14+CD8+**: "These lineages are mutually exclusive... would indicate technical artifact or pathological conditions" ✓
- **CD4+CD8+**: "Double-positive T cells exist in thymic development... rare in peripheral blood" ✓
- **MAIT cells**: "Biologically plausible... canonical MAIT definition" ✓

**Takeaway:** Claude models demonstrate genuine immunological knowledge. They correctly flag impossible lineage combinations and provide nuanced context for rare phenotypes.

### V3 Test Distribution

31 test cases across 10 OMIP panels:
- 14 implausible (lineage conflicts)
- 6 rare but contextually valid
- 5 documented rare subsets
- 6 textbook valid phenotypes

```bash
# List all V3 tests
python scripts/quick_test.py --list-v3

# Run specific test
python scripts/quick_test.py --alien-v3 zorphax-cells --cli --model sonnet
```

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Quick test (~$0.01)
python scripts/quick_test.py --cli --model haiku

# Alien cell V3 test
python scripts/quick_test.py --alien-v3 thymic-remnant --cli --model sonnet

# Full benchmark (not recommended - small dataset, high cost)
python scripts/run_modular_pipeline.py --phase all --models gemini-2.0-flash
```

---

## Project Structure

```
flow_gating_benchmark/
├── src/
│   ├── curation/           # Test case schemas
│   ├── evaluation/         # Scoring (F1, structure)
│   └── experiments/        # LLM clients, prompts
├── data/
│   ├── verified/           # 10 curated OMIP test cases
│   ├── alien_cell/         # V1 tests (deprecated)
│   ├── alien_cell_v2/      # V2 instruction-following tests
│   └── alien_cell_v3/      # V3 biological reasoning tests (31 cases)
├── scripts/
│   ├── quick_test.py       # Single test runner
│   ├── generate_alien_cell_v2.py
│   └── generate_alien_cell_v3.py
└── tests/                  # 291 passing, 8 failing
```

---

## Environment

```bash
GOOGLE_API_KEY=...     # For Gemini models
# Claude CLI uses Max subscription
```

---

## License

MIT

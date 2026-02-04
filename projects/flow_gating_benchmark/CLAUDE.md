# Flow Gating Benchmark

Exploratory project testing LLM prediction of flow cytometry gating strategies.

## Honest Assessment

**This is a learning project, not a rigorous benchmark.**

- 10 test cases (too small for statistical significance)
- F1 metric is fundamentally flawed for this task
- Frequency confound explains ~36% of variance
- No peer review or external validation

## What's Actually Useful

1. **Alien Cell V3 tests** - Probing biological reasoning with novel marker combinations
2. **Methodology lessons** - How NOT to design ablation tests (see V1 failure)
3. **Quick iteration** - Easy to test new ideas with `quick_test.py`

## Quick Commands

```bash
# Single test
python scripts/quick_test.py --cli --model sonnet

# Alien cell reasoning test
python scripts/quick_test.py --alien-v3 zorphax-cells --cli --model sonnet
python scripts/quick_test.py --list-v3  # See all 31 test cases

# Run tests
pytest tests/ -v  # 291 passing, 8 failing
```

## Alien Cell Test Versions

| Version | Description | Status |
|---------|-------------|--------|
| V1 | Change ground truth labels to nonsense | ❌ Invalid (model can't know) |
| V2 | Tell model which names to use | ✓ Valid but trivial |
| V3 | Ask model to insert novel marker combos | ✓ Tests biological reasoning |

**V3 Categories:**
- 14 implausible (lineage conflicts like CD3+CD19+)
- 6 rare_contextual (CD4+CD8+ double positive)
- 5 rare_valid (CD8+ Tregs, Tc17)
- 6 valid (MAIT cells, TSCM)

## Project Structure

```
data/
├── verified/         # 10 OMIP test cases
├── alien_cell/       # V1 (deprecated)
├── alien_cell_v2/    # 9 instruction-following tests
└── alien_cell_v3/    # 31 biological reasoning tests

scripts/
├── quick_test.py     # Main testing tool
├── generate_alien_cell_v2.py
├── generate_alien_cell_v3.py
└── run_modular_pipeline.py
```

## Models

| Model | CLI Flag | Notes |
|-------|----------|-------|
| Haiku | `--model haiku` | Fast, cheap |
| Sonnet | `--model sonnet` | Good balance |
| Opus | `--model opus` | Slower |

## Known Issues

- 8 failing tests (pre-existing, mostly import/mock issues)
- F1 metric penalizes valid alternative naming
- Small dataset limits conclusions

## Environment

```bash
GOOGLE_API_KEY=...  # For Gemini
# Claude CLI uses Max subscription
```

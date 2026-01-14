# Flow Gating Benchmark

Evaluate LLM capabilities in predicting flow cytometry gating strategies from panel information.

> **Status:** Active development. The evaluation framework is functional, but ground truth datasets require manual curation from OMIP papers. Results in any linked posts reflect work-in-progress and may be rerun as data quality improves.

## Latest Results (Full Benchmark - Jan 2026)

See **[results/BENCHMARK_RESULTS_SUMMARY.md](results/BENCHMARK_RESULTS_SUMMARY.md)** for full analysis.

**Benchmark Configuration:**
- 6 models × 12 conditions × 10 test cases × 3 bootstrap = 2,160 predictions
- Judge: gemini-2.5-pro with 5 evaluation styles

### F1 Scores (Hierarchy Matching)

| Model | F1 Score | Parse Rate |
|-------|----------|------------|
| **gemini-2.5-pro** | **0.361** | 100% |
| gemini-2.0-flash | 0.340 | 100% |
| claude-opus-4-20250514 | 0.330 | 100% |
| claude-sonnet-4-20250514 | 0.326 | 100% |
| claude-3-5-haiku-20241022 | 0.306 | 100% |
| gemini-2.5-flash | 0.305 | 100% |

### LLM Judge Quality Scores

| Model | Quality | Consistency |
|-------|---------|-------------|
| **gemini-2.5-pro** | **0.59** | 0.40 |
| claude-opus-4-20250514 | 0.52 | 0.18 |
| gemini-2.5-flash | 0.51 | 0.63 |
| gemini-2.0-flash | 0.41 | **0.65** |
| claude-sonnet-4-20250514 | 0.39 | 0.50 |
| claude-3-5-haiku-20241022 | 0.34 | 0.13 |

### Key Findings

**1. Synthetic vs Real-World Panels:**
Models perform significantly better on template-generated CUSTOM-PBMC-001 (+0.19 F1) compared to OMIP papers, likely due to cleaner structure rather than reasoning vs memorization.

| Model | OMIP F1 | CUSTOM F1 | Delta |
|-------|---------|-----------|-------|
| claude-sonnet-4-20250514 | 0.301 | 0.551 | **+0.25** |
| gemini-2.5-flash | 0.283 | 0.504 | +0.22 |
| claude-opus-4-20250514 | 0.310 | 0.506 | +0.20 |
| gemini-2.5-pro | 0.343 | 0.527 | +0.18 |
| claude-3-5-haiku-20241022 | 0.289 | 0.460 | +0.17 |
| gemini-2.0-flash | 0.325 | 0.469 | +0.14 |

**2. F1 vs Judge Disagreement:**
F1 ranking differs from LLM judge ranking. gemini-2.5-pro leads both, but opus ranks higher by judge quality than F1 suggests.

**3. Model Consistency (Bootstrap Agreement):**

Claude models produce different outputs each run even at temperature=0:

| Model | All Same (3/3) | All Different (3/3) |
|-------|----------------|---------------------|
| gemini-2.0-flash | 28% | **0%** |
| gemini-2.5-flash | 29% | **0%** |
| claude-sonnet-4-20250514 | 35% | 43% |
| gemini-2.5-pro | 4% | 52% |
| claude-3-5-haiku-20241022 | 1% | **92%** |
| claude-opus-4-20250514 | 1% | **96%** |

**Example:** opus produces 3 different hierarchies for CUSTOM-PBMC-001 (same prompt):
```
Bootstrap 1: All Events → Time Gate → Singlets → Live → CD45+ → T Cells → Tregs...
Bootstrap 2: All Events → Singlets → Live → CD45+ → CD3+ T Cells → NKT-like...
Bootstrap 3: All Events → Singlets → Live → Leukocytes → T Cells → Regulatory T...
```
All biologically valid, but different structure and naming.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Dry run (mock API calls)
python scripts/run_modular_pipeline.py --phase all --dry-run

# Real run with Gemini (recommended)
python scripts/run_modular_pipeline.py \
    --phase all \
    --models gemini-2.0-flash \
    --test-cases data/staging \
    --n-bootstrap 10
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MODULAR PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Prediction  │───▶│    Batch     │───▶│  LLM Judge   │          │
│  │  Collector   │    │   Scorer     │    │  (optional)  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│        │                    │                   │                   │
│        ▼                    ▼                   ▼                   │
│  predictions.json    scoring_results.json  judge_results.json      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Recent Additions

- **Multi-Judge Cross-Validation**: Multiple prompt styles for LLM judge reliability
- **Token Breakdown Analysis**: Track thinking vs response tokens for reasoning models
- **Blocked Prediction Recovery**: `scripts/rerun_blocked.py` for MAX_TOKENS failures
- **Per-Provider Rate Limits**: Parallel workers configurable per API provider

---

## CLI Reference

```bash
python scripts/run_modular_pipeline.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--phase` | `all` | `predict`, `score`, `judge`, or `all` |
| `--models` | `claude-sonnet-cli` | Models to test (space-separated) |
| `--test-cases` | `data/verified` | Test case JSON directory |
| `--n-bootstrap` | `1` | Runs per condition (use 10 for variance) |
| `--max-cases` | None | Limit test cases (for testing) |
| `--output` | `results/modular_pipeline` | Output directory |
| `--judge-model` | `gemini-2.5-pro` | Model for qualitative judge |
| `--dry-run` | False | Mock API calls |
| `--resume` | False | Resume from checkpoint |
| `--force` | False | Skip cost confirmation |

### Available Models

| Model | Provider | Cost/Case | Notes |
|-------|----------|-----------|-------|
| `gemini-2.0-flash` | Google | ~$0.01 | **Recommended** - best F1 |
| `gemini-2.5-flash` | Google | ~$0.02 | High failure rate |
| `gemini-2.5-pro` | Google | ~$0.10 | Needs max_tokens=20000 |
| `claude-sonnet-cli` | Anthropic | ~$0.05 | Uses Max subscription |
| `claude-opus-cli` | Anthropic | ~$0.50 | Uses Max subscription |
| `gpt-4o` | OpenAI | ~$0.05 | API |

---

## Experimental Conditions

Each model is tested with **12 conditions** (3 context levels × 2 strategies × 2 reference modes):

### Context Levels

**Minimal** - Markers only:
```
## Panel
Markers: CD3, CD4, CD8, CD45, CD19, Viability
```

**Standard** - Adds experimental metadata:
```
## Experiment Information
Sample Type: Human PBMC
Species: human
Application: T-cell subset analysis

## Panel
- CD3: BUV395 (clone: UCHT1)
- CD4: BV785 (clone: RPA-T4)
- CD8: PE-Cy7 (clone: SK1)
...
```

**Rich** - Adds panel size, complexity, notes:
```
## Experiment Information
...
## Panel
...
## Additional Information
Panel Size: 18 colors
Complexity: hard
Tissue: Peripheral blood
```

*Note: OMIP ID intentionally excluded from rich context to prevent retrieval from training data.*

### Prompt Strategies

**Direct** - Output JSON hierarchy immediately:
```
You are an expert cytometrist. Given the following panel information,
predict the gating hierarchy that an expert would use for data analysis.

{context}

## Task
Predict the complete gating hierarchy, starting from "All Events" through
appropriate quality control gates to final cell population identification.

Return your answer as a JSON object...
```

**Chain-of-Thought (CoT)** - Reasoning before JSON output:
```
...
## Task
Before providing your final answer, briefly consider:
- What technology is this (flow cytometry or mass cytometry/CyTOF)?
- What quality control gates are needed?
- What populations can this panel's markers identify?
- How should gates be organized hierarchically?

Use your expertise to determine the best approach for this specific panel.

After your reasoning, provide the final hierarchy as a JSON object...
```

### Reference Modes

**None** - No additional context

**HIPC** - Injects HIPC 2016 standardized cell definitions (+5.6% F1):
```
## Reference: HIPC 2016 Standardized Cell Definitions
Source: https://www.nature.com/articles/srep20686

### Quality Control Gates (Required)
- Time Gate: Exclude acquisition artifacts
- Singlets: Doublet exclusion (FSC-A vs FSC-H)
- Live cells: Viability dye negative

### Major Lineage Definitions
| Population | Markers | Parent |
|------------|---------|--------|
| T cells | CD3+ CD19- | Lymphocytes |
| CD4+ T cells | CD3+ CD4+ CD8- | T cells |
...

{actual panel context follows}
```

---

## Known Failure Modes

### 1. Token Exhaustion (gemini-2.5-*)
Reasoning models use 70%+ of token budget for "thinking", leaving insufficient output tokens.
- **Symptom**: Truncated JSON, empty responses
- **Fix**: Use `max_tokens=20000` or `scripts/rerun_blocked.py`

### 2. Format Confusion (gemini-2.5-flash)
Model outputs prose descriptions instead of JSON hierarchy.
- **Symptom**: F1 = 0, valid text but unparseable
- **Fix**: None effective - use different model

### 3. Synonym Mismatches (all models)
Ground truth says "CD3+ T cells", model predicts "T lymphocytes".
- **Symptom**: Low F1 despite correct biology
- **Mitigation**: 200+ synonyms in `normalization.py`

### 4. Hallucinated Markers
Model creates gates using markers not in the panel.
- **Symptom**: High hallucination_rate metric
- **Cause**: Training data contamination or confusion

---

## Project Structure

```
flow_gating_benchmark/
├── src/
│   ├── curation/              # Test case schemas
│   │   ├── schemas.py         # TestCase, Panel, GatingHierarchy
│   │   └── omip_extractor.py
│   ├── evaluation/            # Scoring
│   │   ├── metrics.py         # F1, structure, hallucination
│   │   ├── normalization.py   # 200+ gate synonyms
│   │   ├── hierarchy.py       # Tree operations
│   │   ├── task_failure.py    # Refusal detection
│   │   ├── response_parser.py
│   │   └── scorer.py
│   ├── experiments/           # Pipeline
│   │   ├── prediction_collector.py
│   │   ├── batch_scorer.py
│   │   ├── llm_judge.py       # Multi-judge support
│   │   ├── llm_client.py      # Gemini, Claude, OpenAI
│   │   ├── conditions.py
│   │   └── prompts.py
│   └── analysis/              # Hypothesis testing
│       ├── alien_cell.py      # Frequency confound tests
│       └── cognitive_refusal.py
├── data/
│   ├── staging/               # 13 OMIPs (test cases)
│   └── verified/              # Verified test cases
├── scripts/
│   ├── run_modular_pipeline.py
│   └── rerun_blocked.py       # Recover MAX_TOKENS failures
├── results/
│   ├── BENCHMARK_RESULTS_SUMMARY.md  # Latest results
│   └── gemini_benchmark_predictions.json
└── tests/
```

---

## Metrics

### F1 Variants (4 metrics)

We compute multiple F1 scores to understand the gap between string matching and biological equivalence:

| Metric | Method | What it catches |
|--------|--------|-----------------|
| `hierarchy_f1` | String normalization | "CD4+ T Cells" ↔ "CD4 positive T cells" |
| `synonym_f1` | 200+ synonym dictionary | "T Lymphocytes" ↔ "T cells" ↔ "CD3+" |
| `semantic_f1` | MiniLM embeddings | "Helper T cells" ↔ "CD4+ T cells" |
| `weighted_semantic_f1` | Confidence-weighted | Partial credit for similarity 0.7-0.85 |

**Why multiple F1s?** Early analysis showed weak correlation (r≈0.15) between string-based F1 and LLM judge scores. Models often produce biologically correct but linguistically different gate names.

### Other Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `structure_accuracy` | Parent-child relationships correct | 0-1 |
| `critical_gate_recall` | Essential QC gates present | 0-1 |
| `hallucination_rate` | Gates using non-panel markers | 0-1 |

---

## Test Cases (10 verified)

| Test Case | Species | Focus | Markers | Notes |
|-----------|---------|-------|---------|-------|
| CUSTOM-PBMC-001 | Human | Pan-immune | 18 | Synthetic (template-generated) |
| OMIP-008 | Human | Th1/Th2 | 7 | T cell cytokines |
| OMIP-022 | Human | T memory | 12 | Antigen-specific T cells |
| OMIP-074 | Human | B cells | 22 | IgG/IgA subclasses |
| OMIP-076 | Murine | T/B/APC | 18 | Pan-leukocyte |
| OMIP-077 | Human | Pan-leuk | 20 | All principal leukocytes |
| OMIP-083 | Human | Monocytes | 21 | Deep phenotyping |
| OMIP-087 | Human | CyTOF | 32 | Mass cytometry |
| OMIP-095 | Murine | Spectral | 40 | 40-color spectral |
| OMIP-101 | Human | Whole blood | 32 | Myeloid + lymphoid |

---

## Environment

```bash
# Required
GOOGLE_API_KEY=...     # Gemini models + judge

# Optional (CLI models use Max subscription)
ANTHROPIC_API_KEY=...  # For API models
OPENAI_API_KEY=...     # GPT models
```

---

## Development

```bash
# Run tests
pytest tests/ -v

# Quick API test (~$0.01)
python scripts/run_modular_pipeline.py \
    --phase all \
    --models gemini-2.0-flash \
    --max-cases 1
```

---

## License

MIT

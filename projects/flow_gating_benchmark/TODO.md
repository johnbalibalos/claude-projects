# Flow Gating Benchmark - TODO

## Recently Completed (Jan 2026)

### Modular Pipeline Architecture
- [x] `PredictionCollector` - parallel API, rate-limited CLI collection
- [x] `BatchScorer` - score predictions against ground truth
- [x] `LLMJudge` - qualitative assessment (configurable model)
- [x] `--judge-model` flag for flexible judge selection
- [x] Resume/retry for errored predictions on `--resume`
- [x] CLI delay reduced to 0.5s with exponential backoff

### LLM Clients
- [x] `AnthropicClient` (API)
- [x] `ClaudeCLIClient` (Max subscription, OAuth)
- [x] `GeminiClient` (Google API)
- [x] `OpenAIClient`
- [x] `OllamaClient` (local models)
- [x] `MockClient` (dry-run)

### Test Cases
- [x] 13 OMIPs in staging (up from 8)
- [x] Mass cytometry support (OMIP-087)
- [x] Staging → ground truth workflow

### Evaluation
- [x] 200+ gate synonym mappings
- [x] Task failure detection (refusals, meta-questions)
- [x] Critical gate recall (singlets, live)

---

## High Priority

### Multi-Judge Experiment
Design and implement cross-validation with multiple judge models.

**Goal:** Measure inter-judge agreement, detect self-serving bias, find cheapest reliable judge.

- [ ] Design judge personas/prompts (see `docs/JUDGE_EXPERIMENT_DESIGN.md`)
  - Option A: Same prompt, different models
  - Option B: Different prompt styles (validation vs qualitative vs orthogonal)
  - Option C: Hybrid approach
- [ ] Implement `MultiJudge` class for parallel judging
- [ ] Stratified sample: 520 predictions × 3 judges = 1560 calls (same cost)
- [ ] Increase `max_tokens` to 20000 for prediction to reduce token exhaustion
- [ ] Analysis: inter-rater reliability (Fleiss' kappa), self-serving bias, calibration vs auto metrics

**Key question:** Should judges rate same metrics as auto-scoring (F1, structure) or orthogonal dimensions (clinical utility, biological plausibility)?

### Verify Staged OMIPs
- [ ] Manual verification of 13 OMIPs in `data/staging/`
- [ ] Move verified cases to `data/ground_truth/`
- [ ] Validate against original OMIP PDFs

### Statistical Rigor
- [ ] Confidence intervals on aggregate metrics
- [ ] Bootstrap significance testing
- [ ] Multi-run variance estimation (n_bootstrap=10+)

### Expand Model Coverage
- [ ] Run full benchmark on claude-opus-cli
- [ ] Test gpt-4o
- [ ] Local models: llama3.1, qwen2.5, deepseek-r1

---

## Medium Priority

### Scale Test Set
- [ ] Expand to 20+ verified OMIPs
- [ ] Add tissue-specific panels (bone marrow, lymph node)
- [ ] Include rare populations (ILCs, MAITs)

### Integration Tests
- [ ] End-to-end pipeline tests
- [ ] Parser edge cases
- [ ] Cross-validation against .wsp files

---

## Low Priority

### Infrastructure
- [ ] Experiment tracking (MLflow/W&B)
- [ ] Docker for reproducibility
- [ ] CI/CD

### Documentation
- [ ] Tutorial notebook
- [ ] Methods section for publication

---

## Future Work

### Multi-turn Evaluation
- See `design_docs/multi_turn_evaluation_protocol.md` (NOT IMPLEMENTED)

### Confidence Elicitation
- See `design_docs/confidence_elicitation_design.md` (NOT IMPLEMENTED)

### Selective RAG for Rare Cell Types (NOT IMPLEMENTED)

**Context from variance analysis (Jan 2026):**

Analyzed gemini-2.5-pro variance at temperature=0 across 13 OMIPs with 5 bootstrap runs:

| Finding | Detail |
|---------|--------|
| Overall variance | gemini-2.5-flash most deterministic (60%), gemini-2.5-pro least (25%) |
| Primary driver | **Failure rate**, not model stochasticity or training data |
| Failure modes | MAX_TOKENS exhaustion (reasoning too long), no JSON output |
| Valid-only F1 | When 2.5-pro succeeds, it performs well (OMIP-077: 0.45 F1 vs 0.09 overall) |

**Key insight:** Apparent "training data effect" is actually failure rate. Complex panels fail 60-100% of the time, dragging mean F1 to zero. Simpler panels succeed consistently.

**Correlation with marker frequency:**
- Gates with >50k PubMed hits (CD4, CD8, T cells): F1 ≈ 0.54
- Gates with <1k PubMed hits (SLAN, cDC1, HSPC): F1 ≈ 0.10
- Cached frequencies in `data/cache/pubmed_frequencies.json` (100 markers)

**Proposed experiment - Selective RAG:**

1. **Identify rare gates** using PubMed frequency threshold (<1000 hits)
2. **Fetch targeted context** only for rare markers (SLAN, cDC subsets, etc.)
3. **Inject into prompt** as reference knowledge
4. **Measure F1 improvement** on rare-gate panels (OMIP-077, OMIP-083)

**Existing infrastructure:**
- `rag_mode` parameter in conditions.py: "none" | "oracle"
- `HIPC_RAG_CONTEXT` in prompts.py (standardized definitions)
- `src/rag/pmc_client.py` for PubMed Central paper retrieval
- PubMed frequency cache already built

**Implementation sketch:**
```python
# In experiments/prompts.py
def build_prompt_with_selective_rag(test_case, freq_threshold=1000):
    rare_gates = [g for g in test_case.gates if pubmed_freq[g] < freq_threshold]
    if rare_gates:
        context = fetch_marker_context(rare_gates)  # PubMed/OMIP papers
        return inject_rag_context(base_prompt, context)
    return base_prompt
```

**Hypothesis:** Selective RAG should improve F1 by 0.1-0.2 on rare-gate panels with minimal token overhead (~500 tokens). The model CAN reason about these cell types when it succeeds; it just lacks specialized knowledge.

---

*Last updated: 2026-01-12*

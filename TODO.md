# LLM Biology Research - TODO

## Recently Completed (Jan 2026)

### Flow Gating Benchmark
- [x] Task failure detection module (`src/evaluation/task_failure.py`)
  - Detects meta-questions, refusals, instructional responses
  - Integrated into scorer and aggregate metrics
- [x] Manual review report generator (`src/analysis/manual_review_report.py`)
  - Side-by-side comparisons for outlier analysis
  - Configurable outlier thresholds (F1, hallucination, critical recall)
- [x] Refactored evaluation modules
  - Split metrics.py into focused modules (normalization, hierarchy, task_failure)
  - Clean LLM client abstraction (Anthropic, OpenAI, Ollama)
  - Improved response parser with meta-commentary detection
- [x] 8 OMIP test cases with ground truth hierarchies

---

## High Priority

### Flow Gating Benchmark: Ground Truth Fluorophore Condition
- [ ] Create experimental condition that provides exact fluorophore assignments from ground truth
- [ ] Test whether this isolates reasoning failures from information gaps
- [ ] Compare F1 scores: inferred vs. provided fluorophores

### Expand Model Coverage
- [ ] Add Ollama integration for local model inference
- [ ] Test models:
  - [ ] Llama 3.2 (8B, 70B)
  - [ ] DeepSeek-V3
  - [ ] Qwen 2.5
- [ ] Run comparative experiments across all models

### Statistical Rigor
- [ ] Add confidence intervals to all metrics
- [ ] Implement bootstrap significance testing
- [ ] Multi-run experiments for variance estimation

---

## Medium Priority

### Flow Gating Benchmark: Scale Up
- [ ] Expand to 20+ OMIP test cases
- [ ] Add tissue-specific panels (bone marrow, lymph node)
- [ ] Include rare population panels (ILCs, MAITs)

### MCP Server for Gating Extraction
- [ ] Wrap extraction library as MCP server for interactive curation
- [ ] Tools: get_paper_content, parse_marker_table, build_hierarchy, validate_hierarchy
- [ ] Enable Claude-assisted curation sessions

### Testing
- [ ] Add integration tests for experiment runner
- [ ] Add tests for task failure detection edge cases
- [ ] Property-based testing with Hypothesis

### DrugDevBench: Real Figures
- [ ] Curate dataset of ~100 open-access drug development figures
- [ ] Figure types: Western blots, dose-response, PK/PD, ELISA, flow plots
- [ ] Create ground truth annotations

---

## Low Priority

### Documentation
- [ ] API documentation for all libraries
- [ ] Tutorial notebooks for each project
- [ ] Methods section drafts for publication

### Infrastructure
- [ ] Experiment tracking (MLflow or W&B)
- [ ] Reproducibility scripts (Docker, requirements pinning)
- [ ] CI/CD pipeline for automated testing

### Flow Panel Optimizer
- [ ] Validate spreading matrix predictions against real data
- [ ] Add more spectral databases beyond BD

---

## Ideas for Future Work

### Multi-turn Evaluation
- Test iterative refinement: LLM predicts, gets feedback, revises
- Compare single-turn vs. multi-turn F1 scores

### Confidence Elicitation
- Ask models to rate confidence in predictions
- Correlate confidence with actual F1 scores

### Vision Integration
- Extract gating figures from OMIP PDFs
- Use vision LLMs to read gate names from plots

### Cross-validation
- Compare paper-extracted hierarchies with FlowRepository .wsp files
- Assess ground truth reliability

---

*Last updated: 2026-01-10*

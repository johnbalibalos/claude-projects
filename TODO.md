# LLM Biology Research - TODO

## High Priority

### Add Significant Testing
- [ ] Flow Panel Optimizer: Add integration tests for MCP server tools
- [ ] Flow Panel Optimizer: Add edge case tests for spectral calculations (near-zero vectors, identical spectra)
- [ ] Flow Gating Benchmark: Add tests for parser edge cases (malformed JSON, missing fields)
- [ ] Flow Gating Benchmark: Add cross-validation tests against FlowJo exports
- [ ] DrugDevBench: Add tests for each figure type evaluator
- [ ] All projects: Add property-based testing with Hypothesis

### Download Real Figures for Figure Analysis Pipeline (DrugDevBench)
- [ ] Curate dataset of ~100 open-access drug development figures from:
  - [ ] PubMed Central (PMC) open access subset
  - [ ] bioRxiv preprints
  - [ ] FDA drug approval documents (public)
- [ ] Figure types to include:
  - [ ] Western blots (20+)
  - [ ] Dose-response curves (20+)
  - [ ] PK/PD curves (20+)
  - [ ] ELISA results (15+)
  - [ ] Flow cytometry plots (15+)
  - [ ] Histology/IHC images (10+)
- [ ] Create ground truth annotations for each figure
- [ ] Store in `projects/drugdevbench/data/figures/`

### Add Llama/DeepSeek Local Model Support
- [ ] Add Ollama integration for local model inference
- [ ] Supported models:
  - [ ] Llama 3.2 (8B, 70B)
  - [ ] DeepSeek-V3
  - [ ] DeepSeek-Coder
  - [ ] Qwen 2.5
- [ ] Create unified model interface in `libs/models/`
- [ ] Add model comparison conditions to all benchmarks
- [ ] Test local vs API latency and cost tradeoffs

## Medium Priority

### Expand Benchmark Scale
- [ ] Flow Gating: Expand to all 80+ published OMIP panels
- [ ] Flow Panel Optimizer: Add 50+ test cases per category
- [ ] DrugDevBench: Scale to 500+ figures

### Statistical Rigor
- [ ] Add confidence intervals to all metrics
- [ ] Implement bootstrap significance testing
- [ ] Add power analysis for sample size requirements

### Real Data Validation
- [ ] Flow Panel Optimizer: Validate CI predictions against real spreading matrices
- [ ] Flow Gating: Cross-validate against FlowRepository .wsp files
- [ ] DrugDevBench: Compare against expert annotations

## Low Priority

### Documentation
- [ ] Add API documentation for all libraries
- [ ] Create tutorial notebooks for each project
- [ ] Write methods section drafts for publication

### Infrastructure
- [ ] Add experiment tracking (MLflow or Weights & Biases)
- [ ] Create reproducibility scripts (Docker, requirements pinning)
- [ ] Add CI/CD pipeline for automated testing

---

*Last updated: 2026-01-07*

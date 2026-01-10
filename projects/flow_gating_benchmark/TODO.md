# Flow Gating Benchmark - TODO

## Recently Completed (Jan 2026)

### Evaluation Framework Refactoring
- [x] Split metrics.py into focused modules
  - `normalization.py` - Gate name normalization with 200+ cell type synonyms
  - `hierarchy.py` - Tree operations (extract names, parent maps, depth)
  - `task_failure.py` - Meta-question/refusal detection
- [x] Clean LLM client abstraction (`llm_client.py`)
  - Unified interface for Anthropic, OpenAI, Ollama
  - Protocol-based design with create_client() factory
- [x] Improved response parser
  - Better JSON extraction with proper brace matching
  - Meta-term detection in gate names
  - Stricter validation (MIN_GATES, MAX_NAME_LENGTH)

### Task Failure Detection
- [x] `TaskFailureType` enum: META_QUESTIONS, REFUSAL, INSTRUCTIONS, EMPTY, MALFORMED
- [x] Pattern matching for meta-questions, refusals, instructional responses
- [x] Integration into scorer and aggregate metrics
- [x] Confidence scoring based on pattern matches vs. valid gates

### Manual Review Reports
- [x] `ManualReviewReportGenerator` class
- [x] Side-by-side comparison with ASCII tree visualization
- [x] Configurable outlier thresholds (F1, hallucination, critical recall)
- [x] Report levels: summary, outliers, full
- [x] Task failure metrics in reports

### Data Curation
- [x] 8 OMIP test cases with ground truth hierarchies
- [x] OCR/extraction library (marker_logic, paper_parser, auto_extractor)
- [x] Panel entry extraction from XML tables

---

## High Priority

### Ground Truth Fluorophore Condition
Create experimental condition that provides exact fluorophore assignments:
- [ ] Add `fluorophore_provided` condition to experiments
- [ ] Compare F1 scores: inferred vs. provided fluorophores
- [ ] Isolate reasoning failures from information gaps

### Expand Model Coverage
- [ ] Complete Ollama integration testing
- [ ] Run experiments with:
  - [ ] Llama 3.2 (8B, 70B)
  - [ ] DeepSeek-V3
  - [ ] Qwen 2.5
- [ ] Create model comparison report

### Statistical Rigor
- [ ] Add confidence intervals to aggregate metrics
- [ ] Implement bootstrap significance testing
- [ ] Multi-run experiments for variance estimation

---

## Medium Priority

### Scale Up Test Set
- [ ] Expand to 20+ OMIP test cases
- [ ] Add tissue-specific panels (bone marrow, lymph node)
- [ ] Include rare population panels (ILCs, MAITs)
- [ ] Target diverse difficulty levels

### MCP Server for Interactive Curation
Wrap extraction library as MCP server:
- [ ] `get_paper_content` - Extract XML/PDF content
- [ ] `parse_marker_table` - Parse phenotype tables
- [ ] `build_hierarchy` - Build hierarchy from markers
- [ ] `validate_hierarchy` - Check against panel and HIPC
- [ ] `save_test_case` - Persist validated test cases

### Testing
- [ ] Integration tests for experiment runner
- [ ] Edge case tests for task failure detection
- [ ] Property-based testing with Hypothesis
- [ ] Cross-validation against FlowRepository .wsp files

---

## Low Priority

### Documentation
- [ ] API documentation for evaluation modules
- [ ] Tutorial notebook: "Running Your First Experiment"
- [ ] Methods section draft for publication

### Infrastructure
- [ ] Experiment tracking (MLflow or W&B)
- [ ] Docker container for reproducibility
- [ ] CI/CD pipeline

### Analysis Improvements
- [ ] Failure pattern categorization
- [ ] Per-population difficulty analysis
- [ ] Vocabulary familiarity correlation

---

## Future Work Ideas

### Multi-turn Evaluation
- Test iterative refinement: predict → feedback → revise
- Compare single-turn vs. multi-turn F1 scores
- Design feedback strategies (specific vs. general)

### Confidence Elicitation
- Ask models to rate prediction confidence
- Correlate confidence with actual F1 scores
- Test calibration across difficulty levels

### Vision Integration
- Extract gating figures from OMIP PDFs
- Use vision LLMs to read gate names from plots
- OCR fallback for non-vision models

### FlowRepository Cross-validation
- Automatic download of .wsp files
- Parse hierarchies using FlowKit
- Compare paper-extracted vs. .wsp hierarchies
- Assess ground truth reliability

---

*Last updated: 2026-01-10*

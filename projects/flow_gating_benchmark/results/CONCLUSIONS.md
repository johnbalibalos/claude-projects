# Flow Gating Benchmark: Key Conclusions

## Primary Finding

**Claude Sonnet achieves 67.3% hierarchy F1 on gating strategy prediction**, with performance ranging from 81.2% on simple panels to 53.4% on complex 40-color panels.

## What We Learned

### 1. LLMs "Know" Standard Gating Conventions

Critical gate recall (82.3%) demonstrates that Claude has learned standard flow cytometry conventions:
- Time gate for acquisition artifacts
- Singlet gate for doublet exclusion
- Live/Dead discrimination

These are reliably predicted even with minimal context.

### 2. Context Quality Matters More Than Prompting Strategy

| Intervention | F1 Improvement |
|--------------|----------------|
| Minimal → Rich context | **+24.5pp** |
| Direct → CoT prompting | +10.0pp |

Providing sample type, species, and application context is more impactful than sophisticated prompting.

### 3. Complexity Creates a Hard Ceiling

```
Simple (≤15 colors):  81.2% F1
Medium (16-25):       67.8% F1
Complex (26+):        53.4% F1
```

This isn't just more gates to predict - complex panels require tracking marker co-expression patterns Claude cannot reliably maintain.

### 4. Hallucinations are Predictable

15.6% hallucination rate, concentrated in:
- Memory markers (CD45RO) when not in panel
- Chemokine receptors (CCR7, CXCR5)
- Differentiation markers

Claude "assumes" panels are more comprehensive than they are.

### 5. Rare Populations are a Blind Spot

| Population | Recall |
|------------|--------|
| T cells, B cells, NK | 85.6% |
| ILCs, MAIT, pDCs | **31.2%** |

Rare populations require specialized knowledge that either:
- Isn't in training data
- Can't be reliably retrieved

## Implications for Anthropic's Life Sciences Team

### This Benchmark is Diagnostic

1. **Difficulty gradient**: Panel complexity provides natural curriculum
2. **Interpretable failures**: Hallucinations map to specific biological concepts
3. **Measurable progress**: F1 and hallucination rate track model improvements

### Practical Utility Assessment

| Panel Type | Recommendation |
|------------|----------------|
| Simple (≤15) | LLM suggestions useful, light review |
| Medium (16-25) | LLM as starting point, expert refinement |
| Complex (26+) | Expert-led, LLM for verification only |

### Key Metric: Hallucination Rate

For scientific applications, **hallucination rate is the critical safety metric**. A gating strategy that references non-existent markers would produce nonsensical results.

Current 15.6% hallucination rate is **too high for unsupervised use** but acceptable with expert review.

## What's Missing

1. **Multi-model comparison**: Is this Claude-specific or general LLM behavior?
2. **Few-shot learning**: Do examples help complex panels?
3. **Tool augmentation**: Can tools reduce hallucinations?
4. **Real .wsp validation**: Ground truth from workspace files, not papers

## The Path to Reliable Gating Prediction

Based on our findings, reliable (>90% F1, <5% hallucination) gating prediction requires:

1. **Rich context** (sample type, species, application) - ✅ achievable now
2. **Panel validation** (tool to check marker existence) - needs MCP
3. **Complexity handling** (hierarchical reasoning) - needs model improvement
4. **Rare population knowledge** (specialized training) - needs domain fine-tuning

## Bottom Line

> Claude can predict gating strategies for simple panels with 81% accuracy, but complex panels and rare populations remain challenging. Hallucination rate (15.6%) requires expert oversight.

This benchmark establishes a reproducible measure of biological reasoning capability that can track improvements over time.

---

*This document summarizes findings from the gating strategy prediction benchmark. Full methodology and results in REPORT.md.*

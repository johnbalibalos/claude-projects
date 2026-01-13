# Multi-Turn Evaluation Protocol for Flow Gating

> **STATUS: PROPOSED - NOT YET IMPLEMENTED**
>
> This is a design proposal. No multi-turn evaluation code has been implemented. Current pipeline is single-turn only.

## Overview

This document specifies a multi-turn evaluation protocol for testing LLM interactive reasoning capabilities in flow cytometry gating. Unlike single-turn evaluation (prompt → hierarchy), multi-turn tests the model's ability to:

1. Build hierarchies incrementally
2. Respond to guidance/corrections
3. Reason about marker constraints
4. Self-assess and revise predictions

## Protocol Design

### Turn Structure

```
Turn 1 (Initialization)
├── User provides: Panel, sample type, experimental context
├── User asks: "What are the first-level gates you would apply?"
└── Expected: Live/Dead, Singlets, basic scatter gates

Turn 2 (Expansion)
├── User acknowledges first-level gates
├── User asks: "For [Lymphocytes], what subpopulations would you identify?"
└── Expected: T cells, B cells, NK cells based on markers

Turn 3 (Refinement)
├── User introduces a constraint: "The panel includes [marker]. How does this affect gating?"
├── Or points out an issue: "Your hierarchy doesn't include [gate]. Should it?"
└── Expected: Adjustment or reasoned justification

Turn 4 (Verification)
├── User shows ground truth: "Here is the expected hierarchy: [...]"
├── User asks: "How does yours compare? What did you miss?"
└── Expected: Self-assessment, error acknowledgment, learning
```

### Detailed Turn Specifications

#### Turn 1: Initial Gating

**User prompt:**
```
I have a flow cytometry panel for {sample_type} analysis.

Panel markers:
{panel_markers}

What initial quality control and top-level gating would you recommend?
Focus on the first 2-3 levels of the hierarchy. Don't go deeper yet.
```

**Expected response elements:**
- Time gate (optional but good practice)
- Singlets (FSC and/or SSC based)
- Live/Dead discrimination
- Major population identification (e.g., Lymphocytes, Monocytes)

**Scoring:**
- Gate name match (fuzzy)
- Logical ordering (Live → Singlets is wrong; should be reverse)
- Appropriate for sample type

#### Turn 2: Population Expansion

**User prompt (adaptive based on Turn 1):**
```
Good. You identified {gates_from_turn_1}.

Let's focus on {selected_population}. Given the markers in the panel,
what subpopulations would you identify within {selected_population}?

Panel markers available:
{relevant_markers}
```

**Expected response elements:**
- Subpopulations appropriate for the parent gate
- Correct marker usage for discrimination
- Appropriate hierarchy depth (2-3 levels)

**Scoring:**
- Subset relationship correctness
- Marker-population alignment
- Hallucination check (populations not supported by panel)

#### Turn 3: Constraint Handling

**User prompt (one of several variants):**

**Variant A (Missing gate):**
```
I notice you didn't include a gate for {missing_gate}.
Given that the panel has {relevant_marker}, should we add this?
```

**Variant B (Marker clarification):**
```
The panel includes {ambiguous_marker}. In your hierarchy, you used it for
{usage}. Is this the only way to use this marker, or could it inform
other populations too?
```

**Variant C (Error correction):**
```
I think there might be an issue with your hierarchy. {population_A} and
{population_B} seem to be at the same level, but one should be a subset
of the other. Can you explain your reasoning or revise?
```

**Expected response elements:**
- Acknowledgment of the constraint/issue
- Reasoned response (agreement or disagreement with justification)
- Hierarchy revision if appropriate

**Scoring:**
- Reasoning quality (1-5 scale)
- Appropriate response to constraint
- Revision accuracy (if revised)

#### Turn 4: Self-Assessment

**User prompt:**
```
Here is the ground truth gating hierarchy from the OMIP paper:

{ground_truth_hierarchy}

Compare this with your prediction:

{model_prediction}

1. What gates did you get right?
2. What did you miss or get wrong?
3. What would you do differently next time?
```

**Expected response elements:**
- Accurate identification of matches
- Acknowledgment of misses/errors
- Reflection on reasoning process

**Scoring:**
- Self-assessment accuracy (matches actual F1)
- Error acknowledgment quality
- Learning/improvement potential

### Scoring Framework

#### Per-Turn Scores

| Turn | Metric | Weight |
|------|--------|--------|
| 1 | Initial gate F1 | 0.25 |
| 2 | Expansion accuracy | 0.25 |
| 3 | Constraint handling | 0.20 |
| 4 | Self-assessment accuracy | 0.15 |
| 4 | Error acknowledgment | 0.15 |

#### Aggregate Scores

- **Final Hierarchy F1**: Standard metric on complete hierarchy
- **Self-Correction Score**: (Turn 3 accuracy - Turn 2 accuracy) if revision occurred
- **Meta-Cognition Score**: Correlation between self-assessment and actual performance
- **Interactive Reasoning Score**: Weighted sum of all turn scores

### Adaptive Turn Selection

The protocol adapts based on model performance:

```python
def select_turn_3_variant(turn_2_result):
    """Select appropriate Turn 3 based on Turn 2 performance."""
    if turn_2_result.has_missing_critical_gates:
        return "missing_gate", turn_2_result.missing_critical[0]
    elif turn_2_result.has_hallucinations:
        return "error_correction", turn_2_result.hallucinations[0]
    elif turn_2_result.has_ambiguous_markers:
        return "marker_clarification", turn_2_result.ambiguous_markers[0]
    else:
        # Model did well, ask about edge case
        return "edge_case", select_edge_case_question(turn_2_result)
```

## Implementation Outline

### Core Classes

```python
@dataclass
class TurnResult:
    turn_number: int
    prompt: str
    response: str
    parsed_gates: list[str]
    metrics: dict[str, float]
    elapsed_time: float

@dataclass
class MultiTurnSession:
    test_case_id: str
    model: str
    turns: list[TurnResult]
    final_hierarchy: dict
    aggregate_scores: dict[str, float]

class MultiTurnEvaluator:
    def __init__(self, model: str, max_turns: int = 4):
        self.model = model
        self.max_turns = max_turns
        self.client = Anthropic()

    def run_session(self, test_case: TestCase) -> MultiTurnSession:
        turns = []
        context = self._build_initial_context(test_case)

        # Turn 1
        turn1 = self._run_turn_1(test_case, context)
        turns.append(turn1)

        # Turn 2 (adaptive)
        turn2 = self._run_turn_2(test_case, turn1, context)
        turns.append(turn2)

        # Turn 3 (adaptive)
        variant, target = select_turn_3_variant(turn2)
        turn3 = self._run_turn_3(test_case, turn2, variant, target, context)
        turns.append(turn3)

        # Turn 4
        turn4 = self._run_turn_4(test_case, turns, context)
        turns.append(turn4)

        return MultiTurnSession(
            test_case_id=test_case.test_case_id,
            model=self.model,
            turns=turns,
            final_hierarchy=self._extract_final_hierarchy(turns),
            aggregate_scores=self._compute_aggregate_scores(turns),
        )
```

### Message History Management

```python
def _build_messages(self, turns: list[TurnResult], new_prompt: str):
    """Build message history for API call."""
    messages = []

    for turn in turns:
        messages.append({"role": "user", "content": turn.prompt})
        messages.append({"role": "assistant", "content": turn.response})

    messages.append({"role": "user", "content": new_prompt})
    return messages
```

## Cost Estimate

| Component | Tokens | Cost (Sonnet) |
|-----------|--------|---------------|
| Turn 1 prompt + response | ~2,000 | $0.006 |
| Turn 2 prompt + response | ~2,500 | $0.008 |
| Turn 3 prompt + response | ~2,000 | $0.006 |
| Turn 4 prompt + response | ~3,000 | $0.010 |
| **Total per test case** | ~9,500 | **$0.030** |

For 15 test cases × 3 models = 45 sessions:
- Total tokens: ~430,000
- Estimated cost: **$15-25**

Compare to single-turn: ~$5-8 for same coverage.

## Success Criteria

The multi-turn protocol is considered successful if:

1. **Improved accuracy via iteration**: Turn 3 corrections improve final F1 by >5%
2. **Meaningful self-assessment**: Self-assessment correlates with actual F1 (r > 0.5)
3. **Useful error patterns**: Turn 3 reveals systematic weaknesses not visible in single-turn

## Limitations

1. **Cost**: 3-5x more expensive than single-turn
2. **Complexity**: More moving parts, harder to reproduce
3. **Not realistic use case**: Users typically want one-shot prediction
4. **Path dependence**: Results depend on Turn 3 variant selection

## When to Use

**Use multi-turn when:**
- Investigating interactive reasoning capabilities
- Debugging systematic failures
- Comparing models on meta-cognitive abilities
- Research on self-correction

**Prefer single-turn when:**
- Benchmark reporting (standard metrics)
- Cost-constrained evaluation
- Simple pass/fail evaluation

## Files to Create

| File | Purpose |
|------|---------|
| `src/evaluation/multi_turn.py` | Core protocol implementation |
| `src/evaluation/turn_prompts.py` | Prompt templates for each turn |
| `scripts/run_multi_turn_eval.py` | CLI for running multi-turn evaluation |

## Next Steps (If Proceeding)

1. Implement `MultiTurnEvaluator` class
2. Create turn prompt templates
3. Pilot on 3 test cases with 1 model
4. Analyze results and refine protocol
5. Full evaluation if pilot successful

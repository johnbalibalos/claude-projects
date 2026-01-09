# MCP Tester

Generic framework for running MCP/tool ablation studies with automatic checkpointing.

## Features

- Multi-condition experiment runner
- Tool execution with Anthropic API
- Automatic checkpointing of results
- Built-in evaluators for common metrics
- Report generation

## Installation

```bash
cd libs/mcp_tester
pip install -e .
```

Requires `anthropic` and the `checkpoint` library.

## Usage

### Basic Ablation Study

```python
from mcp_tester import AblationStudy, TestCase, Condition, CaseType

# Define test cases
test_cases = [
    TestCase(
        id="test_1",
        prompt="Calculate the similarity between PE and FITC",
        ground_truth={"similarity": 0.85},
        case_type=CaseType.SIMILARITY
    ),
]

# Define conditions
conditions = [
    Condition(
        name="baseline",
        tools_enabled=False
    ),
    Condition(
        name="with_tools",
        tools_enabled=True,
        tools=[
            {
                "name": "calculate_similarity",
                "description": "Calculate spectral similarity",
                "input_schema": {...}
            }
        ]
    ),
]

# Run study
study = AblationStudy(
    name="similarity_test",
    model="claude-sonnet-4-20250514",
    test_cases=test_cases,
    conditions=conditions,
    tool_executor=my_tool_fn
)

results = study.run()
study.generate_report(results)
```

### Custom Evaluator

```python
from mcp_tester import Evaluator

class MyEvaluator(Evaluator):
    def extract(self, response: str) -> dict:
        # Parse the response
        return {"value": parse_value(response)}

    def score(self, extracted: dict, ground_truth: dict) -> dict:
        # Compare extracted to ground truth
        return {
            "accuracy": 1.0 if extracted == ground_truth else 0.0
        }

study = AblationStudy(
    name="my_study",
    evaluator=MyEvaluator(),
    ...
)
```

### Tool Executor

```python
def my_tool_executor(tool_name: str, arguments: dict) -> dict:
    if tool_name == "calculate_similarity":
        return {"similarity": compute_similarity(**arguments)}
    return {"error": f"Unknown tool: {tool_name}"}

study = AblationStudy(
    tool_executor=my_tool_executor,
    ...
)
```

## API Reference

### TestCase

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier |
| `prompt` | str | The prompt to send to the model |
| `ground_truth` | dict | Expected answer for evaluation |
| `case_type` | CaseType | Category of test case |

### Condition

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Condition name |
| `tools_enabled` | bool | Whether tools are available |
| `tools` | list | Tool definitions (Anthropic format) |
| `context` | str | Optional context to prepend to prompt |

### AblationStudy

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Study name (used for checkpointing) |
| `model` | str | Anthropic model ID |
| `test_cases` | list[TestCase] | Test cases to run |
| `conditions` | list[Condition] | Conditions to compare |
| `evaluator` | Evaluator | Custom evaluator (default: AccuracyEvaluator) |
| `tool_executor` | Callable | Function to execute tools |
| `max_tool_calls` | int | Max tool calls per trial (default: 30) |

### Built-in Evaluators

- `AccuracyEvaluator`: Simple exact-match accuracy
- `SimilarityEvaluator`: Fuzzy string similarity

## Output

### StudyResults

```python
results = study.run()

# Access results
print(f"Success rate: {results.success_rate():.1%}")

# Filter by condition
baseline_trials = results.filter_by_condition("baseline")

# Generate report
report = study.generate_report(results)
```

### Report Example

```markdown
# Ablation Study Report: my_study

**Model:** claude-sonnet-4-20250514
**Total Trials:** 20
**Success Rate:** 95.0%

## Results by Condition

### baseline
- **accuracy:** 0.650
- **Avg Latency:** 2.3s

### with_tools
- **accuracy:** 0.850
- **Avg Latency:** 8.7s
```

## License

MIT

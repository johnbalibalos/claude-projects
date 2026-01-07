# Tool Error Analysis Implementation Plan

## Overview

Add comprehensive error analysis for tool use in the MCP ablation study. This will help understand not just *whether* tools help, but *how* they're being used (and misused).

## Research Questions

1. **When did Claude call tools?** - Timing and sequencing of tool invocations
2. **Did it call them correctly?** - Input validation and argument quality
3. **Did it interpret outputs correctly?** - Response utilization analysis

---

## Implementation Components

### 1. Enhanced Tool Call Logging

**File:** `src/flow_panel_optimizer/evaluation/runner.py`

Extend `tool_calls_made` to capture richer context:

```python
@dataclass
class ToolCallRecord:
    """Enhanced tool call record with context."""
    tool_name: str
    arguments: dict
    result: dict

    # Timing context
    call_index: int              # Which call in sequence (0, 1, 2...)
    turn_number: int             # Which conversation turn
    preceding_text: str          # Claude's reasoning before the call (truncated)

    # Input validation
    input_valid: bool            # Were arguments syntactically valid?
    input_errors: list[str]      # List of validation errors

    # Output analysis (populated post-hoc)
    output_referenced: bool      # Did Claude reference this output later?
    output_correctly_interpreted: bool  # Did Claude interpret it correctly?
    interpretation_notes: str    # Explanation of interpretation quality
```

### 2. Input Validation Module

**New file:** `src/flow_panel_optimizer/evaluation/tool_validation.py`

```python
class ToolInputValidator:
    """Validate tool call inputs against expected schemas and domain knowledge."""

    def validate_analyze_panel(self, args: dict) -> ValidationResult:
        """
        Checks:
        - fluorophores is a list
        - Each fluorophore name exists in database
        - No duplicates in list
        - List length is reasonable (2-40)
        """

    def validate_check_compatibility(self, args: dict) -> ValidationResult:
        """
        Checks:
        - candidate is a string and exists in database
        - existing_panel is a list of valid fluorophores
        - candidate is not already in existing_panel
        """

    def validate_suggest_fluorophores(self, args: dict) -> ValidationResult:
        """
        Checks:
        - existing_panel is valid
        - expression_level is one of [high, medium, low]
        - Panel size is reasonable
        """

    def validate_get_fluorophore_info(self, args: dict) -> ValidationResult:
        """
        Checks:
        - name exists in database
        - Fuzzy match suggestions if not found
        """

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    suggestions: list[str]  # e.g., "Did you mean 'PE-Cy7' instead of 'PE-CY7'?"
```

### 3. Output Interpretation Analysis

**New file:** `src/flow_panel_optimizer/evaluation/interpretation_analysis.py`

Analyze whether Claude correctly used tool outputs:

```python
class InterpretationAnalyzer:
    """Analyze whether tool outputs were correctly interpreted."""

    def analyze_trial(self, trial: TrialResult) -> InterpretationReport:
        """
        For each tool call, check if:
        1. The output was referenced in subsequent reasoning
        2. The interpretation was correct
        3. The final decision aligned with tool recommendations
        """

    def check_analyze_panel_interpretation(
        self,
        tool_output: dict,
        subsequent_text: str,
        final_assignments: dict
    ) -> InterpretationResult:
        """
        Check if:
        - Problematic pairs identified by tool were addressed
        - Quality rating was acknowledged
        - CI was mentioned/used in reasoning
        """

    def check_compatibility_interpretation(
        self,
        tool_output: dict,
        subsequent_text: str,
        final_assignments: dict
    ) -> InterpretationResult:
        """
        Check if:
        - AVOID recommendations were heeded
        - CAUTION recommendations acknowledged
        - Specific conflicts mentioned were addressed
        """

    def check_suggestion_interpretation(
        self,
        tool_output: dict,
        subsequent_text: str,
        final_assignments: dict
    ) -> InterpretationResult:
        """
        Check if:
        - Top-ranked suggestions were considered
        - Reasons for rejecting suggestions were valid
        - Expression level matching was respected
        """

@dataclass
class InterpretationResult:
    output_referenced: bool
    correctly_interpreted: bool
    error_type: Optional[str]  # e.g., "ignored_warning", "misread_value", "inverted_recommendation"
    notes: str
```

### 4. Error Taxonomy

Define categories of tool use errors:

```python
class ToolErrorType(Enum):
    # Input errors
    INVALID_FLUOROPHORE_NAME = "invalid_fluorophore_name"
    MISSPELLED_FLUOROPHORE = "misspelled_fluorophore"
    WRONG_EXPRESSION_LEVEL = "wrong_expression_level"
    DUPLICATE_IN_PANEL = "duplicate_in_panel"
    EMPTY_PANEL = "empty_panel"

    # Timing errors
    PREMATURE_ANALYSIS = "premature_analysis"  # Analyzed before panel was complete
    REDUNDANT_CALL = "redundant_call"          # Same call made multiple times
    MISSED_OPPORTUNITY = "missed_opportunity"  # Should have called tool but didn't

    # Interpretation errors
    IGNORED_WARNING = "ignored_warning"        # Tool said AVOID, Claude used anyway
    MISREAD_VALUE = "misread_value"            # Confused similarity scores, etc.
    INVERTED_RECOMMENDATION = "inverted"       # Did opposite of recommendation
    PARTIAL_INTERPRETATION = "partial"         # Only used part of output
    NO_REFERENCE = "no_reference"              # Never mentioned tool output
```

### 5. Aggregate Analysis Report

**New file:** `src/flow_panel_optimizer/evaluation/tool_analysis_report.py`

```python
class ToolAnalysisReport:
    """Aggregate tool use analysis across trials."""

    def generate_report(self, results: ExperimentResults) -> dict:
        return {
            "summary": {
                "total_tool_calls": int,
                "valid_input_rate": float,  # % of calls with valid inputs
                "correct_interpretation_rate": float,
                "avg_calls_per_trial": float,
            },
            "by_tool": {
                "analyze_panel": ToolStats,
                "check_compatibility": ToolStats,
                "suggest_fluorophores": ToolStats,
                "get_fluorophore_info": ToolStats,
            },
            "error_breakdown": {
                "input_errors": Counter[ToolErrorType],
                "interpretation_errors": Counter[ToolErrorType],
            },
            "timing_analysis": {
                "avg_first_call_turn": float,
                "call_sequence_patterns": list[str],  # Common sequences
            },
            "correlation_with_accuracy": {
                "valid_inputs_vs_accuracy": float,
                "correct_interpretation_vs_accuracy": float,
            }
        }

@dataclass
class ToolStats:
    total_calls: int
    valid_input_rate: float
    correct_interpretation_rate: float
    common_errors: list[tuple[ToolErrorType, int]]
```

---

## Implementation Tasks

### Phase 1: Data Collection Enhancement

- [ ] Add `ToolCallRecord` dataclass to runner.py
- [ ] Capture `preceding_text` before each tool call (last N chars of Claude's response)
- [ ] Store full conversation history in TrialResult for post-hoc analysis
- [ ] Add turn_number tracking in tool use loop

### Phase 2: Input Validation

- [ ] Create `tool_validation.py` module
- [ ] Implement validators for each tool type
- [ ] Add fuzzy matching for fluorophore names (handle typos)
- [ ] Run validation during trial execution
- [ ] Store validation results in ToolCallRecord

### Phase 3: Interpretation Analysis

- [ ] Create `interpretation_analysis.py` module
- [ ] Implement text search for tool output references
- [ ] Build heuristics for correct interpretation detection:
  - Check if AVOID recommendations appear in final panel
  - Check if suggested fluorophores were used
  - Compare tool scores with final decisions
- [ ] Handle edge cases (multiple conflicting tool outputs)

### Phase 4: Reporting & Visualization

- [ ] Create `tool_analysis_report.py` module
- [ ] Generate aggregate statistics
- [ ] Add per-trial breakdown option
- [ ] Create markdown report generator
- [ ] Add CSV export for further analysis
- [ ] Create visualization helpers (optional: charts/graphs)

### Phase 5: Integration & Testing

- [ ] Add --analyze-tools flag to CLI
- [ ] Create test cases for each error type
- [ ] Validate analysis on existing results
- [ ] Document analysis methodology

---

## Example Output

```
TOOL ERROR ANALYSIS REPORT
==========================

Summary (112 trials, 8 conditions)
----------------------------------
Total tool calls: 1,310
Average per MCP trial: 11.7
Valid input rate: 94.2%
Correct interpretation rate: 78.3%

Tool Breakdown
--------------
| Tool                  | Calls | Valid% | Interpreted% | Common Error          |
|-----------------------|-------|--------|--------------|----------------------|
| check_compatibility   | 564   | 96.1%  | 82.4%        | ignored_warning      |
| suggest_fluorophores  | 406   | 91.4%  | 71.2%        | partial_interpretation |
| analyze_panel         | 236   | 98.3%  | 85.1%        | no_reference         |
| get_fluorophore_info  | 104   | 89.4%  | 76.9%        | misspelled_fluorophore |

Input Error Breakdown
---------------------
- misspelled_fluorophore: 41 (52.6%)
- invalid_fluorophore_name: 18 (23.1%)
- duplicate_in_panel: 12 (15.4%)
- empty_panel: 7 (9.0%)

Interpretation Error Breakdown
------------------------------
- ignored_warning: 89 (39.2%)
- partial_interpretation: 72 (31.7%)
- no_reference: 45 (19.8%)
- misread_value: 21 (9.3%)

Timing Analysis
---------------
- First tool call: Turn 1 (98% of trials)
- Most common sequence: suggest → check → suggest → check → analyze
- Redundant calls: 3.2% of total

Correlation with Accuracy
-------------------------
- Trials with 100% valid inputs: 74.2% accuracy
- Trials with <90% valid inputs: 58.1% accuracy
- Trials with >80% correct interpretation: 81.3% accuracy
```

---

## Key Metrics to Track

1. **Input Validity Rate** - What % of tool calls had valid arguments?
2. **Interpretation Rate** - What % of tool outputs were correctly used?
3. **Ignored Warning Rate** - How often did Claude ignore AVOID/CAUTION?
4. **Tool Efficiency** - Calls needed vs. minimum necessary
5. **Error-Accuracy Correlation** - Do input/interpretation errors predict worse outcomes?

---

## Future Extensions

- **Automatic error correction**: Detect and fix misspellings before execution
- **Tool call suggestions**: Recommend tools Claude should have called
- **Reasoning trace analysis**: Deeper NLP analysis of Claude's reasoning
- **Comparative analysis**: How do error patterns differ across models?
- **Human annotation interface**: Manual review of edge cases

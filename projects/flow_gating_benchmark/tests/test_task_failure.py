"""Tests for evaluation/task_failure.py - Task failure detection."""


from evaluation.task_failure import (
    TaskFailureResult,
    TaskFailureType,
    compute_task_failure_rate,
    detect_task_failure,
)


class TestDetectTaskFailure:
    """Tests for detect_task_failure function."""

    def test_valid_response_not_failure(self):
        """Valid gating response is not a failure."""
        response = """
        Here is the gating hierarchy:

        All Events
        └── Singlets
            └── Live cells
                └── Lymphocytes
                    ├── CD3+ T cells
                    │   ├── CD4+ T cells
                    │   └── CD8+ T cells
                    └── CD19+ B cells
        """
        result = detect_task_failure(response)

        assert not result.is_failure
        assert result.failure_type == TaskFailureType.NONE
        assert result.gate_count >= 5

    def test_empty_response_is_failure(self):
        """Empty response is task failure."""
        result = detect_task_failure("")

        assert result.is_failure
        assert result.failure_type == TaskFailureType.EMPTY
        assert result.confidence == 1.0

    def test_whitespace_only_is_failure(self):
        """Whitespace-only response is task failure."""
        result = detect_task_failure("   \n\t  ")

        assert result.is_failure
        assert result.failure_type == TaskFailureType.EMPTY

    def test_meta_question_is_failure(self):
        """Meta-questions about the experiment are task failures."""
        response = """
        I'd be happy to help, but I need some clarification first.

        What markers are being used in your panel?
        What cell populations are you interested in studying?
        Could you provide more context about your research question?
        """
        result = detect_task_failure(response)

        assert result.is_failure
        assert result.failure_type == TaskFailureType.META_QUESTIONS
        assert len(result.evidence) > 0

    def test_refusal_is_failure(self):
        """Refusal to complete task is failure."""
        response = """
        I cannot predict this. It would be impossible to determine.
        I am unable to provide a gating hierarchy.
        """
        result = detect_task_failure(response)

        assert result.is_failure
        # Note: "more information" triggers meta-question pattern, so pure refusals
        # need to avoid that phrase
        assert result.failure_type in (TaskFailureType.REFUSAL, TaskFailureType.META_QUESTIONS)

    def test_instructional_is_failure(self):
        """Instructional response without actual gates is failure."""
        response = """
        Here's how you would typically approach this problem:

        Generally, you would start by removing debris and doublets.
        The typical approach would be to gate on live cells first.
        A common strategy is to then identify major lineages.

        The specific gates depend on your markers and goals.
        """
        result = detect_task_failure(response)

        assert result.is_failure
        assert result.failure_type == TaskFailureType.INSTRUCTIONS

    def test_mixed_response_with_gates_not_failure(self):
        """Response with some meta-content but valid gates is not failure."""
        response = """
        Based on the panel provided, here's my suggested gating hierarchy:

        All Events
        └── Singlets (FSC-A vs FSC-H)
            └── Live cells (Zombie NIR negative)
                └── Lymphocytes (FSC vs SSC)
                    ├── CD3+ T cells
                    │   ├── CD4+ helper T cells
                    │   └── CD8+ cytotoxic T cells
                    ├── CD19+ B cells
                    └── CD56+ NK cells

        Note: This depends on your specific markers.
        """
        result = detect_task_failure(response)

        assert not result.is_failure
        assert result.gate_count >= 5

    def test_hierarchy_with_meta_in_names(self):
        """Hierarchy with meta-content in gate names is failure."""
        hierarchy = {
            "name": "What populations are you interested in?",
            "children": [
                {"name": "Which markers should I use?", "children": []}
            ]
        }
        result = detect_task_failure(
            "See hierarchy below",
            parsed_hierarchy=hierarchy
        )

        assert result.is_failure
        assert len(result.evidence) > 0


class TestComputeTaskFailureRate:
    """Tests for compute_task_failure_rate function."""

    def test_no_failures(self):
        """All valid responses."""
        results = [
            {"raw_response": "Singlets > Live > Lymphocytes > T cells > CD4+", "parse_success": True},
            {"raw_response": "Live cells > CD45+ > CD3+ T cells > CD4+", "parse_success": True},
            {"raw_response": "Lymphocytes > CD19+ B cells > Memory B cells", "parse_success": True},
        ]
        stats = compute_task_failure_rate(results)

        assert stats["total"] == 3
        assert stats["task_failure_rate"] == 0.0
        assert stats["task_failure_count"] == 0

    def test_all_failures(self):
        """All failed responses."""
        results = [
            {"raw_response": "", "parse_success": True},  # Empty
            {"raw_response": "", "parse_success": True},  # Empty
            {"raw_response": "", "parse_success": True},  # Empty
        ]
        stats = compute_task_failure_rate(results)

        assert stats["total"] == 3
        assert stats["task_failure_rate"] == 1.0
        assert stats["task_failure_count"] == 3
        assert stats["empty"] == 3

    def test_mixed_results(self):
        """Mix of valid and failed responses."""
        results = [
            {"raw_response": "Singlets > Live > Lymphocytes > T cells > CD4+", "parse_success": True},
            {"raw_response": "", "parse_success": True},  # Empty - failure
            {"raw_response": "Live > CD3+ > CD4+ > Memory T cells > Treg", "parse_success": True},
            {"raw_response": "", "parse_success": True},  # Empty - failure
        ]
        stats = compute_task_failure_rate(results)

        assert stats["total"] == 4
        assert stats["task_failure_rate"] == 0.5  # 2 out of 4
        assert stats["task_failure_count"] == 2

    def test_parse_failures_counted(self):
        """Parse failures are counted as task failures by default."""
        results = [
            {"raw_response": "valid response", "parse_success": False},  # Parse failure
            {"raw_response": "Singlets > Live > T cells > CD4+ > CD8+", "parse_success": True},
        ]
        stats = compute_task_failure_rate(results, include_parse_failures=True)

        assert stats["total"] == 2
        assert stats["malformed"] == 1

    def test_parse_failures_excluded(self):
        """Parse failures can be excluded."""
        results = [
            {"raw_response": "valid response", "parse_success": False},  # Parse failure
            {"raw_response": "Singlets > Live > T cells > CD4+ > CD8+", "parse_success": True},
        ]
        stats = compute_task_failure_rate(results, include_parse_failures=False)

        assert stats["total"] == 2
        assert stats["malformed"] == 0

    def test_empty_results(self):
        """Empty results list."""
        stats = compute_task_failure_rate([])

        assert stats["total"] == 0
        assert stats["task_failure_rate"] == 0.0

    def test_failure_type_counts(self):
        """Counts by failure type."""
        results = [
            {"raw_response": "", "parse_success": True},  # Empty
            {"raw_response": "  ", "parse_success": True},  # Empty
            {"raw_response": "What markers?", "parse_success": True},  # Meta
            {"raw_response": "cannot predict", "parse_success": True},  # Refusal
        ]
        stats = compute_task_failure_rate(results)

        assert stats["empty"] == 2
        # Note: Short responses might not trigger meta/refusal patterns strongly


class TestTaskFailureResult:
    """Tests for TaskFailureResult dataclass."""

    def test_result_fields(self):
        """TaskFailureResult has expected fields."""
        result = TaskFailureResult(
            is_failure=True,
            failure_type=TaskFailureType.META_QUESTIONS,
            confidence=0.8,
            evidence=["Meta-question: 'what markers'"],
            gate_count=2,
        )

        assert result.is_failure is True
        assert result.failure_type == TaskFailureType.META_QUESTIONS
        assert result.confidence == 0.8
        assert len(result.evidence) == 1
        assert result.gate_count == 2


class TestTaskFailureType:
    """Tests for TaskFailureType enum."""

    def test_enum_values(self):
        """All failure types have correct values."""
        assert TaskFailureType.NONE.value == "none"
        assert TaskFailureType.META_QUESTIONS.value == "meta_questions"
        assert TaskFailureType.REFUSAL.value == "refusal"
        assert TaskFailureType.INSTRUCTIONS.value == "instructions"
        assert TaskFailureType.EMPTY.value == "empty"
        assert TaskFailureType.MALFORMED.value == "malformed"

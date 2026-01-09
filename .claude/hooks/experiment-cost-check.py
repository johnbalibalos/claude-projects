#!/usr/bin/env python3
"""
PreToolUse hook to intercept experiment scripts and require cost confirmation.

This hook:
1. Detects when an experiment script is being run
2. If --force flag is used: allow without confirmation
3. If no --force: block and require user to confirm costs first

Hook input (stdin JSON):
{
  "tool_name": "Bash",
  "tool_input": {"command": "python run_experiment.py ..."}
}

Exit codes:
- 0: Allow
- 2: Block (stderr shown as error, requires user confirmation)
"""

import json
import sys
import re

# Patterns that indicate experiment scripts (may incur API costs)
EXPERIMENT_PATTERNS = [
    r"run_experiment",
    r"run_.*experiment",
    r"run_.*ablation",
    r"run_benchmark",
    r"run_sonnet",
    r"run_opus",
    r"run_local_models",
    r"extract_multi_method.*--methods.*llm",  # LLM extraction costs money
    r"hypothesis_pipeline",
]

# Skip confirmation patterns (user explicitly wants to skip)
SKIP_CONFIRMATION_PATTERNS = [
    r"--force",
    r"--skip-cost-check",
    r"--dry-run",  # Dry run doesn't cost anything
    r"--help",
    r"-h\b",
]


def main():
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)  # Allow if can't parse

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})

    # Only check Bash commands
    if tool_name != "Bash":
        sys.exit(0)

    command = tool_input.get("command", "")
    if not command:
        sys.exit(0)

    # Check if this is an experiment script
    is_experiment = False
    for pattern in EXPERIMENT_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            is_experiment = True
            break

    if not is_experiment:
        sys.exit(0)  # Not an experiment, allow

    # Check if user wants to skip confirmation
    for pattern in SKIP_CONFIRMATION_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            sys.exit(0)  # User explicitly skipping, allow

    # Block and require confirmation
    message = """
⚠️  EXPERIMENT COST CHECK ⚠️

This command may incur API costs. Before running:

1. Show the user a cost estimate:
   python -c "from hypothesis_pipeline.cost import estimate_experiment_cost; ..."

2. Ask the user to confirm they want to proceed

3. If confirmed, re-run with --force to skip this check

Alternatively, use --dry-run to test without API calls.
"""
    print(message, file=sys.stderr)
    sys.exit(2)  # Block


if __name__ == "__main__":
    main()

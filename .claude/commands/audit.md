# Quick Code Audit

Fast audit for critical anti-patterns. Use `/code-review` for comprehensive analysis.

## Quick Checks (run in parallel)

Search for these patterns using Grep:

| Pattern | Severity | Grep Pattern |
|---------|----------|--------------|
| sys.path hack | CRITICAL | `sys\.path\.(insert\|append)` |
| Bare except | HIGH | `except\s*:` |
| subprocess CLI wrap | HIGH | `subprocess\.run\(\["(claude\|gh\|git)` |
| print() usage | MEDIUM | `print\(` (count only) |
| TODO/FIXME | INFO | `TODO\|FIXME\|HACK\|XXX` |

## Output

Provide a quick summary table:

```
## Quick Audit Results

| Check | Files | Instances | Action |
|-------|-------|-----------|--------|
| sys.path hacks | 12 | 15 | Fix with pyproject.toml |
| Bare except | 0 | 0 | OK |
| CLI wrappers | 1 | 1 | Mark as experimental |
| print() statements | 35 | 487 | Consider logging |
| TODOs | 8 | 12 | Review backlog |

**Verdict:** [CLEAN / NEEDS WORK / CRITICAL ISSUES]
```

Run these checks now on Python files in the current directory.

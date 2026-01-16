# Code Review: Anti-Patterns & Vibe Coding Detection

You are a senior engineer performing a code review. Scan the specified files or directory for anti-patterns, code smells, and "vibe coding" indicators.

## What to Scan

If the user specifies a path, scan that. Otherwise, scan the current project's Python files.

## Anti-Patterns to Detect

### 1. **sys.path Hacks** (CRITICAL)
```python
# BAD - fragile, breaks when files move
sys.path.insert(0, str(Path(__file__).parent.parent / "libs"))
```
**Fix:** Use proper `pyproject.toml` with `pip install -e .`

### 2. **CLI Wrapper Clients** (HIGH)
```python
# BAD - wrapping CLI tools meant for humans
result = subprocess.run(["claude", "-p", prompt], ...)
result = subprocess.run(["gh", "api", ...], ...)
```
**Fix:** Use official SDKs/APIs. Mark as EXPERIMENTAL if unavoidable.

### 3. **Hardcoded Paths** (HIGH)
```python
# BAD - not portable
data_dir = Path("data/verified/real")
config_path = "configs/settings.yaml"
```
**Fix:** Use environment variables, config files, or `__file__`-relative paths.

### 4. **print() Instead of Logging** (MEDIUM)
```python
# BAD - can't control verbosity
print(f"Processing {item}...")
print(f"  Calling API...", end=" ", flush=True)
```
**Fix:** Use `logging` module with proper levels.

### 5. **Inline Progress Patterns** (MEDIUM)
```python
# BAD - doesn't work in non-TTY, can't be silenced
print(f"\r  Progress: {i}/{n} ({pct:.1f}%)", end="", flush=True)
```
**Fix:** Use `rich.progress`, `tqdm`, or structured logging.

### 6. **Bare except Clauses** (MEDIUM)
```python
# BAD - catches KeyboardInterrupt, SystemExit
try:
    do_something()
except:
    pass
```
**Fix:** Use `except Exception:` at minimum, prefer specific exceptions.

### 7. **Generic RuntimeError** (MEDIUM)
```python
# BAD - caller can't distinguish error types
if not api_key:
    raise RuntimeError("API key not set")
```
**Fix:** Create custom exception classes (e.g., `ConfigurationError`).

### 8. **Missing Type Hints on Public Functions** (LOW)
```python
# BAD - unclear contract
def process_data(items, config):
    ...
```
**Fix:** Add type hints: `def process_data(items: list[Item], config: Config) -> Result:`

### 9. **Functions Over 50 Lines** (LOW)
Long functions indicate too many responsibilities.
**Fix:** Extract helper functions, use early returns.

### 10. **Global Mutable State** (MEDIUM)
```python
# BAD - non-deterministic, hard to test
CACHE = {}  # Module-level mutable dict
```
**Fix:** Use `functools.lru_cache`, dependency injection, or class encapsulation.

### 11. **Hardcoded Domain Logic** (MEDIUM)
```python
# BAD - mixing code and configuration
MARKER_DB = {"CD3": {"lineage": "T cells"}, ...}
```
**Fix:** Move to external JSON/YAML config file.

### 12. **Missing Dry-Run Mode** (LOW)
Scripts that make API calls or write files should support `--dry-run`.
**Fix:** Add `--dry-run` flag that uses mock clients and skips file I/O.

## Output Format

Provide a structured report:

```
## Code Review: [path]

### Critical Issues (fix immediately)
- [ ] `file.py:23` - sys.path hack
- [ ] `client.py:45` - CLI wrapper pattern

### High Priority
- [ ] `config.py:12` - Hardcoded path "/data/cache"

### Medium Priority
- [ ] `runner.py:89` - print() instead of logging (15 instances)
- [ ] `utils.py:34` - Bare except clause

### Low Priority
- [ ] `analysis.py:156` - Function `process_all()` is 87 lines

### Summary
| Category | Count |
|----------|-------|
| Critical | 2 |
| High | 1 |
| Medium | 16 |
| Low | 3 |

### Quick Wins
1. Replace `print()` with `logging` - 15 files affected
2. Add `--dry-run` flag to runner scripts
3. Move hardcoded paths to environment variables
```

## Instructions

1. Use `Grep` and `Glob` to find patterns efficiently
2. Read suspicious files to confirm issues
3. Prioritize by severity
4. Provide specific line numbers
5. Suggest concrete fixes
6. Identify "quick wins" that are easy to fix

Run this review now on the specified path or current directory.

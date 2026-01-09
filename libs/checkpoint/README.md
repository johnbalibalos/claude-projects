# Checkpoint

Resumable workflow runner with automatic checkpointing for long-running experiments.

## Features

- Automatic progress saving to disk (JSONL format)
- Skip already-completed items on resume
- Error tracking and recovery
- Progress reporting

## Installation

```bash
cd libs/checkpoint
pip install -e .
```

## Usage

### Basic Usage

```python
from checkpoint import CheckpointedRunner

runner = CheckpointedRunner("my_experiment")

items = [{"id": 1, "data": "..."}, {"id": 2, "data": "..."}]

for item in runner.iterate(items, key_fn=lambda x: x['id']):
    result = expensive_api_call(item)
    runner.save_result(item['id'], result)

# If interrupted and restarted, will skip already-completed items
all_results = runner.get_all_results()
```

### With Progress Reporting

```python
from checkpoint import BatchCheckpointedRunner

runner = BatchCheckpointedRunner("experiment")

for idx, total, item in runner.iterate_with_progress(
    items,
    key_fn=lambda x: x['id'],
    desc="Processing"
):
    result = process(item)
    runner.save_result(item['id'], result)
```

### Error Handling

```python
runner = CheckpointedRunner("experiment")

for item in runner.iterate(items, key_fn=lambda x: x['id']):
    try:
        result = process(item)
        runner.save_result(item['id'], result)
    except Exception as e:
        runner.save_error(item['id'], str(e))
        continue

# Review errors
errors = runner.get_all_errors()
```

## API Reference

### CheckpointedRunner

| Method | Description |
|--------|-------------|
| `iterate(items, key_fn)` | Yield items that haven't been completed |
| `save_result(key, result)` | Save a completed result |
| `save_error(key, error)` | Save an error for an item |
| `is_completed(key)` | Check if an item has been completed |
| `get_result(key)` | Get result for a specific key |
| `get_all_results()` | Get all completed results |
| `get_all_errors()` | Get all errors |
| `progress()` | Return (completed, total) counts |
| `clear()` | Clear all checkpoint data |

### BatchCheckpointedRunner

Extends `CheckpointedRunner` with:

| Method | Description |
|--------|-------------|
| `iterate_with_progress(items, key_fn, desc)` | Iterate with progress info: (idx, total, item) |

## File Structure

Checkpoints are saved to `./checkpoints/` by default:

```
checkpoints/
├── experiment_meta.json     # Metadata (timestamps, counts)
├── experiment_results.jsonl # Completed results
└── experiment_errors.jsonl  # Errors
```

## License

MIT

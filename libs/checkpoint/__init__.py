"""
Checkpoint - Resumable workflow library for long-running experiments.

Usage:
    from checkpoint import CheckpointedRunner, checkpoint

    # Method 1: Decorator for individual function caching
    @checkpoint("my_cache")
    def expensive_computation(x, y):
        return x + y

    # Method 2: CheckpointedRunner for iterative workflows
    runner = CheckpointedRunner("experiment_name", checkpoint_dir="./checkpoints")

    for item in runner.iterate(items, key_fn=lambda x: x['id']):
        result = process(item)
        runner.save_result(item['id'], result)

    all_results = runner.get_all_results()
"""

from .runner import CheckpointedRunner
from .decorator import checkpoint

__all__ = ["CheckpointedRunner", "checkpoint"]
__version__ = "0.1.0"

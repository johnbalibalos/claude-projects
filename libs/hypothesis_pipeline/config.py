"""
Configuration management for hypothesis pipelines.

Supports:
- YAML config files as templates
- Config inheritance/composition
- CLI overrides
- Config validation
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import yaml

from .models import ReasoningType, ContextLevel, RAGMode, DataSource


@dataclass
class PipelineConfig:
    """Configuration for the hypothesis pipeline."""

    name: str

    # Dimensions to test (cartesian product)
    models: list[str] = field(default_factory=lambda: ["claude-sonnet-4-20250514"])
    reasoning_types: list[ReasoningType] = field(default_factory=lambda: [ReasoningType.DIRECT])
    context_levels: list[ContextLevel] = field(default_factory=lambda: [ContextLevel.STANDARD])
    rag_modes: list[RAGMode] = field(default_factory=lambda: [RAGMode.NONE])
    tool_configs: list[list[str]] = field(default_factory=lambda: [[]])

    # Model parameters
    max_tokens: int = 4096
    temperature: float = 0.0
    max_tool_calls: int = 30

    # Experiment execution settings
    n_bootstrap_runs: int = 1  # Number of times to run each condition (for statistical power)
    data_source: DataSource = DataSource.SYNTHETIC  # Type of test data

    # Pipeline settings
    checkpoint_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    output_dir: Path = field(default_factory=lambda: Path("./results"))

    # Custom configs (keyed by enum value)
    strategy_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    context_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    rag_configs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Experiment metadata
    description: str = ""
    tags: list[str] = field(default_factory=list)
    hypothesis: str = ""  # What you're testing

    # Inherit from another config
    extends: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Path):
                result[f.name] = str(value)
            elif isinstance(value, list) and value and hasattr(value[0], 'value'):
                result[f.name] = [v.value for v in value]
            elif hasattr(value, 'value'):
                result[f.name] = value.value
            else:
                result[f.name] = value
        return result

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def save(self, path: Path | str) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_yaml())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary."""
        # Convert enum strings to enums
        if "reasoning_types" in data:
            data["reasoning_types"] = [
                ReasoningType(r) if isinstance(r, str) else r
                for r in data["reasoning_types"]
            ]
        if "context_levels" in data:
            data["context_levels"] = [
                ContextLevel(c) if isinstance(c, str) else c
                for c in data["context_levels"]
            ]
        if "rag_modes" in data:
            data["rag_modes"] = [
                RAGMode(r) if isinstance(r, str) else r
                for r in data["rag_modes"]
            ]
        if "data_source" in data:
            data["data_source"] = (
                DataSource(data["data_source"])
                if isinstance(data["data_source"], str)
                else data["data_source"]
            )

        # Convert paths
        if "checkpoint_dir" in data:
            data["checkpoint_dir"] = Path(data["checkpoint_dir"])
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])

        # Filter to valid fields
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered)


class ConfigLoader:
    """
    Loads and manages pipeline configurations.

    Supports:
    - Loading from YAML files
    - Config inheritance (extends)
    - CLI overrides
    - Environment variable substitution
    """

    def __init__(self, config_dir: Path | str | None = None):
        """
        Args:
            config_dir: Directory containing config templates
        """
        self.config_dir = Path(config_dir) if config_dir else Path("./configs")
        self._cache: dict[str, dict[str, Any]] = {}

    def load(
        self,
        path_or_name: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> PipelineConfig:
        """
        Load a config file with optional overrides.

        Args:
            path_or_name: Path to config file or name (without .yaml)
            overrides: Dictionary of values to override

        Returns:
            Loaded and merged PipelineConfig
        """
        # Resolve path
        path = self._resolve_path(path_or_name)

        # Load base config
        data = self._load_yaml(path)

        # Handle inheritance
        if "extends" in data and data["extends"]:
            parent = self.load(data["extends"])
            parent_data = parent.to_dict()
            # Merge: parent values overwritten by child values
            data = self._deep_merge(parent_data, data)
            del data["extends"]  # Remove extends from final config

        # Apply overrides
        if overrides:
            data = self._deep_merge(data, overrides)

        # Substitute environment variables
        data = self._substitute_env(data)

        return PipelineConfig.from_dict(data)

    def _resolve_path(self, path_or_name: str | Path) -> Path:
        """Resolve a path or config name to a full path."""
        path = Path(path_or_name)

        # If it's already a valid path, use it
        if path.exists():
            return path

        # Try adding .yaml extension
        if path.with_suffix(".yaml").exists():
            return path.with_suffix(".yaml")

        # Try in config directory
        config_path = self.config_dir / path_or_name
        if config_path.exists():
            return config_path

        if config_path.with_suffix(".yaml").exists():
            return config_path.with_suffix(".yaml")

        raise FileNotFoundError(f"Config not found: {path_or_name}")

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Load a YAML file."""
        if str(path) in self._cache:
            return self._cache[str(path)].copy()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        self._cache[str(path)] = data
        return data.copy()

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _substitute_env(self, data: Any) -> Any:
        """Substitute ${VAR} patterns with environment variables."""
        if isinstance(data, str):
            # Replace ${VAR} or ${VAR:default}
            import re
            pattern = r'\$\{(\w+)(?::([^}]*))?\}'

            def replace(match):
                var_name = match.group(1)
                default = match.group(2)
                return os.environ.get(var_name, default or match.group(0))

            return re.sub(pattern, replace, data)

        elif isinstance(data, dict):
            return {k: self._substitute_env(v) for k, v in data.items()}

        elif isinstance(data, list):
            return [self._substitute_env(v) for v in data]

        return data

    def list_configs(self) -> list[str]:
        """List available config files in config directory."""
        if not self.config_dir.exists():
            return []

        return [
            p.stem for p in self.config_dir.glob("*.yaml")
        ]


def parse_cli_overrides(args: list[str]) -> dict[str, Any]:
    """
    Parse CLI override arguments.

    Format: --key=value or --key.subkey=value

    Examples:
        --name=my_experiment
        --models=claude-sonnet-4-20250514,gpt-4o
        --reasoning_types=direct,cot
        --temperature=0.5
        --strategy_configs.cot.reasoning_prompt="Think step by step"
    """
    overrides = {}

    for arg in args:
        if not arg.startswith("--"):
            continue

        arg = arg[2:]  # Remove --
        if "=" not in arg:
            continue

        key, value = arg.split("=", 1)

        # Parse value type
        parsed_value = _parse_value(value)

        # Handle nested keys (key.subkey)
        keys = key.split(".")
        current = overrides

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = parsed_value

    return overrides


def _parse_value(value: str) -> Any:
    """Parse a string value into appropriate Python type."""
    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # None
    if value.lower() == "none":
        return None

    # Number
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # List (comma-separated)
    if "," in value:
        return [_parse_value(v.strip()) for v in value.split(",")]

    # String
    return value


# =============================================================================
# BUILT-IN CONFIG TEMPLATES
# =============================================================================


def create_minimal_config(name: str, **overrides: Any) -> PipelineConfig:
    """Create a minimal config for quick testing."""
    return PipelineConfig(
        name=name,
        models=["claude-sonnet-4-20250514"],
        reasoning_types=[ReasoningType.DIRECT],
        context_levels=[ContextLevel.STANDARD],
        rag_modes=[RAGMode.NONE],
        tool_configs=[[]],
        description="Minimal config for quick testing",
        **overrides,
    )


def create_ablation_config(name: str, **overrides: Any) -> PipelineConfig:
    """Create a config for ablation studies."""
    return PipelineConfig(
        name=name,
        models=["claude-sonnet-4-20250514"],
        reasoning_types=[ReasoningType.DIRECT, ReasoningType.COT],
        context_levels=[ContextLevel.MINIMAL, ContextLevel.STANDARD, ContextLevel.RICH],
        rag_modes=[RAGMode.NONE],
        tool_configs=[[]],
        description="Ablation study across reasoning and context",
        **overrides,
    )


def create_full_config(name: str, **overrides: Any) -> PipelineConfig:
    """Create a comprehensive config testing all dimensions."""
    return PipelineConfig(
        name=name,
        models=["claude-sonnet-4-20250514"],
        reasoning_types=[
            ReasoningType.DIRECT,
            ReasoningType.COT,
            ReasoningType.WOT,
        ],
        context_levels=[
            ContextLevel.MINIMAL,
            ContextLevel.STANDARD,
            ContextLevel.RICH,
        ],
        rag_modes=[RAGMode.NONE, RAGMode.ORACLE],
        tool_configs=[[], ["default_tools"]],
        description="Full experiment across all dimensions",
        **overrides,
    )

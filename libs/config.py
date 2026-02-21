"""
Shared configuration base classes for experiment configs.

Provides a Pydantic BaseModel that standardizes common experiment
configuration fields across all projects. Project-specific configs
should inherit from BaseExperimentConfig.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class BaseExperimentConfig(BaseModel):
    """Base configuration for all experiments.

    Subclass this in project-specific configs to inherit common fields
    while adding domain-specific parameters.

    Example::

        class FlowGatingConfig(BaseExperimentConfig):
            panel_type: str = "omip"
            num_markers: int = 30
    """

    model_config = {"extra": "allow"}

    # Model settings
    model_name: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="LLM model identifier",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens in response",
    )

    # Execution settings
    output_dir: Path = Field(
        default=Path("output"),
        description="Directory for experiment outputs",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    num_workers: int = Field(
        default=1,
        ge=1,
        description="Number of parallel workers",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, skip actual API calls",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

"""
Define experimental conditions for ablation study.

Conditions vary:
1. OMIP retrieval weighting (none, 1x, 2x, 5x, 10x, exclusive)
2. MCP tool access (enabled/disabled)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import json


class RetrievalMode(Enum):
    NONE = "none"              # No OMIP retrieval
    STANDARD = "standard"      # Weight = 1.0
    WEIGHTED_2X = "weighted_2x"
    WEIGHTED_5X = "weighted_5x"
    WEIGHTED_10X = "weighted_10x"
    EXCLUSIVE = "exclusive"    # ONLY retrieve from OMIP corpus


@dataclass
class ExperimentalCondition:
    """A single experimental condition."""
    name: str
    retrieval_mode: RetrievalMode
    retrieval_weight: float
    mcp_enabled: bool
    description: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "retrieval_mode": self.retrieval_mode.value,
            "retrieval_weight": self.retrieval_weight,
            "mcp_enabled": self.mcp_enabled,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentalCondition":
        return cls(
            name=data["name"],
            retrieval_mode=RetrievalMode(data["retrieval_mode"]),
            retrieval_weight=data["retrieval_weight"],
            mcp_enabled=data["mcp_enabled"],
            description=data["description"]
        )


# Define all experimental conditions
CONDITIONS = [
    ExperimentalCondition(
        name="baseline",
        retrieval_mode=RetrievalMode.NONE,
        retrieval_weight=0.0,
        mcp_enabled=False,
        description="No retrieval, no tools - pure LLM knowledge"
    ),
    ExperimentalCondition(
        name="retrieval_standard",
        retrieval_mode=RetrievalMode.STANDARD,
        retrieval_weight=1.0,
        mcp_enabled=False,
        description="Standard RAG with OMIP corpus"
    ),
    ExperimentalCondition(
        name="retrieval_2x",
        retrieval_mode=RetrievalMode.WEIGHTED_2X,
        retrieval_weight=2.0,
        mcp_enabled=False,
        description="OMIP panels weighted 2x in retrieval"
    ),
    ExperimentalCondition(
        name="retrieval_5x",
        retrieval_mode=RetrievalMode.WEIGHTED_5X,
        retrieval_weight=5.0,
        mcp_enabled=False,
        description="OMIP panels weighted 5x in retrieval"
    ),
    ExperimentalCondition(
        name="retrieval_10x",
        retrieval_mode=RetrievalMode.WEIGHTED_10X,
        retrieval_weight=10.0,
        mcp_enabled=False,
        description="OMIP panels weighted 10x in retrieval"
    ),
    ExperimentalCondition(
        name="retrieval_exclusive",
        retrieval_mode=RetrievalMode.EXCLUSIVE,
        retrieval_weight=float('inf'),
        mcp_enabled=False,
        description="Only retrieve from OMIP corpus, nothing else"
    ),
    ExperimentalCondition(
        name="mcp_only",
        retrieval_mode=RetrievalMode.NONE,
        retrieval_weight=0.0,
        mcp_enabled=True,
        description="MCP tool access, no OMIP retrieval"
    ),
    ExperimentalCondition(
        name="mcp_plus_retrieval",
        retrieval_mode=RetrievalMode.STANDARD,
        retrieval_weight=1.0,
        mcp_enabled=True,
        description="MCP tool + standard OMIP retrieval"
    ),
]


# Subset for quick testing
QUICK_TEST_CONDITIONS = [
    CONDITIONS[0],  # baseline
    CONDITIONS[6],  # mcp_only
]


# Core conditions for main comparison
CORE_CONDITIONS = [
    CONDITIONS[0],  # baseline
    CONDITIONS[1],  # retrieval_standard
    CONDITIONS[6],  # mcp_only
    CONDITIONS[7],  # mcp_plus_retrieval
]


def get_condition_by_name(name: str) -> Optional[ExperimentalCondition]:
    """Get condition by name."""
    for cond in CONDITIONS:
        if cond.name == name:
            return cond
    return None


def get_conditions_by_names(names: list[str]) -> list[ExperimentalCondition]:
    """Get multiple conditions by name."""
    return [c for c in CONDITIONS if c.name in names]

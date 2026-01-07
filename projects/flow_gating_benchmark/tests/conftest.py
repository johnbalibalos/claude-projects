"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_hierarchy():
    """Sample gating hierarchy for testing."""
    return {
        "name": "All Events",
        "markers": [],
        "children": [
            {
                "name": "Singlets",
                "markers": ["FSC-A", "FSC-H"],
                "children": [
                    {
                        "name": "Live",
                        "markers": ["Live/Dead"],
                        "children": [
                            {
                                "name": "CD45+",
                                "markers": ["CD45"],
                                "children": [
                                    {"name": "T cells", "markers": ["CD3"], "children": []},
                                    {"name": "B cells", "markers": ["CD19"], "children": []},
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_panel():
    """Sample panel for testing."""
    return [
        {"marker": "CD3", "fluorophore": "BUV395"},
        {"marker": "CD19", "fluorophore": "BV605"},
        {"marker": "CD45", "fluorophore": "BV421"},
        {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
    ]


@pytest.fixture
def ground_truth_dir():
    """Path to ground truth test cases."""
    return Path(__file__).parent.parent / "data" / "ground_truth"

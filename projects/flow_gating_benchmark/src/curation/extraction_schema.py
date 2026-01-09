"""
Schema for multi-method gating hierarchy extraction with concordance tracking.

Design goals:
1. Track which method produced each extraction (xml, llm, manual, vision)
2. Store multiple extractions per paper for concordance testing
3. Calculate agreement metrics between methods
4. Support incremental extraction (add new methods without re-running others)

Example structure:
{
    "omip_id": "OMIP-077",
    "pmc_id": "PMC9292053",
    "extractions": {
        "panel": {
            "xml": {"entries": [...], "confidence": 0.9, "timestamp": "..."},
            "llm": {"entries": [...], "confidence": 0.7, "timestamp": "..."},
        },
        "gating_hierarchy": {
            "xml": {"hierarchy": {...}, "confidence": 0.3},
            "llm": {"hierarchy": {...}, "confidence": 0.7},
            "manual": {"hierarchy": {...}, "confidence": 1.0},
        }
    },
    "concordance": {
        "panel": {"xml_vs_llm": {"jaccard": 0.85, "mismatches": [...]}},
        "gating_hierarchy": {"xml_vs_llm": {"tree_similarity": 0.6}}
    },
    "best_extraction": {
        "panel": "xml",
        "gating_hierarchy": "llm"
    }
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ExtractionMethod(str, Enum):
    """Methods for extracting data from papers."""
    XML = "xml"           # Direct XML table parsing
    LLM = "llm"           # LLM extraction from text
    VISION = "vision"     # Vision LLM from figures
    MANUAL = "manual"     # Human curation
    WSP = "wsp"           # FlowJo workspace file


@dataclass
class MethodExtraction:
    """A single extraction from one method."""
    method: ExtractionMethod
    data: dict  # The extracted data (panel entries, hierarchy, etc.)
    confidence: float  # 0.0 to 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model: str | None = None  # For LLM: model used
    source_file: str | None = None  # Source file path
    notes: str | None = None  # Any extraction notes
    raw_response: str | None = None  # For LLM: raw response for debugging


@dataclass
class ConcordanceResult:
    """Result of comparing two extractions."""
    method_a: ExtractionMethod
    method_b: ExtractionMethod
    metric_name: str  # e.g., "jaccard", "tree_edit_distance"
    score: float  # 0.0 to 1.0 (higher = more agreement)
    details: dict = field(default_factory=dict)  # Mismatches, etc.


@dataclass
class MultiMethodExtraction:
    """
    Container for all extractions of a single paper across methods.

    Supports:
    - Multiple extraction methods per data type
    - Concordance calculation between methods
    - Best extraction selection
    """
    omip_id: str
    pmc_id: str | None = None
    doi: str | None = None
    title: str | None = None

    # Extractions by data type, then by method
    panel_extractions: dict[str, MethodExtraction] = field(default_factory=dict)
    hierarchy_extractions: dict[str, MethodExtraction] = field(default_factory=dict)
    context_extractions: dict[str, MethodExtraction] = field(default_factory=dict)

    # Concordance results
    panel_concordance: list[ConcordanceResult] = field(default_factory=list)
    hierarchy_concordance: list[ConcordanceResult] = field(default_factory=list)

    # Best extraction selection (method name)
    best_panel_method: str | None = None
    best_hierarchy_method: str | None = None

    def add_panel_extraction(self, extraction: MethodExtraction):
        """Add a panel extraction from a method."""
        self.panel_extractions[extraction.method.value] = extraction

    def add_hierarchy_extraction(self, extraction: MethodExtraction):
        """Add a hierarchy extraction from a method."""
        self.hierarchy_extractions[extraction.method.value] = extraction

    def get_best_panel(self) -> MethodExtraction | None:
        """Get the best panel extraction."""
        if self.best_panel_method and self.best_panel_method in self.panel_extractions:
            return self.panel_extractions[self.best_panel_method]
        # Default: highest confidence
        if not self.panel_extractions:
            return None
        return max(self.panel_extractions.values(), key=lambda x: x.confidence)

    def get_best_hierarchy(self) -> MethodExtraction | None:
        """Get the best hierarchy extraction."""
        if self.best_hierarchy_method and self.best_hierarchy_method in self.hierarchy_extractions:
            return self.hierarchy_extractions[self.best_hierarchy_method]
        if not self.hierarchy_extractions:
            return None
        return max(self.hierarchy_extractions.values(), key=lambda x: x.confidence)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "omip_id": self.omip_id,
            "pmc_id": self.pmc_id,
            "doi": self.doi,
            "title": self.title,
            "extractions": {
                "panel": {
                    method: {
                        "method": ext.method.value,
                        "data": ext.data,
                        "confidence": ext.confidence,
                        "timestamp": ext.timestamp,
                        "model": ext.model,
                        "source_file": ext.source_file,
                        "notes": ext.notes,
                    }
                    for method, ext in self.panel_extractions.items()
                },
                "gating_hierarchy": {
                    method: {
                        "method": ext.method.value,
                        "data": ext.data,
                        "confidence": ext.confidence,
                        "timestamp": ext.timestamp,
                        "model": ext.model,
                        "source_file": ext.source_file,
                        "notes": ext.notes,
                    }
                    for method, ext in self.hierarchy_extractions.items()
                },
                "context": {
                    method: {
                        "method": ext.method.value,
                        "data": ext.data,
                        "confidence": ext.confidence,
                        "timestamp": ext.timestamp,
                        "model": ext.model,
                    }
                    for method, ext in self.context_extractions.items()
                }
            },
            "concordance": {
                "panel": [
                    {
                        "methods": f"{r.method_a.value}_vs_{r.method_b.value}",
                        "metric": r.metric_name,
                        "score": r.score,
                        "details": r.details
                    }
                    for r in self.panel_concordance
                ],
                "gating_hierarchy": [
                    {
                        "methods": f"{r.method_a.value}_vs_{r.method_b.value}",
                        "metric": r.metric_name,
                        "score": r.score,
                        "details": r.details
                    }
                    for r in self.hierarchy_concordance
                ]
            },
            "best_extraction": {
                "panel": self.best_panel_method,
                "gating_hierarchy": self.best_hierarchy_method
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MultiMethodExtraction":
        """Load from dictionary."""
        obj = cls(
            omip_id=data["omip_id"],
            pmc_id=data.get("pmc_id"),
            doi=data.get("doi"),
            title=data.get("title"),
        )

        # Load extractions
        extractions = data.get("extractions", {})

        for method, ext_data in extractions.get("panel", {}).items():
            obj.panel_extractions[method] = MethodExtraction(
                method=ExtractionMethod(ext_data["method"]),
                data=ext_data["data"],
                confidence=ext_data["confidence"],
                timestamp=ext_data.get("timestamp", ""),
                model=ext_data.get("model"),
                source_file=ext_data.get("source_file"),
                notes=ext_data.get("notes"),
            )

        for method, ext_data in extractions.get("gating_hierarchy", {}).items():
            obj.hierarchy_extractions[method] = MethodExtraction(
                method=ExtractionMethod(ext_data["method"]),
                data=ext_data["data"],
                confidence=ext_data["confidence"],
                timestamp=ext_data.get("timestamp", ""),
                model=ext_data.get("model"),
                source_file=ext_data.get("source_file"),
                notes=ext_data.get("notes"),
            )

        # Load concordance
        for conc in data.get("concordance", {}).get("panel", []):
            methods = conc["methods"].split("_vs_")
            obj.panel_concordance.append(ConcordanceResult(
                method_a=ExtractionMethod(methods[0]),
                method_b=ExtractionMethod(methods[1]),
                metric_name=conc["metric"],
                score=conc["score"],
                details=conc.get("details", {})
            ))

        obj.best_panel_method = data.get("best_extraction", {}).get("panel")
        obj.best_hierarchy_method = data.get("best_extraction", {}).get("gating_hierarchy")

        return obj

    def save(self, output_dir: Path | str):
        """Save to JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_id = self.omip_id.lower().replace("-", "_").replace(" ", "_")
        output_path = output_dir / f"{safe_id}.json"

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        return output_path

    @classmethod
    def load(cls, path: Path | str) -> "MultiMethodExtraction":
        """Load from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def calculate_panel_concordance(
    ext_a: MethodExtraction,
    ext_b: MethodExtraction
) -> ConcordanceResult:
    """
    Calculate concordance between two panel extractions.

    Uses Jaccard similarity on marker names, with alias normalization.
    """
    from .marker_aliases import calculate_marker_concordance_with_aliases

    # Extract marker names from both
    markers_a = set()
    markers_b = set()

    data_a = ext_a.data if ext_a.data else {}
    data_b = ext_b.data if ext_b.data else {}

    for entry in data_a.get("entries", []):
        if isinstance(entry, dict) and entry.get("marker"):
            markers_a.add(entry["marker"].upper())

    for entry in data_b.get("entries", []):
        if isinstance(entry, dict) and entry.get("marker"):
            markers_b.add(entry["marker"].upper())

    # Calculate with alias normalization
    result = calculate_marker_concordance_with_aliases(markers_a, markers_b)
    jaccard = result["normalized_jaccard"]  # Use normalized score

    return ConcordanceResult(
        method_a=ext_a.method,
        method_b=ext_b.method,
        metric_name="jaccard_markers_normalized",
        score=jaccard,
        details={
            "raw_jaccard": result["raw_jaccard"],
            "normalized_jaccard": result["normalized_jaccard"],
            "raw_markers_a": sorted(markers_a),
            "raw_markers_b": sorted(markers_b),
            "canonical_intersection": result["canonical_intersection"],
            "canonical_only_a": result["canonical_only_a"],
            "canonical_only_b": result["canonical_only_b"],
        }
    )


def calculate_hierarchy_concordance(
    ext_a: MethodExtraction,
    ext_b: MethodExtraction
) -> ConcordanceResult:
    """
    Calculate concordance between two hierarchy extractions.

    Uses gate name overlap and structural similarity.
    """
    def get_all_gates(hierarchy: dict, gates: set = None) -> set:
        """Recursively extract all gate names."""
        if gates is None:
            gates = set()

        name = hierarchy.get("name", "")
        if name:
            gates.add(name.lower().strip())

        for child in hierarchy.get("children", []):
            get_all_gates(child, gates)

        return gates

    data_a = ext_a.data if ext_a.data else {}
    data_b = ext_b.data if ext_b.data else {}
    hierarchy_a = data_a.get("hierarchy", {}) or {}
    hierarchy_b = data_b.get("hierarchy", {}) or {}

    gates_a = get_all_gates(hierarchy_a)
    gates_b = get_all_gates(hierarchy_b)

    # Jaccard on gate names
    if not gates_a and not gates_b:
        jaccard = 1.0
    elif not gates_a or not gates_b:
        jaccard = 0.0
    else:
        intersection = gates_a & gates_b
        union = gates_a | gates_b
        jaccard = len(intersection) / len(union)

    return ConcordanceResult(
        method_a=ext_a.method,
        method_b=ext_b.method,
        metric_name="jaccard_gates",
        score=jaccard,
        details={
            "gates_a": sorted(gates_a),
            "gates_b": sorted(gates_b),
            "intersection": sorted(gates_a & gates_b),
            "only_in_a": sorted(gates_a - gates_b),
            "only_in_b": sorted(gates_b - gates_a),
        }
    )


def calculate_all_concordance(extraction: MultiMethodExtraction):
    """Calculate concordance between all method pairs."""
    # Panel concordance
    panel_methods = list(extraction.panel_extractions.keys())
    for i, method_a in enumerate(panel_methods):
        for method_b in panel_methods[i+1:]:
            result = calculate_panel_concordance(
                extraction.panel_extractions[method_a],
                extraction.panel_extractions[method_b]
            )
            extraction.panel_concordance.append(result)

    # Hierarchy concordance
    hierarchy_methods = list(extraction.hierarchy_extractions.keys())
    for i, method_a in enumerate(hierarchy_methods):
        for method_b in hierarchy_methods[i+1:]:
            result = calculate_hierarchy_concordance(
                extraction.hierarchy_extractions[method_a],
                extraction.hierarchy_extractions[method_b]
            )
            extraction.hierarchy_concordance.append(result)

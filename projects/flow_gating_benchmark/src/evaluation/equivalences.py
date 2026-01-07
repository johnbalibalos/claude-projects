"""
Gate name equivalence handling for improved fuzzy matching.

This module provides:
- EquivalenceRegistry: Loads and applies expert-verified equivalences
- AnnotationCapture: Captures near-miss pairs for expert review

The workflow is:
1. During scoring, use EquivalenceRegistry to match gate names
2. When no match found, check if it's a near-miss (similarity > threshold)
3. Near-misses are saved to pending_annotations.jsonl for expert review
4. Experts review and add verified equivalences to verified_equivalences.yaml
5. Re-run scoring with expanded equivalences
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class EquivalenceClass:
    """A set of gate names that are semantically equivalent."""

    canonical: str
    variants: set[str]
    domain: str | None = None
    notes: str | None = None


@dataclass
class PatternRule:
    """A regex-based equivalence pattern."""

    pattern: re.Pattern
    replacement: str
    bidirectional: bool = True


class EquivalenceRegistry:
    """
    Loads and applies expert-verified gate name equivalences.

    Usage:
        registry = EquivalenceRegistry(Path("data/annotations/verified_equivalences.yaml"))

        # Check if two names are equivalent
        if registry.are_equivalent("gd t cells", "γδ T cells"):
            print("Match!")

        # Get canonical form
        canonical = registry.get_canonical("Tregs")  # -> "regulatory t cells"

        # Normalize a gate name (applies patterns + lookup)
        normalized = registry.normalize("CD4 positive")  # -> "cd4+"
    """

    def __init__(self, equivalences_path: Path | None = None):
        self.equivalences_path = equivalences_path
        self.classes: list[EquivalenceClass] = []
        self.lookup: dict[str, str] = {}  # variant -> canonical
        self.patterns: list[PatternRule] = []

        if equivalences_path and equivalences_path.exists():
            self._load()

    def _load(self) -> None:
        """Load equivalences from YAML file."""
        with open(self.equivalences_path) as f:
            data = yaml.safe_load(f)

        if not data:
            return

        # Load equivalence classes
        for eq in data.get("equivalence_classes", []):
            canonical = eq["canonical"].lower().strip()
            variants = {v.lower().strip() for v in eq.get("variants", [])}
            variants.add(canonical)

            ec = EquivalenceClass(
                canonical=canonical,
                variants=variants,
                domain=eq.get("domain"),
                notes=eq.get("notes"),
            )
            self.classes.append(ec)

            # Build lookup table
            for v in variants:
                self.lookup[v] = canonical

        # Load pattern rules
        for p in data.get("patterns", []):
            try:
                pattern = re.compile(p["pattern"], re.IGNORECASE)
                self.patterns.append(
                    PatternRule(
                        pattern=pattern,
                        replacement=p["equivalent"],
                        bidirectional=p.get("bidirectional", True),
                    )
                )
            except re.error as e:
                print(f"Invalid regex pattern '{p['pattern']}': {e}")

    def get_canonical(self, gate_name: str) -> str:
        """
        Get canonical form of a gate name.

        Args:
            gate_name: The gate name to canonicalize

        Returns:
            The canonical form, or normalized input if no match found
        """
        normalized = gate_name.lower().strip()

        # Direct lookup first (fastest)
        if normalized in self.lookup:
            return self.lookup[normalized]

        # Try pattern matching
        for rule in self.patterns:
            match = rule.pattern.match(normalized)
            if match:
                canonical = rule.pattern.sub(rule.replacement, normalized)
                # Check if the result maps to a known canonical form
                if canonical in self.lookup:
                    return self.lookup[canonical]
                # For bidirectional patterns, also check if we should return as-is
                if rule.bidirectional:
                    return canonical

        return normalized

    def normalize(self, gate_name: str) -> str:
        """
        Normalize a gate name by applying patterns and basic cleanup.

        This is a lighter-weight operation than get_canonical() that
        applies pattern transformations without requiring a lookup hit.

        Args:
            gate_name: The gate name to normalize

        Returns:
            Normalized form
        """
        normalized = gate_name.lower().strip()

        # Apply pattern transformations
        for rule in self.patterns:
            if rule.pattern.match(normalized):
                normalized = rule.pattern.sub(rule.replacement, normalized)
                break  # Apply only first matching pattern

        # Basic cleanup
        normalized = " ".join(normalized.split())  # Normalize whitespace

        return normalized

    def are_equivalent(self, gate1: str, gate2: str) -> bool:
        """
        Check if two gate names are equivalent.

        Args:
            gate1: First gate name
            gate2: Second gate name

        Returns:
            True if the gates are equivalent
        """
        canonical1 = self.get_canonical(gate1)
        canonical2 = self.get_canonical(gate2)
        return canonical1 == canonical2

    def find_match(
        self, gate_name: str, candidates: set[str]
    ) -> tuple[str | None, str | None]:
        """
        Find a matching gate name from a set of candidates.

        Args:
            gate_name: The gate name to match
            candidates: Set of candidate gate names

        Returns:
            Tuple of (matched_candidate, canonical_form) or (None, None)
        """
        gate_canonical = self.get_canonical(gate_name)

        for candidate in candidates:
            candidate_canonical = self.get_canonical(candidate)
            if gate_canonical == candidate_canonical:
                return candidate, gate_canonical

        return None, None

    def get_all_variants(self, gate_name: str) -> set[str]:
        """
        Get all known variants of a gate name.

        Args:
            gate_name: The gate name

        Returns:
            Set of all equivalent variants (including the input)
        """
        canonical = self.get_canonical(gate_name)

        # Find the equivalence class
        for ec in self.classes:
            if ec.canonical == canonical:
                return ec.variants.copy()

        # No class found, return just the normalized input
        return {gate_name.lower().strip()}

    def __len__(self) -> int:
        """Return number of equivalence classes."""
        return len(self.classes)

    def __repr__(self) -> str:
        return f"EquivalenceRegistry({len(self.classes)} classes, {len(self.patterns)} patterns)"


@dataclass
class PendingAnnotation:
    """A candidate equivalence pair awaiting expert review."""

    id: str
    predicted: str
    ground_truth: str
    similarity: float
    test_case: str
    parent_context: str | None = None
    status: str = "pending"  # pending, verified, rejected
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "predicted": self.predicted,
            "ground_truth": self.ground_truth,
            "similarity": self.similarity,
            "test_case": self.test_case,
            "parent_context": self.parent_context,
            "status": self.status,
            "created": self.created,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PendingAnnotation:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            predicted=data["predicted"],
            ground_truth=data["ground_truth"],
            similarity=data["similarity"],
            test_case=data["test_case"],
            parent_context=data.get("parent_context"),
            status=data.get("status", "pending"),
            created=data.get("created", datetime.now().isoformat()),
            reviewed_by=data.get("reviewed_by"),
            reviewed_at=data.get("reviewed_at"),
            notes=data.get("notes"),
        )


class AnnotationCapture:
    """
    Captures near-miss gate pairs for expert review.

    Near-misses are pairs where:
    - The names don't match via equivalence lookup
    - But string similarity is above a threshold

    These are saved to pending_annotations.jsonl for expert review.

    Usage:
        capture = AnnotationCapture(Path("data/annotations/pending_annotations.jsonl"))

        # During scoring, check for near-misses
        if capture.check_and_capture(
            predicted="gd t cells",
            ground_truth="gamma-delta T cells",
            test_case_id="OMIP-044",
            parent_context="CD3+ T cells"
        ):
            print("Near-miss captured for review")

        # Save at end of run
        capture.save()
    """

    # Similarity thresholds for capturing near-misses
    SIMILARITY_THRESHOLD_LOW = 0.5  # Below this, definitely not a match
    SIMILARITY_THRESHOLD_HIGH = 0.9  # Above this, should already match

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.pending: list[PendingAnnotation] = []
        self._existing_pairs: set[tuple[str, str]] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing pending annotations."""
        if not self.output_path.exists():
            return

        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        ann = PendingAnnotation.from_dict(data)
                        self.pending.append(ann)
                        # Track existing pairs to avoid duplicates
                        self._existing_pairs.add(
                            (ann.predicted.lower(), ann.ground_truth.lower())
                        )
                    except json.JSONDecodeError:
                        continue

    def compute_similarity(self, s1: str, s2: str) -> float:
        """
        Compute string similarity ratio.

        Uses SequenceMatcher which handles insertions, deletions, and substitutions.
        """
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def check_and_capture(
        self,
        predicted: str,
        ground_truth: str,
        test_case_id: str,
        parent_context: str | None = None,
    ) -> bool:
        """
        Check if pair is a near-miss and capture for annotation.

        Args:
            predicted: The predicted gate name
            ground_truth: The ground truth gate name
            test_case_id: ID of the test case
            parent_context: Optional parent gate name for context

        Returns:
            True if captured as near-miss (should not count as match yet)
        """
        similarity = self.compute_similarity(predicted, ground_truth)

        # Check if it's in the near-miss range
        if not (self.SIMILARITY_THRESHOLD_LOW < similarity < self.SIMILARITY_THRESHOLD_HIGH):
            return False

        # Check if already captured
        pair_key = (predicted.lower(), ground_truth.lower())
        if pair_key in self._existing_pairs:
            return True  # Already captured, still a near-miss

        # Create new annotation
        annotation = PendingAnnotation(
            id=f"ann_{len(self.pending):04d}",
            predicted=predicted,
            ground_truth=ground_truth,
            similarity=round(similarity, 3),
            test_case=test_case_id,
            parent_context=parent_context,
        )

        self.pending.append(annotation)
        self._existing_pairs.add(pair_key)

        return True

    def save(self) -> None:
        """Save pending annotations to file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w") as f:
            for ann in self.pending:
                f.write(json.dumps(ann.to_dict()) + "\n")

    def get_stats(self) -> dict:
        """Get annotation statistics."""
        pending = sum(1 for a in self.pending if a.status == "pending")
        verified = sum(1 for a in self.pending if a.status == "verified")
        rejected = sum(1 for a in self.pending if a.status == "rejected")

        return {
            "total": len(self.pending),
            "pending": pending,
            "verified": verified,
            "rejected": rejected,
        }

    def get_pending(self) -> list[PendingAnnotation]:
        """Get all pending annotations awaiting review."""
        return [a for a in self.pending if a.status == "pending"]

    def mark_verified(
        self,
        annotation_id: str,
        reviewer: str,
        notes: str | None = None,
    ) -> bool:
        """
        Mark an annotation as verified (gates are equivalent).

        Args:
            annotation_id: ID of the annotation
            reviewer: Name/email of the reviewer
            notes: Optional notes

        Returns:
            True if annotation was found and updated
        """
        for ann in self.pending:
            if ann.id == annotation_id:
                ann.status = "verified"
                ann.reviewed_by = reviewer
                ann.reviewed_at = datetime.now().isoformat()
                ann.notes = notes
                return True
        return False

    def mark_rejected(
        self,
        annotation_id: str,
        reviewer: str,
        notes: str | None = None,
    ) -> bool:
        """
        Mark an annotation as rejected (gates are NOT equivalent).

        Args:
            annotation_id: ID of the annotation
            reviewer: Name/email of the reviewer
            notes: Optional notes

        Returns:
            True if annotation was found and updated
        """
        for ann in self.pending:
            if ann.id == annotation_id:
                ann.status = "rejected"
                ann.reviewed_by = reviewer
                ann.reviewed_at = datetime.now().isoformat()
                ann.notes = notes
                return True
        return False


def create_enhanced_matcher(
    equivalences_path: Path | None = None,
    annotations_path: Path | None = None,
) -> tuple[EquivalenceRegistry, AnnotationCapture | None]:
    """
    Create an equivalence registry and optional annotation capture.

    Convenience function for setting up the matching system.

    Args:
        equivalences_path: Path to verified_equivalences.yaml
        annotations_path: Path to pending_annotations.jsonl (optional)

    Returns:
        Tuple of (registry, capture) where capture may be None
    """
    registry = EquivalenceRegistry(equivalences_path)

    capture = None
    if annotations_path:
        capture = AnnotationCapture(annotations_path)

    return registry, capture

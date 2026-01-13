"""
Semantic similarity matching for gate names using sentence transformers.

This module replaces manual synonym sets with embedding-based matching,
addressing the F1 bias problem where verbose naming styles are unfairly
favored over terse but biologically equivalent names.

Usage:
    from evaluation.semantic_similarity import SemanticMatcher, compute_semantic_f1

    matcher = SemanticMatcher()
    f1, precision, recall, matches = compute_semantic_f1(
        predicted_gates, ground_truth_gates, matcher
    )
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Lazy-loaded model to avoid import overhead
_model = None
_model_name = "all-MiniLM-L6-v2"


def get_embedding_model():
    """Lazy-load sentence transformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for semantic matching. "
                "Install with: pip install sentence-transformers"
            )
    return _model


def compute_embedding(text: str) -> NDArray[np.float32]:
    """Compute embedding for a single text string."""
    model = get_embedding_model()
    return model.encode(text, convert_to_numpy=True)


def compute_embeddings(texts: list[str]) -> NDArray[np.float32]:
    """Compute embeddings for multiple texts efficiently."""
    if not texts:
        return np.array([])
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(
    embeddings_a: NDArray[np.float32],
    embeddings_b: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute pairwise cosine similarities between two sets of embeddings."""
    if embeddings_a.size == 0 or embeddings_b.size == 0:
        return np.array([[]])

    # Normalize embeddings
    norms_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    # Avoid division by zero
    norms_a = np.where(norms_a == 0, 1, norms_a)
    norms_b = np.where(norms_b == 0, 1, norms_b)

    normalized_a = embeddings_a / norms_a
    normalized_b = embeddings_b / norms_b

    return np.dot(normalized_a, normalized_b.T)


@dataclass
class SemanticMatch:
    """Represents a semantic match between predicted and ground truth gates."""
    predicted: str
    ground_truth: str
    similarity: float
    match_type: str  # 'exact', 'high_similarity', 'medium_similarity'


@dataclass
class SemanticMatchResult:
    """Result of semantic matching between gate sets."""
    matches: list[SemanticMatch] = field(default_factory=list)
    unmatched_predicted: list[str] = field(default_factory=list)
    unmatched_ground_truth: list[str] = field(default_factory=list)

    # Similarity matrix for analysis
    similarity_matrix: NDArray[np.float32] | None = None
    predicted_gates: list[str] = field(default_factory=list)
    ground_truth_gates: list[str] = field(default_factory=list)


class SemanticMatcher:
    """
    Matcher for gate names using semantic embeddings.

    Uses sentence transformers to compute embeddings and find
    semantically similar gate names even when surface forms differ.

    Attributes:
        high_threshold: Similarity threshold for confident matches (default: 0.85)
        medium_threshold: Similarity threshold for possible matches (default: 0.70)
        use_cache: Whether to cache embeddings (default: True)
    """

    def __init__(
        self,
        high_threshold: float = 0.85,
        medium_threshold: float = 0.70,
        use_cache: bool = True,
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.use_cache = use_cache
        self._embedding_cache: dict[str, NDArray[np.float32]] = {}

    def _preprocess_gate_name(self, name: str) -> str:
        """Preprocess gate name for embedding."""
        # Add context to help the model understand this is a cell population
        # This improves matching for domain-specific terms
        processed = name.lower().strip()

        # Expand common abbreviations
        expansions = {
            'mono': 'monocyte',
            'lymph': 'lymphocyte',
            'dc': 'dendritic cell',
            'nk': 'natural killer',
            'treg': 'regulatory t cell',
            'th1': 't helper 1',
            'th2': 't helper 2',
            'th17': 't helper 17',
            'pdc': 'plasmacytoid dendritic cell',
            'mdc': 'myeloid dendritic cell',
        }

        for abbrev, expansion in expansions.items():
            # Only expand if it's a standalone word or at word boundary
            if processed == abbrev or processed.startswith(abbrev + ' ') or processed.endswith(' ' + abbrev):
                processed = processed.replace(abbrev, expansion, 1)

        return f"cell population: {processed}"

    def get_embedding(self, gate_name: str) -> NDArray[np.float32]:
        """Get embedding for a gate name, using cache if available."""
        if self.use_cache and gate_name in self._embedding_cache:
            return self._embedding_cache[gate_name]

        processed = self._preprocess_gate_name(gate_name)
        embedding = compute_embedding(processed)

        if self.use_cache:
            self._embedding_cache[gate_name] = embedding

        return embedding

    def get_embeddings(self, gate_names: list[str]) -> NDArray[np.float32]:
        """Get embeddings for multiple gate names efficiently."""
        if not gate_names:
            return np.array([])

        # Check cache for all names
        if self.use_cache:
            uncached = [g for g in gate_names if g not in self._embedding_cache]
            if uncached:
                processed = [self._preprocess_gate_name(g) for g in uncached]
                embeddings = compute_embeddings(processed)
                for name, emb in zip(uncached, embeddings):
                    self._embedding_cache[name] = emb

            return np.array([self._embedding_cache[g] for g in gate_names])

        processed = [self._preprocess_gate_name(g) for g in gate_names]
        return compute_embeddings(processed)

    def compute_similarity(self, gate1: str, gate2: str) -> float:
        """Compute semantic similarity between two gate names."""
        # Fast path for exact match
        if gate1.lower().strip() == gate2.lower().strip():
            return 1.0

        emb1 = self.get_embedding(gate1)
        emb2 = self.get_embedding(gate2)
        return cosine_similarity(emb1, emb2)

    def find_best_match(
        self,
        gate: str,
        candidates: list[str],
        min_threshold: float | None = None,
    ) -> tuple[str | None, float]:
        """
        Find the best matching gate from candidates.

        Args:
            gate: Gate name to match
            candidates: List of candidate gate names
            min_threshold: Minimum similarity threshold (defaults to medium_threshold)

        Returns:
            Tuple of (best_match, similarity) or (None, 0.0) if no match above threshold
        """
        if not candidates:
            return None, 0.0

        threshold = min_threshold if min_threshold is not None else self.medium_threshold

        gate_emb = self.get_embedding(gate)
        candidate_embs = self.get_embeddings(candidates)

        similarities = np.array([
            cosine_similarity(gate_emb, c_emb) for c_emb in candidate_embs
        ])

        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]

        if best_sim >= threshold:
            return candidates[best_idx], float(best_sim)
        return None, 0.0

    def match_gate_sets(
        self,
        predicted: set[str] | list[str],
        ground_truth: set[str] | list[str],
    ) -> SemanticMatchResult:
        """
        Match predicted gates to ground truth using semantic similarity.

        Uses Hungarian algorithm for optimal assignment when sets are similar size,
        or greedy matching for large sets.

        Args:
            predicted: Predicted gate names
            ground_truth: Ground truth gate names

        Returns:
            SemanticMatchResult with matches and unmatched gates
        """
        pred_list = list(predicted)
        gt_list = list(ground_truth)

        if not pred_list or not gt_list:
            return SemanticMatchResult(
                unmatched_predicted=pred_list,
                unmatched_ground_truth=gt_list,
                predicted_gates=pred_list,
                ground_truth_gates=gt_list,
            )

        # Compute all pairwise similarities
        pred_embs = self.get_embeddings(pred_list)
        gt_embs = self.get_embeddings(gt_list)
        sim_matrix = cosine_similarity_matrix(pred_embs, gt_embs)

        # Use Hungarian algorithm for optimal matching
        matches, matched_pred_idx, matched_gt_idx = self._hungarian_match(
            sim_matrix, pred_list, gt_list
        )

        unmatched_pred = [
            pred_list[i] for i in range(len(pred_list))
            if i not in matched_pred_idx
        ]
        unmatched_gt = [
            gt_list[i] for i in range(len(gt_list))
            if i not in matched_gt_idx
        ]

        return SemanticMatchResult(
            matches=matches,
            unmatched_predicted=unmatched_pred,
            unmatched_ground_truth=unmatched_gt,
            similarity_matrix=sim_matrix,
            predicted_gates=pred_list,
            ground_truth_gates=gt_list,
        )

    def _hungarian_match(
        self,
        sim_matrix: NDArray[np.float32],
        pred_list: list[str],
        gt_list: list[str],
    ) -> tuple[list[SemanticMatch], set[int], set[int]]:
        """Use Hungarian algorithm for optimal bipartite matching."""
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            # Fall back to greedy matching
            return self._greedy_match(sim_matrix, pred_list, gt_list)

        # Convert similarities to costs (Hungarian minimizes)
        cost_matrix = 1 - sim_matrix

        # Handle non-square matrices by padding
        n_pred, n_gt = sim_matrix.shape
        if n_pred != n_gt:
            max_dim = max(n_pred, n_gt)
            padded_cost = np.ones((max_dim, max_dim))
            padded_cost[:n_pred, :n_gt] = cost_matrix
            cost_matrix = padded_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        matched_pred_idx = set()
        matched_gt_idx = set()

        for pred_idx, gt_idx in zip(row_ind, col_ind):
            # Skip padding matches
            if pred_idx >= n_pred or gt_idx >= n_gt:
                continue

            similarity = sim_matrix[pred_idx, gt_idx]

            # Only accept matches above threshold
            if similarity >= self.medium_threshold:
                if similarity >= self.high_threshold:
                    match_type = 'high_similarity'
                else:
                    match_type = 'medium_similarity'

                # Check for exact match
                if pred_list[pred_idx].lower().strip() == gt_list[gt_idx].lower().strip():
                    match_type = 'exact'
                    similarity = 1.0

                matches.append(SemanticMatch(
                    predicted=pred_list[pred_idx],
                    ground_truth=gt_list[gt_idx],
                    similarity=float(similarity),
                    match_type=match_type,
                ))
                matched_pred_idx.add(pred_idx)
                matched_gt_idx.add(gt_idx)

        return matches, matched_pred_idx, matched_gt_idx

    def _greedy_match(
        self,
        sim_matrix: NDArray[np.float32],
        pred_list: list[str],
        gt_list: list[str],
    ) -> tuple[list[SemanticMatch], set[int], set[int]]:
        """Fallback greedy matching when scipy is not available."""
        matches = []
        matched_pred_idx: set[int] = set()
        matched_gt_idx: set[int] = set()

        # Get all similarities above threshold, sorted by similarity
        candidates = []
        for i in range(len(pred_list)):
            for j in range(len(gt_list)):
                if sim_matrix[i, j] >= self.medium_threshold:
                    candidates.append((sim_matrix[i, j], i, j))

        candidates.sort(reverse=True)

        for similarity, pred_idx, gt_idx in candidates:
            if pred_idx in matched_pred_idx or gt_idx in matched_gt_idx:
                continue

            if similarity >= self.high_threshold:
                match_type = 'high_similarity'
            else:
                match_type = 'medium_similarity'

            if pred_list[pred_idx].lower().strip() == gt_list[gt_idx].lower().strip():
                match_type = 'exact'

            matches.append(SemanticMatch(
                predicted=pred_list[pred_idx],
                ground_truth=gt_list[gt_idx],
                similarity=float(similarity),
                match_type=match_type,
            ))
            matched_pred_idx.add(pred_idx)
            matched_gt_idx.add(gt_idx)

        return matches, matched_pred_idx, matched_gt_idx

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()


def compute_semantic_f1(
    predicted: set[str] | list[str],
    ground_truth: set[str] | list[str],
    matcher: SemanticMatcher | None = None,
) -> tuple[float, float, float, SemanticMatchResult]:
    """
    Compute F1 score using semantic similarity matching.

    Args:
        predicted: Predicted gate names
        ground_truth: Ground truth gate names
        matcher: Optional SemanticMatcher instance (created if not provided)

    Returns:
        Tuple of (f1, precision, recall, match_result)
    """
    if matcher is None:
        matcher = SemanticMatcher()

    result = matcher.match_gate_sets(predicted, ground_truth)

    n_pred = len(list(predicted))
    n_gt = len(list(ground_truth))
    n_matched = len(result.matches)

    precision = n_matched / n_pred if n_pred > 0 else 0.0
    recall = n_matched / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, precision, recall, result


def compute_weighted_semantic_f1(
    predicted: set[str] | list[str],
    ground_truth: set[str] | list[str],
    matcher: SemanticMatcher | None = None,
) -> tuple[float, float, float, SemanticMatchResult]:
    """
    Compute F1 score weighted by match confidence.

    Higher similarity matches contribute more to the score than
    lower similarity (but still above threshold) matches.

    Args:
        predicted: Predicted gate names
        ground_truth: Ground truth gate names
        matcher: Optional SemanticMatcher instance

    Returns:
        Tuple of (weighted_f1, weighted_precision, weighted_recall, match_result)
    """
    if matcher is None:
        matcher = SemanticMatcher()

    result = matcher.match_gate_sets(predicted, ground_truth)

    n_pred = len(list(predicted))
    n_gt = len(list(ground_truth))

    if not result.matches:
        return 0.0, 0.0, 0.0, result

    # Sum of similarities for matched pairs
    total_similarity = sum(m.similarity for m in result.matches)

    # Weighted precision: average similarity of matches / predicted count
    weighted_precision = total_similarity / n_pred if n_pred > 0 else 0.0

    # Weighted recall: average similarity of matches / ground truth count
    weighted_recall = total_similarity / n_gt if n_gt > 0 else 0.0

    weighted_f1 = (
        2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
        if (weighted_precision + weighted_recall) > 0 else 0.0
    )

    return weighted_f1, weighted_precision, weighted_recall, result


# Convenience function for quick similarity checks
@functools.lru_cache(maxsize=1000)
def are_semantically_equivalent(gate1: str, gate2: str, threshold: float = 0.85) -> bool:
    """
    Check if two gate names are semantically equivalent.

    Cached for performance on repeated comparisons.

    Args:
        gate1: First gate name
        gate2: Second gate name
        threshold: Similarity threshold for equivalence

    Returns:
        True if gates are semantically equivalent
    """
    # Fast path for exact match
    if gate1.lower().strip() == gate2.lower().strip():
        return True

    matcher = SemanticMatcher(high_threshold=threshold, use_cache=True)
    similarity = matcher.compute_similarity(gate1, gate2)
    return similarity >= threshold

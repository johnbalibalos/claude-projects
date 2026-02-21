"""
Inter-judge agreement metrics for LLM-as-Judge evaluation.

Implements Cohen's kappa (2 judges) and Fleiss' kappa (multiple judges)
along with exact/majority agreement rates and pairwise correlations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InterJudgeAgreement:
    """Inter-judge agreement metrics."""

    n_judges: int
    n_items: int
    exact_agreement_rate: float  # All judges agree exactly
    majority_agreement_rate: float  # Majority agrees
    cohens_kappa: float  # For 2 judges
    fleiss_kappa: float  # For multiple judges
    correlation: float  # Pearson correlation of scores


def compute_inter_judge_agreement(
    judgments: list[list[float]],  # [judge][item] -> score
) -> InterJudgeAgreement:
    """
    Compute inter-judge agreement metrics.

    Args:
        judgments: List of score lists, one per judge

    Returns:
        InterJudgeAgreement with various agreement metrics
    """
    judgments_array = np.array(judgments)
    n_judges, n_items = judgments_array.shape

    # Exact agreement: all judges give same score (within tolerance)
    exact_agreements = 0
    for i in range(n_items):
        item_scores = judgments_array[:, i]
        if np.max(item_scores) - np.min(item_scores) < 0.5:
            exact_agreements += 1
    exact_agreement_rate = exact_agreements / n_items

    # Majority agreement
    majority_agreements = 0
    for i in range(n_items):
        item_scores = judgments_array[:, i]
        # Round to nearest integer for majority calculation
        rounded = np.round(item_scores)
        _unique, counts = np.unique(rounded, return_counts=True)
        if np.max(counts) > n_judges / 2:
            majority_agreements += 1
    majority_agreement_rate = majority_agreements / n_items

    # Cohen's kappa (for 2 judges)
    if n_judges == 2:
        kappa = _cohens_kappa(judgments_array[0], judgments_array[1])
    else:
        kappa = 0.0

    # Fleiss' kappa
    fleiss = _fleiss_kappa(judgments_array)

    # Correlation (average pairwise)
    correlations = []
    for i in range(n_judges):
        for j in range(i + 1, n_judges):
            corr = np.corrcoef(judgments_array[i], judgments_array[j])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    avg_correlation = float(np.mean(correlations)) if correlations else 0.0

    return InterJudgeAgreement(
        n_judges=n_judges,
        n_items=n_items,
        exact_agreement_rate=exact_agreement_rate,
        majority_agreement_rate=majority_agreement_rate,
        cohens_kappa=kappa,
        fleiss_kappa=fleiss,
        correlation=avg_correlation,
    )


def _cohens_kappa(scores1: np.ndarray, scores2: np.ndarray) -> float:
    """Compute Cohen's kappa for two raters."""
    # Discretize scores
    categories = np.unique(np.concatenate([scores1, scores2]))
    n = len(scores1)

    # Build confusion matrix
    matrix = np.zeros((len(categories), len(categories)))
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    for s1, s2 in zip(scores1, scores2):
        i, j = cat_to_idx[s1], cat_to_idx[s2]
        matrix[i, j] += 1

    # Observed agreement
    po = np.trace(matrix) / n

    # Expected agreement
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    pe = np.sum(row_sums * col_sums) / (n * n)

    if pe == 1:
        return 1.0

    return float((po - pe) / (1 - pe))


def _fleiss_kappa(ratings: np.ndarray) -> float:
    """Compute Fleiss' kappa for multiple raters."""
    n_subjects, n_raters = ratings.shape

    # Get unique categories
    categories = np.unique(ratings)
    n_categories = len(categories)

    # Build rating matrix: n_subjects x n_categories
    # Each cell = number of raters who assigned that category
    rating_matrix = np.zeros((n_subjects, n_categories))
    for i, cat in enumerate(categories):
        rating_matrix[:, i] = np.sum(ratings == cat, axis=0)

    # P_i for each subject
    n = n_raters
    P_i = (np.sum(rating_matrix ** 2, axis=1) - n) / (n * (n - 1))

    # P_bar (mean of P_i)
    P_bar = np.mean(P_i)

    # p_j for each category
    p_j = np.sum(rating_matrix, axis=0) / (n_subjects * n)

    # P_e_bar
    P_e_bar = np.sum(p_j ** 2)

    if P_e_bar == 1:
        return 1.0

    return float((P_bar - P_e_bar) / (1 - P_e_bar))


"""Tests for cosine similarity calculations."""

import numpy as np
import pytest

from flow_panel_optimizer.spectral.similarity import (
    cosine_similarity,
    build_similarity_matrix,
    find_high_similarity_pairs,
    similarity_risk_level,
)


class TestCosineSimilarity:
    """Tests for the cosine_similarity function."""

    def test_identical_spectra_return_one(self):
        """Identical spectra should have similarity of 1.0."""
        spectrum = np.array([0.1, 0.5, 1.0, 0.5, 0.1])
        assert cosine_similarity(spectrum, spectrum) == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal_spectra_return_zero(self):
        """Orthogonal spectra should have similarity of 0.0."""
        spec_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        spec_b = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        assert cosine_similarity(spec_a, spec_b) == pytest.approx(0.0, abs=1e-10)

    def test_high_similarity_spectra(self, high_similarity_spectra):
        """Similar spectra should have high similarity (>0.95)."""
        spec_a, spec_b = high_similarity_spectra
        sim = cosine_similarity(spec_a, spec_b)
        assert sim > 0.95
        assert sim < 1.0

    def test_low_similarity_spectra(self, low_similarity_spectra):
        """Dissimilar spectra should have low similarity (<0.5)."""
        spec_a, spec_b = low_similarity_spectra
        sim = cosine_similarity(spec_a, spec_b)
        assert sim < 0.5

    def test_symmetric(self):
        """Similarity should be symmetric: sim(a,b) == sim(b,a)."""
        spec_a = np.array([0.2, 0.5, 1.0, 0.3])
        spec_b = np.array([0.1, 0.8, 0.6, 0.4])
        assert cosine_similarity(spec_a, spec_b) == pytest.approx(
            cosine_similarity(spec_b, spec_a), abs=1e-10
        )

    def test_different_lengths_raises_error(self):
        """Should raise ValueError for different length arrays."""
        spec_a = np.array([1.0, 2.0, 3.0])
        spec_b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="same length"):
            cosine_similarity(spec_a, spec_b)

    def test_zero_vector_returns_zero(self):
        """Zero vector should return 0 similarity."""
        spec_a = np.array([0.0, 0.0, 0.0])
        spec_b = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(spec_a, spec_b) == 0.0

    def test_with_spectrum_objects(self, fitc_like_spectrum, pe_like_spectrum):
        """Should work with Spectrum objects."""
        sim = cosine_similarity(fitc_like_spectrum, pe_like_spectrum)
        assert 0 <= sim <= 1

    def test_scale_invariant(self):
        """Similarity should be scale-invariant."""
        spec_a = np.array([1.0, 2.0, 3.0])
        spec_b = np.array([1.0, 2.0, 3.0])
        spec_c = np.array([2.0, 4.0, 6.0])  # spec_b * 2

        assert cosine_similarity(spec_a, spec_b) == pytest.approx(
            cosine_similarity(spec_a, spec_c), abs=1e-10
        )


class TestBuildSimilarityMatrix:
    """Tests for building similarity matrices."""

    def test_matrix_shape(self, test_spectra_dict):
        """Matrix should be NxN."""
        names, matrix = build_similarity_matrix(test_spectra_dict)
        n = len(test_spectra_dict)
        assert matrix.shape == (n, n)
        assert len(names) == n

    def test_diagonal_is_one(self, test_spectra_dict):
        """Diagonal should be all 1.0 (self-similarity)."""
        names, matrix = build_similarity_matrix(test_spectra_dict)
        for i in range(len(names)):
            assert matrix[i, i] == pytest.approx(1.0, abs=1e-10)

    def test_symmetric(self, test_spectra_dict):
        """Matrix should be symmetric."""
        names, matrix = build_similarity_matrix(test_spectra_dict)
        assert np.allclose(matrix, matrix.T)

    def test_preserves_names_order(self, test_spectra_dict):
        """Names should be in insertion order."""
        names, matrix = build_similarity_matrix(test_spectra_dict)
        assert names == list(test_spectra_dict.keys())

    def test_single_fluorophore(self):
        """Should handle single fluorophore."""
        spectra = {"FITC": np.array([0.5, 1.0, 0.5])}
        names, matrix = build_similarity_matrix(spectra)
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 1.0


class TestFindHighSimilarityPairs:
    """Tests for finding high similarity pairs."""

    def test_finds_high_pairs(self, test_spectra_dict):
        """Should find FITC-BB515 as high similarity."""
        names, matrix = build_similarity_matrix(test_spectra_dict)
        pairs = find_high_similarity_pairs(matrix, names, threshold=0.90)

        # FITC and BB515 should be in the high pairs
        pair_names = [(a, b) for a, b, _ in pairs]
        assert any(
            ("FITC" in pair and "BB515" in pair)
            for pair in pair_names
        )

    def test_excludes_low_pairs(self, test_spectra_dict):
        """Should exclude pairs below threshold."""
        names, matrix = build_similarity_matrix(test_spectra_dict)
        pairs = find_high_similarity_pairs(matrix, names, threshold=0.90)

        # APC and FITC should NOT be in high pairs (too different)
        for a, b, sim in pairs:
            assert sim >= 0.90

    def test_sorted_descending(self, test_spectra_dict):
        """Results should be sorted by similarity descending."""
        names, matrix = build_similarity_matrix(test_spectra_dict)
        pairs = find_high_similarity_pairs(matrix, names, threshold=0.50)

        if len(pairs) >= 2:
            similarities = [sim for _, _, sim in pairs]
            assert similarities == sorted(similarities, reverse=True)

    def test_no_self_pairs(self, test_spectra_dict):
        """Should not include self-pairs."""
        names, matrix = build_similarity_matrix(test_spectra_dict)
        pairs = find_high_similarity_pairs(matrix, names, threshold=0.90)

        for a, b, _ in pairs:
            assert a != b


class TestSimilarityRiskLevel:
    """Tests for risk level classification."""

    def test_critical_threshold(self):
        """>= 0.98 should be critical."""
        assert similarity_risk_level(0.99) == "critical"
        assert similarity_risk_level(0.98) == "critical"

    def test_high_threshold(self):
        """>= 0.95 (but < 0.98) should be high."""
        assert similarity_risk_level(0.97) == "high"
        assert similarity_risk_level(0.95) == "high"

    def test_moderate_threshold(self):
        """>= 0.90 (but < 0.95) should be moderate."""
        assert similarity_risk_level(0.94) == "moderate"
        assert similarity_risk_level(0.90) == "moderate"

    def test_low_threshold(self):
        """>= 0.80 (but < 0.90) should be low."""
        assert similarity_risk_level(0.85) == "low"
        assert similarity_risk_level(0.80) == "low"

    def test_minimal_threshold(self):
        """< 0.80 should be minimal."""
        assert similarity_risk_level(0.79) == "minimal"
        assert similarity_risk_level(0.50) == "minimal"

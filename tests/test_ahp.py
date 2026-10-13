"""Tests for the AHP weighting."""

import numpy as np
import pytest

from everglades_flood import ahp


def test_matrix_is_reciprocal():
    matrix = ahp.build_matrix()
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix * matrix.T, 1.0)


def test_matrix_has_declared_judgements():
    matrix = ahp.build_matrix()
    i = ahp.FACTORS.index("sar_water_frequency")
    j = ahp.FACTORS.index("landcover")
    assert matrix[i, j] == 7
    assert matrix[j, i] == pytest.approx(1 / 7)


def test_perfectly_consistent_matrix_has_zero_inconsistency():
    """A matrix built from a true ratio scale must give CR = 0 exactly."""
    priorities = np.array([0.5, 0.25, 0.15, 0.10])
    matrix = priorities[:, None] / priorities[None, :]

    result = ahp.solve(matrix, factors=list("abcd"))

    assert result.lambda_max == pytest.approx(4.0, abs=1e-9)
    assert result.consistency_ratio == pytest.approx(0.0, abs=1e-9)
    assert result.is_consistent
    assert np.allclose(result.weights, priorities)


def test_weights_sum_to_one():
    result = ahp.default_weights()
    assert sum(result.as_dict().values()) == pytest.approx(1.0)


def test_default_judgements_are_consistent():
    """The model's own weights must pass Saaty's CR < 0.10 test.

    If a future edit to PAIRWISE_JUDGEMENTS contradicts itself, this fails
    rather than silently producing weights nobody can defend.
    """
    result = ahp.default_weights()
    assert result.is_consistent, result.report()
    assert result.consistency_ratio < 0.10


def test_declared_ranking_is_respected():
    """The weights must order the factors the way the judgements claim."""
    weights = ahp.default_weights().as_dict()
    assert weights["sar_water_frequency"] > weights["elevation"]
    assert weights["elevation"] > weights["twi"]
    assert weights["twi"] > weights["slope"]
    assert weights["slope"] > weights["landcover"]


def test_grossly_inconsistent_matrix_is_flagged():
    """a > b > c > a is a cycle; AHP must notice."""
    matrix = np.array([
        [1.0, 9.0, 1 / 9],
        [1 / 9, 1.0, 9.0],
        [9.0, 1 / 9, 1.0],
    ])
    result = ahp.solve(matrix, factors=list("abc"))
    assert not result.is_consistent


def test_unknown_factor_raises():
    with pytest.raises(KeyError):
        ahp.build_matrix(judgements={("nonsense", "elevation"): 3})


def test_non_positive_judgement_raises():
    with pytest.raises(ValueError):
        ahp.build_matrix(judgements={("elevation", "slope"): 0})


def test_non_square_matrix_raises():
    with pytest.raises(ValueError):
        ahp.solve(np.ones((3, 4)))


def test_report_mentions_consistency():
    text = ahp.default_weights().report()
    assert "consistency ratio" in text
    assert "Sentinel-1 water frequency" in text

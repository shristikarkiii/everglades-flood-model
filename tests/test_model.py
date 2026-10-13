"""Tests for normalisation, overlay and classification."""

import numpy as np
import pytest

from everglades_flood import model


def _mask(shape):
    return np.ones(shape, dtype=bool)


def test_normalise_spans_zero_to_one():
    array = np.linspace(10, 20, 101).reshape(101, 1)
    out = model.normalise(array, _mask(array.shape), percentiles=(0, 100))
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_normalise_inverts():
    array = np.array([[0.0, 5.0, 10.0]])
    out = model.normalise(array, _mask(array.shape), invert=True,
                          percentiles=(0, 100))
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 2] == pytest.approx(0.0)


def test_percentile_clipping_resists_a_single_outlier():
    """One extreme cell must not collapse the rest of the distribution.

    A single cell six orders of magnitude above the others -- which is exactly
    what raw flow accumulation looks like -- breaks plain min-max scaling, so
    that every other pixel lands indistinguishably near zero.
    """
    array = np.concatenate([np.linspace(0, 10, 999), [1e6]]).reshape(-1, 1)
    mask = _mask(array.shape)

    naive = model.normalise(array, mask, percentiles=(0, 100))
    assert (naive[:-1] < 0.01).mean() > 0.99      # everything but the outlier

    clipped = model.normalise(array, mask, percentiles=(1, 99))
    assert 0.3 < float(np.median(clipped)) < 0.7  # spread across the range


def test_normalise_masks_outside_pixels():
    array = np.arange(9, dtype="float64").reshape(3, 3)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, :] = True
    out = model.normalise(array, mask, percentiles=(0, 100))
    assert np.isnan(out[0, 0])
    assert np.isfinite(out[1, 1])


def test_normalise_rejects_a_degenerate_range():
    array = np.full((5, 5), 3.0)
    with pytest.raises(ValueError):
        model.normalise(array, _mask(array.shape))


def test_normalise_rejects_an_empty_mask():
    with pytest.raises(ValueError):
        model.normalise(np.zeros((4, 4)), np.zeros((4, 4), dtype=bool))


def test_index_is_a_weighted_mean():
    factors = {"a": np.full((2, 2), 1.0), "b": np.full((2, 2), 0.0)}
    weights = {"a": 0.75, "b": 0.25}
    index = model.susceptibility_index(factors, weights, _mask((2, 2)))
    assert np.allclose(index, 0.75)


def test_weights_are_renormalised_over_supplied_factors():
    """Dropping a factor must not systematically lower the index.

    The terrain-only ablation drops the SAR term; if the remaining weights
    were not rescaled it would score lower purely because a term went missing,
    and the comparison against the full model would be meaningless.
    """
    factors = {"a": np.full((2, 2), 0.8), "b": np.full((2, 2), 0.8)}
    weights = {"a": 0.5, "b": 0.2, "dropped": 0.3}
    index = model.susceptibility_index(factors, weights, _mask((2, 2)))
    assert np.allclose(index, 0.8)


def test_missing_pixels_fall_back_to_available_factors():
    a = np.array([[1.0, 1.0]])
    b = np.array([[0.0, np.nan]])
    index = model.susceptibility_index(
        {"a": a, "b": b}, {"a": 0.5, "b": 0.5}, _mask((1, 2)))
    assert index[0, 0] == pytest.approx(0.5)
    assert index[0, 1] == pytest.approx(1.0)   # only 'a' contributed


def test_index_requires_a_weight_for_every_factor():
    with pytest.raises(KeyError):
        model.susceptibility_index(
            {"a": np.zeros((2, 2))}, {"b": 1.0}, _mask((2, 2)))


def test_classify_populates_every_class():
    rng = np.random.default_rng(0)
    index = rng.random((200, 200)).astype("float32")
    classes, summary = model.classify(index, _mask(index.shape), n_classes=5)

    assert set(np.unique(classes)) == {1, 2, 3, 4, 5}
    assert len(summary) == 5
    for row in summary:
        assert row["share"] == pytest.approx(0.2, abs=0.01)


def test_classify_handles_a_skewed_index():
    """Quantile classes stay populated where equal intervals would not.

    This distribution is heavily right-skewed -- most of the mass sits near
    zero, which is what a flood susceptibility index actually looks like.
    Equal-interval breaks would leave the upper classes nearly empty.
    """
    rng = np.random.default_rng(3)
    index = rng.exponential(0.05, 40_000).astype("float32").reshape(-1, 1)
    classes, summary = model.classify(index, _mask(index.shape))

    assert all(row["pixels"] > 0 for row in summary)
    for row in summary:
        assert row["share"] == pytest.approx(0.2, abs=0.02)


def test_classify_warns_when_the_index_is_heavily_tied():
    """Ties can make quantile classes collapse; that must not pass silently."""
    index = np.concatenate([
        np.full(9000, 0.01), np.linspace(0.02, 1.0, 1000)
    ]).reshape(-1, 1).astype("float32")

    with pytest.warns(RuntimeWarning, match="heavily tied"):
        model.classify(index, _mask(index.shape))


def test_classify_leaves_masked_pixels_as_zero():
    index = np.random.default_rng(1).random((50, 50)).astype("float32")
    mask = np.zeros((50, 50), dtype=bool)
    mask[:25] = True
    classes, _ = model.classify(index, mask)
    assert (classes[25:] == 0).all()
    assert (classes[:25] > 0).all()

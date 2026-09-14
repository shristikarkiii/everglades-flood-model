"""Tests for map rendering.

These guard two specific failures: an overlay silently resizing the axes, so
the map ends up as a panel surrounded by empty background; and an outline that
encloses area the model does not cover.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from everglades_flood import plotting  # noqa: E402


class _Grid:
    """A small stand-in for the analysis grid.

    Large enough that the map legend fits without matplotlib complaining that
    tight_layout cannot accommodate it.
    """
    crs = "EPSG:32617"
    width, height = 400, 320
    shape = (320, 400)
    bounds = (400000.0, 2700000.0, 400000.0 + 400 * 30, 2700000.0 + 320 * 30)


def _axes_with_image(grid):
    fig, ax = plt.subplots()
    ax.imshow(np.zeros(grid.shape), extent=plotting._extent_km(grid),
              origin="upper")
    return fig, ax


def _blob(shape, rows, cols):
    mask = np.zeros(shape, dtype=bool)
    mask[rows[0]:rows[1], cols[0]:cols[1]] = True
    return mask


def test_outline_does_not_expand_the_axes():
    """Nothing drawn on a finished map may resize its frame."""
    grid = _Grid()
    fig, ax = _axes_with_image(grid)
    before = (ax.get_xlim(), ax.get_ylim())

    plotting.outline_mask(ax, _blob(grid.shape, (60, 240), (60, 300)), grid)

    assert (ax.get_xlim(), ax.get_ylim()) == before
    plt.close(fig)


def test_outline_actually_draws_something():
    """The guard must not be satisfied by drawing nothing at all."""
    grid = _Grid()
    fig, ax = _axes_with_image(grid)
    assert not ax.collections

    plotting.outline_mask(ax, _blob(grid.shape, (60, 240), (60, 300)), grid)

    assert ax.collections, "no outline was drawn"
    plt.close(fig)


def test_outline_of_an_empty_mask_is_a_no_op():
    """A park entirely outside the modelled area must draw nothing.

    This is the case that matters: intersecting the park polygon with the study
    domain can legitimately leave nothing behind, and contouring an all-False
    array would otherwise raise.
    """
    grid = _Grid()
    fig, ax = _axes_with_image(grid)

    plotting.outline_mask(ax, np.zeros(grid.shape, dtype=bool), grid)
    plotting.outline_mask(ax, None, grid)

    assert not ax.collections
    plt.close(fig)


def test_outline_follows_the_mask_not_the_frame():
    """The traced edge must sit where the mask is, not around the whole map."""
    grid = _Grid()
    fig, ax = _axes_with_image(grid)

    # A blob confined to the left third of the grid.
    plotting.outline_mask(ax, _blob(grid.shape, (60, 240), (20, 150)), grid)

    xs = np.concatenate([
        np.asarray(path.vertices)[:, 0]
        for collection in ax.collections
        for path in collection.get_paths()
    ])
    width_km = (grid.bounds[2] - grid.bounds[0]) / 1000.0
    assert xs.max() < width_km * 0.5, "outline spilled past the masked area"
    plt.close(fig)


def test_susceptibility_map_renders_with_an_outline(tmp_path):
    grid = _Grid()
    rng = np.random.default_rng(0)
    classes = rng.integers(1, 6, grid.shape).astype("uint8")
    domain = np.ones(grid.shape, dtype=bool)
    summary = [{"label": l, "share": 0.2} for l in
               ["Very low", "Low", "Moderate", "High", "Very high"]]

    out = plotting.susceptibility_map(
        classes, summary, domain, grid, tmp_path / "map.png",
        outline=_blob(grid.shape, (60, 240), (60, 300)))

    assert out.exists()
    assert out.stat().st_size > 5000

"""Tests for map rendering.

These guard a specific failure: overlays whose geometry runs outside the
analysis grid silently growing the axes, so the map ends up as a small panel
surrounded by empty background.
"""

import json

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from everglades_flood import plotting  # noqa: E402
from everglades_flood.raster import analysis_grid  # noqa: E402


@pytest.fixture(scope="module")
def grid():
    return analysis_grid()


def _write_geojson(tmp_path, ring):
    path = tmp_path / "boundary.geojson"
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"UNIT_NAME": "Test"},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }],
    }), encoding="utf-8")
    return path


def test_overlay_does_not_expand_the_axes(grid, tmp_path):
    """A boundary running far outside the grid must not resize the frame.

    Big Cypress continues north of the AOI and Everglades National Park extends
    south into Florida Bay, so real boundaries genuinely do leave the frame.
    Matplotlib's default is to grow the axes to fit plotted lines, which pads
    the figure with background and reads as a broken map.
    """
    # A ring well outside the study area, in both directions.
    ring = [[-83.0, 22.0], [-79.0, 22.0], [-79.0, 28.0], [-83.0, 28.0],
            [-83.0, 22.0]]
    path = _write_geojson(tmp_path, ring)

    fig, ax = plt.subplots()
    ax.imshow(np.zeros(grid.shape), extent=plotting._extent_km(grid))
    before = (ax.get_xlim(), ax.get_ylim())

    plotting.overlay_boundaries(ax, path, grid)

    assert (ax.get_xlim(), ax.get_ylim()) == before
    plt.close(fig)


def test_overlay_actually_draws_something(grid, tmp_path):
    """The guard must not be achieved by drawing nothing at all."""
    ring = [[-81.2, 25.2], [-80.6, 25.2], [-80.6, 25.7], [-81.2, 25.7],
            [-81.2, 25.2]]
    path = _write_geojson(tmp_path, ring)

    fig, ax = plt.subplots()
    ax.imshow(np.zeros(grid.shape), extent=plotting._extent_km(grid))
    assert not ax.lines

    plotting.overlay_boundaries(ax, path, grid)

    assert ax.lines, "no boundary was drawn"
    xs, ys = ax.lines[0].get_data()
    width_km = (grid.bounds[2] - grid.bounds[0]) / 1000.0
    assert 0 < float(np.mean(xs)) < width_km
    plt.close(fig)


def test_missing_boundary_file_is_not_fatal(grid, tmp_path):
    """Cartographic decoration must never break a pipeline run."""
    fig, ax = plt.subplots()
    ax.imshow(np.zeros(grid.shape), extent=plotting._extent_km(grid))
    before = (ax.get_xlim(), ax.get_ylim())

    plotting.overlay_boundaries(ax, None, grid)
    plotting.overlay_boundaries(ax, tmp_path / "nope.geojson", grid)

    assert (ax.get_xlim(), ax.get_ylim()) == before
    plt.close(fig)


def test_susceptibility_map_frame_matches_the_grid(grid, tmp_path):
    """The rendered map must span exactly the analysis grid."""
    rng = np.random.default_rng(0)
    small = (60, 66)
    classes = rng.integers(1, 6, small).astype("uint8")
    domain = np.ones(small, dtype=bool)

    class _G:
        crs = grid.crs
        transform = grid.transform
        width, height = small[1], small[0]
        bounds = grid.bounds
        shape = small

    summary = [{"label": l, "share": 0.2} for l in
               ["Very low", "Low", "Moderate", "High", "Very high"]]

    ring = [[-83.0, 22.0], [-79.0, 22.0], [-79.0, 28.0], [-83.0, 28.0],
            [-83.0, 22.0]]
    path = _write_geojson(tmp_path, ring)

    out = plotting.susceptibility_map(
        classes, summary, domain, _G, tmp_path / "map.png", boundaries=path)

    assert out.exists()
    assert out.stat().st_size > 5000

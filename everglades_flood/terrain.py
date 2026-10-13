"""Terrain and hydrological derivatives.

A standing caveat applies to everything in this module. The Everglades falls
on the order of 3 cm per kilometre. Every algorithm here was designed for
hillslopes, and on ground this flat the quantities they compute are much closer
to the noise floor of the DEM than they would be in terrain with relief. The
derivatives are still informative -- relative differences of a few centimetres
genuinely do decide where water sits in a wetland -- but they are reported with
that in mind, and the weighting in `ahp.py` deliberately ranks slope low for
this reason.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import config


def slope_radians(dem, pixel_size=None):
    """Slope by Horn's (1981) method, in radians.

    Horn's kernel weights the diagonal neighbours at half the cardinal ones,
    which makes it markedly less sensitive to single-pixel DEM noise than a
    simple central difference. On this terrain that matters more than usual.
    """
    pixel_size = pixel_size or config.PIXEL_SIZE

    filled = _fill_nan_for_filtering(dem)

    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float64")
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype="float64")

    dz_dx = ndimage.convolve(filled, kernel_x, mode="nearest") / (8.0 * pixel_size)
    dz_dy = ndimage.convolve(filled, kernel_y, mode="nearest") / (8.0 * pixel_size)

    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    slope[~np.isfinite(dem)] = np.nan
    return slope.astype("float32")


def _fill_nan_for_filtering(array):
    """Replace NaN with the nearest finite value so convolution stays defined."""
    array = np.asarray(array, dtype="float64")
    mask = ~np.isfinite(array)
    if not mask.any():
        return array
    idx = ndimage.distance_transform_edt(
        mask, return_distances=False, return_indices=True
    )
    return array[tuple(idx)]


def _patch_numpy_for_pysheds():
    """Restore np.in1d for pysheds.

    pysheds 0.4 calls numpy.in1d, which NumPy 2.0 removed in favour of
    numpy.isin. Downgrading NumPy is not an option here because rasterio 1.5
    is built against 2.x, so the alias is restored instead.

    The two are equivalent at the one call site that matters:
    `np.in1d(fdir.ravel(), dirmap).reshape(fdir.shape)` passes an already
    flattened array, and np.isin on a 1-D input returns the same 1-D result.
    """
    if not hasattr(np, "in1d"):
        np.in1d = np.isin


def hydrology(dem, pixel_size=None):
    """Fill sinks, route flow and accumulate, using pysheds.

    Returns a dict with `filled`, `flow_direction` and `flow_accumulation`
    (in cells).
    """
    import tempfile
    from pathlib import Path

    import rasterio

    _patch_numpy_for_pysheds()
    from pysheds.grid import Grid as PyshedsGrid

    pixel_size = pixel_size or config.PIXEL_SIZE

    # pysheds reads from a file and carries its own georeferencing, so the DEM
    # goes out to a temporary GeoTIFF with an explicit nodata value.
    nodata = -32768.0
    work = np.where(np.isfinite(dem), dem, nodata).astype("float32")

    tmpdir = Path(tempfile.mkdtemp(prefix="everglades_hydro_"))
    tmp = tmpdir / "dem.tif"
    profile = {
        "driver": "GTiff", "height": work.shape[0], "width": work.shape[1],
        "count": 1, "dtype": "float32", "nodata": nodata,
        "crs": config.ANALYSIS_CRS,
        "transform": rasterio.transform.from_origin(0, 0, pixel_size, pixel_size),
    }
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(work, 1)

    grid = PyshedsGrid.from_raster(str(tmp))
    raster = grid.read_raster(str(tmp))

    print("      filling pits...")
    pit_filled = grid.fill_pits(raster)
    print("      filling depressions...")
    depression_filled = grid.fill_depressions(pit_filled)
    print("      resolving flats...")
    inflated = grid.resolve_flats(depression_filled)
    print("      routing flow...")
    flow_direction = grid.flowdir(inflated)
    print("      accumulating...")
    accumulation = grid.accumulation(flow_direction)

    invalid = ~np.isfinite(dem)
    filled = np.asarray(inflated, dtype="float32")
    accum = np.asarray(accumulation, dtype="float32")
    filled[invalid] = np.nan
    accum[invalid] = np.nan

    try:
        tmp.unlink()
        tmpdir.rmdir()
    except OSError:
        pass

    return {
        "filled": filled,
        "flow_direction": np.asarray(flow_direction),
        "flow_accumulation": accum,
    }


def topographic_wetness_index(flow_accumulation, slope_rad, pixel_size=None):
    """TWI = ln(a / tan(beta)), with a floor on tan(beta).

    `a` is specific catchment area: upslope contributing area per unit contour
    length, which for a D8 grid is (cells * cellsize^2) / cellsize.

    The floor is the important part. Over ground this flat, tan(beta) reaches
    values on the order of 1e-6, and dividing by it produces a handful of cells
    with astronomically large TWI and an enormous dynamic range. Normalising
    that array pins virtually every pixel near zero and leaves a sparse scatter
    of extreme values -- a map that looks like noise because, by then, it is.
    """
    pixel_size = pixel_size or config.PIXEL_SIZE

    contributing_area = (flow_accumulation + 1.0) * pixel_size
    tan_beta = np.maximum(np.tan(slope_rad), np.tan(config.MIN_SLOPE_RAD))

    with np.errstate(divide="ignore", invalid="ignore"):
        twi = np.log(contributing_area / tan_beta)

    twi[~np.isfinite(flow_accumulation)] = np.nan
    return twi.astype("float32")


def channel_network(flow_accumulation, threshold_cells=None):
    """Boolean channel mask from a flow-accumulation threshold."""
    threshold_cells = threshold_cells or config.CHANNEL_ACCUMULATION_CELLS
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(flow_accumulation, nan=0.0) >= threshold_cells


def distance_to_channel(channels, pixel_size=None):
    """Euclidean distance from every cell to the nearest channel, in metres.

    Computing this in a projected CRS is what makes the result a distance. On a
    geographic grid the same transform returns degrees, and anisotropic ones at
    that: at this latitude a step east covers about 10% less ground than a step
    north, so the "distance" would be shorter in one direction than the other.
    """
    pixel_size = pixel_size or config.PIXEL_SIZE

    if not channels.any():
        raise ValueError(
            "no channel cells found; lower CHANNEL_ACCUMULATION_CELLS"
        )

    distance = ndimage.distance_transform_edt(~channels, sampling=pixel_size)
    return distance.astype("float32")

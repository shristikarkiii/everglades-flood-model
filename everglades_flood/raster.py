"""Analysis grid definition and raster I/O helpers.

Every layer in the model is resampled onto one common grid, defined once here.
Co-registration is not a detail: a weighted overlay of layers that are not on
the same grid silently compares a pixel against its neighbour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds

from . import config


@dataclass(frozen=True)
class Grid:
    """The analysis grid: CRS, affine transform and shape."""

    crs: str
    transform: Affine
    width: int
    height: int

    @property
    def shape(self):
        return self.height, self.width

    @property
    def bounds(self):
        left = self.transform.c
        top = self.transform.f
        right = left + self.width * self.transform.a
        bottom = top + self.height * self.transform.e
        return left, bottom, right, top

    def profile(self, dtype="float32", nodata=np.nan, count=1):
        return {
            "driver": "GTiff",
            "crs": self.crs,
            "transform": self.transform,
            "width": self.width,
            "height": self.height,
            "count": count,
            "dtype": dtype,
            "nodata": nodata,
            "compress": "deflate",
            "predictor": 2 if str(dtype).startswith("float") else 1,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }


def analysis_grid() -> Grid:
    """Build the 30 m UTM 17N grid covering the study area.

    Bounds are snapped outward to whole multiples of the pixel size so the grid
    origin is stable no matter how the AOI is nudged.
    """
    west, south, east, north = transform_bounds(
        "EPSG:4326", config.ANALYSIS_CRS, *config.AOI_4326, densify_pts=64
    )

    px = config.PIXEL_SIZE
    west = np.floor(west / px) * px
    south = np.floor(south / px) * px
    east = np.ceil(east / px) * px
    north = np.ceil(north / px) * px

    width = int(round((east - west) / px))
    height = int(round((north - south) / px))

    return Grid(
        crs=config.ANALYSIS_CRS,
        transform=Affine(px, 0.0, west, 0.0, -px, north),
        width=width,
        height=height,
    )


def write(path, array, grid: Grid, dtype="float32", nodata=np.nan):
    """Write a single-band raster on the analysis grid."""
    array = np.asarray(array)
    if dtype.startswith("float"):
        array = array.astype(dtype)
    else:
        array = array.astype(dtype)
    profile = grid.profile(dtype=dtype, nodata=nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
    return path


def read(path, masked=True):
    """Read a single-band raster, returning float32 with NaN for nodata."""
    with rasterio.open(path) as src:
        arr = src.read(1, masked=masked)
    if np.ma.isMaskedArray(arr):
        # Cast before filling: an integer masked array cannot take NaN as its
        # fill value, which is how categorical rasters arrive here.
        out = arr.astype("float32").filled(np.nan)
    else:
        out = arr.astype("float32")
    return out


def reproject_file(src_path, grid: Grid, resampling=Resampling.bilinear,
                   src_nodata=None):
    """Reproject a raster file onto the analysis grid, returning an array."""
    with rasterio.open(src_path) as src:
        dst = np.full(grid.shape, np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata if src_nodata is not None else src.nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst


def reproject_array(array, src_transform, src_crs, grid: Grid,
                    resampling=Resampling.bilinear, src_nodata=np.nan):
    """Reproject an in-memory array onto the analysis grid."""
    dst = np.full(grid.shape, np.nan, dtype="float32")
    reproject(
        source=np.ascontiguousarray(array.astype("float32")),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst

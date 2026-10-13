"""Sentinel-1 water frequency.

Every Sentinel-1 scene over the study area for a full year is thresholded for
open water, and the fraction of valid observations in which a pixel appears as
water becomes a conditioning factor.

The point of including it is that it *observes* inundation rather than
inferring it from the shape of the land. A model built only from DEM
derivatives -- flow accumulation, slope, channel proximity, elevation -- is
asking topography to stand in for hydrology, which on terrain with a metre of
total relief is asking a great deal of it.

Radiometrically terrain-corrected (RTC) gamma0 is used rather than raw GRD, so
calibration, terrain flattening and geocoding are already applied consistently
across the archive.

A caveat that belongs in any SAR wetland study: this detects *open* water by
its specular, low-backscatter signature. Water beneath emergent vegetation --
sawgrass marsh, much of the Everglades -- can instead raise backscatter through
double-bounce with the stems, and is systematically under-detected. The
resulting layer should be read as "how often is this pixel open water", which
is a strong but incomplete proxy for inundation.
"""

from __future__ import annotations

import concurrent.futures as futures
import warnings

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from . import config
from .raster import Grid


def search_scenes(limit=None):
    """Find Sentinel-1 RTC scenes covering the AOI over the configured year."""
    import planetary_computer
    from pystac_client import Client

    catalog = Client.open(config.STAC_API)
    search = catalog.search(
        collections=[config.S1_COLLECTION],
        bbox=config.AOI_4326,
        datetime="{}/{}".format(config.S1_START, config.S1_END),
    )
    items = list(search.items())
    items.sort(key=lambda it: it.datetime)
    if limit:
        # Even subsample across the year rather than the first N, so a limited
        # run still spans the wet and dry seasons.
        step = max(1, len(items) // limit)
        items = items[::step][:limit]
    return items, planetary_computer


def _pick_overview(src, target_resolution):
    """Choose the overview level closest to the target resolution.

    A 10 m scene warped onto a 30 m grid does not need to be read at full
    resolution. Reading from an overview cuts the transfer by roughly the
    square of the factor, which is the difference between this stage taking
    minutes and taking hours.
    """
    overviews = src.overviews(1)
    if not overviews:
        return None
    native = abs(src.res[0])
    best = None
    for factor in sorted(overviews):
        if native * factor <= target_resolution:
            best = factor
    if best is None:
        return None
    return sorted(overviews).index(best)


def otsu_threshold(values, lo=None, hi=None, bins=None):
    """Otsu's threshold and its between-class variance ratio.

    The ratio (often written eta) is the share of total variance explained by
    the two-class split. It is the separability score used to decide whether a
    scene's histogram is genuinely bimodal, or whether Otsu has simply cut a
    single mode down the middle.
    """
    lo = config.OTSU_SEARCH_RANGE_DB[0] if lo is None else lo
    hi = config.OTSU_SEARCH_RANGE_DB[1] if hi is None else hi
    bins = bins or config.OTSU_BINS

    v = values[(values > lo) & (values < hi)]
    if v.size < 10_000:
        return None, 0.0

    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    p = hist.astype("float64")
    p /= p.sum()
    centres = (edges[:-1] + edges[1:]) / 2.0

    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_total = mu[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_total * omega - mu) ** 2 / (omega * (1.0 - omega))
    between[~np.isfinite(between)] = 0.0

    k = int(np.argmax(between))
    total_variance = float(np.sum(p * (centres - mu_total) ** 2))
    eta = float(between[k] / total_variance) if total_variance > 0 else 0.0
    return float(centres[k]), eta


def _read_scene(href, grid: Grid):
    """Read one scene's VV band onto the analysis grid, in decibels.

    Returns None when the scene does not actually intersect the grid.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        with rasterio.open(href) as probe:
            bounds = transform_bounds(probe.crs, grid.crs, *probe.bounds)
            left, bottom, right, top = grid.bounds
            if (bounds[2] < left or bounds[0] > right
                    or bounds[3] < bottom or bounds[1] > top):
                return None
            level = _pick_overview(probe, config.S1_READ_RESOLUTION)

        opener = (
            rasterio.open(href, overview_level=level)
            if level is not None
            else rasterio.open(href)
        )
        with opener as src:
            with WarpedVRT(
                src,
                crs=grid.crs,
                transform=grid.transform,
                width=grid.width,
                height=grid.height,
                resampling=Resampling.average,
            ) as vrt:
                linear = vrt.read(1, masked=True)

    arr = linear.astype("float32").filled(np.nan)

    # RTC gamma0 arrives as linear power. Zero and negative values are fill,
    # not measurements, and log10 of them is not a number worth propagating.
    arr[arr <= 0] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(arr)


def screen_scene(db, land_mask, grid: Grid):
    """Decide whether a scene can be thresholded, and at what value.

    Returns (threshold, diagnostics). `threshold` is None when the scene is
    rejected; the diagnostics say why, and are reported so the screening is
    auditable rather than silent.
    """
    valid = np.isfinite(db)
    coverage = float(valid.mean())
    diag = {"coverage": coverage, "land_fraction": 0.0,
            "threshold": None, "eta": 0.0, "reason": None}

    if coverage < config.MIN_SCENE_COVERAGE:
        diag["reason"] = "coverage"
        return None, diag

    valid_count = int(valid.sum())
    land_fraction = float((valid & land_mask).sum()) / max(1, valid_count)
    diag["land_fraction"] = land_fraction
    if land_fraction < config.MIN_SCENE_LAND_FRACTION:
        diag["reason"] = "land_fraction"
        return None, diag

    threshold, eta = otsu_threshold(db[valid])
    diag["threshold"] = threshold
    diag["eta"] = eta

    if threshold is None:
        diag["reason"] = "too_few_samples"
        return None, diag
    if eta < config.MIN_OTSU_SEPARABILITY:
        diag["reason"] = "not_bimodal"
        return None, diag
    lo, hi = config.OTSU_PLAUSIBLE_RANGE_DB
    if not (lo <= threshold <= hi):
        diag["reason"] = "implausible_threshold"
        return None, diag

    return threshold, diag


def water_frequency(grid: Grid, land_mask, limit=None, max_workers=6,
                    progress=True):
    """Fraction of accepted Sentinel-1 observations in which a pixel is water.

    Returns (frequency, observations, scene_log). Pixels with fewer than
    `config.S1_MIN_OBSERVATIONS` accepted looks are left as NaN -- a frequency
    of 1.0 derived from two observations is not a measurement.
    """
    items, planetary_computer = search_scenes(limit=limit)
    print("   {} Sentinel-1 {} scene(s) {} to {}".format(
        len(items), config.S1_COLLECTION, config.S1_START, config.S1_END))
    if not items:
        raise RuntimeError("no Sentinel-1 scenes found for the AOI and dates")

    water = np.zeros(grid.shape, dtype="int32")
    observations = np.zeros(grid.shape, dtype="int32")
    scene_log = []

    def job(item):
        signed = planetary_computer.sign(item)
        asset = signed.assets[config.S1_POLARIZATION]
        return _read_scene(asset.href, grid)

    done = 0
    accepted = 0
    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending = {pool.submit(job, it): it for it in items}
        for future in futures.as_completed(pending):
            item = pending[future]
            done += 1
            record = {"id": item.id, "date": str(item.datetime.date())}

            try:
                db = future.result()
            except Exception as exc:  # one bad scene must not stop the run
                record.update(reason="read_error", error=str(exc)[:160])
                scene_log.append(record)
                continue

            if db is None:
                record.update(reason="no_overlap")
                scene_log.append(record)
                continue

            threshold, diag = screen_scene(db, land_mask, grid)
            record.update(diag)
            scene_log.append(record)

            if threshold is not None:
                valid = np.isfinite(db)
                observations += valid
                water += valid & (db < threshold)
                accepted += 1

            if progress:
                print("\r   {}/{} scenes screened, {} accepted".format(
                    done, len(items), accepted), end="")

    print()
    if accepted == 0:
        raise RuntimeError("no Sentinel-1 scene passed screening")

    reasons = {}
    for record in scene_log:
        reasons[record.get("reason")] = reasons.get(record.get("reason"), 0) + 1
    print("   screening: " + ", ".join(
        "{}={}".format(k or "accepted", v) for k, v in sorted(
            reasons.items(), key=lambda kv: (kv[0] is not None, kv[0]))))

    thresholds = [r["threshold"] for r in scene_log
                  if r.get("reason") is None and r.get("threshold") is not None]
    if thresholds:
        print("   accepted thresholds: {:.1f} to {:.1f} dB (median {:.1f})"
              .format(min(thresholds), max(thresholds),
                      float(np.median(thresholds))))

    with np.errstate(invalid="ignore", divide="ignore"):
        frequency = water.astype("float32") / observations
    frequency[observations < config.S1_MIN_OBSERVATIONS] = np.nan

    print("   looks per pixel: median {:.0f}, max {}".format(
        float(np.median(observations)), int(observations.max())))
    return frequency, observations, scene_log

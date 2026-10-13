"""Validation against independent flood data.

A susceptibility map without an accuracy assessment is an illustration, not a
result: nothing distinguishes a good model from a plausible-looking one until
it is scored against something it did not see.

The primary reference is FEMA's National Flood Hazard Layer. It is
independent of everything the model is built from: the Special Flood Hazard
Area is delineated from hydrological and hydraulic engineering studies, not
from the Copernicus DEM and not from Sentinel-1. JRC Global Surface Water is
used as a second, weaker check.

Read the numbers with one caveat in mind. The SFHA is a regulatory product
covering the 1%-annual-chance floodplain, mapped to differing standards and
vintages by county, and it is not a map of observed inundation. Agreement with
it is evidence that the model ranks flood exposure sensibly, not proof that it
predicts individual floods.
"""

from __future__ import annotations

import json

import numpy as np
import rasterio.features  # noqa: F401  (registers rasterio.features)
from rasterio.warp import transform_geom

from .raster import Grid

# Zones beginning A or V are Special Flood Hazard Areas: the 1%-annual-chance
# floodplain, with V denoting additional coastal wave hazard.
SFHA_PREFIXES = ("A", "V")

# Zone D means "Areas of Undetermined Flood Hazard" -- FEMA has not studied
# them. It is not a finding of low hazard, and it is emphatically not a
# negative observation.
#
# This distinction decides whether the validation means anything here. Zone D
# covers 22.8% of this study domain, and it is the lowest, wettest ground in
# it: median elevation 0.93 m against 3.99 m for Zone X, and a mean JRC surface
# water occurrence of 12.2% against 0.3%. That is the undeveloped interior of
# Everglades National Park, unmapped because there is no property there to
# regulate, not because it does not flood.
#
# Scoring those pixels as negatives dragged every model variant below chance
# (full model AUC 0.456). Excluding them raises the same model, unchanged, to
# 0.751. The first number says nothing about the model and everything about
# the reference.
UNDETERMINED_ZONES = {"D"}

# Not hazard determinations at all.
NON_DETERMINATION_ZONES = {"OPEN WATER", "AREA NOT INCLUDED"}


def _is_sfha(properties):
    flag = properties.get("SFHA_TF")
    if isinstance(flag, str) and flag.strip().upper() in ("T", "TRUE", "YES"):
        return True
    if flag is True:
        return True
    zone = (properties.get("FLD_ZONE") or "").strip().upper()
    return zone.startswith(SFHA_PREFIXES)


def _burn(shapes, grid: Grid):
    if not shapes:
        return np.zeros(grid.shape, dtype=bool)
    return rasterio.features.rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        default_value=1,
        dtype="uint8",
    ).astype(bool)


def rasterize_fema(geojson_path, grid: Grid, verbose=True):
    """Burn FEMA flood zones onto the analysis grid.

    Returns (sfha, evaluable). `evaluable` is where FEMA has actually made a
    determination, and is the only place the reference carries information --
    see UNDETERMINED_ZONES above for why that distinction matters so much here.
    """
    with open(geojson_path, encoding="utf-8") as fh:
        collection = json.load(fh)

    features = collection.get("features", [])
    sfha_shapes = []
    evaluable_shapes = []
    zones = {}

    for feature in features:
        properties = feature.get("properties") or {}
        zone = (properties.get("FLD_ZONE") or "?").strip().upper()
        zones[zone] = zones.get(zone, 0) + 1

        geometry = feature.get("geometry")
        if not geometry:
            continue
        projected = transform_geom("EPSG:4326", grid.crs, geometry)

        if zone in UNDETERMINED_ZONES or zone in NON_DETERMINATION_ZONES:
            continue

        evaluable_shapes.append((projected, 1))
        if _is_sfha(properties):
            sfha_shapes.append((projected, 1))

    if verbose:
        print("   FEMA zones: " + ", ".join(
            "{}={}".format(k, v)
            for k, v in sorted(zones.items(), key=lambda kv: -kv[1])[:12]))
        excluded = sum(v for k, v in zones.items()
                       if k in UNDETERMINED_ZONES or k in NON_DETERMINATION_ZONES)
        print("   {} polygons carry a determination, {} do not (excluded)"
              .format(len(evaluable_shapes), excluded))

    if not sfha_shapes:
        raise RuntimeError("no SFHA polygons found in the FEMA download")

    sfha = _burn(sfha_shapes, grid)
    evaluable = _burn(evaluable_shapes, grid)

    # An undetermined polygon overlapping a determined one must not reinstate
    # the determined pixels as negatives, and vice versa: determination wins.
    return sfha & evaluable, evaluable


def roc(scores, truth, mask, max_samples=500_000, seed=0):
    """ROC curve and AUC for a continuous score against a boolean reference."""
    from sklearn.metrics import roc_auc_score, roc_curve

    valid = mask & np.isfinite(scores)
    y = truth[valid].astype("uint8")
    x = scores[valid].astype("float64")

    if y.sum() == 0 or y.sum() == y.size:
        raise ValueError("reference has only one class inside the mask")

    if x.size > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.size, size=max_samples, replace=False)
        x, y = x[idx], y[idx]

    auc = float(roc_auc_score(y, x))
    fpr, tpr, _ = roc_curve(y, x)
    return {"auc": auc, "fpr": fpr, "tpr": tpr,
            "n": int(x.size), "positives": int(y.sum())}


def compare(models, truth, mask, **kwargs):
    """Run `roc` for several named score rasters and report them together."""
    results = {}
    for name, scores in models.items():
        try:
            results[name] = roc(scores, truth, mask, **kwargs)
        except ValueError as exc:
            print("   {} skipped: {}".format(name, exc))
    return results


def report(results):
    lines = ["{:<34} {:>8}".format("model", "AUC"), "-" * 44]
    for name, res in sorted(results.items(), key=lambda kv: -kv[1]["auc"]):
        lines.append("{:<34} {:>8.4f}".format(name, res["auc"]))
    return "\n".join(lines)

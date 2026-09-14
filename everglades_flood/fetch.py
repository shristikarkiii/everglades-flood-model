"""Download the open datasets the model runs on.

Nothing here needs credentials. Each function is idempotent: if the output
already exists it is left alone, so a re-run costs nothing and a partial run
can be resumed.
"""

from __future__ import annotations

import json
import time

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.merge import merge

from . import config
from .raster import Grid, reproject_file, write

CHUNK = 1 << 20


def _download(url, dest, label=None):
    """Stream a file to disk unless it is already there."""
    if dest.exists() and dest.stat().st_size > 0:
        print("   have   {}".format(dest.name))
        return dest

    label = label or dest.name
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                got += len(chunk)
                if total:
                    pct = 100 * got / total
                    print("\r   fetch  {} {:5.1f}%".format(label, pct), end="")
        print("\r   fetch  {} done ({:.0f} MB)".format(label, got / 1e6))
        tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# Copernicus DEM GLO-30
# ---------------------------------------------------------------------------


def fetch_dem(grid: Grid):
    """Download, mosaic and reproject the Copernicus DEM onto the grid."""
    out = config.INTERIM / "dem.tif"
    if out.exists():
        print("   have   dem.tif")
        return out

    print("== Copernicus DEM GLO-30 ==")
    tile_dir = config.RAW / "copernicus_dem"
    tile_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for lat, lon in config.COP_DEM_TILES:
        name = "Copernicus_DSM_COG_10_{}_00_{}_00_DEM".format(lat, lon)
        url = "{}/{}/{}.tif".format(config.COP_DEM_BASE, name, name)
        dest = tile_dir / (name + ".tif")
        try:
            _download(url, dest, label=name[-12:])
        except requests.HTTPError as exc:
            print("   skip   {} ({})".format(name, exc.response.status_code))
            continue
        # A tile can exist but hold no land; merge would still work, but an
        # unreadable file would not.
        try:
            with rasterio.open(dest) as src:
                if src.width and src.height:
                    paths.append(dest)
        except rasterio.RasterioIOError as exc:
            print("   skip   {} (unreadable: {})".format(name, exc))

    if not paths:
        raise RuntimeError("no usable Copernicus DEM tiles were downloaded")

    mosaic_path = config.RAW / "dem_mosaic.tif"
    if not mosaic_path.exists():
        srcs = [rasterio.open(p) for p in paths]
        try:
            arr, transform = merge(srcs)
            meta = srcs[0].meta.copy()
            meta.update(
                height=arr.shape[1], width=arr.shape[2], transform=transform,
                compress="deflate",
            )
            with rasterio.open(mosaic_path, "w", **meta) as dst:
                dst.write(arr)
        finally:
            for s in srcs:
                s.close()
        print("   merged {} tile(s)".format(len(paths)))

    dem = reproject_file(mosaic_path, grid, resampling=Resampling.bilinear)
    write(out, dem, grid)
    print("   wrote  dem.tif")
    return out


# ---------------------------------------------------------------------------
# ESA WorldCover
# ---------------------------------------------------------------------------


def fetch_worldcover(grid: Grid):
    """Download and reproject ESA WorldCover land cover onto the grid."""
    out = config.INTERIM / "landcover.tif"
    if out.exists():
        print("   have   landcover.tif")
        return out

    print("== ESA WorldCover 2021 ==")
    tile_dir = config.RAW / "worldcover"
    tile_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for tile in config.WORLDCOVER_TILES:
        name = "ESA_WorldCover_10m_2021_v200_{}_Map.tif".format(tile)
        dest = tile_dir / name
        _download("{}/{}".format(config.WORLDCOVER_BASE, name), dest, label=tile)
        paths.append(dest)

    mosaic_path = config.RAW / "worldcover_mosaic.tif"
    if not mosaic_path.exists():
        srcs = [rasterio.open(p) for p in paths]
        try:
            arr, transform = merge(srcs)
            meta = srcs[0].meta.copy()
            meta.update(height=arr.shape[1], width=arr.shape[2],
                        transform=transform, compress="deflate")
            with rasterio.open(mosaic_path, "w", **meta) as dst:
                dst.write(arr)
        finally:
            for s in srcs:
                s.close()

    # Land cover is categorical: nearest neighbour, never bilinear.
    lc = reproject_file(mosaic_path, grid, resampling=Resampling.nearest)
    write(out, np.nan_to_num(lc, nan=0).astype("uint8"), grid,
          dtype="uint8", nodata=0)
    print("   wrote  landcover.tif")
    return out


# ---------------------------------------------------------------------------
# JRC Global Surface Water
# ---------------------------------------------------------------------------


def fetch_gsw(grid: Grid):
    """Download JRC Global Surface Water occurrence onto the grid."""
    out = config.INTERIM / "gsw_occurrence.tif"
    if out.exists():
        print("   have   gsw_occurrence.tif")
        return out

    print("== JRC Global Surface Water ==")
    dest = config.RAW / "gsw_occurrence_90W_30N.tif"
    _download(config.GSW_URL, dest, label="GSW occurrence")

    occ = reproject_file(dest, grid, resampling=Resampling.average,
                         src_nodata=255)
    write(out, occ, grid)
    print("   wrote  gsw_occurrence.tif")
    return out


# ---------------------------------------------------------------------------
# FEMA National Flood Hazard Layer
# ---------------------------------------------------------------------------


def fetch_fema(grid: Grid):
    """Download FEMA flood hazard polygons covering the AOI as GeoJSON.

    The service caps a response at 1000 features, so results are paged. Only
    the zone attributes are requested; geometry comes back in EPSG:4326 and is
    rasterised later.
    """
    out = config.RAW / "fema_flood_zones.geojson"
    if out.exists():
        print("   have   fema_flood_zones.geojson")
        return out

    print("== FEMA National Flood Hazard Layer ==")
    url = "{}/{}/query".format(config.FEMA_NFHL, config.FEMA_FLOOD_HAZARD_LAYER)
    features = []
    offset = 0

    # The service returns HTTP 500 rather than a paging error when a single
    # response would carry too much geometry; 1000 polygons is past that line
    # here, 200 is comfortably inside it.
    page = 200

    while True:
        params = {
            "where": "1=1",
            "geometry": "{},{},{},{}".format(*config.AOI_4326),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "FLD_ZONE,SFHA_TF,ZONE_SUBTY",
            "returnGeometry": "true",
            "resultOffset": offset,
            "resultRecordCount": page,
            "f": "geojson",
        }

        batch = None
        for attempt in range(4):
            try:
                r = requests.get(url, params=params, timeout=300)
                if r.status_code == 500 and page > 25:
                    # Too much geometry for one response: ask for less.
                    page = max(25, page // 2)
                    params["resultRecordCount"] = page
                    continue
                r.raise_for_status()
                batch = r.json().get("features", [])
                break
            except (requests.HTTPError, requests.Timeout,
                    requests.ConnectionError, ValueError) as exc:
                if attempt == 3:
                    raise RuntimeError(
                        "FEMA query failed at offset {} after 4 attempts: {}"
                        .format(offset, exc)
                    )
                time.sleep(2 ** attempt)

        features.extend(batch)
        print("\r   fetch  FEMA zones {} feature(s)".format(len(features)), end="")
        if len(batch) < page:
            break
        offset += len(batch)

    print()
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh)
    print("   wrote  fema_flood_zones.geojson ({} features)".format(len(features)))
    return out


# ---------------------------------------------------------------------------
# National Park Service boundaries
# ---------------------------------------------------------------------------


def fetch_park_boundaries():
    """Everglades National Park and Big Cypress National Preserve outlines.

    Used only for cartography. The study domain is a rectangle of south
    Florida, and about 6% of it is developed land outside any park; drawing the
    boundaries lets a reader see exactly which part of the map is protected
    wetland instead of taking the title's word for it.
    """
    out = config.RAW / "park_boundaries.geojson"
    if out.exists():
        print("   have   park_boundaries.geojson")
        return out

    print("== NPS park boundaries ==")
    url = ("https://services1.arcgis.com/fBc8EJBxQRMcHlei/ArcGIS/rest/services/"
           "NPS_Land_Resources_Division_Boundary_and_Tract_Data_Service/"
           "FeatureServer/2/query")
    try:
        r = requests.get(url, params={
            "where": "UNIT_CODE IN ('EVER','BICY')",
            "outFields": "UNIT_CODE,UNIT_NAME",
            "outSR": "4326", "returnGeometry": "true", "f": "geojson",
        }, timeout=180)
        r.raise_for_status()
        payload = r.json()
    except (requests.RequestException, ValueError) as exc:
        # Cartographic decoration only: a failure here must not stop the run.
        print("   skip   park boundaries unavailable ({})".format(
            str(exc)[:80]))
        return None

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    names = [f.get("properties", {}).get("UNIT_NAME")
             for f in payload.get("features", [])]
    print("   wrote  park_boundaries.geojson ({})".format(", ".join(
        n for n in names if n)))
    return out

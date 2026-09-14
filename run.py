#!/usr/bin/env python3
"""Run the Everglades flood susceptibility pipeline end to end.

    python run.py                # run everything, reusing cached stages
    python run.py --force sar    # recompute one stage
    python run.py --force all

Every stage caches to data/interim, so a re-run costs only the stages that
changed. Outputs land in outputs/.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from everglades_flood import (ahp, config, domain, fetch, model, plotting,
                              sar, sensitivity, terrain, validate)
from everglades_flood.raster import analysis_grid, read, write

STAGES = ("fetch", "terrain", "sar", "model", "validate", "figures")


def _cached(path, force):
    return path.exists() and not force


def stage_fetch(grid, force=False):
    print("\n[1/6] data")
    fetch.fetch_dem(grid)
    fetch.fetch_worldcover(grid)
    fetch.fetch_gsw(grid)
    fetch.fetch_fema(grid)
    fetch.fetch_park_boundaries()

    dem = read(config.INTERIM / "dem.tif")
    landcover = read(config.INTERIM / "landcover.tif")
    study = domain.study_domain(dem, landcover)
    write(config.INTERIM / "domain.tif", study.astype("uint8"), grid,
          dtype="uint8", nodata=255)
    print("   study domain: {:,} px ({:.1f}% of grid); {:.1f}% of the AOI is sea"
          .format(int(study.sum()), 100 * study.mean(),
                  100 * (1 - study.mean())))
    return dem, landcover, study


def stage_terrain(grid, dem, study, force=False):
    print("\n[2/6] terrain")
    paths = {
        "slope": config.INTERIM / "slope.tif",
        "filled": config.INTERIM / "filled_dem.tif",
        "accumulation": config.INTERIM / "flow_accumulation.tif",
        "twi": config.INTERIM / "twi.tif",
        "distance": config.INTERIM / "distance_to_channel.tif",
    }
    if all(_cached(p, force) for p in paths.values()):
        print("   cached")
        return {k: read(p) for k, p in paths.items()}

    # Flow is routed over land only. Including the sea asks pysheds to resolve
    # an eight-million-cell plateau at exactly 0 m -- slow, and meaningless.
    dem_land = np.where(study, dem, np.nan).astype("float32")

    slope = terrain.slope_radians(dem_land)
    print("   slope: median {:.4f} deg".format(
        float(np.degrees(np.nanmedian(slope)))))

    hydro = terrain.hydrology(dem_land)
    accumulation = hydro["flow_accumulation"]
    twi = terrain.topographic_wetness_index(accumulation, slope)
    channels = terrain.channel_network(accumulation)
    distance = terrain.distance_to_channel(channels)
    print("   channels: {:,} cells; distance median {:.0f} m".format(
        int((channels & study).sum()), float(np.median(distance[study]))))

    out = {"slope": slope, "filled": hydro["filled"],
           "accumulation": accumulation, "twi": twi, "distance": distance}
    for key, path in paths.items():
        write(path, out[key], grid)
    return out


def stage_sar(grid, landcover, force=False):
    print("\n[3/6] sentinel-1 water frequency")
    freq_path = config.INTERIM / "sar_water_frequency.tif"
    obs_path = config.INTERIM / "sar_observations.tif"
    if _cached(freq_path, force) and _cached(obs_path, force):
        print("   cached")
        return read(freq_path), read(obs_path)

    land = domain.land_mask(landcover)
    frequency, observations, log = sar.water_frequency(grid, land)
    write(freq_path, frequency, grid)
    write(obs_path, observations.astype("float32"), grid)
    with open(config.INTERIM / "sar_scene_log.json", "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    return frequency, observations.astype("float32")


def stage_model(grid, dem, landcover, study, terrain_layers, water_frequency):
    print("\n[4/6] weighted overlay")

    raw = {
        "sar_water_frequency": water_frequency,
        "elevation": dem,
        "twi": terrain_layers["twi"],
        "distance_to_channel": terrain_layers["distance"],
        "slope": terrain_layers["slope"],
        "landcover": domain.landcover_susceptibility(landcover),
    }
    factors = model.normalise_factors(raw, study)

    result = ahp.default_weights()
    print(result.report())
    if not result.is_consistent:
        raise RuntimeError("AHP judgements are inconsistent; CR >= 0.10")

    weights = result.as_dict()
    index = model.susceptibility_index(factors, weights, study)
    classes, summary = model.classify(index, study)

    write(config.OUTPUTS / "flood_susceptibility_index.tif", index, grid)
    write(config.OUTPUTS / "flood_susceptibility_classes.tif", classes, grid,
          dtype="uint8", nodata=0)
    for name, layer in factors.items():
        write(config.INTERIM / "factor_{}.tif".format(name), layer, grid)

    print()
    for row in summary:
        print("   {:<10} {:>5.1f}%  index {:.3f} to {:.3f}".format(
            row["label"], 100 * row["share"], row["lower"], row["upper"]))

    # Ablations, to show what each part of the model is worth.
    terrain_only = {k: v for k, v in factors.items()
                    if k != "sar_water_frequency"}
    ablations = {
        "Full model (terrain + Sentinel-1)": index,
        "Terrain only (no SAR)": model.susceptibility_index(
            terrain_only, weights, study),
        "Sentinel-1 water frequency alone": factors["sar_water_frequency"],
        "Inverted elevation alone": factors["elevation"],
    }
    return factors, index, classes, summary, result, ablations


def stage_validate(grid, study, ablations):
    print("\n[5/6] validation")
    sfha, evaluable = validate.rasterize_fema(
        config.RAW / "fema_flood_zones.geojson", grid)

    evaluation = study & evaluable
    print("   determined area inside the study domain: {:,} px ({:.1f}% of it)"
          .format(int(evaluation.sum()), 100 * evaluation.sum() / study.sum()))
    print("   of which SFHA: {:.1f}%".format(
        100 * (sfha & evaluation).sum() / max(1, evaluation.sum())))

    results = validate.compare(ablations, sfha, evaluation)
    print()
    print(validate.report(results))

    gsw = read(config.INTERIM / "gsw_occurrence.tif")
    secondary = {}
    gsw_mask = study & np.isfinite(gsw)
    if gsw_mask.sum() > 10000:
        frequently_wet = gsw > 50
        secondary = validate.compare(ablations, frequently_wet, gsw_mask)
        print("\n   secondary reference (JRC surface water > 50% occurrence)")
        print(validate.report(secondary))

    return results, secondary, sfha, evaluation


def _boundaries_path():
    """Park outlines for the map, or None if the NPS service was unreachable."""
    path = config.RAW / "park_boundaries.geojson"
    return path if path.exists() else None


def stage_figures(grid, study, factors, classes, summary, ahp_result,
                  terrain_layers, results):
    print("\n[6/6] figures")
    labels = {k: ahp.FACTOR_LABELS[k] for k in factors}

    paths = [
        plotting.susceptibility_map(
            classes, summary, study, grid,
            config.FIGURES / "01-flood-susceptibility-map.png",
            boundaries=_boundaries_path()),
        plotting.factor_panel(
            factors, study, grid,
            config.FIGURES / "02-conditioning-factors.png", labels=labels),
        plotting.weights_bar(
            ahp_result, config.FIGURES / "03-ahp-weights.png"),
        plotting.roc_curves(
            results, config.FIGURES / "04-validation-roc.png"),
        plotting.normalisation_comparison(
            terrain_layers["accumulation"], terrain_layers["twi"], study, grid,
            config.FIGURES / "05-normalisation-comparison.png"),
    ]
    for path in paths:
        print("   {}".format(path.name))
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", choices=STAGES + ("all",), action="append",
                        default=[], help="recompute a cached stage")
    args = parser.parse_args(argv)
    force = set(args.force)
    forced = lambda name: name in force or "all" in force  # noqa: E731

    started = time.time()
    grid = analysis_grid()
    print("analysis grid: {}x{} px at {:.0f} m, {}".format(
        grid.width, grid.height, grid.transform.a, grid.crs))

    dem, landcover, study = stage_fetch(grid, force=forced("fetch"))
    terrain_layers = stage_terrain(grid, dem, study, force=forced("terrain"))
    water_frequency, _ = stage_sar(grid, landcover, force=forced("sar"))

    factors, index, classes, summary, ahp_result, ablations = stage_model(
        grid, dem, landcover, study, terrain_layers, water_frequency)

    results, secondary, sfha, evaluation = stage_validate(
        grid, study, ablations)

    print("\n[5b/6] weight sensitivity")
    gsw = read(config.INTERIM / "gsw_occurrence.tif")
    gsw_mask = study & np.isfinite(gsw)
    references = {
        "FEMA SFHA": (sfha, evaluation),
        "JRC surface water": (gsw > 50, gsw_mask),
    }
    records = sensitivity.sweep(factors, ahp_result.as_dict(), study, references)
    print(sensitivity.report(records, list(references)))
    for label in references:
        top = sensitivity.best(records, label)
        if top:
            print("   best against {:<20} w(SAR)={:.2f}  AUC {:.4f}".format(
                label, top["weight"], top["auc_" + label]))

    stage_figures(grid, study, factors, classes, summary, ahp_result,
                  terrain_layers, results)
    plotting.sensitivity_curve(
        records, list(references),
        config.FIGURES / "06-weight-sensitivity.png",
        ahp_weight=ahp_result.as_dict()["sar_water_frequency"])
    print("   06-weight-sensitivity.png")

    payload = {
        "grid": {"width": grid.width, "height": grid.height,
                 "crs": grid.crs, "pixel_size": grid.transform.a},
        "study_domain_pixels": int(study.sum()),
        "weights": ahp_result.as_dict(),
        "consistency_ratio": ahp_result.consistency_ratio,
        "classes": summary,
        "validation_fema_sfha": {k: v["auc"] for k, v in results.items()},
        "validation_gsw": {k: v["auc"] for k, v in secondary.items()},
    }
    with open(config.OUTPUTS / "results.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\ndone in {:.0f}s -> {}".format(time.time() - started,
                                           config.OUTPUTS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

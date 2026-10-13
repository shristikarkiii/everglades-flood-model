"""Tests for water thresholding, scene screening and the study domain."""

import numpy as np
import pytest

from everglades_flood import config, domain, sar, validate


# ---------------------------------------------------------------------------
# Otsu thresholding
# ---------------------------------------------------------------------------


def _bimodal(n=200_000, water_db=-22.0, land_db=-8.0, spread=1.5, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.normal(water_db, spread, n // 2),
        rng.normal(land_db, spread, n // 2),
    ])


def test_otsu_finds_the_split_between_two_modes():
    threshold, eta = sar.otsu_threshold(_bimodal())
    assert -22.0 < threshold < -8.0
    assert threshold == pytest.approx(-15.0, abs=1.5)
    assert eta > 0.8


def test_otsu_tracks_a_shifted_water_mode():
    """Wind roughening moves the water mode; the threshold must follow it.

    This is the whole reason the pipeline does not use a fixed dB cut: the
    same water measured -23 dB in August and -16 dB in January.
    """
    calm, _ = sar.otsu_threshold(_bimodal(water_db=-23.0))
    windy, _ = sar.otsu_threshold(_bimodal(water_db=-16.0, land_db=-6.0))
    assert windy > calm + 2.0


def test_otsu_reports_low_separability_for_one_mode():
    """A scene that is all water has no split to find, and must say so."""
    rng = np.random.default_rng(2)
    unimodal = rng.normal(-20.0, 1.5, 200_000)
    _, eta = sar.otsu_threshold(unimodal)
    assert eta < config.MIN_OTSU_SEPARABILITY


def test_otsu_needs_enough_samples():
    threshold, eta = sar.otsu_threshold(np.array([-20.0, -8.0]))
    assert threshold is None
    assert eta == 0.0


# ---------------------------------------------------------------------------
# Scene screening
# ---------------------------------------------------------------------------


class _Grid:
    shape = (400, 400)


def _scene(coverage=1.0, land_fraction=0.5, seed=0):
    """Build a synthetic dB scene and matching land mask."""
    shape = _Grid.shape
    rng = np.random.default_rng(seed)
    n = shape[0] * shape[1]

    db = np.full(n, np.nan)
    valid = int(n * coverage)
    land_n = int(valid * land_fraction)
    db[:land_n] = rng.normal(-8.0, 1.5, land_n)
    db[land_n:valid] = rng.normal(-22.0, 1.5, valid - land_n)

    land = np.zeros(n, dtype=bool)
    land[:land_n] = True
    return db.reshape(shape), land.reshape(shape)


def test_good_scene_is_accepted():
    db, land = _scene()
    threshold, diag = sar.screen_scene(db, land, _Grid)
    assert threshold is not None
    assert diag["reason"] is None
    assert config.OTSU_PLAUSIBLE_RANGE_DB[0] <= threshold <= config.OTSU_PLAUSIBLE_RANGE_DB[1]


def test_sliver_scene_is_rejected_on_coverage():
    db, land = _scene(coverage=0.04)
    threshold, diag = sar.screen_scene(db, land, _Grid)
    assert threshold is None
    assert diag["reason"] == "coverage"


def test_all_water_scene_is_rejected_on_land_fraction():
    """The failure mode measured on real data: corner scenes see only sea."""
    db, land = _scene(land_fraction=0.01)
    threshold, diag = sar.screen_scene(db, land, _Grid)
    assert threshold is None
    assert diag["reason"] == "land_fraction"


# ---------------------------------------------------------------------------
# Study domain
# ---------------------------------------------------------------------------


def test_sea_is_removed_but_an_inland_lake_is_kept():
    landcover = np.full((60, 60), 90, dtype="float32")   # herbaceous wetland
    landcover[:, :12] = domain.WATER                      # sea, touches edge
    landcover[30:36, 30:36] = domain.WATER                # inland lake

    sea = domain.sea_mask(landcover)
    assert sea[:, :12].all()
    assert not sea[30:36, 30:36].any()

    dem = np.zeros((60, 60), dtype="float32")
    study = domain.study_domain(dem, landcover)
    assert not study[:, :12].any()
    assert study[30:36, 30:36].all()


def test_domain_excludes_missing_elevation():
    landcover = np.full((20, 20), 90, dtype="float32")
    dem = np.zeros((20, 20), dtype="float32")
    dem[0, 0] = np.nan
    assert not domain.study_domain(dem, landcover)[0, 0]


def test_landcover_susceptibility_is_bounded_and_ordered():
    codes = np.array([[50, 10, 95, 90, 80]], dtype="float32")
    scores = domain.landcover_susceptibility(codes)
    assert np.nanmin(scores) >= 0.0
    assert np.nanmax(scores) <= 1.0
    # built-up < tree cover < mangrove < wetland < open water
    assert list(scores[0]) == sorted(scores[0])


def test_unknown_landcover_code_is_not_scored():
    scores = domain.landcover_susceptibility(np.array([[7]], dtype="float32"))
    assert np.isnan(scores[0, 0])


# ---------------------------------------------------------------------------
# FEMA zone interpretation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("properties,expected", [
    ({"FLD_ZONE": "AE", "SFHA_TF": "T"}, True),
    ({"FLD_ZONE": "VE", "SFHA_TF": "T"}, True),
    ({"FLD_ZONE": "AH", "SFHA_TF": "T"}, True),
    ({"FLD_ZONE": "A", "SFHA_TF": None}, True),
    ({"FLD_ZONE": "X", "SFHA_TF": "F"}, False),
    ({"FLD_ZONE": "D", "SFHA_TF": "F"}, False),
])
def test_sfha_classification(properties, expected):
    assert validate._is_sfha(properties) is expected


def test_zone_d_is_treated_as_undetermined():
    """Zone D is 'not studied', not 'not at risk'.

    Counting it as a negative put every model variant below chance against
    FEMA; excluding it raised the unchanged model from 0.456 to 0.751.
    """
    assert "D" in validate.UNDETERMINED_ZONES

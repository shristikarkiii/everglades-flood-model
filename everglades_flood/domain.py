"""Definition of the study domain.

The AOI is a rectangle over the southern tip of Florida, and 51% of it is open
sea -- Florida Bay and the Gulf of Mexico. Modelling flood susceptibility over
ocean is meaningless, and leaving it in corrupts everything downstream: it
dominates the normalisation ranges, it drags the validation statistics towards
a trivial land/sea discrimination, and it makes the resulting map look far more
skilful than it is.

Sea is identified as permanent water connected to the edge of the AOI, rather
than as all permanent water. Inland lakes, canals and the Water Conservation
Area impoundments are genuinely part of the flood system and stay in.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# ESA WorldCover class codes.
WATER = 80
LAND_CLASSES = (10, 20, 30, 40, 50, 60, 90, 95, 100)

# Land cover to flood-susceptibility score, 0 (least) to 1 (most).
#
# This is the weakest-evidenced layer in the model, which is why AHP gives it
# the smallest weight. In a wetland, land cover is about as much a consequence
# of the flooding regime as a cause of it.
LANDCOVER_SUSCEPTIBILITY = {
    80: 1.00,   # permanent water (inland)
    90: 0.90,   # herbaceous wetland
    95: 0.80,   # mangroves
    60: 0.55,   # bare / sparse vegetation
    30: 0.45,   # grassland
    40: 0.40,   # cropland
    20: 0.35,   # shrubland
    10: 0.30,   # tree cover
    50: 0.25,   # built-up (drained, but impervious)
    100: 0.30,  # moss and lichen
}


def sea_mask(landcover):
    """Permanent water connected to the edge of the grid: the open sea.

    Mangrove and marsh classes break the connection between the sea and the
    interior, so a flood fill from the border does not leak inland.
    """
    water = (landcover == WATER) | ~np.isfinite(landcover)
    labels, count = ndimage.label(water)
    if count == 0:
        return np.zeros(landcover.shape, dtype=bool)

    border = np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1],
    ])
    edge_labels = set(np.unique(border)) - {0}
    if not edge_labels:
        return np.zeros(landcover.shape, dtype=bool)

    return np.isin(labels, list(edge_labels))


def study_domain(dem, landcover):
    """Boolean mask of pixels the model is defined over."""
    valid = np.isfinite(dem) & np.isfinite(landcover)
    return valid & ~sea_mask(landcover)


def land_mask(landcover):
    """Non-water land cover, used to screen Sentinel-1 scenes."""
    return np.isin(landcover, LAND_CLASSES)


def landcover_susceptibility(landcover):
    """Map land cover codes to a 0-1 susceptibility score."""
    out = np.full(landcover.shape, np.nan, dtype="float32")
    for code, score in LANDCOVER_SUSCEPTIBILITY.items():
        out[landcover == code] = score
    return out

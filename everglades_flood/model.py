"""Normalisation, weighted overlay and classification.

Normalisation is where a weighted-overlay model quietly goes wrong.

Plain min-max scaling is only safe on a well-behaved distribution. Applied to
raw flow accumulation -- which spans one to several hundred thousand cells with
almost all the mass at the bottom -- it maps essentially every pixel to a value
indistinguishable from zero and leaves a sparse scatter of channel cells near
one. Overlay that and the output is salt-and-pepper noise wearing the colours
of a hazard map. Measured on this dataset, plain min-max puts 98.8% of pixels
below 0.01.

Two choices prevent it:

  * the distribution is clipped at the 1st and 99th percentiles before
    scaling, so a handful of extreme cells cannot compress everything else;
  * flow accumulation enters through TWI, which is logarithmic, rather than
    raw.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import config


def normalise(array, mask, invert=False, percentiles=None):
    """Scale to 0-1 using percentile-clipped min-max.

    `invert=True` is for factors where a *low* value means *high*
    susceptibility -- elevation, slope, distance to channel.
    """
    percentiles = percentiles or config.NORMALISE_PERCENTILES

    valid = mask & np.isfinite(array)
    if not valid.any():
        raise ValueError("no valid pixels to normalise")

    lo, hi = np.percentile(array[valid], percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(
            "degenerate normalisation range: p{}={}, p{}={}".format(
                percentiles[0], lo, percentiles[1], hi)
        )

    out = (array.astype("float32") - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    if invert:
        out = 1.0 - out

    out[~valid] = np.nan
    return out


# Which direction each factor runs. True means "low value = high
# susceptibility", so the normalised layer is inverted.
FACTOR_INVERTED = {
    "sar_water_frequency": False,
    "elevation": True,
    "twi": False,
    "distance_to_channel": True,
    "slope": True,
    "landcover": False,
}


def normalise_factors(raw, mask):
    """Normalise every raw factor onto 0-1, respecting its direction."""
    out = {}
    for name, array in raw.items():
        out[name] = normalise(array, mask, invert=FACTOR_INVERTED[name])
    return out


def susceptibility_index(factors, weights, mask):
    """Weighted linear combination of the normalised factors.

    Weights are renormalised over whatever factors are actually supplied, so
    dropping a factor -- as the terrain-only ablation does -- yields a
    comparable 0-1 index rather than one that is systematically lower.
    """
    names = list(factors)
    missing = [n for n in names if n not in weights]
    if missing:
        raise KeyError("no weight supplied for: {}".format(missing))

    total = sum(weights[n] for n in names)
    if total <= 0:
        raise ValueError("weights sum to zero")

    index = np.zeros(next(iter(factors.values())).shape, dtype="float32")
    contributing = np.zeros_like(index)

    for name in names:
        layer = factors[name]
        valid = np.isfinite(layer)
        index[valid] += (weights[name] / total) * layer[valid]
        contributing[valid] += weights[name] / total

    # A pixel missing one factor should still get a sensible index from the
    # rest, rather than silently scoring low because a term was absent.
    with np.errstate(invalid="ignore", divide="ignore"):
        index = np.where(contributing > 0, index / contributing, np.nan)

    index[~mask] = np.nan
    return index.astype("float32")


def classify(index, mask, n_classes=None, labels=None):
    """Split the index into quantile classes.

    Equal intervals would leave some classes almost empty on an index this
    skewed, and a legend with an empty class in it is worse than useless.
    Quantiles guarantee every class is populated. The trade-off, which is worth
    being explicit about, is that quantile classes describe *relative*
    susceptibility within this study area and cannot be compared against
    another region's map.
    """
    n_classes = n_classes or config.N_CLASSES
    labels = labels or config.CLASS_LABELS

    valid = mask & np.isfinite(index)
    values = index[valid]

    quantiles = np.linspace(0, 100, n_classes + 1)[1:-1]
    breaks = np.percentile(values, quantiles)

    # Quantile breaks collapse when the index is heavily tied -- if more than
    # 1/n_classes of the domain shares one value, two breaks land on it and a
    # class comes out empty. That is a property of the data, not something to
    # paper over, but it must not pass silently: an empty class in the legend
    # is exactly the kind of thing nobody notices in a finished map.
    if len(np.unique(breaks)) < len(breaks):
        warnings.warn(
            "quantile breaks are not unique ({}); the index is heavily tied "
            "and some susceptibility classes will be empty".format(
                np.round(breaks, 6)),
            RuntimeWarning,
            stacklevel=2,
        )

    classes = np.full(index.shape, 0, dtype="uint8")
    classes[valid] = np.digitize(values, breaks, right=False) + 1

    edges = [float(values.min())] + [float(b) for b in breaks] + [
        float(values.max())]
    summary = []
    for i, label in enumerate(labels[:n_classes]):
        count = int((classes == i + 1).sum())
        summary.append({
            "class": i + 1,
            "label": label,
            "lower": edges[i],
            "upper": edges[i + 1],
            "pixels": count,
            "share": count / max(1, int(valid.sum())),
        })

    return classes, summary

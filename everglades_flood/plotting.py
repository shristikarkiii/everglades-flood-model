"""Figure generation.

Maps are the deliverable, so they are exported as figures rather than captured
from an application window. Every map is drawn on the analysis grid with the
sea masked out, axes in kilometres, and a scale bar.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from . import config

# ColorBrewer RdYlBu, reversed: cool for low susceptibility, hot for high.
CLASS_COLOURS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]

OUTSIDE = "#e9e9e9"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "savefig.bbox": "tight",
    "savefig.dpi": 160,
})


def _extent_km(grid):
    left, bottom, right, top = grid.bounds
    return [0, (right - left) / 1000.0, 0, (top - bottom) / 1000.0]


def _masked(array, domain):
    return np.ma.masked_array(array, mask=~(domain & np.isfinite(array)))


def _scale_bar(ax, grid, length_km=25):
    left, bottom, right, top = grid.bounds
    width_km = (right - left) / 1000.0
    height_km = (top - bottom) / 1000.0
    x0 = width_km * 0.06
    y0 = height_km * 0.06
    ax.plot([x0, x0 + length_km], [y0, y0], color="black", lw=2.5,
            solid_capstyle="butt", zorder=5)
    ax.text(x0 + length_km / 2, y0 + height_km * 0.018, "{} km".format(length_km),
            ha="center", va="bottom", fontsize=8, zorder=5)


def outline_mask(ax, mask, grid, color="#111111", lw=1.1):
    """Trace the edge of a boolean mask on a map drawn in km coordinates.

    Outlining the *mask* rather than the source polygon matters here. The park
    polygons run well outside the modelled area -- Everglades National Park is
    roughly a third open water, and its boundary sweeps through Florida Bay,
    while Big Cypress continues north of the AOI. Drawing those polygons
    directly puts a large closed outline around sea that the model deliberately
    does not cover, which reads as an unfinished map. Passing
    `park & study_domain` instead traces the park exactly where there are
    results to enclose.

    Axis limits are captured and restored: nothing drawn on a finished map
    should be able to resize its frame.
    """
    if mask is None or not np.any(mask):
        return

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.contour(
        mask.astype("float32"),
        levels=[0.5],
        extent=_extent_km(grid),
        origin="upper",
        colors=[color],
        linewidths=lw,
        zorder=4,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def show_map(ax, array, domain, grid, cmap="viridis", vmin=None, vmax=None,
             title=None, scale_bar=True):
    ax.set_facecolor(OUTSIDE)
    im = ax.imshow(_masked(array, domain), extent=_extent_km(grid),
                   origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#999999")
    if title:
        ax.set_title(title)
    if scale_bar:
        _scale_bar(ax, grid)
    return im


def factor_panel(factors, domain, grid, path, labels=None, outline=None):
    """Grid of the normalised conditioning factors."""
    labels = labels or {}
    names = list(factors)
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        im = show_map(ax, factors[name], domain, grid, cmap="magma",
                      vmin=0, vmax=1, title=labels.get(name, name),
                      scale_bar=False)
        outline_mask(ax, outline, grid, color="#00d0ff", lw=0.7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("normalised 0-1", fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    for ax in axes[len(names):]:
        ax.axis("off")

    fig.suptitle("Normalised conditioning factors", fontsize=13,
                 fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def susceptibility_map(classes, summary, domain, grid, path,
                       title="Flood susceptibility, Everglades wetland",
                       outline=None, outline_lw=1.2,
                       outside_label="Outside study area (sea)"):
    """The headline five-class map."""
    cmap = ListedColormap(CLASS_COLOURS)
    norm = BoundaryNorm(np.arange(0.5, len(CLASS_COLOURS) + 1.5), cmap.N)

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_facecolor(OUTSIDE)
    data = np.ma.masked_array(classes.astype("float32"),
                              mask=~(domain & (classes > 0)))
    ax.imshow(data, extent=_extent_km(grid), origin="upper", cmap=cmap,
              norm=norm, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, pad=12)

    # Lock the frame to the grid before anything else is drawn onto it.
    extent = _extent_km(grid)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    _scale_bar(ax, grid)
    outline_mask(ax, outline, grid, color="#111111", lw=outline_lw)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CLASS_COLOURS[i],
                      edgecolor="#666666", linewidth=0.4)
        for i in range(len(summary))
    ]
    labels = [
        "{}  ({:.0f}%)".format(row["label"], 100 * row["share"])
        for row in summary
    ]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=OUTSIDE,
                                 edgecolor="#666666", linewidth=0.4))
    labels.append(outside_label)

    if outline is not None and np.any(outline):
        handles.append(plt.Line2D([0], [0], color="#111111", lw=1.2))
        labels.append("Park / Preserve (analysed extent)")

    ax.legend(handles, labels, loc="lower right", frameon=True,
              framealpha=0.95, fontsize=9, title="Susceptibility",
              title_fontsize=9)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def roc_curves(results, path, title="Validation against FEMA flood hazard zones"):
    """ROC curves for the model and its ablations."""
    fig, ax = plt.subplots(figsize=(7, 6.6))

    ordered = sorted(results.items(), key=lambda kv: -kv[1]["auc"])
    colours = plt.cm.viridis(np.linspace(0.05, 0.8, len(ordered)))

    for (name, res), colour in zip(ordered, colours):
        ax.plot(res["fpr"], res["tpr"], lw=2, color=colour,
                label="{}  (AUC {:.3f})".format(name, res["auc"]))

    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#999999",
            label="No skill  (AUC 0.500)")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def weights_bar(ahp_result, path):
    """AHP weights with the consistency ratio annotated."""
    from .ahp import FACTOR_LABELS

    items = sorted(ahp_result.as_dict().items(), key=lambda kv: kv[1])
    names = [FACTOR_LABELS[k] for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    bars = ax.barh(names, values, color="#4a7fb5", edgecolor="#2c4f73")
    for bar, value in zip(bars, values):
        ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2,
                "{:.3f}".format(value), va="center", fontsize=9)

    ax.set_xlabel("Weight")
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_title("Conditioning factor weights (AHP)")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.06,
            "consistency ratio = {:.3f}\n(acceptable below 0.10)".format(
                ahp_result.consistency_ratio),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f2f2f2",
                      edgecolor="#cccccc"))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def sensitivity_curve(records, labels, path, ahp_weight=None,
                      title="Sensitivity to the Sentinel-1 weight"):
    """AUC against each reference as the SAR weight varies."""
    weights = [r["weight"] for r in records]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    colours = ["#d7191c", "#2c7bb6", "#4daf4a"]

    for (label, colour) in zip(labels, colours):
        values = [r.get("auc_" + label, np.nan) for r in records]
        ax.plot(weights, values, "o-", color=colour, lw=2, ms=4, label=label)
        finite = [(w, v) for w, v in zip(weights, values) if np.isfinite(v)]
        if finite:
            w_best, v_best = max(finite, key=lambda t: t[1])
            ax.plot([w_best], [v_best], "*", color=colour, ms=15,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=5)

    if ahp_weight is not None:
        ax.axvline(ahp_weight, ls="--", lw=1.4, color="#666666")
        ax.text(ahp_weight, ax.get_ylim()[0], "  AHP prior = {:.2f}".format(
            ahp_weight), rotation=90, va="bottom", ha="left", fontsize=9,
            color="#444444")

    ax.set_xlabel("Weight given to Sentinel-1 water frequency")
    ax.set_ylabel("AUC")
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def normalisation_comparison(raw_accumulation, twi, domain, grid, path,
                             outline=None):
    """Why this pipeline does not normalise raw flow accumulation.

    Left: raw flow accumulation under plain min-max scaling. Right: the same
    information carried by TWI with percentile-clipped normalisation.
    """
    from .model import normalise

    naive = raw_accumulation.astype("float32").copy()
    valid = domain & np.isfinite(naive)
    lo = float(np.nanmin(naive[valid]))
    hi = float(np.nanmax(naive[valid]))
    naive = (naive - lo) / (hi - lo)
    naive[~valid] = np.nan

    fixed = normalise(twi, domain)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))

    im0 = show_map(axes[0], naive, domain, grid, cmap="magma", vmin=0, vmax=1,
                   title="Raw flow accumulation, plain min-max\n"
                         "(rejected)")
    fig.colorbar(im0, ax=axes[0], fraction=0.04, pad=0.02)

    im1 = show_map(axes[1], fixed, domain, grid, cmap="magma", vmin=0, vmax=1,
                   title="TWI, percentile-clipped normalisation\n"
                         "(adopted)")
    fig.colorbar(im1, ax=axes[1], fraction=0.04, pad=0.02)

    for ax in axes:
        outline_mask(ax, outline, grid, color="#00d0ff", lw=0.7)

    share = float(np.nanmean(naive[valid] < 0.01))
    axes[0].text(0.02, 0.97,
                 "{:.2f}% of pixels fall below 0.01".format(100 * share),
                 transform=axes[0].transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                           edgecolor="#cccccc", alpha=0.9))

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path

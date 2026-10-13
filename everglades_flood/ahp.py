"""Analytic Hierarchy Process weighting.

Weights picked by hand -- 0.4 for this factor, 0.25 for that one -- are not so
much wrong as unfalsifiable. A reader has no way to check them, and nothing
reveals whether the set is even self-consistent with the reasoning that
produced it.

AHP addresses both. The analyst states pairwise judgements -- "how much more
important is elevation than slope?" -- on Saaty's 1-9 scale, the weights fall
out as the principal eigenvector of that matrix, and the consistency ratio
measures whether those judgements contradict each other. A CR above 0.10
conventionally means the judgements should be revisited. The judgements
themselves are declared below, so the argument is on the page rather than in
the analyst's head.

Saaty, T.L. (1980). The Analytic Hierarchy Process. McGraw-Hill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Saaty's random consistency index: the mean CI of randomly generated
# reciprocal matrices of order n.
RANDOM_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}

# The conditioning factors, in matrix order.
FACTORS = [
    "sar_water_frequency",
    "elevation",
    "twi",
    "distance_to_channel",
    "slope",
    "landcover",
]

FACTOR_LABELS = {
    "sar_water_frequency": "Sentinel-1 water frequency",
    "elevation": "Elevation (inverted)",
    "twi": "Topographic wetness index",
    "distance_to_channel": "Distance to channel (inverted)",
    "slope": "Slope (inverted)",
    "landcover": "Land cover susceptibility",
}

# Upper triangle of the pairwise comparison matrix, read as
# "row factor is N times as important as column factor".
#
# The reasoning, which is the part a reader is entitled to argue with:
#
#   * Sentinel-1 water frequency is the only factor that observes inundation
#     rather than inferring it from shape, so it leads.
#   * Elevation is the dominant physical control on a floodplain with a metre
#     of total relief -- what floods is what is low.
#   * TWI and distance to channel describe where water collects and travels,
#     and matter, but both are derived from the same near-flat surface.
#   * Slope is deliberately ranked low. Over ground falling ~3 cm/km the
#     computed gradient is closer to DEM noise than to topography.
#   * Land cover is ranked last because in a wetland it is as much a
#     consequence of flooding as a cause of it.
PAIRWISE_JUDGEMENTS = {
    ("sar_water_frequency", "elevation"): 2,
    ("sar_water_frequency", "twi"): 4,
    ("sar_water_frequency", "distance_to_channel"): 4,
    ("sar_water_frequency", "slope"): 6,
    ("sar_water_frequency", "landcover"): 7,
    ("elevation", "twi"): 3,
    ("elevation", "distance_to_channel"): 3,
    ("elevation", "slope"): 5,
    ("elevation", "landcover"): 6,
    ("twi", "distance_to_channel"): 1,
    ("twi", "slope"): 2,
    ("twi", "landcover"): 3,
    ("distance_to_channel", "slope"): 2,
    ("distance_to_channel", "landcover"): 3,
    ("slope", "landcover"): 2,
}


@dataclass(frozen=True)
class AHPResult:
    factors: list
    matrix: np.ndarray
    weights: np.ndarray
    lambda_max: float
    consistency_index: float
    consistency_ratio: float

    @property
    def is_consistent(self) -> bool:
        return self.consistency_ratio < 0.10

    def as_dict(self):
        return dict(zip(self.factors, (float(w) for w in self.weights)))

    def report(self) -> str:
        lines = ["Factor weights (AHP)", "-" * 52]
        for factor, weight in sorted(
            self.as_dict().items(), key=lambda kv: -kv[1]
        ):
            lines.append("  {:<34} {:>6.3f}".format(FACTOR_LABELS[factor], weight))
        lines.append("-" * 52)
        lines.append("  lambda_max          {:.4f}".format(self.lambda_max))
        lines.append("  consistency index   {:.4f}".format(self.consistency_index))
        lines.append("  consistency ratio   {:.4f}  ({})".format(
            self.consistency_ratio,
            "consistent" if self.is_consistent else "INCONSISTENT, CR >= 0.10",
        ))
        return "\n".join(lines)


def build_matrix(factors=None, judgements=None) -> np.ndarray:
    """Assemble a reciprocal pairwise comparison matrix."""
    factors = list(factors or FACTORS)
    judgements = PAIRWISE_JUDGEMENTS if judgements is None else judgements

    index = {name: i for i, name in enumerate(factors)}
    n = len(factors)
    matrix = np.ones((n, n), dtype="float64")

    for (a, b), value in judgements.items():
        if a not in index or b not in index:
            raise KeyError("unknown factor in judgement: {!r} vs {!r}".format(a, b))
        if value <= 0:
            raise ValueError("judgement must be positive, got {}".format(value))
        i, j = index[a], index[b]
        matrix[i, j] = float(value)
        matrix[j, i] = 1.0 / float(value)

    return matrix


def solve(matrix, factors=None) -> AHPResult:
    """Derive weights and the consistency ratio from a pairwise matrix."""
    matrix = np.asarray(matrix, dtype="float64")
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("matrix must be square, got {}".format(matrix.shape))
    if n not in RANDOM_INDEX:
        raise ValueError("no random index published for n={}".format(n))

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    principal = int(np.argmax(eigenvalues.real))
    lambda_max = float(eigenvalues[principal].real)

    vector = np.abs(eigenvectors[:, principal].real)
    weights = vector / vector.sum()

    consistency_index = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    random_index = RANDOM_INDEX[n]
    consistency_ratio = (
        consistency_index / random_index if random_index > 0 else 0.0
    )

    return AHPResult(
        factors=list(factors or FACTORS),
        matrix=matrix,
        weights=weights,
        lambda_max=lambda_max,
        consistency_index=consistency_index,
        consistency_ratio=consistency_ratio,
    )


def default_weights() -> AHPResult:
    """The model's weights, from the judgements declared in this module."""
    return solve(build_matrix(), FACTORS)

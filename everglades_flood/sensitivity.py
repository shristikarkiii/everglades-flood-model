"""Weight sensitivity analysis.

AHP weights are a prior: they encode what the analyst believed before seeing
any outcome data. That is a virtue -- it is what stops the weights being tuned
until the map looks nice -- but it means the prior can be wrong, and the
honest thing to do is measure how much the result depends on it.

This sweeps the weight given to the Sentinel-1 layer from 0 to 0.7, holding
the other five factors in their AHP proportions, and scores every resulting
model against both independent references. Where the two references disagree
about the best weight, that disagreement is the finding.
"""

from __future__ import annotations

import numpy as np

from . import model, validate


def sweep(factors, base_weights, study, references, target="sar_water_frequency",
          values=None):
    """Score the model across a range of weights for one factor.

    `references` maps a name to (truth_mask, evaluation_mask).
    Returns a list of {weight, auc_<reference>...} records.
    """
    values = values if values is not None else np.linspace(0.0, 0.7, 15)

    others = {k: v for k, v in base_weights.items() if k != target}
    others_total = sum(others.values())
    if others_total <= 0:
        raise ValueError("the non-target weights sum to zero")

    records = []
    for weight in values:
        weights = {target: float(weight)}
        for name, value in others.items():
            weights[name] = (1.0 - float(weight)) * value / others_total

        index = model.susceptibility_index(factors, weights, study)

        record = {"weight": float(weight)}
        for label, (truth, evaluation) in references.items():
            try:
                record["auc_" + label] = validate.roc(
                    index, truth, evaluation, max_samples=200_000)["auc"]
            except ValueError:
                record["auc_" + label] = float("nan")
        records.append(record)

    return records


def best(records, label):
    """The weight maximising AUC against one reference."""
    key = "auc_" + label
    scored = [r for r in records if np.isfinite(r.get(key, np.nan))]
    if not scored:
        return None
    return max(scored, key=lambda r: r[key])


def report(records, labels):
    header = "{:>8}".format("w(SAR)") + "".join(
        "{:>16}".format(l) for l in labels)
    lines = [header, "-" * len(header)]
    for record in records:
        row = "{:>8.2f}".format(record["weight"])
        for label in labels:
            row += "{:>16.4f}".format(record.get("auc_" + label, float("nan")))
        lines.append(row)
    return "\n".join(lines)

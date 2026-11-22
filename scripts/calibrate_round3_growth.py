"""Fit empirical growth candidates and validate future observations.

Original observations stay private. The exported model contains fitted
parameters and aggregated validation only, permitting simulation without
redistributing original images or observations.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit, logit

from scripts.extract_round3_culture_data import COMMIT, REPOSITORY, write_json


MODELS = ("logistic", "lag_logistic", "exponential")


def curve(hours, params):
    t = np.asarray(hours, dtype=float)
    active = np.maximum(0, t - params.get("lag_hours", 0))
    if params["model"] == "exponential":
        return np.minimum(1.0, params["initial_density"] *
                          np.exp(np.minimum(700, params["rate_per_hour"] * active)))
    return expit(logit(params["initial_density"]) + params["rate_per_hour"] * active)


def crossing_hours(target, params):
    if target <= params["initial_density"]:
        return 0.0
    if params["model"] == "exponential":
        distance = np.log(target / params["initial_density"])
    else:
        distance = logit(target) - logit(params["initial_density"])
    return float(params.get("lag_hours", 0) + distance / params["rate_per_hour"])


def fit_curve(times, values, model):
    """Bounded least squares on measured density, with positive growth only."""
    t, y = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    minimum = 4 if model == "lag_logistic" else 3
    if len(t) < minimum or not np.all(np.diff(t) > 0):
        raise ValueError("Insufficient distinct ordered observations")
    if np.any((y <= 0) | (y >= 1)):
        raise ValueError("Density outside open unit interval")
    slope = float(np.polyfit(t, logit(y), 1)[0])
    if slope <= 0:
        raise ValueError("Nonpositive fitted growth trend")
    upper_lag = max(1.0, float(t[-2]))
    slope = np.clip(slope, 1e-5, 0.9)
    initial = np.clip(float(expit(logit(y[0]) - slope * t[0])), 1e-7, 0.95)
    lower, upper = [1e-7, 1e-5], [0.99, 1.0]
    starts = [0]
    if model == "lag_logistic":
        lower.append(0)
        upper.append(upper_lag)
        starts = [0, min(t[0], upper_lag * .9), upper_lag * .5]

    def unpack(x):
        return {"model": model, "initial_density": float(x[0]),
                "rate_per_hour": float(x[1]),
                "lag_hours": float(x[2]) if len(x) == 3 else 0.0}

    best = None
    for lag in starts:
        guess = [initial if not lag else float(y[0]), float(slope)]
        if model == "lag_logistic":
            guess.append(float(lag))
        fit = least_squares(lambda x: curve(t, unpack(x)) - y, guess,
                            bounds=(lower, upper), max_nfev=600,
                            ftol=1e-9, xtol=1e-9, gtol=1e-9)
        if fit.success and (best is None or fit.cost < best.cost):
            best = fit
    if best is None:
        raise ValueError("Growth fitting did not converge")
    params = unpack(best.x)
    params["fit_rmse"] = float(np.sqrt(np.mean((curve(t, params) - y) ** 2)))
    return params


def segments(frame):
    """Retain normal pre-interruption measurements, without inventing cutoffs."""
    result = []
    for identity, rows in frame.groupby("culture_id", sort=True):
        rows = rows.sort_values("time")
        passages = rows[rows.operation.str.endswith("Passage")]
        times = passages.time.to_numpy()
        for generation, passage in enumerate(times):
            end = times[generation + 1] if generation + 1 < len(times) else np.inf
            images = rows[(rows.time > passage) & (rows.time < end) &
                          rows.density.notna() &
                          rows.operation.str.endswith("GetImage")]
            result.append({
                "culture_id": identity, "cell_type": rows.cell_type.iloc[0],
                "generation": generation, "passage_time": int(passage),
                "times": (images.time.to_numpy() - passage) / 60,
                "values": images.density.to_numpy(),
                "point_count": len(images),
                "terminal": generation + 1 == len(times),
                "sampled": bool(rows.operation.str.endswith("Sampling").any()),
            })
    return result


def evaluate(cases):
    """Forward validation: fit prefixes, predict observations not used in fitting."""
    records = []
    for case in cases:
        t, y = case["times"], case["values"]
        target = .8 if case["cell_type"] == "HEK" else .3
        for model in MODELS:
            minimum = 4 if model == "lag_logistic" else 3
            for count in range(minimum, len(t)):
                try:
                    params = fit_curve(t[:count], y[:count], model)
                    prediction = float(curve([t[count]], params)[0])
                    row = {
                        "cell_type": case["cell_type"], "model": model,
                        "culture_id": case["culture_id"], "generation": case["generation"],
                        "prefix_points": count, "absolute_error": abs(prediction - y[count]),
                    }
                    if y[count - 1] < target <= y[count]:
                        actual = t[count - 1] + ((target - y[count - 1]) /
                                 (y[count] - y[count - 1])) * (t[count] - t[count - 1])
                        row["crossing_error_hours"] = crossing_hours(target, params) - actual
                        row["crossing_reference"] = "linear bracket interpolation, not ground truth"
                    records.append(row)
                except ValueError:
                    continue
    summaries = {}
    for kind in ("HEK", "iPS"):
        summaries[kind] = {}
        for model in MODELS:
            selected = [r for r in records if r["cell_type"] == kind and r["model"] == model]
            error = [r["absolute_error"] for r in selected]
            crossing = [abs(r["crossing_error_hours"]) for r in selected
                        if "crossing_error_hours" in r]
            summaries[kind][model] = {
                "forward_predictions": len(error),
                "density_mae": float(np.mean(error)) if error else None,
                "density_median_absolute_error": float(np.median(error)) if error else None,
                "density_p95_absolute_error": float(np.quantile(error, .95)) if error else None,
                "bracketed_crossings": len(crossing),
                "crossing_mae_hours": float(np.mean(crossing)) if crossing else None,
            }
    # Compare the candidates on matched prefixes, since lag needs one extra point.
    matched = {}
    for row in records:
        key = (row["cell_type"], row["culture_id"], row["generation"], row["prefix_points"])
        matched.setdefault(key, {})[row["model"]] = row
    for kind in ("HEK", "iPS"):
        common = [r for key, r in matched.items()
                  if key[0] == kind and all(model in r for model in MODELS)]
        for model in MODELS:
            summaries[kind][model]["matched_forward_predictions"] = len(common)
            summaries[kind][model]["matched_cultures"] = len({
                key[1] for key, r in matched.items()
                if key[0] == kind and all(m in r for m in MODELS)})
            summaries[kind][model]["matched_density_mae"] = (
                float(np.mean([r[model]["absolute_error"] for r in common])) if common else None)
    return records, summaries


def leave_culture_out(fitted, cases, chosen):
    """Forecast withheld cultures with median parameters from other cultures."""
    result = {}
    for kind in ("HEK", "iPS"):
        rows = []
        identities = sorted({c["culture_id"] for c in cases if c["cell_type"] == kind})
        for identity in identities:
            pool = [c for c in fitted if c["cell_type"] == kind and
                    c["culture_id"] != identity and c["model"] == chosen[kind]]
            if not pool:
                continue
            median = {"model": chosen[kind]}
            for field in ("initial_density", "rate_per_hour", "lag_hours"):
                median[field] = float(np.median([c[field] for c in pool]))
            errors = []
            for case in cases:
                if case["culture_id"] == identity and len(case["times"]):
                    errors.extend(abs(curve(case["times"], median) - case["values"]))
            rows.append({"culture": hashlib.sha256(identity.encode()).hexdigest()[:12],
                         "density_mae": float(np.mean(errors)), "points": len(errors)})
        result[kind] = {"cultures": rows,
                        "macro_density_mae": float(np.mean([r["density_mae"] for r in rows]))}
    return result


def calibrate(source, output, private_output):
    output.mkdir(parents=True, exist_ok=True)
    private_output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source / "normalized_history.csv")
    cases = segments(frame)
    fits, exclusions = [], []
    for case in cases:
        for model in MODELS:
            try:
                params = fit_curve(case["times"], case["values"], model)
                fits.append({**params, **{k: case[k] for k in
                            ("culture_id", "cell_type", "generation", "point_count")}})
            except ValueError as error:
                exclusions.append({"culture_id": case["culture_id"],
                                   "cell_type": case["cell_type"],
                                   "generation": case["generation"], "model": model,
                                   "points": case["point_count"], "reason": str(error)})
    validation_rows, validation = evaluate(cases)
    chosen = {}
    for kind in ("HEK", "iPS"):
        comparison = validation[kind]
        lag_supported = (comparison["lag_logistic"]["matched_forward_predictions"] >= 10
                         and comparison["lag_logistic"]["matched_cultures"] >= 3)
        eligible = list(MODELS) if lag_supported else ["logistic", "exponential"]
        field = "matched_density_mae" if lag_supported else "density_mae"
        valid = [(comparison[model][field], model) for model in eligible
                 if comparison[model][field] is not None]
        if not valid:
            # Small HEK groups may not permit a matched three-parameter comparison.
            valid = [(v["density_mae"], model) for model, v in validation[kind].items()
                     if v["density_mae"] is not None]
        chosen[kind] = min(valid)[1]
    pool = []
    for fit in fits:
        if fit["model"] == chosen[fit["cell_type"]]:
            exported = {k: v for k, v in fit.items() if k != "culture_id"}
            exported["donor"] = hashlib.sha256(fit["culture_id"].encode()).hexdigest()[:12]
            pool.append(exported)
    write_json(output / "growth_model.json", {
        "source_repository": REPOSITORY, "source_commit": COMMIT,
        "normalized_input_sha256": hashlib.sha256(
            (source / "normalized_history.csv").read_bytes()).hexdigest(),
        "chosen_models": chosen,
        "selection": ("Lowest matched-prefix forward density MAE; lag candidate requires "
                      "at least 10 matched forecasts across 3 cultures, an agent-selected "
                      "minimum evidence rule. Otherwise compare two-parameter models "
                      "on all eligible forward forecasts. Exploratory, no final test set."),
        "carrying_capacity": 1.0,
        "parameter_pool": pool,
        "parameter_sampling": "Uniform donor block; generation-matched joint parameters",
        "missing_generation": "Draw a complete joint record from same-type generation pool",
        "variation": "Empirical fitted variation, not independently established population variance",
        "bounds": {"initial_density": [1e-7, .99], "rate_per_hour": [1e-5, 1.0],
                   "lag_hours": "0 to penultimate observation time"},
        "validation": validation,
        "leave_culture_out": leave_culture_out(fits, cases, chosen),
        "point_counts": {"original_density_points": int(frame.density.notna().sum()),
                         "segments": len(cases), "chosen_fitted_segments": len(pool)},
        "limitations": [
            "Model selection uses forward validation; no independent final validation dataset",
            "Only five source cultures per type; repetitions are synthetic, not biological replicates",
            "Normal pre-interruption values retained; interruption causes/cutoffs not independently known",
            "Sparse or nonpositive-trend intervals excluded from parameter estimation, not labeled failures",
            "Image-derived density scale interpreted from source; not recalibrated against cell counts",
        ],
    })
    write_json(private_output / "fit_audit.json", {
        "fits": fits, "exclusions": exclusions, "forward_validation": validation_rows,
        "segments": [{**c, "times": c["times"].tolist(), "values": c["values"].tolist()}
                     for c in cases],
    })
    print(json.dumps({"chosen": chosen, "fitted_pool": len(pool),
                      "validation": validation}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()
    calibrate(args.source, args.output, args.private_output)


if __name__ == "__main__":
    main()

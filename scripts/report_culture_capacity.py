"""Create a portable visual report from verified culture-capacity records.

All figures and the standalone HTML are written to the requested local output.
No source observations are embedded. Private fit-validation supplements are
produced separately when requested.
"""

import argparse
import base64
from collections import Counter
import gzip
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gems30-mpl")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.calibrate_round3_growth import curve, crossing_hours
from scripts.extract_round3_culture_data import write_json
from scripts.measure_culture_capacity import fingerprint


ROLE_LABELS = {"HEKPassage": "HEK passage", "HEKSampling": "HEK sampling",
               "iPSPassage": "iPS passage", "iPSSampling": "iPS sampling"}


def wilson(successes, trials):
    if not trials:
        return None, None
    z = 1.959963984540054
    proportion = successes/trials
    denominator = 1+z*z/trials
    center = (proportion+z*z/(2*trials))/denominator
    half = z*math.sqrt(proportion*(1-proportion)/trials+z*z/(4*trials*trials))/denominator
    return max(0, center-half), min(1, center+half)


def read_records(root):
    records, targets, representative, medium_gaps = [], [], {}, []
    for path in sorted(root.glob("h*_i*_r*/result.json")):
        try:
            result = json.loads(path.read_text())
            events = json.load(gzip.open(path.with_name("events.json.gz"), "rt"))
            request = json.loads(path.with_name("input.json").read_text())
        except (json.JSONDecodeError, FileNotFoundError, EOFError):
            continue  # Explicit partial previews may encounter an unfinished write.
        records.append(result)
        h, i, r = result["hek"], result["ips"], result["repetition"]
        origins = {key: 0 for key in request["cultures"]}
        last_medium = {}
        for event in events:
            if event["target"] is not None:
                diagnostic = dict(event)
                if "nominal_density_error" not in diagnostic:
                    identity = event["culture_id"]
                    nominal_start = event["group_optimal_start"] + event["start"]-event["group_start"]
                    parameters = request["cultures"][identity]["parameters"][event["generation"]]
                    nominal = float(curve([(nominal_start-origins[identity])/60], parameters)[0])
                    diagnostic.update(nominal_density_error=nominal-event["target"],
                                      nominal_action_density=nominal,
                                      schedule_density_change=event["density"]-nominal)
                targets.append({**diagnostic, "hek": h, "ips": i, "repetition": r,
                                "completed_run": result["completed"],
                                "genome_hash": fingerprint(
                                    request["cultures"][event["culture_id"]]["parameters"])})
            if event["operation"].endswith("Passage"):
                origins[event["culture_id"]] = event["end"]
                if event["cell_type"] == "iPS":
                    last_medium[event["culture_id"]] = event["end"]
            elif event["operation"].endswith("MediumChange"):
                identity = event["culture_id"]
                if identity in last_medium:
                    medium_gaps.append({
                        "hek": h, "ips": i, "repetition": r, "culture_id": identity,
                        "generation": event["generation"], "group_state": event["group_state"],
                        "previous_medium_end": last_medium[identity],
                        "medium_start": event["start"],
                        "gap_hours": (event["start"]-last_medium[identity])/60,
                        "completed_run": result["completed"],
                    })
                last_medium[identity] = event["end"]
        representative.setdefault((h, i), []).append({
            "result": result, "events": events, "input": request,
        })
    chosen = {}
    for pair, cases in representative.items():
        complete = [c for c in cases if c["result"]["completed"]]
        eligible = complete or cases
        median_time = np.median([c["result"]["virtual_minutes"] for c in eligible])
        case = min(eligible, key=lambda c: (
            abs(c["result"]["virtual_minutes"]-median_time), c["result"]["repetition"]))
        chosen[pair] = case
    return records, pd.DataFrame(targets), chosen, pd.DataFrame(medium_gaps)


def summarize(records):
    groups = {}
    for row in records:
        groups.setdefault((row["hek"], row["ips"]), []).append(row)
    conditions = []
    for (h, i), rows in sorted(groups.items()):
        n = len(rows)
        complete = [r for r in rows if r["completed"]]
        accepted = sum(r["primary_accepted"] for r in rows)
        alternate = sum(r["supplementary_accepted"] for r in rows)
        lower, upper = wilson(accepted, n)
        conditions.append({
            "hek": h, "ips": i, "repetitions_recorded": n,
            "completed": len(complete), "primary_accepted": accepted,
            "supplementary_accepted": alternate,
            "primary_fraction": accepted/n, "primary_wilson_low": lower,
            "primary_wilson_high": upper,
            "median_virtual_days": float(np.median([r["virtual_minutes"]/1440 for r in complete]))
            if complete else None,
            "median_utilization": float(np.median([r["machine_utilization"] for r in complete]))
            if complete else None,
            "median_wall_seconds": float(np.median([r["wall_seconds"] for r in rows])),
            "max_wall_seconds": max(r["wall_seconds"] for r in rows),
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
        })
    return conditions


def operation_diagnostics(targets, output, reference_targets=None):
    """Keep culture-level acceptance and matched parameter-prefix effects visible."""
    culture_rows = []
    completed = targets[targets.completed_run] if len(targets) else targets
    if len(completed):
        for (h, i, r, identity), rows in completed.groupby(
                ["hek", "ips", "repetition", "culture_id"]):
            culture_rows.append({
                "hek": int(h), "ips": int(i), "repetition": int(r), "culture_id": identity,
                "cell_type": rows.cell_type.iloc[0], "scored_operations": len(rows),
                "primary_accepted": bool(len(rows) == 3 and rows.primary_pass.all()),
                "supplementary_accepted": bool(len(rows) == 3 and rows.supplementary_pass.all()),
                "mean_absolute_error": float(rows.error.abs().mean()),
            })
    cultures = pd.DataFrame(culture_rows)
    cultures.to_csv(output / "culture_acceptance.csv", index=False)
    diagnostics = {"completed_cultures": len(culture_rows), "by_cell_type": {},
                   "paired_prefix": []}
    if len(cultures):
        for kind, rows in cultures.groupby("cell_type"):
            diagnostics["by_cell_type"][kind] = {
                "cultures": len(rows), "primary_accepted": int(rows.primary_accepted.sum()),
                "supplementary_accepted": int(rows.supplementary_accepted.sum())}
    if reference_targets is None or not len(reference_targets) or not len(completed):
        return diagnostics
    reference = reference_targets[reference_targets.completed_run]
    main_prefix = completed[completed.culture_id.isin(["HEK-00", "iPS-00"])]
    keys = ["repetition", "culture_id", "generation", "operation"]
    matched = main_prefix.merge(reference[keys+["error", "primary_pass", "genome_hash"]],
                                on=keys, suffixes=("", "_reference"), validate="many_to_one")
    if not (matched.genome_hash == matched.genome_hash_reference).all():
        raise ValueError("Matched reference cultures do not have identical parameter draws")
    matched["absolute_error_change"] = matched.error.abs()-matched.error_reference.abs()
    matched.to_csv(output / "paired_prefix_operations.csv", index=False)
    for (h, i), rows in matched.groupby(["hek", "ips"]):
        six_operation_passes = [
            bool(len(block) == 6 and block.primary_pass.all())
            for _, block in rows.groupby("repetition")]
        diagnostics["paired_prefix"].append({
            "hek": int(h), "ips": int(i), "matched_operations": len(rows),
            "matched_repetitions": int(rows.repetition.nunique()),
            "prefix_primary_accepted": sum(six_operation_passes),
            "mean_absolute_error_change": float(rows.absolute_error_change.mean()),
            "median_absolute_error_change": float(rows.absolute_error_change.median()),
        })
    return diagnostics


def medium_gap_figure(gaps, output):
    if not len(gaps):
        return None
    gaps.to_csv(output / "medium_change_gaps.csv", index=False)
    complete = gaps[gaps.completed_run]
    if not len(complete):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.arange(0, max(74, math.ceil(complete.gap_hours.max())+3), 2)
    for state, group in complete.groupby("group_state"):
        axes[0].hist(group.gap_hours, bins=bins, alpha=.55,
                     label=state.replace("IPS", "iPS "))
    axes[0].set(xlabel="Hours from previous medium/passage end to medium-change start",
                ylabel="Operation count", title="Source protocol includes a first ~48-hour gap")
    axes[0].legend(fontsize=8)
    condition_gaps = complete.groupby(["hek", "ips"]).gap_hours.quantile([.5, .95]).unstack()
    axes[1].scatter([h+i for h, i in condition_gaps.index], condition_gaps[.5],
                    label="Condition median", alpha=.5)
    axes[1].scatter([h+i for h, i in condition_gaps.index], condition_gaps[.95],
                    label="Condition 95th percentile", alpha=.5)
    axes[1].set(xlabel="Total cultures", ylabel="Medium gap (hours)",
                title="Timing diagnostics; no additional pass/fail threshold")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, output, "medium_gaps")
    return {f"p{p}": float(complete.gap_hours.quantile(p/100)) for p in [0, 50, 95, 100]}


def paired_prefix_figure(diagnostics, config, output):
    rows = diagnostics["paired_prefix"]
    if not rows:
        return False
    hs, iss = config["hek_counts"], config["ips_counts"]
    fraction = np.full((len(iss), len(hs)), np.nan)
    change = fraction.copy()
    labels = {}
    for row in rows:
        if row["hek"] not in hs or row["ips"] not in iss:
            continue
        x, y = hs.index(row["hek"]), iss.index(row["ips"])
        fraction[y, x] = row["prefix_primary_accepted"]/row["matched_repetitions"]
        change[y, x] = row["mean_absolute_error_change"]
        labels[(y, x)] = f"{row['prefix_primary_accepted']}/{row['matched_repetitions']}"
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.7))
    a = axes[0].imshow(fraction, origin="lower", vmin=0, vmax=1, cmap="viridis")
    finite = change[np.isfinite(change)]
    extent = max(.02, float(np.max(np.abs(finite)))) if len(finite) else .1
    b = axes[1].imshow(change, origin="lower", vmin=-extent, vmax=extent, cmap="RdBu_r")
    for (y, x), label in labels.items():
        axes[0].text(x, y, label, ha="center", va="center",
                     color="white" if fraction[y, x] < .55 else "#111827", fontsize=8)
        axes[1].text(x, y, f"{change[y,x]:+.3f}", ha="center", va="center", fontsize=8)
    for axis in axes:
        axis.set(xticks=range(len(hs)), xticklabels=hs, yticks=range(len(iss)), yticklabels=iss,
                 xlabel="HEK cultures", ylabel="iPS cultures")
    axes[0].set_title("Fixed HEK-00 / iPS-00 prefix: all six operations pass")
    axes[1].set_title("Mean absolute density error change from paired 1+1")
    fig.colorbar(a, ax=axes[0], fraction=.04)
    fig.colorbar(b, ax=axes[1], fraction=.04)
    fig.text(.5, .01, "Identical parameter draws; matched completed runs only. "
             "Positive change means larger error than 1+1.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save_figure(fig, output, "paired_prefix")
    return True


def save_figure(figure, output, name):
    figure.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)


def heatmaps(conditions, config, output):
    hs, iss = config["hek_counts"], config["ips_counts"]
    items = [
        ("completed", "Completion through sampling", "completion"),
        ("primary_accepted", "Primary density criterion: target ±0.1", "primary_acceptance"),
        ("supplementary_accepted", "Supplementary density criterion", "supplementary_acceptance"),
    ]
    for field, title, name in items:
        matrix = np.full((len(iss), len(hs)), np.nan)
        labels = {}
        for row in conditions:
            if row["hek"] not in hs or row["ips"] not in iss:
                continue
            x, y = hs.index(row["hek"]), iss.index(row["ips"])
            matrix[y, x] = row[field]/row["repetitions_recorded"]
            labels[(y, x)] = f"{row[field]}/{row['repetitions_recorded']}"
        fig, ax = plt.subplots(figsize=(7.3, 6.4))
        image = ax.imshow(matrix, origin="lower", vmin=0, vmax=1, cmap="viridis")
        ax.set(xticks=range(len(hs)), xticklabels=hs, yticks=range(len(iss)),
               yticklabels=iss, xlabel="HEK cultures", ylabel="iPS cultures", title=title)
        for (y, x), label in labels.items():
            ax.text(x, y, label, ha="center", va="center",
                    color="white" if matrix[y, x] < .55 else "#111827", fontsize=10)
        fig.colorbar(image, ax=ax, label="Fraction of recorded runs")
        fig.text(.5, .015, "All cultures must finish; density acceptance also checks every targeted operation.",
                 ha="center", fontsize=8)
        save_figure(fig, output, name)
    matrix = np.full((len(iss), len(hs)), np.nan)
    for row in conditions:
        if row["hek"] in hs and row["ips"] in iss and row["median_utilization"] is not None:
            matrix[iss.index(row["ips"]), hs.index(row["hek"])] = row["median_utilization"]
    fig, ax = plt.subplots(figsize=(7.3, 6.4))
    image = ax.imshow(matrix, origin="lower", vmin=0, vmax=1, cmap="cividis")
    ax.set(xticks=range(len(hs)), xticklabels=hs, yticks=range(len(iss)), yticklabels=iss,
           xlabel="HEK cultures", ylabel="iPS cultures", title="Median shared-resource utilization")
    for y in range(len(iss)):
        for x in range(len(hs)):
            if np.isfinite(matrix[y, x]):
                ax.text(x, y, f"{matrix[y,x]:.0%}", ha="center", va="center",
                        color="white" if matrix[y,x] < .5 else "#111827")
    fig.colorbar(image, ax=ax, label="Busy fraction; includes refill, completed runs only")
    save_figure(fig, output, "utilization")


def density_figures(targets, config, output):
    complete = targets[targets.completed_run] if len(targets) else targets
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    rng = np.random.default_rng(30)
    roles = list(ROLE_LABELS)
    actual_target = {"HEKPassage": .8, "HEKSampling": .4,
                     "iPSPassage": .3, "iPSSampling": .3}
    for index, role in enumerate(roles):
        selected = complete[complete.operation == role] if len(complete) else complete
        density = selected.density.to_numpy() if len(selected) else np.array([])
        target = actual_target[role]
        width = min(.1, .2*target)
        axes[0].fill_between([index-.38, index+.38], [target-.1]*2, [target+.1]*2,
                             color="#cbd5e1", alpha=.7)
        axes[0].fill_between([index-.38, index+.38], [target-width]*2, [target+width]*2,
                             color="#93c5fd", alpha=.7)
        axes[0].plot([index-.38, index+.38], [target, target], color="#111827", lw=1.2)
        if len(density):
            axes[0].boxplot([density], positions=[index], widths=.28, showfliers=False,
                            patch_artist=True, boxprops={"facecolor": "white"})
            sample = rng.choice(density, min(350, len(density)), replace=False)
            axes[0].scatter(index+rng.uniform(-.2, .2, len(sample)), sample,
                            s=5, color="#334155", alpha=.18)
        with_nominal = selected[selected.nominal_density_error.notna()] if (
            len(selected) and "nominal_density_error" in selected) else pd.DataFrame()
        if len(with_nominal):
            sample = with_nominal.sample(min(1200, len(with_nominal)), random_state=30)
            axes[1].scatter(sample.nominal_density_error, sample.error, s=8, alpha=.22,
                            label=ROLE_LABELS[role])
    axes[0].set(xticks=range(4), xticklabels=[ROLE_LABELS[r] for r in roles],
                ylim=(0, 1), ylabel="Simulated pre-operation density",
                title="Density bands and actual operation densities")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].axhspan(-.1, .1, color="#e2e8f0", alpha=.6)
    axes[1].axvspan(-.1, .1, color="#e2e8f0", alpha=.4)
    axes[1].axhline(0, color="#94a3b8", lw=.7)
    axes[1].axvline(0, color="#94a3b8", lw=.7)
    axes[1].plot([-.5, .8], [-.5, .8], ls="--", lw=1, color="#475569")
    axes[1].set(xlabel="Density error at nominal action time",
                ylabel="Density error at actual action time",
                title="Forecast/time convention versus scheduling displacement")
    if axes[1].collections:
        axes[1].legend(fontsize=8)
    fig.text(.5, .015, "Completed runs. Points may be thinned for display; bands are provisional.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save_figure(fig, output, "density_errors")


def calibration_figures(model, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    names = ["logistic", "lag_logistic", "exponential"]
    validation = model.get("validation", {})
    for offset, kind in enumerate(("HEK", "iPS")):
        values = [validation.get(kind, {}).get(name, {}).get("matched_density_mae")
                  for name in names]
        values = [v if v is not None else 0 for v in values]
        axes[0].bar(np.arange(3)+(offset-.5)*.35, values, width=.35, label=kind)
    axes[0].set(xticks=range(3), xticklabels=["Logistic", "Lag logistic", "Exponential"],
                ylabel="Matched-prefix forward density MAE", title="Growth-model candidates")
    axes[0].legend()
    for offset, kind in enumerate(("HEK", "iPS")):
        pool = [p for p in model["parameter_pool"] if p["cell_type"] == kind
                and (kind != "HEK" or p["generation"] > 0)]
        x = [p["generation"]-(1 if kind == "HEK" else 0)+1+(offset-.5)*.18 for p in pool]
        axes[1].scatter(x, [p["rate_per_hour"] for p in pool], label=kind, s=28, alpha=.7)
        axes[2].scatter(x, [p["initial_density"] for p in pool], label=kind, s=28, alpha=.7)
    axes[1].set(xlabel="Protocol cycle", ylabel="Fitted rate (1/hour)",
                xticks=[1, 2, 3], title="Joint empirical growth variation")
    axes[2].set(xlabel="Protocol cycle", ylabel="Fitted post-passage density",
                xticks=[1, 2, 3], yscale="log", title="Empirical reseeding variation")
    axes[1].legend()
    fig.text(.5, .01, "Source cultures are technical culture units, not human donors. "
             "HEK lag comparison has insufficient matched forecasts.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save_figure(fig, output, "calibration")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind, axis in zip(("HEK", "iPS"), axes):
        rows = model["leave_culture_out"][kind]["cultures"]
        axis.bar(range(len(rows)), [r["density_mae"] for r in rows], color="#64748b")
        chosen = model["validation"][kind][model["chosen_models"][kind]]
        axis.axhline(chosen["density_mae"], color="#2563eb", ls="--",
                    label="Online forward MAE within source histories")
        axis.set(xticks=range(len(rows)), xticklabels=[f"C{j+1}" for j in range(len(rows))],
                 ylabel="Density MAE", title=f"{kind}: leave-one-source-culture-out")
        axis.legend(fontsize=8)
    fig.text(.5, .01, "Median remaining-culture parameters predict the held-out culture. "
             "Different information from an online history fit.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save_figure(fig, output, "generalization")


def tolerance_figure(model, output):
    """Translate provisional bands into durations on the fitted normal curves."""
    definitions = {
        "HEKPassage": ("HEK", [1, 2], .8),
        "HEKSampling": ("HEK", [3], .4),
        "iPSPassage": ("iPS", [0, 1], .3),
        "iPSSampling": ("iPS", [2], .3),
    }
    rows = []
    for role, (kind, generations, target) in definitions.items():
        for p in model["parameter_pool"]:
            if p["cell_type"] != kind or p["generation"] not in generations:
                continue
            for name, half_width in [("Primary ±0.1", .1),
                                     ("Supplementary capped ±20%", min(.1, .2*target))]:
                lower, upper = crossing_hours(target-half_width, p), crossing_hours(target+half_width, p)
                rows.append({"operation": role, "criterion": name, "source_culture": p["donor"],
                             "source_generation": p["generation"], "target": target,
                             "half_width": half_width, "lower_crossing_hours": lower,
                             "upper_crossing_hours": upper, "window_hours": upper-lower})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "tolerance_windows.csv", index=False)
    fig, axis = plt.subplots(figsize=(10, 4.5))
    for index, role in enumerate(definitions):
        for offset, criterion, color in [
                (-.16, "Primary ±0.1", "#94a3b8"),
                (.16, "Supplementary capped ±20%", "#60a5fa")]:
            subset = frame[(frame.operation == role) & (frame.criterion == criterion)]
            axis.boxplot([subset.window_hours], positions=[index+offset], widths=.24,
                         showfliers=True, patch_artist=True, boxprops={"facecolor": color})
    axis.set(xticks=range(4), xticklabels=[ROLE_LABELS[r] for r in definitions],
             ylabel="Hours spent inside provisional density band",
             title="Density tolerance gives different time margins across normal curves")
    axis.text(.01, .96, "Grey: primary ±0.1   Blue: capped ±20%", transform=axis.transAxes,
              va="top", fontsize=9)
    fig.tight_layout()
    save_figure(fig, output, "tolerance_windows")


def representative_figures(case, output, prefix):
    result, events, cultures = case["result"], case["events"], case["input"]["cultures"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 1, .65]},
                             sharex=True)
    palette = plt.colormaps["tab10"]
    for kind, ax in zip(("HEK", "iPS"), axes):
        for index, (identity, record) in enumerate(
                (pair for pair in cultures.items() if pair[1]["cell_type"] == kind)):
            own = [e for e in events if e["culture_id"] == identity]
            passages = [e["end"] for e in own if e["operation"].endswith("Passage")]
            origins = [0]+passages if kind == "HEK" else passages
            sample = next((e for e in own if e["operation"].endswith("Sampling")), None)
            ending = sample["start"] if sample else result["virtual_minutes"]
            for generation, beginning in enumerate(origins):
                if generation >= len(record["parameters"]):
                    break
                last = origins[generation+1] if generation+1 < len(origins) else ending
                times = np.linspace(beginning, last, 75)
                ax.plot(times/1440, curve((times-beginning)/60, record["parameters"][generation]),
                        color=palette(index % 10), lw=1.1,
                        label=identity if generation == 0 else None)
            scored = [e for e in own if e["target"] is not None]
            for event in scored:
                ax.scatter(event["start"]/1440, event["density"], s=25,
                           color=palette(index % 10), marker="o" if event["primary_pass"] else "x")
        targets = [.8, .4] if kind == "HEK" else [.3]
        for target in targets:
            ax.axhspan(target-.1, target+.1, alpha=.1, color="#64748b")
            ax.axhline(target, lw=.8, ls="--", color="#64748b")
        ax.set(ylim=(0, 1), ylabel=f"{kind} density")
        ax.legend(ncol=5, fontsize=7, loc="upper right")
    days = math.ceil(result["virtual_minutes"]/1440)
    cell_busy, refill_busy = np.zeros(days), np.zeros(days)
    for event in events:
        if event["duration"] == 0:
            continue
        for day in range(int(event["start"]//1440), min(days, int(event["end"]//1440)+1)):
            overlap = max(0, min(event["end"], (day+1)*1440)-max(event["start"], day*1440))
            (refill_busy if event["cell_type"] == "refill" else cell_busy)[day] += overlap/1440
    axes[2].bar(np.arange(days)+.5, cell_busy, width=.85, label="Culture operations")
    axes[2].bar(np.arange(days)+.5, refill_busy, bottom=cell_busy, width=.85, label="Refill")
    axes[2].set(xlabel="Virtual days since enrollment", ylabel="Daily busy fraction", ylim=(0, 1))
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{result['hek']} HEK + {result['ips']} iPS, repetition {result['repetition']} "
                 f"— {'accepted' if result['primary_accepted'] else 'not accepted'}")
    fig.tight_layout(rect=(0, 0, 1, .97))
    save_figure(fig, output, prefix)


def private_fit_supplement(audit_path, model, output):
    audit = json.loads(audit_path.read_text())
    cases = audit["segments"]
    fig, axes = plt.subplots(math.ceil(len(cases)/5), 5, figsize=(16, 18))
    for axis, case in zip(axes.flat, cases):
        axis.scatter(np.array(case["times"])/24, case["values"], s=9, color="#334155")
        selected = next((p for p in audit["fits"] if p["culture_id"] == case["culture_id"] and
                         p["generation"] == case["generation"] and
                         p["model"] == model["chosen_models"][case["cell_type"]]), None)
        if selected:
            times = np.linspace(0, max(case["times"]), 100)
            axis.plot(times/24, curve(times, selected), color="#2563eb")
        axis.set_title(f"{case['culture_id']} cycle {case['generation']}", fontsize=8)
        axis.set_ylim(0, 1)
    for axis in axes.flat[len(cases):]:
        axis.set_visible(False)
    fig.suptitle("PRIVATE source-observation fit check — excluded from the portable report")
    fig.supxlabel("Days since recorded passage")
    fig.supylabel("Recorded image-derived density")
    fig.tight_layout(rect=(0, 0, 1, .97))
    save_figure(fig, output, "private_observation_fits")


def write_report(root, model_path, output, reference=None, allow_partial=False, private_audit=None):
    if output.exists():
        raise FileExistsError("Use a new report directory to preserve previous artifacts")
    metadata = json.loads((root / "metadata.json").read_text())
    model = json.loads(model_path.read_text())
    if fingerprint(model) != metadata["model_hash"]:
        raise ValueError("Report model differs from recorded simulation model")
    records, targets, representatives, medium_gaps = read_records(root)
    if not allow_partial:
        verification = json.loads((root / "verification.json").read_text())
        if not verification["valid"] or len(records) != len(metadata["cases"]):
            raise ValueError("Final report requires verified outcomes for every planned run")
    output.mkdir(parents=True)
    figures = output / "figures"
    figures.mkdir()
    conditions = summarize(records)
    heatmaps(conditions, metadata["config"], figures)
    density_figures(targets, metadata["config"], figures)
    calibration_figures(model, figures)
    tolerance_figure(model, figures)
    for pair in [(3, 3), (5, 5), (10, 10)]:
        if pair in representatives:
            representative_figures(representatives[pair], figures, f"timeline_{pair[0]}_{pair[1]}")
    flat_records = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))}
                    for row in records]
    pd.DataFrame(flat_records).to_csv(output / "runs.csv", index=False)
    pd.DataFrame([{**c, "outcomes": json.dumps(c["outcomes"], sort_keys=True)}
                  for c in conditions]).to_csv(output / "conditions.csv", index=False)
    if len(targets):
        targets.to_csv(output / "targeted_operations.csv", index=False)
    ref_summary, reference_records, reference_targets = None, [], None
    if reference is not None:
        reference_metadata = json.loads((reference / "metadata.json").read_text())
        if any(reference_metadata[k] != metadata[k] for k in ("config_hash", "model_hash")):
            raise ValueError("Reference config/model differs from the main study")
        reference_verification = json.loads((reference / "verification.json").read_text())
        if not reference_verification["valid"]:
            raise ValueError("Reference records require independent verification")
        reference_records, reference_targets, _, _ = read_records(reference)
        ref_summary = {
            "runs": len(reference_records),
            "completed": sum(r["completed"] for r in reference_records),
            "primary_accepted": sum(r["primary_accepted"] for r in reference_records),
            "supplementary_accepted": sum(r["supplementary_accepted"] for r in reference_records),
            "interpretation": "Same policy/refill/windows, paired source-parameter prefixes; not a no-constraint oracle",
        }
    diagnostics = operation_diagnostics(targets, output, reference_targets)
    diagnostics["medium_gap_hours"] = medium_gap_figure(medium_gaps, figures)
    has_paired = paired_prefix_figure(diagnostics, metadata["config"], figures)
    write_json(output / "diagnostics.json", diagnostics)
    robust = [c for c in conditions if c["primary_accepted"] == c["repetitions_recorded"]
              and c["repetitions_recorded"] == metadata["config"]["repetitions"]]
    summary = {
        "scope": metadata["scope"], "planned_runs": len(metadata["cases"]),
        "recorded_runs": len(records), "all_records_present": len(records) == len(metadata["cases"]),
        "completed": sum(r["completed"] for r in records),
        "primary_accepted": sum(r["primary_accepted"] for r in records),
        "supplementary_accepted": sum(r["supplementary_accepted"] for r in records),
        "outcomes": dict(Counter(r["outcome"] for r in records)),
        "all_30_primary_pairs": [(c["hek"], c["ips"]) for c in robust],
        "conditions": conditions, "reference": ref_summary,
        "diagnostics": diagnostics,
        "model_hash": metadata["model_hash"], "config_hash": metadata["config_hash"],
        "source_commit": model["source_commit"], "simulation_code_commit": metadata["environment"]["git_commit"],
        "interpretation": "Conditional Monte Carlo frequencies; no wet-culture probability or upper capacity guarantee",
    }
    role_rows = []
    if len(targets):
        for role, rows in targets[targets.completed_run].groupby("operation"):
            role_rows.append({
                "operation": ROLE_LABELS[role], "operations": len(rows),
                "primary_within_band": int(rows.primary_pass.sum()),
                "supplementary_within_band": int(rows.supplementary_pass.sum()),
                "mean_absolute_density_error": float(rows.error.abs().mean()),
                "nominal_outside_primary": int((rows.nominal_density_error.abs() >
                                               metadata["config"]["primary_density_half_width"]+1e-12).sum()),
                "actual_outside_primary": int((~rows.primary_pass).sum()),
            })
    summary["operation_roles"] = role_rows
    summary["report_generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(output / "summary.json", summary)
    captions = {
        "completion": "Fraction of runs that reached sampling for every culture.",
        "primary_acceptance": "All cultures sampled and every targeted passage/sampling density within target ±0.1.",
        "supplementary_acceptance": "Same runs, rescored using ±20% of target with absolute half-width capped at 0.1.",
        "utilization": "Completed runs only. Includes culture work and refill; mean utilization does not reveal all timing bottlenecks.",
        "density_errors": "Grey and blue bands are primary and supplementary criteria. Nominal error includes forecast and time conventions; displacement also includes readiness, time-window preferences and solver choices.",
        "calibration": "Matched-prefix validation and fitted parameter variation. HEK has only two matched lag forecasts; an extra-parameter model was not selected on that evidence.",
        "generalization": "Cross-culture prediction is less accurate than extrapolation from a culture's own earlier observations. Five technical culture units per type do not establish a population distribution.",
        "tolerance_windows": "Widths use the fitted normal curves for the protocol's relevant cycles. They are descriptive time margins, not independently validated biological tolerances. Absolute or relative density bands alone do not give every cell type the same processing-time margin.",
    }
    if has_paired:
        captions["paired_prefix"] = (
            "A fixed two-culture prefix controls the number of scored operations. "
            "Both conditions use identical fitted parameter draws. These paired policy outcomes "
            "help interpret load effects; they do not prove that resource contention alone caused a difference.")
    if diagnostics["medium_gap_hours"] is not None:
        captions["medium_gaps"] = (
            "Passage resets medium age. Gaps run from the previous medium-change/passage end "
            "to the next medium-change start. The source's initial state route contains two "
            "24-hour image waits before the following medium change, so some gaps are about 48 hours "
            "even at low load. No biological timing cutoff was selected.")
    image_sections = ""
    for name, caption in captions.items():
        content = base64.b64encode((figures / f"{name}.png").read_bytes()).decode()
        image_sections += (f'<figure><img src="data:image/png;base64,{content}" '
                           f'alt="{html.escape(name)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    for pair in [(3, 3), (5, 5), (10, 10)]:
        name = f"timeline_{pair[0]}_{pair[1]}"
        if (figures / f"{name}.png").exists():
            content = base64.b64encode((figures / f"{name}.png").read_bytes()).decode()
            image_sections += (f'<figure><img src="data:image/png;base64,{content}" alt="{name}">'
                               '<figcaption>Representative run selected near median completion time, '
                               'independently of acceptance. Crosses indicate out-of-band operations.</figcaption></figure>')
    payload = {
        "conditions": conditions,
        "runs": [{k: row[k] for k in ["hek", "ips", "repetition", "outcome", "completed",
                 "primary_accepted", "supplementary_accepted", "primary_violations",
                 "supplementary_violations", "virtual_minutes", "wall_seconds"]}
                 for row in records],
        "representatives": {
            f"{h}_{i}": {"repetition": case["result"]["repetition"],
                         "end": case["result"]["virtual_minutes"],
                         "events": [[e["start"], e["end"], e["culture_id"],
                                     e["operation"], e["density"], e["target"]]
                                    for e in case["events"]]}
            for (h, i), case in representatives.items()},
        "hek_counts": metadata["config"]["hek_counts"],
        "ips_counts": metadata["config"]["ips_counts"],
    }
    data_json = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    condition_table = pd.DataFrame([{k: c[k] for k in ["hek", "ips", "repetitions_recorded",
                         "completed", "primary_accepted", "supplementary_accepted",
                         "primary_wilson_low", "primary_wilson_high",
                         "median_utilization", "median_virtual_days"]} for c in conditions])
    model_rows = []
    for kind, models in model.get("validation", {}).items():
        for name, row in models.items():
            model_rows.append({"cell_type": kind, "candidate": name,
                "selected": model["chosen_models"][kind] == name,
                "forward_predictions": row["forward_predictions"],
                "matched_predictions": row["matched_forward_predictions"],
                "matched_cultures": row["matched_cultures"],
                "density_mae": row["density_mae"],
                "matched_density_mae": row["matched_density_mae"],
                "crossing_mae_hours": row["crossing_mae_hours"]})
    model_table = pd.DataFrame(model_rows).to_html(index=False, float_format=lambda x:f"{x:.4f}")
    prefix = ("<strong>PARTIAL / PILOT PREVIEW.</strong> " if allow_partial else
              "<strong>PILOT STUDY.</strong> " if metadata["scope"] == "pilot" else "")
    text = REPORT_TEMPLATE.replace("__DATA__", data_json)
    text = text.replace("__FIGURES__", image_sections)
    text = text.replace("__CONDITIONS__", condition_table.to_html(index=False,
                        float_format=lambda x:f"{x:.3f}"))
    text = text.replace("__MODEL_TABLE__", model_table)
    text = text.replace("__SUMMARY__", (
        f"{prefix}{len(records):,}/{len(metadata['cases']):,} planned outcomes recorded. "
        f"{summary['completed']:,} runs completed through sampling; "
        f"{summary['primary_accepted']:,} met the primary density rule; "
        f"{summary['supplementary_accepted']:,} met the supplementary rule."))
    text = text.replace("__REFERENCE__", (
        f"1 HEK + 1 iPS: {ref_summary['completed']}/{ref_summary['runs']} completed; "
        f"{ref_summary['primary_accepted']}/{ref_summary['runs']} primary accepted; "
        f"{ref_summary['supplementary_accepted']}/{ref_summary['runs']} supplementary accepted.")
        if ref_summary else "No matched reference supplied.")
    finished_pairs = [c for c in conditions if c["repetitions_recorded"] == metadata["config"]["repetitions"]]
    findings = []
    if finished_pairs:
        best = max(finished_pairs, key=lambda c: (c["primary_fraction"], c["hek"]+c["ips"]))
        findings.append(
            f"Among count pairs with all 30 outcomes recorded, the highest observed primary "
            f"acceptance is {best['hek']} HEK + {best['ips']} iPS: "
            f"{best['primary_accepted']}/30 "
            f"(95% conditional Wilson interval {best['primary_wilson_low']:.1%}–"
            f"{best['primary_wilson_high']:.1%}). This descriptive ranking is not "
            "a statistically proven optimum.")
        if not robust:
            findings.append("No fully recorded pair has 30/30 primary acceptance. "
                            "The study therefore does not establish a count pair guaranteed "
                            "to meet every density target under this policy.")
    ceiling = next((c for c in conditions if c["hek"] == 10 and c["ips"] == 10), None)
    if ceiling:
        findings.append(f"At the requested 10+10 ceiling, {ceiling['completed']}/"
                        f"{ceiling['repetitions_recorded']} runs completed, and "
                        f"{ceiling['primary_accepted']}/{ceiling['repetitions_recorded']} "
                        "met the primary density rule. Completion alone is insufficient "
                        "evidence for acceptable cultivation.")
    if ref_summary:
        findings.append(
            f"The 1+1 reference met the primary rule in {ref_summary['primary_accepted']}/"
            f"{ref_summary['runs']} runs. A baseline density miss already at low load "
            "points to forecast/time-convention/window effects as well as possible contention "
            "at higher loads.")
    findings.append(
        "An absolute ±0.1 band has relative half-widths of 12.5%, 25% and 33.3% at "
        "targets 0.8, 0.4 and 0.3. Keep it as the predeclared exploratory result and "
        "inspect the capped ±20% sensitivity separately; neither criterion establishes "
        "biological suitability.")
    text = text.replace("__FINDINGS__", "<ul>"+ "".join(
        f"<li>{html.escape(f)}</li>" for f in findings)+"</ul>")
    text = text.replace("__ROLE_TABLE__", pd.DataFrame(role_rows).to_html(
        index=False, float_format=lambda x:f"{x:.4f}"))
    text = text.replace("__PROVENANCE__", html.escape(json.dumps({
        "source_commit": model["source_commit"],
        "model_hash": metadata["model_hash"], "config_hash": metadata["config_hash"],
        "code_commit": metadata["environment"]["git_commit"],
        "packages": metadata["environment"]["packages"],
        "report_generator_sha256": summary["report_generator_sha256"],
    }, indent=2)))
    (output / "report.html").write_text(text)
    (output / "README.md").write_text(
        "# ROUND3 culture-capacity evaluation\n\n"
        f"{summary['recorded_runs']:,}/{summary['planned_runs']:,} outcomes recorded; "
        f"{summary['completed']:,} complete; {summary['primary_accepted']:,} primary accepted; "
        f"{summary['supplementary_accepted']:,} supplementary accepted.\n\n"
        "Open `report.html` for the standalone report, condition selector and timelines. "
        "PNG/SVG figures are in `figures/`; CSV tables retain every run and targeted operation.\n\n"
        + "\n\n".join(findings) + "\n\n"
        f"All-30 primary pairs: {summary['all_30_primary_pairs']}.\n\n"
        "These are conditional simulated frequencies, not guarantees for real cultures. "
        "The range ceiling is 10+10, not a proven upper limit. Biological density tolerances, "
        "storage/inventory sufficiency, morphology and long-term cell quality remain unverified.\n\n"
        f"Source `{model['source_commit']}`, simulation code `{metadata['environment']['git_commit']}`.\n")
    if private_audit is not None:
        private_dir = output / "private_supplement"
        private_dir.mkdir()
        private_fit_supplement(private_audit, model, private_dir)
    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            artifacts.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(output / "report_manifest.json", artifacts)
    print(json.dumps({k: summary[k] for k in ["planned_runs", "recorded_runs", "completed",
                      "primary_accepted", "supplementary_accepted", "all_30_primary_pairs"]},
                     indent=2))


REPORT_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ROUND3 culture capacity — empirical simulation</title>
<style>
:root{font-family:system-ui,sans-serif;color:#172033;background:#f1f5f9}
body{margin:0}main{max-width:1180px;margin:auto;padding:28px}h1{font-size:32px}
h2{margin-top:36px}.lead{background:#fff;border-left:5px solid #2563eb;padding:20px;line-height:1.7}
.panel,figure{background:white;border-radius:10px;padding:20px;margin:18px 0;box-shadow:0 2px 10px #0f172a09}
figure img{width:100%;height:auto}figcaption{font-size:14px;color:#475569;line-height:1.6;margin-top:10px}
.twocol{display:grid;grid-template-columns:1fr 1fr;gap:20px}
table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:7px;border-bottom:1px solid #e2e8f0;text-align:right}
th{background:#f8fafc}.scroll{overflow:auto}button,select,input{font:inherit}
button{border:0;border-radius:5px;padding:9px;cursor:pointer}.grid button{width:100%;min-width:43px;color:#fff}
.grid td{padding:3px}.selected{outline:3px solid #f97316}
.note{color:#475569;font-size:14px;line-height:1.7}svg{width:100%;background:#fff}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap}.controls input{width:260px}
pre{white-space:pre-wrap;font-size:12px;background:#f8fafc;padding:18px;border-radius:7px}
@media(max-width:850px){.twocol{grid-template-columns:1fr}main{padding:15px}h1{font-size:25px}}
</style></head><body><main>
<h1>ROUND3 concurrent HEK / iPS culture capacity</h1>
<p class="note">Empirical growth variation · one shared operation resource · refill included · endpoint: sampling</p>
<div class="lead">__SUMMARY__</div>
<div class="panel">__FINDINGS__</div>
<p>Completion and density acceptance answer different questions. A completed culture may have been
processed outside its provisional density band. Interpret the grid as outcomes under this protocol,
growth model and planning policy, rather than a single guaranteed physical capacity.</p>
<h2>Explore count pairs</h2><div class="twocol">
<div class="panel"><label>Criterion <select id="criterion">
<option value="primary_accepted">Primary: target ±0.1</option>
<option value="supplementary_accepted">Supplementary: ±20%, absolute width ≤0.1</option>
<option value="completed">Completed through sampling</option></select></label>
<div id="grid" class="grid scroll"></div><p class="note">Cells show accepted / recorded repetitions.
Orange outline selects the count pair. Grey indicates no recorded data.</p></div>
<div class="panel"><h3 id="pairtitle">Select a pair</h3><p id="pairstats"></p>
<div class="scroll" id="replicates"></div><p class="note">A 95% Wilson interval describes
Monte Carlo uncertainty conditional on this fitted empirical pool. It does not quantify
uncertainty about actual wet-culture success. Count conditions share paired parameter prefixes.</p></div></div>
<h2>Representative event timeline</h2><div class="panel">
<p id="timelinecaption"></p><div class="controls">
<label>Window <select id="window"><option value="all">Whole run</option>
<option value="day">One-day zoom</option></select></label>
<label>Start day <input id="day" type="range" min="0" max="40" value="0"></label><span id="daylabel"></span>
</div><div id="timeline"></div>
<p class="note">Blue: imaging; orange: passage; green: medium change; purple: sampling;
grey: coating/refill. Hover a reservation for its operation and density. Grey row lines
indicate the timeline grid. Day zoom exposes short reservations hidden in the full view.
Group-start work-hour preferences are soft costs; they do not prohibit operations crossing a rest interval.</p></div>
<h2>Results and model checks</h2>__FIGURES__
<h2>Targeted operation diagnostics</h2><div class="panel scroll">__ROLE_TABLE__</div>
<p class="note">Nominal and actual out-of-band counts overlap. The nominal diagnostic includes
forecast and operation-time conventions; it is not a mutually exclusive causal attribution.</p>
<h2>Matched small-cohort reference</h2><p class="note">__REFERENCE__</p>
<p>The 1+1 reference keeps the same prediction policy, refill and work-hour preferences.
It shares the HEK-00 / iPS-00 parameter prefixes with each main repetition. It helps distinguish
baseline forecast/window effects from the operational effect of adding cultures, without acting as an oracle.</p>
<h2>Condition table</h2><div class="panel scroll">__CONDITIONS__</div>
<h2>Growth validation</h2><div class="panel scroll">__MODEL_TABLE__</div>
<p class="note">Forward validation excludes the predicted observation from fitting. Crossing errors
use interpolation between bracketing observations, not an independently measured crossing time.
Model selection is exploratory; there is no independent final validation dataset. Source cultures
are technical culture units, not genetic or human donors.</p>
<h2>Assumptions and limits</h2><div class="panel">
<ul><li>All cultures are initially enrolled. HEK starts post-passage; iPS runs its initial preparation.
No failures, interruptions or additional observation noise are injected.</li>
<li>Growth is an image-density proxy. Post-passage parameters are drawn from a small empirical pool;
unavailable source generations use explicitly recorded same-type/generation fallbacks.</li>
<li>The unmodeled initial iPS parent stock is held at nominal density 0.3 until initial passage.
Post-passage starting density is an empirical draw, without mechanistic feedback from pre-passage
density. Time zero uses a fixed clock phase; start-phase sensitivity is not tested.</li>
<li>The density bands are provisional, not biologically validated. Medium-change gaps, storage,
consumables, morphology and long-term quality are not additional acceptance guarantees.</li>
<li>Sampler/passage groups commit remaining task times once started. An image inside a started
group does not automatically postpone the following treatment.</li>
<li>Nominal-versus-actual density diagnostics include readiness, time preferences, contention
and solver approximation. Mean utilization below 100% does not prove that all timing requirements
can be met.</li>
<li>FEASIBLE schedules are validated without proof of optimality. UNKNOWN and execution censoring
are separate from biological violations. Search latency is wall time, not an added virtual operation.</li>
<li>Thirty repetitions are simulated draws from the fitted source pool, not thirty new biological
experiments. A 10+10 success reaches the requested range ceiling, not a measured limit above it.</li></ul></div>
<h2>Reproducibility</h2><pre>__PROVENANCE__</pre>
<p class="note">Original observations and the private fit-check supplement are excluded from this portable
report. PNG/SVG figures and CSV tables accompany the HTML. Derived artifacts retain the source's
publication boundary until explicitly approved for release.</p>
</main><script>const DATA=__DATA__;
let selected=null;const criterion=document.getElementById('criterion');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=x=>x==null?'—':(100*x).toFixed(1)+'%';
function drawGrid(){let out='<table><thead><tr><th>iPS ↓ / HEK →</th>';
for(const h of DATA.hek_counts)out+='<th>'+h+'</th>';out+='</tr></thead><tbody>';
for(const i of [...DATA.ips_counts].reverse()){out+='<tr><th>'+i+'</th>';
for(const h of DATA.hek_counts){const row=DATA.conditions.find(c=>c.hek===h&&c.ips===i);
const k=row?row[criterion.value]:0,n=row?row.repetitions_recorded:0,f=n?k/n:0;
const color=n?`hsl(${220-110*f} 55% ${38+12*f}%)`:'#cbd5e1';
out+=`<td><button style="background:${color}" data-h="${h}" data-i="${i}"
class="${selected&&selected.h===h&&selected.i===i?'selected':''}">${n?k+'/'+n:'—'}</button></td>`;}
out+='</tr>';}document.getElementById('grid').innerHTML=out+'</tbody></table>';
for(const button of document.querySelectorAll('#grid button'))button.onclick=()=>choose(+button.dataset.h,+button.dataset.i);}
function choose(h,i){selected={h,i};drawGrid();
const row=DATA.conditions.find(c=>c.hek===h&&c.ips===i);
document.getElementById('pairtitle').textContent=h+' HEK + '+i+' iPS';
document.getElementById('pairstats').textContent=row?
`Complete ${row.completed}/${row.repetitions_recorded}; primary ${row.primary_accepted}/${row.repetitions_recorded};
supplementary ${row.supplementary_accepted}/${row.repetitions_recorded}. Primary Wilson interval
${pct(row.primary_wilson_low)}–${pct(row.primary_wilson_high)}. Median utilization ${pct(row.median_utilization)}.`:'No recorded runs.';
const rows=DATA.runs.filter(r=>r.hek===h&&r.ips===i);
let table='<table><tr><th>Rep.</th><th>Outcome</th><th>Primary</th><th>Supplement</th><th>Violations</th></tr>';
for(const r of rows)table+=`<tr><td>${r.repetition}</td><td>${esc(r.outcome)}</td><td>${r.primary_accepted?'pass':'—'}</td>
<td>${r.supplementary_accepted?'pass':'—'}</td><td>${r.primary_violations}</td></tr>`;
document.getElementById('replicates').innerHTML=table+'</table>';drawTimeline();}
function drawTimeline(){const key=selected?selected.h+'_'+selected.i:'',data=DATA.representatives[key];
if(!data){document.getElementById('timeline').innerHTML='<p>No representative available.</p>';return;}
const slider=document.getElementById('day'),mode=document.getElementById('window').value;
slider.max=Math.max(0,Math.ceil(data.end/1440)-1);slider.disabled=mode==='all';
const beginning=mode==='day'?Number(slider.value)*1440:0;
const ending=mode==='day'?Math.min(data.end,beginning+1440):data.end;
document.getElementById('daylabel').textContent=mode==='day'?'Day '+slider.value:'All days';
document.getElementById('timelinecaption').textContent=
`Representative repetition ${data.repetition}, selected near median completion time independently of acceptance.`;
const ids=[...new Set(data.events.map(e=>e[2]))].sort((a,b)=>a==='refill'?1:b==='refill'?-1:a.localeCompare(b));
const width=1040,left=110,right=20,row=24,height=60+ids.length*row;
const x=t=>left+(t-beginning)/Math.max(1,ending-beginning)*(width-left-right);
let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Recorded operation timeline">`;
for(let n=0;n<ids.length;n++){const y=42+n*row;svg+=`<text x="4" y="${y+12}" font-size="11">${esc(ids[n])}</text>
<line x1="${left}" x2="${width-right}" y1="${y+8}" y2="${y+8}" stroke="#e2e8f0"/>`;}
for(let tick=0;tick<=8;tick++){const t=beginning+(ending-beginning)*tick/8;
svg+=`<line x1="${x(t)}" x2="${x(t)}" y1="28" y2="${height-12}" stroke="#f1f5f9"/>
<text x="${x(t)}" y="18" text-anchor="middle" font-size="11">${(t/1440).toFixed(mode==='day'?2:1)} d</text>`;}
for(const e of data.events){if(e[1]<=beginning||e[0]>=ending||e[1]===e[0])continue;
const color=e[3].endsWith('GetImage')?'#2563eb':e[3].endsWith('Passage')?'#f97316':
e[3].endsWith('MediumChange')?'#16a34a':e[3].endsWith('Sampling')?'#7c3aed':'#64748b';
const start=Math.max(beginning,e[0]),end=Math.min(ending,e[1]),y=42+ids.indexOf(e[2])*row;
const title=e[2]+' '+e[3]+'; '+(e[0]/1440).toFixed(3)+'–'+(e[1]/1440).toFixed(3)+' d'+
(e[4]==null?'':'; density '+e[4].toFixed(3))+(e[5]==null?'':'; target '+e[5]);
svg+=`<rect x="${x(start)}" y="${y}" width="${Math.max(.6,x(end)-x(start))}" height="16" fill="${color}">
<title>${esc(title)}</title></rect>`;}document.getElementById('timeline').innerHTML=svg+'</svg>';}
criterion.onchange=drawGrid;document.getElementById('window').onchange=drawTimeline;
document.getElementById('day').oninput=drawTimeline;drawGrid();
if(DATA.conditions.length)choose(DATA.conditions[0].hek,DATA.conditions[0].ips);
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--private-audit", type=Path)
    args = parser.parse_args()
    write_report(args.root, args.model, args.output, args.reference,
                 args.allow_partial, args.private_audit)


if __name__ == "__main__":
    main()

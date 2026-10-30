"""Summarize verified Issue #24 records without scheduling or changing raw data."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


KEYS = ("series", "pattern", "budget", "mode", "parents", "children",
        "group_interval", "generations")
METRICS = ("planning_seconds", "preparation_seconds", "fixture_seconds",
           "supplemental_initialization_seconds",
           "save_seconds", "total_pipeline_seconds", "measured_phase_seconds",
           "peak_rss_kib", "input_json_bytes", "own_history_rows",
           "inherited_history_rows", "archived_history_rows", "timing_penalty",
           "delay_median_minutes", "delay_p95_minutes", "delay_max_minutes")


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple("scan_and_boundary" if k == "series" and row[k] in ("scan", "boundary")
                    else row[k] for k in KEYS)
        groups[key].append(row)
    output = []
    for key, group in sorted(groups.items()):
        entry = dict(zip(KEYS, key))
        entry.update(repetitions=len(group),
                     on_time=sum(r["outcome"] == "VALID_ON_TIME" for r in group),
                     late=sum(r["outcome"] == "VALID_LATE" for r in group),
                     outcomes=json.dumps(dict(Counter(r["outcome"] for r in group)), sort_keys=True))
        phases = sorted({r.get("storage_phase", "unspecified") for r in group})
        entry["storage_phases"] = ";".join(phases)
        for metric in METRICS:
            if metric in ("save_seconds", "total_pipeline_seconds", "measured_phase_seconds") \
                    and len(phases) > 1:
                # Keep I/O from tmpfs and the workspace filesystem separate.
                entry[f"{metric}_by_storage_phase"] = json.dumps({
                    phase: {name: function(values) for name, function in (
                        ("min", min), ("median", statistics.median), ("max", max))}
                    for phase in phases
                    if (values := [r[metric] for r in group if r.get(metric) is not None and
                                   r.get("storage_phase", "unspecified") == phase])
                }, sort_keys=True)
                continue
            values = [r[metric] for r in group if r.get(metric) is not None]
            if values:
                for name, value in (("min", min(values)), ("median", statistics.median(values)),
                                    ("max", max(values))):
                    entry[f"{metric}_{name}"] = value
        dimensions = group[0].get("dimensions", {})
        assert all(r.get("dimensions", {}) == dimensions for r in group)
        entry.update(dimensions)
        output.append(entry)
    return output


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(source, output):
    verified = json.loads((source / "verification.json").read_text())
    rows = [json.loads(line) for line in (source / "results.jsonl").read_text().splitlines()]
    assert len(rows) == verified["trial_count"]
    profiles = 0
    for row in rows:
        profile = source / row["artifact_directory"] / "initialization_profile.json"
        if profile.exists():
            measurement = json.loads(profile.read_text())
            assert measurement["input_hash"] == row["input_hash"]
            row["supplemental_initialization_seconds"] = measurement["initialization_seconds"]
            profiles += 1
    stats = summarize(rows)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "conditions.csv", stats)
    trials = []
    for row in rows:
        trial = {k: row[k] for k in (*KEYS, "repetition", "outcome", "artifact_directory")}
        trial.update({k: row.get(k) for k in METRICS})
        trial["storage_phase"] = row.get("storage_phase", "unspecified")
        trial.update(row.get("dimensions", {}))
        trial["solver_status"] = ";".join(r["status"] for r in row.get("reports", []))
        trials.append(trial)
    write_csv(output / "trials.csv", trials)
    metadata = json.loads((source / "metadata.json").read_text())
    (output / "summary.json").write_text(json.dumps(dict(
        source_commit=metadata["commit"], verification=verified,
        timing=metadata["timing"], conditions=stats, initialization_profiles=profiles,
        initialization_note="Supplemental cold-process measurements; excluded from original pipeline timers",
        storage_note="I/O and pipeline metrics from different storage phases are not pooled"), indent=2))
    colors = dict(zip(sorted({r["pattern"] for r in rows}), plt.get_cmap("tab10").colors))
    budgets = sorted({r["budget"] for r in rows})
    fig, axes = plt.subplots(1, len(budgets), figsize=(13, 5.5), squeeze=False)
    for ax, budget in zip(axes[0], budgets):
        for pattern, color in colors.items():
            for mode, style in (("lifecycle", "-"), ("flat", "--")):
                points = sorted((r for r in stats if r["series"] == "scan_and_boundary" and
                                 r["pattern"] == pattern and r["budget"] == budget and
                                 r["mode"] == mode and "planning_seconds_median" in r),
                                key=lambda r: r["active_experiments"])
                ax.plot([r["active_experiments"] for r in points],
                        [r["planning_seconds_median"] for r in points],
                        color=color, linestyle=style, marker="o", markersize=3)
        ax.axhline(budget, color="black", linestyle=":", linewidth=1)
        ax.set(xscale="log", yscale="log", xlabel="Active child experiments",
               ylabel="Total planning seconds (median)", title=f"{budget:g}s total-time condition")
        ax.grid(True, alpha=.2)
    handles = [Line2D([0], [0], color=color, label=pattern) for pattern, color in colors.items()]
    handles += [Line2D([0], [0], color="black", linestyle=style, label=mode)
                for mode, style in (("lifecycle", "-"), ("flat", "--"))]
    fig.legend(handles=handles, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, .87))
    for suffix in ("png", "svg"):
        fig.savefig(output / f"planning.{suffix}", dpi=160)
    plt.close(fig)
    print(json.dumps(dict(trials=len(rows), conditions=len(stats), source_commit=metadata["commit"])))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.directory, args.output)

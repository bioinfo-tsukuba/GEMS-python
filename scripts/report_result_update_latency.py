"""Tabulate and plot saved received-result latency records without running a solver."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analyze_scheduler_capacity import describe


def report(root):
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    inputs = json.loads((root / "inputs.json").read_text())
    env = json.loads((root / "environment.json").read_text())
    metrics = [
        "response_seconds", "progress_seconds", "update_api_seconds", "json_seconds",
        "search_seconds", "non_search_seconds",
    ]
    flat = [
        {**{key: row[key] for key in
            ("pattern", "size", "search_limit_seconds", "repetition", "outcome",
             "input_hash", "response_sha256", "response_bytes", "solver_calls") + tuple(metrics)},
         "completes_group": row["event"]["completes_group"]}
        for row in rows
    ]
    with (root / "trial_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    condition_stats = []
    for pattern in inputs:
        for limit in [1.0, 60.0]:
            selected = [r for r in rows if r["pattern"] == pattern
                        and r["search_limit_seconds"] == limit]
            success = [r for r in selected if r["outcome"] == "VALID"]
            failed = [r for r in selected if r["outcome"] != "VALID"]
            item = {
                "pattern": pattern, "size": inputs[pattern]["size"],
                "search_limit_seconds": limit, "runs": len(selected), "valid": len(success),
                "failed_response_seconds": describe(r["response_seconds"] for r in failed),
            }
            for metric in metrics:
                item[metric] = describe(r[metric] for r in success)
            condition_stats.append(item)
    (root / "statistics.json").write_text(json.dumps(condition_stats, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    patterns = list(inputs)
    names = ["contention", "heterogeneous", "candidates", "penalties", "replanning"]
    for ax, limit in zip(axes, [1.0, 60.0]):
        for position, pattern in enumerate(patterns):
            selected = [r for r in rows if r["pattern"] == pattern
                        and r["search_limit_seconds"] == limit]
            valid = [r["response_seconds"] for r in selected if r["outcome"] == "VALID"]
            failed = [r["response_seconds"] for r in selected if r["outcome"] != "VALID"]
            for values, color, marker in [(valid, "#0072B2", "o"), (failed, "#D55E00", "x")]:
                offsets = np.linspace(-.12, .12, len(values)) if len(values) > 1 else [0]
                ax.scatter(position + np.array(offsets[:len(values)]), values,
                           color=color, marker=marker, s=35)
            if valid:
                stats = describe(valid)
                ax.errorbar(position, stats["median"],
                            yerr=[[stats["median"] - stats["min"]],
                                  [stats["max"] - stats["median"]]],
                            fmt="D", color="#222222", capsize=5, markersize=4)
            ax.text(position, max(valid + failed, default=0) + limit * .025,
                    f"{len(valid)}/{len(selected)} valid", ha="center", fontsize=9)
        ax.set_xticks(range(len(patterns)),
                      [f"{name}\nn={inputs[p]['size']}" for name, p in zip(names, patterns)],
                      rotation=22, ha="right", fontsize=9)
        ax.set_ylim(0, limit * 1.12)
        ax.set_ylabel("Real seconds to response JSON")
        ax.set_title(f"{limit:g}-second CP-SAT limit; one search per received result")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"Received result -> schedule JSON: {env['repetitions']} repeats of the first completion event")
    handles = [
        plt.Line2D([], [], color="#0072B2", marker="o", ls="", label="Valid schedule"),
        plt.Line2D([], [], color="#D55E00", marker="x", ls="", label="UNKNOWN / error response"),
        plt.Line2D([], [], color="#222222", marker="D", ls="", label="Success median; min–max"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9)
    fig.savefig(root / "response_latency.png", dpi=170)
    fig.savefig(root / "response_latency.svg")
    plt.close(fig)
    lines = [
        "# Experiment result receipt to schedule JSON generation",
        "",
        f"Measured source: `{env['commit']}`. Saved plan reference: `{env['source_archive_commit']}`.",
        f"Started: `{env['started_at_utc']}`. CPU: {env['cpu']}, CP-SAT worker 1 / seed 0.",
        "Dependency versions and the full installed-package list are recorded in the environment file.",
        "",
        "Timing starts when a received DataFrame is passed to the application and covers fixing started work, updating history, completion processing, "
        "required state transitions and task generation, CP-SAT replanning, and full schedule JSON generation.",
        "Startup, input preparation, hardware communication, rendering and file saving are excluded. Debug output is captured in memory. "
        "The existing update API is called once, with no replanning before result receipt and one search per response.",
        "",
        "Inputs are the first result events from the previous valid plans. "
        "All events occur partway through groups, so state-transition and next-group-generation branches were not exercised in these measurements. "
        "Those branches were checked in separate small regression tests. Complex experiment analysis is also not simulated.",
        "",
        "![Response time after result receipt](response_latency.png)",
        "",
        "| Workload | Scale | Search seconds | Successes/trials | Successful response median / p95 / maximum (seconds) | Search component median (seconds) | Other components median (seconds) | Failed response median (seconds) |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in condition_stats:
        success = row["response_seconds"]
        failure = row["failed_response_seconds"]
        def value(stats, key="median"):
            return f"{stats[key]:.6f}" if stats["n"] else "—"
        lines.append(
            f"| {row['pattern']} | {row['size']} | {row['search_limit_seconds']:g} | "
            f"{row['valid']}/{row['runs']} | {value(success)} / {value(success, 'p95')} / "
            f"{value(success, 'max')} | {value(row['search_seconds'])} | "
            f"{value(row['non_search_seconds'])} | {value(failure)} |")
    lines += [
        "",
        f"These are descriptive statistics from {env['repetitions']} repetitions each, passing the same received result and plan to the 1-second / 60-second settings. "
        "The p95 is linearly interpolated from few points and does not guarantee future performance. "
        "Component medians are aggregated separately, so their sum may not exactly equal the total-time median.",
        "UNKNOWN timings are not pooled with successful schedule-generation timings.",
        "",
        "The previous operational times of about 2/120 seconds include search before receipt. This measurement times one search after receipt. "
        "Success on this first response does not guarantee successful continued updates.",
        "",
        "trial_metrics.csv contains all trials; statistics.json contains per-condition medians, interquartile ranges, p95, "
        "minima, maxima, means and sample standard deviations. raw_results.tar.gz retains inputs, outputs, searches, failures and "
        "logs. Input/output hashes, environment information and verification are also provided.",
        "",
        "```bash",
        "MPLCONFIGDIR=/tmp/gems-latency-mpl MPLBACKEND=Agg poetry run python \\",
        "  scripts/measure_result_update_latency.py \\",
        "  --output /tmp/gems-result-latency-new --repetitions 3",
        "",
        "# Reaggregate and plot saved results only",
        "poetry run python -m scripts.report_result_update_latency /tmp/gems-result-latency-new",
        "poetry run python -m scripts.verify_result_update_latency /tmp/gems-result-latency-new",
        "```",
        "",
        "New measurements require about 15 minutes of search. Choose a new output directory.",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))
    return condition_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    report(parser.parse_args().directory)

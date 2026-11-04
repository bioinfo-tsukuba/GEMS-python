"""Rebuild detailed tables and a success-rate figure from preserved study records."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(root):
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    cfg = json.loads((root / "config.json").read_text())
    trials, calls, updates = [], [], []
    groups = defaultdict(list)
    for row in rows:
        common = {key: row[key] for key in
                  ("pattern", "size", "phase", "repetition", "search_limit_seconds", "input_hash")}
        solver_calls = row.get("reports", [])
        trial = {**common, "outcome": row["outcome"], "error": row.get("error"),
                 "artifact_directory": row["artifact_directory"],
                 **row.get("dimensions", {})}
        for key in ("total_seconds", "worker_process_seconds", "memory_peak_mib",
                    "timing_penalty", "movable_penalty", "delay_median_minutes",
                    "delay_p95_minutes", "delay_max_minutes", "schedule_span_minutes",
                    "machine_utilization", "save_seconds", "draw_seconds"):
            trial[key] = row.get(key)
        for metric in ("build_seconds", "search_seconds", "apply_seconds",
                       "validation_seconds", "other_seconds"):
            trial[metric] = sum(call.get(metric, 0) for call in solver_calls)
        trial["solver_calls"] = len(solver_calls)
        trial["solver_statuses"] = ",".join(dict.fromkeys(c["status"] for c in solver_calls))
        trials.append(trial)
        for index, call in enumerate(solver_calls):
            calls.append({**common, "call_index": index, **call})
        for update in row.get("updates", []):
            updates.append({
                **common, "step": update["step"], "update_input_hash": update["input_hash"],
                "update_seconds": update["update_seconds"], "status": update["result"]["status"],
                "simulated_time": update["result"].get("simulated_time"),
                "added_experiments": ",".join(update["added_experiments"]),
                "solver_calls": len(update["calls"]), **update["dimensions"],
                "search_seconds": sum(c["search_seconds"] for c in update["calls"]),
            })
        if row["phase"] == "scan":
            groups[(row["pattern"], row["search_limit_seconds"], row["size"])].append(row)
    write_csv(root / "trial_metrics.csv", trials)
    write_csv(root / "solver_calls.csv", calls)
    write_csv(root / "update_metrics.csv", updates)

    overview = [
        "# Scale evaluation with 1-second / 60-second search limits",
        "",
        "1 or 60 seconds are search limits passed directly to CP-SAT. Model construction, application and validation are excluded from the success criterion.",
        "Success means returning a valid plan. FEASIBLE means optimality is unproven; UNKNOWN means no solution was found.",
        "",
        "| Workload | Search limit (seconds) | Largest measured scale with all repetitions successful | Successes | Range of maximum delay (minutes) at that scale |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for pattern in cfg["patterns"]:
        for limit in cfg["search_limits_seconds"]:
            successful = [size for size in cfg["sizes"]
                          if groups[(pattern, limit, size)]
                          and all(r["outcome"] == "VALID" for r in groups[(pattern, limit, size)])]
            size = max(successful, default=0)
            selected = groups[(pattern, limit, size)]
            delays = [r["delay_max_minutes"] for r in selected]
            span = f"{min(delays):g}–{max(delays):g}" if delays else "—"
            overview.append(f"| {pattern} | {limit:g} | {size} | "
                            f"{len(selected)}/{len(selected)} | {span} |")
    overview += [
        "",
        "Success at 512 reaches the configured measurement ceiling, not the true capacity limit. "
        "No monotonic relationship between failure and scale is assumed. Outside boundary conditions, only one initial trial was run.",
        "Delay is in virtual-experiment minutes; runtime is in actual seconds. Equipment throughput and scientific validity require separate evaluation.",
        "",
        "## Operational updates",
        "",
        "| Workload | Scale | Search limit (seconds) | Result | Successful/attempted updates | Planning calls | Successful update time median (seconds, reference) |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["phase"] == "operation":
            events = row.get("updates", [])
            times = [u["update_seconds"] for u in events if u["result"]["status"] == "ok"]
            median = f"{statistics.median(times):.3f}" if times else "—"
            overview.append(f"| {row['pattern']} | {row['size']} | {row['search_limit_seconds']:g} | "
                            f"{row['outcome']} | {len(times)}/{len(events)} | "
                            f"{len(row.get('reports', []))} | {median} |")
    overview += [
        "",
        "Operational updates make multiple planning calls per update. The search limit applies to each call. "
        "Saving and rendering were measured separately in small tests.",
        "Success in static scale evaluation does not guarantee successful continued updates. Updates that fail partway through do not count as successes.",
        "",
        "trial_metrics.csv records time, quality and memory; solver_calls.csv records all planning calls, including bounds in the same units as the objective; "
        "update_metrics.csv records update times and scale. Raw data is retained in results.jsonl and trials/.",
        "",
    ]
    (root / "measurement_summary.md").write_text("\n".join(overview))

    labels = [(pattern, limit) for pattern in cfg["patterns"] for limit in cfg["search_limits_seconds"]]
    image = np.full((len(labels), len(cfg["sizes"])), np.nan)
    for y, (pattern, limit) in enumerate(labels):
        for x, size in enumerate(cfg["sizes"]):
            selected = groups[(pattern, limit, size)]
            if selected:
                image[y, x] = sum(r["outcome"] == "VALID" for r in selected) / len(selected)
    fig, ax = plt.subplots(figsize=(12, 6.5), layout="constrained")
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad("#e4e4e4")
    heatmap = ax.imshow(image, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(cfg["sizes"])), cfg["sizes"])
    ax.set_yticks(range(len(labels)), [f"{p} / {limit:g} s" for p, limit in labels])
    ax.set_xlabel("Active experiments (task/fixed counts in CSV)")
    ax.set_title("Fixed CP-SAT search limits: valid plans / measured trials")
    for y, (pattern, limit) in enumerate(labels):
        for x, size in enumerate(cfg["sizes"]):
            selected = groups[(pattern, limit, size)]
            text = (f"{sum(r['outcome'] == 'VALID' for r in selected)}/{len(selected)}"
                    if selected else "—")
            fraction = image[y, x]
            color = "white" if fraction < .2 or fraction > .8 else "black"
            ax.text(x, y, text, ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(heatmap, ax=ax, label="Valid fraction")
    fig.savefig(root / "success_rates.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    report(parser.parse_args().directory)

"""Reproduce the OR-Tools virtual-experiment Pandas/Polars comparison."""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.simulators.dataframe_comparison import assert_same, cases, run_case
from gems_python.common.cp_sat_scheduler import SOLVER_TIME_LIMIT_SECONDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Capture source state before creating the generated, possibly tracked output.
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ("pandas", "polars", "pyarrow", "ortools")},
        "scheduler": "o", "time_limit_seconds": SOLVER_TIME_LIMIT_SECONDS,
        "solver_seed": 0, "workers": 1, "command": sys.argv,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    results, summary = [], []
    for case in cases():
        runs, seconds = {}, {}
        for backend in ("pandas", "polars"):
            start = time.perf_counter()
            runs[backend] = run_case(case, backend)
            seconds[backend] = time.perf_counter() - start
        assert_same(case, runs["pandas"], runs["polars"])
        fingerprint = runs["pandas"]["input_sha256"]
        results.append({"case": case, "input_sha256": fingerprint, "runs": runs, "seconds": seconds})
        summary.append({"case": case["name"], "steps": case["steps"], "input_sha256": fingerprint,
                        "equal": True, "pandas_seconds": seconds["pandas"], "polars_seconds": seconds["polars"],
                        "solver_calls_per_backend": len(runs["pandas"]["solver"])})
        print(f"{case['name']}: equal ({case['steps']} steps)")
    (args.output / "runs.json").write_text(json.dumps({"metadata": metadata, "cases": results}, indent=2) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    (args.output / "summary.md").write_text(
        "# Pandas / Polars virtual experiments\n\n"
        f"All {len(results)} paired cases matched exactly; "
        f"{sum(c['case']['steps'] for c in results)} steps per backend.\n\n"
        "Compared row/column order and values, transitions, task schedules, assignments, "
        "penalties and solver status/objectives. All solver results were OPTIMAL. "
        "Wall time was recorded but excluded from equality.\n"
    )


if __name__ == "__main__":
    main()

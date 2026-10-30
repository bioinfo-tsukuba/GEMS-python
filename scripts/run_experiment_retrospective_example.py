"""Run a virtual protocol to completion, save, reload, and render its records."""
import argparse
import contextlib
import json
from pathlib import Path

from gems_python.common.persistence import load_snapshot, save_snapshot
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments
from tests.simulators.retrospective_review import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--backend", choices=("pandas", "polars"), default="pandas")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "simulation.log").open("w") as log, contextlib.redirect_stdout(log):
        lab, events = run(args.output / "state", args.backend)
    save_snapshot(lab, args.output / "snapshot")
    restored = load_snapshot(args.output / "snapshot", Experiments)
    restored.render_retrospective(args.output / "figures", font=args.font)
    (args.output / "events.json").write_text(json.dumps(events, indent=2) + "\n")
    print(f"{len(restored.ended_experiments)} Experiment / "
          f"{len(restored.retrospective_trace['tasks'])} Task / {restored.reference_time} min")


if __name__ == "__main__":
    main()

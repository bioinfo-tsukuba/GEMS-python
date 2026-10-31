"""Render a completed saved run (JSON/pickle snapshot directory)."""
import argparse
from pathlib import Path

from gems_python.common.persistence import load_snapshot
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--time-unit", default="min")
    parser.add_argument("--source", choices=("auto", "json", "pickle"), default="auto")
    args = parser.parse_args()
    collection = load_snapshot(args.snapshot, Experiments, source=args.source)
    paths = collection.render_retrospective(args.output, font=args.font, time_unit=args.time_unit)
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()

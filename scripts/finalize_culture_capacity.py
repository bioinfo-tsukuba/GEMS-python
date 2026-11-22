"""Wait for complete simulation outcomes, verify them, and create a local report."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def wait_for_execution(root, wait):
    execution = root / "execution.json"
    metadata = json.loads((root / "metadata.json").read_text())
    last_count = -1
    while not execution.exists():
        if not wait:
            raise FileNotFoundError(f"Driver has not finished: {execution}")
        count = sum(1 for _ in (root / "results.jsonl").open()) if (
            root / "results.jsonl").exists() else 0
        if count != last_count:
            print(f"{root.name}: {count}/{len(metadata['cases'])} outcomes recorded", flush=True)
            last_count = count
        process = json.loads((root / "process.json").read_text())
        try:
            os.kill(process["pid"], 0)
        except ProcessLookupError as error:
            # Allow the driver's final manifest to be flushed before diagnosing
            # a vanished process; no incomplete outcome is invented.
            time.sleep(1)
            if not execution.exists():
                raise RuntimeError("Simulation driver vanished without final execution manifest") from error
        time.sleep(15)
    result = json.loads(execution.read_text())
    if not result["all_cases_recorded"] or result["recorded_cases"] != len(metadata["cases"]):
        raise ValueError("Driver finished without every planned outcome; inspect and explicitly resume")


def finalize(root, model, output, reference=None, private_audit=None, wait=False, browser=False):
    from scripts.extract_round3_culture_data import write_json

    state_path = root / "finalization.json"
    state = {"status": "waiting", "report_directory": str(output), "publication": "local only"}
    write_json(state_path, state)
    try:
        for directory in [root]+([reference] if reference else []):
            wait_for_execution(directory, wait)
            command = [sys.executable, "scripts/verify_culture_capacity.py", str(directory)]
            subprocess.run(command, check=True)
        from scripts.report_culture_capacity import write_report
        write_report(root, model, output, reference, private_audit=private_audit)
        if browser:
            subprocess.run([sys.executable, "scripts/verify_culture_capacity_report.py",
                            "--html", str(output / "report.html"),
                            "--output", str(root / "browser_verification")], check=True)
        state.update(status="verified_report_created", browser_verified=browser)
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--private-audit", type=Path)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--browser", action="store_true",
                        help="Requires optional Playwright and a configured Chromium environment")
    args = parser.parse_args()
    finalize(args.root, args.model, args.output, args.reference, args.private_audit,
             args.wait, args.browser)


if __name__ == "__main__":
    main()

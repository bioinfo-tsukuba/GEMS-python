"""Check saved response timings, paired inputs and physical assignments independently."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments
from scripts.measure_scheduler_capacity import fingerprint, snapshot
from scripts.measure_result_update_latency import freeze_begun_work


def verify(root):
    env = json.loads((root / "environment.json").read_text())
    bases = json.loads((root / "inputs.json").read_text())
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    hashes = json.loads((root / "measurement_source_sha256.json").read_text())
    for path, expected in hashes.items():
        source = subprocess.run(
            ["git", "show", f"{env['commit']}:{path}"], capture_output=True, check=True,
        ).stdout
        assert hashlib.sha256(source).hexdigest() == expected, path
    pairs = defaultdict(list)
    valid_count = 0
    for row in rows:
        pattern, limit = row["pattern"], row["search_limit_seconds"]
        directory = root / f"{pattern}-{limit:g}-{row['repetition']}"
        request = json.loads((directory / "request.json").read_text())
        manager = Experiments.from_dict(request["input"])
        assert fingerprint(snapshot(manager)) == row["input_hash"] == bases[pattern]["input_hash"]
        assert request["event"] == row["event"] == bases[pattern]["event"]
        assert row["solver_calls"] == len(row["reports"]) == 1
        assert row["reports"][0]["solver_time_limit_seconds"] == limit
        assert abs(row["response_seconds"] - sum(row[key] for key in
                   ["progress_seconds", "update_api_seconds", "json_seconds"])) < 1e-9
        assert row["response_seconds"] >= row["search_seconds"]
        content = (directory / "response.json").read_bytes()
        assert hashlib.sha256(content).hexdigest() == row["response_sha256"]
        assert len(content) == row["response_bytes"]
        payload = json.loads(content)
        if row["outcome"] == "VALID":
            assert row["reports"][0]["status"] in ["FEASIBLE", "OPTIMAL"]
            freeze_begun_work(manager, row["event"]["event_time"])
            old = {(g.task_group_id, t.task_id): (g, t)
                   for g in manager.task_groups for t in g.tasks}
            assigned = {(t["task_group_id"], t["task_id"]): t for t in payload["tasks"]}
            assert len(assigned) == len(payload["tasks"]) == len(old)
            assert payload["reference_time"] == row["event"]["event_time"]
            machine_types = {m.machine_id: m.machine_type for m in manager.machine_list.machines}
            reservations = defaultdict(list)
            for key, task in assigned.items():
                group, previous = old[key]
                assert task["experiment_uuid"] == group.experiment_uuid
                assert machine_types[task["allocated_machine_id"]] == previous.optimal_machine_type
                assert task["processing_time"] == previous.processing_time
                if not group.is_not_started():
                    assert task["scheduled_time"] == previous.scheduled_time
                    assert task["allocated_machine_id"] == previous.allocated_machine_id
                if task["processing_time"] > 0:
                    reservations[task["allocated_machine_id"]].append((
                        task["scheduled_time"], task["scheduled_time"] + task["processing_time"]))
            for intervals in reservations.values():
                intervals.sort()
                assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
            result_key = (row["event"]["task_group_id"], row["event"]["task_id"])
            assert assigned[result_key]["task_status"] == "Completed"
            valid_count += 1
        else:
            assert payload["outcome"] == row["outcome"] == "NO_SOLUTION"
            assert row["reports"][0]["status"] == "UNKNOWN"
        pairs[(pattern, limit)].append(row)
    assert len(rows) == len(bases) * 2 * env["repetitions"]
    assert all(sorted(row["repetition"] for row in selected) == list(range(env["repetitions"]))
               for selected in pairs.values())
    for item in json.loads((root / "statistics.json").read_text()):
        selected = [r["response_seconds"] for r in pairs[
            (item["pattern"], item["search_limit_seconds"])] if r["outcome"] == "VALID"]
        selected.sort()
        assert item["valid"] == len(selected)
        for key, quantile in [("median", .5), ("q1", .25), ("q3", .75), ("p95", .95)]:
            if not selected:
                continue
            position = (len(selected) - 1) * quantile
            low = int(position)
            high = min(low + 1, len(selected) - 1)
            value = selected[low] + (selected[high] - selected[low]) * (position - low)
            assert abs(value - item["response_seconds"][key]) < 1e-9
    verification = {
        "source_commit": env["commit"], "trials": len(rows), "valid_outputs": valid_count,
        "unknown_responses": len(rows) - valid_count, "solver_calls": len(rows),
        "condition_count": len(pairs), "repetitions": env["repetitions"],
        "paired_inputs_events": "same restored input hash and event at both limits",
        "source_files": "four measured files agree with recorded clean source commit",
        "output_checks": "JSON hash/size, task/type compatibility, fixed fields, physical nonoverlap",
        "time_checks": "one search, requested limit, phase sum, independent linear quantiles",
    }
    (root / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    return verification


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))

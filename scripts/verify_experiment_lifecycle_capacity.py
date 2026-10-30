"""Independently check recorded lifecycle capacity results without a solver call."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gems_python.common.persistence import load_snapshot
from gems_python.common.scheduling import validate_groups
from gems_python.common.serialization import equivalent
from gems_python.multi_machine_problem_interval_task.lifecycle import fingerprint
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments
from scripts.measure_experiment_lifecycle import conditions
from scripts.measure_scheduler_capacity import dimensions, quality


def task_specifications(lab):
    # Numeric group IDs may be assigned by planning; persistent IDs and physical
    # specifications must remain unchanged.
    return [(g.task_group_uuid, g.experiment_uuid, g.optimal_start_time, g.penalty_type, g.status,
             [(t.task_id, t.processing_time, t.interval, t.optimal_machine_type, t.task_status)
              for t in g.tasks]) for g in lab.task_groups]


def verify(root):
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    completion = json.loads((root / "completion.json").read_text())
    assert len(rows) == completion["trial_count"]
    pairs = defaultdict(list)
    counts = Counter()
    boundaries = defaultdict(list)
    for row in rows:
        directory = root / row["artifact_directory"]
        job = json.loads((directory / "job.json").read_text())
        for key, value in job.items():
            assert row[key] == value
        if row["outcome"] in ("WORKER_ERROR", "WORKER_TIMEOUT_OR_ERROR"):
            counts["worker_failures"] += 1
            continue
        source = Experiments.from_json((directory / "input.json").read_text())
        assert fingerprint(source) == row["input_hash"]
        assert fingerprint(conditions(source)) == row["condition_hash"]
        assert dimensions(source) == row["dimensions"]
        target = Experiments.from_json((directory / "output.json").read_text())
        assert fingerprint(task_specifications(source)) == fingerprint(task_specifications(target)), \
            "Planner changed task specification"
        assert {e.experiment_uuid for e in source.experiments} == \
            {e.experiment_uuid for e in target.experiments}
        assert fingerprint(source.ended_experiments) == fingerprint(target.ended_experiments)
        assert fingerprint(source.split_requests) == fingerprint(target.split_requests)
        for before, after in zip(source.experiments, target.experiments):
            assert before.experiment_uuid == after.experiment_uuid
            assert fingerprint(before.history_for_state()) == fingerprint(after.history_for_state())
            assert fingerprint(before.states) == fingerprint(after.states)
        assert equivalent(target, load_snapshot(directory / "saved", Experiments))
        assert len(source.experiments) == row["parents"] * row["children"] ** row["generations"]
        assert len({e.experiment_uuid for e in source.experiments}) == len(source.experiments)
        if row["mode"] == "lifecycle":
            expected_archives = sum(row["parents"] * row["children"] ** g
                                    for g in range(row["generations"]))
            assert len(source.ended_experiments) == expected_archives
            assert all(e.parent_uuid in source.ended_experiments for e in source.experiments)
            assert all(r.status == "COMMITTED" for r in source.split_requests.values())
            assert all(r.child_uuids and len(set(r.child_uuids)) == row["children"]
                       for r in source.split_requests.values())
        else:
            assert not source.ended_experiments and not source.split_requests
            assert all(e.parent_uuid is None and e._inherited_history is None for e in source.experiments)
        assert len(row["reports"]) == 1, "Capacity comparison must not retry"
        assert row["reports"][0]["solver_time_limit_seconds"] == row["budget"]
        if row["outcome"].startswith("VALID_"):
            fixed = {id(new): (old.scheduled_time, old.allocated_machine_id)
                     for old_group, new_group in zip(source.task_groups, target.task_groups)
                     if not old_group.is_not_started()
                     for old, new in zip(old_group.tasks, new_group.tasks)}
            validate_groups(target.task_groups, target.reference_time, target.machine_list, fixed)
            assert row["reports"][0]["valid"]
            assert row["reports"][0]["status"] in ("OPTIMAL", "FEASIBLE")
            assert (row["planning_seconds"] <= row["budget"]) == (row["outcome"] == "VALID_ON_TIME")
            for key, value in quality(target).items():
                assert row[key] == value, f"Invalid derived metric: {key}"
            counts["valid_outputs"] += 1
        else:
            assert (directory / "error.json").exists()
            assert target.manager_status == "STOPPED"
            assert target.stop_diagnostic["kind"] == row["outcome"]
            counts["preserved_planning_failures"] += 1
        key = tuple(row[k] for k in ("series", "pattern", "budget", "parents", "children",
                                    "group_interval", "generations", "repetition"))
        pairs[key].append(row)
        counts["verified_inputs"] += 1
        counts["verified_snapshots"] += 1
        if row["series"] in ("scan", "boundary") and row["mode"] == "lifecycle":
            boundaries[(row["pattern"], row["budget"], row["parents"])].append(row["outcome"])
    for paired in pairs.values():
        if len(paired) == 2:
            assert {r["mode"] for r in paired} == {"lifecycle", "flat"}
            assert paired[0]["condition_hash"] == paired[1]["condition_hash"]
            counts["verified_pairs"] += 1
    report = dict(counts, outcomes=dict(Counter(r["outcome"] for r in rows)),
                  trial_count=len(rows), fully_measured=completion["fully_measured"],
                  skipped=len(completion["skipped"]))
    summary = []
    for pattern, budget in sorted({(k[0], k[1]) for k in boundaries}):
        tested = {parents: outcomes for (p, b, parents), outcomes in boundaries.items()
                  if (p, b) == (pattern, budget)}
        all_success = [n for n, outcomes in tested.items() if len(outcomes) >= 3 and
                       all(o == "VALID_ON_TIME" for o in outcomes)]
        summary.append(dict(pattern=pattern, budget=budget,
                            maximum_confirmed_on_time_parents=max(all_success, default=None),
                            tested={str(k): v for k, v in sorted(tested.items())}))
    (root / "verification.json").write_text(json.dumps(report, indent=2))
    (root / "boundaries.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)

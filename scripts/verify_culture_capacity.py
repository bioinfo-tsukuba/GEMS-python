"""Independently verify recorded culture runs without rerunning the solver."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def latent_density(parameters, elapsed_hours):
    active = max(0.0, elapsed_hours-parameters.get("lag_hours", 0))
    initial, rate = parameters["initial_density"], parameters["rate_per_hour"]
    if parameters["model"] == "exponential":
        return min(1.0, initial*math.exp(min(700, rate*active)))
    return 1/(1+(1/initial-1)*math.exp(-rate*active))


def verify_trial(directory, config, model_hash):
    request = json.loads((directory / "input.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    events = json.load(gzip.open(directory / "events.json.gz", "rt"))
    calls = json.load(gzip.open(directory / "solver_calls.json.gz", "rt"))
    issues = []
    cultures = request["cultures"]
    if request["config_hash"] != canonical_hash(config) or request["model_hash"] != model_hash:
        issues.append("Input config/model identity mismatch")
    if result["input_hash"] != canonical_hash(cultures):
        issues.append("Sampled-input fingerprint mismatch")
    origins = {key: [0 if value["cell_type"] == "HEK" else -1, 0]
               for key, value in cultures.items()}
    groups = defaultdict(list)
    primary_errors, supplementary_errors, sampled = 0, 0, set()
    targeted = 0
    last_end = -1
    occupation, refill_occupation = 0, 0
    for event in events:
        if event["end"] < last_end:
            issues.append("Result delivery order is not monotonic")
        last_end = event["end"]
        if event["end"]-event["start"] != event["duration"]:
            issues.append("Incorrect operation duration")
        if event["machine"] != "ROUND3-shared-resource":
            issues.append("Invalid machine ID")
        groups[event["group_uuid"]].append(event)
        occupation += event["duration"]
        if event["culture_id"] == "refill":
            refill_occupation += event["duration"]
            if event["duration"] not in (0, config["round3"]["refill_processing_minutes"]):
                issues.append("Refill time omitted or altered")
            continue
        identity, kind = event["culture_id"], event["cell_type"]
        if identity not in cultures:
            issues.append("Unknown culture identity")
            continue
        if cultures[identity]["cell_type"] != kind:
            issues.append("Culture type mismatch")
        operation = event["operation"].removeprefix(kind)
        expected_duration = config["round3"][kind]["processing_minutes"].get(operation)
        if expected_duration is None or event["duration"] != expected_duration:
            issues.append("Configured operation duration omitted or altered")
        generation, passage_end = origins[identity]
        expected = (config["round3"]["iPS"]["passage_density"] if generation < 0 else
                    latent_density(cultures[identity]["parameters"][generation],
                                   (event["start"]-passage_end)/60))
        if abs(expected-event["density"]) > 1e-10:
            issues.append("Latent density disagrees with independent elapsed-time formula")
        if event["generation"] != generation:
            issues.append("Passage generation mismatch")
        is_targeted = generation >= 0 and event["operation"].endswith(("Passage", "Sampling"))
        if is_targeted:
            targeted += 1
            target = config["round3"][kind][
                "sampling_density" if event["operation"].endswith("Sampling")
                else "passage_density"]
            error = expected-target
            if abs(event["target"]-target) > 1e-12 or abs(event["error"]-error) > 1e-10:
                issues.append("Target/error mismatch")
            if "nominal_action_density" in event:
                nominal_start = event["group_optimal_start"] + event["start"]-event["group_start"]
                nominal = latent_density(cultures[identity]["parameters"][generation],
                                         (nominal_start-passage_end)/60)
                if abs(nominal-event["nominal_action_density"]) > 1e-10:
                    issues.append("Nominal-density diagnostic mismatch")
                if abs(event["nominal_density_error"]+event["schedule_density_change"] -
                       event["error"]) > 1e-10:
                    issues.append("Signed error decomposition mismatch")
            primary = abs(error) <= config["primary_density_half_width"]+1e-12
            width = min(config["primary_density_half_width"],
                        config["supplementary_relative_half_width"]*target)
            supplementary = abs(error) <= width+1e-12
            if event["primary_pass"] != primary or event["supplementary_pass"] != supplementary:
                issues.append("Acceptance criterion mismatch")
            primary_errors += not primary
            supplementary_errors += not supplementary
        elif event["target"] is not None:
            issues.append("Untargeted startup operation incorrectly scored")
        if event["operation"].endswith("Passage"):
            origins[identity] = [generation+1, event["end"]]
        if event["operation"].endswith("Sampling"):
            if identity in sampled:
                issues.append("Duplicate sampling")
            sampled.add(identity)
    slots = sorted([e for e in events if e["duration"] > 0],
                   key=lambda event: (event["start"], event["end"]))
    for before, after in zip(slots, slots[1:]):
        if before["end"] > after["start"]:
            issues.append("Physical-machine occupancy overlap")
    for rows in groups.values():
        rows.sort(key=lambda row: row["task_id"])
        expected = rows[0]["group_start"]
        for index, row in enumerate(rows):
            if row["task_id"] != index:
                issues.append("Missing/reordered operation within completed prefix")
            expected += row["interval"]
            if row["start"] != expected or row["group_start"] != rows[0]["group_start"]:
                issues.append("Changed group start or fixed interval")
            expected += row["duration"]
    if result["primary_violations"] != primary_errors or (
            result["supplementary_violations"] != supplementary_errors):
        issues.append("Reported violation count mismatch")
    if result["scored_operations"] != targeted or result["completed_cultures"] != len(sampled):
        issues.append("Reported operation/completion count mismatch")
    if result["completed"]:
        if sampled != set(cultures) or targeted != 3*len(cultures):
            issues.append("Completed run does not include all passage/sampling endpoints")
        if len(calls) != len(events)+1:
            issues.append("Completed run is not one initial solve plus one update per event")
    if result["primary_accepted"] != (result["completed"] and primary_errors == 0):
        issues.append("Primary run decision mismatch")
    if result["supplementary_accepted"] != (result["completed"] and supplementary_errors == 0):
        issues.append("Supplementary run decision mismatch")
    if result["refill_occupancy_minutes"] != refill_occupation or (
            result["culture_occupancy_minutes"]+refill_occupation != occupation):
        issues.append("Occupancy aggregation mismatch")
    expected_utilization = occupation/result["virtual_minutes"] if result["virtual_minutes"] else 0
    if abs(expected_utilization-result["machine_utilization"]) > 1e-12:
        issues.append("Reported resource utilization mismatch")
    if result["completed"] != (result["outcome"] == "COMPLETED"):
        issues.append("Completion/outcome classification mismatch")
    statuses = dict(Counter(row["status"] for row in calls))
    if statuses != result["solver_statuses"] or len(calls) != result["solver_calls"]:
        issues.append("Solver call/status aggregation mismatch")
    if any(row["solver_time_limit_seconds"] != config["solver_time_limit_seconds"] for row in calls):
        issues.append("Changed solver search limit")
    if result["event_check_errors"] or result["fixed_reservation_errors"]:
        issues.append("Runtime verification recorded errors")
    return issues, {"events": len(events), "solver_calls": len(calls),
                    "targeted_operations": targeted, "outcome": result["outcome"]}


def verify(root):
    metadata = json.loads((root / "metadata.json").read_text())
    config = metadata["config"]
    expected = {tuple(case) for case in metadata["cases"]}
    records, failures, totals, manifest = {}, {}, Counter(), []
    for directory in sorted(root.glob("h*_i*_r*")):
        if not (directory / "result.json").exists():
            failures[directory.name] = ["No recorded outcome"]
            continue
        issues, counts = verify_trial(directory, config, metadata["model_hash"])
        result = json.loads((directory / "result.json").read_text())
        case = (result["hek"], result["ips"], result["repetition"])
        if case in records:
            issues.append("Duplicate run case")
        records[case] = counts
        totals.update({key: counts[key] for key in ("events", "solver_calls", "targeted_operations")})
        if issues:
            failures[directory.name] = issues
        for name in ("input.json", "result.json", "events.json.gz", "solver_calls.json.gz"):
            path = directory / name
            manifest.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    missing = sorted(expected-set(records))
    unexpected = sorted(set(records)-expected)
    output = {
        "valid": not failures and not missing and not unexpected,
        "expected_cases": len(expected), "recorded_cases": len(records),
        "missing_cases": missing, "unexpected_cases": unexpected, "failures": failures,
        "totals": dict(totals), "outcomes": dict(Counter(r["outcome"] for r in records.values())),
        "config_hash": canonical_hash(config), "model_hash": metadata["model_hash"],
        "method": "Independent math replay, group intervals, machine occupancy, counts and hashes; no solver replay",
    }
    (root / "verification.json").write_text(json.dumps(output, indent=2)+"\n")
    (root / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(output, indent=2))
    return output["valid"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if not verify(args.root):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

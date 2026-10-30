"""Run final checks after measurements, without concurrent solver workloads."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

root = Path(__file__).resolve().parent
repo = root.parent
study = root / "capacity-a719506"


def status(stage, **details):
    (root / "final_checks_status.json").write_text(json.dumps(
        dict(stage=stage, time=datetime.now(timezone.utc).isoformat(), **details), indent=2))


def run(command, log, cwd=repo):
    with (root / log).open("w") as stream:
        return subprocess.run(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                              stdout=stream, stderr=subprocess.STDOUT).returncode


status("waiting_for_postprocessing")
while True:
    pipeline = json.loads((root / "pipeline_status.json").read_text())
    if pipeline["stage"] == "complete":
        break
    if pipeline["stage"] == "failed":
        status("failed", error="Postprocessing failed", pipeline=pipeline)
        raise RuntimeError("Postprocessing must succeed before final checks")
    time.sleep(5)

environment = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(root / "mpl-final"))
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
isolated = root / ("full-final-" + head[:8])
isolated.mkdir()
with tarfile.open(fileobj=io.BytesIO(
        subprocess.check_output(["git", "archive", head], cwd=repo))) as archive:
    archive.extractall(isolated, filter="data")

status("full_suite", source_commit=head, source_directory=str(isolated))
command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
started = datetime.now(timezone.utc).isoformat()
returncode = run(command, "full-final-tests.txt", cwd=isolated)
finished = datetime.now(timezone.utc).isoformat()
baseline = (root / "baseline-tests.txt").read_text()
actual = (root / "full-final-tests.txt").read_text()
errors = lambda text: sorted(re.findall(r"^(?:ERROR|FAIL): (.+)$", text, re.MULTILINE))
summary = re.search(r"Ran (\d+) tests in ([\d.]+)s", actual)
assert summary, "Missing unittest summary"
test_receipt = dict(source_commit=head, command=command, started=started, finished=finished,
                    returncode=returncode, count=int(summary[1]), seconds=float(summary[2]),
                    baseline_errors=errors(baseline), actual_errors=errors(actual),
                    baseline_count=109, baseline_commit="2140ab1919adb453122fd964cfdb3a7c216c74b9")
(study / "final_tests.json").write_text(json.dumps(test_receipt, indent=2))
assert returncode == 1 and len(errors(actual)) == 5 and errors(actual) == errors(baseline), (
    "Final suite differs from the five known baseline errors")
assert int(summary[1]) == 147, "Unexpected final test coverage"

status("coverage_and_profiles")
rows = [json.loads(line) for line in (study / "results.jsonl").read_text().splitlines()]
keys = ("series", "pattern", "budget", "parents", "children", "group_interval",
        "generations", "mode", "repetition")
signatures = [tuple(row[k] for k in keys) for row in rows]
assert len(set(signatures)) == len(rows), "Duplicate measurements"
pairs = defaultdict(set)
for row in rows:
    pairs[tuple(row[k] for k in keys if k != "mode")].add(row["mode"])
assert all(modes == {"flat", "lifecycle"} for modes in pairs.values()), "Unpaired conditions"
profiles = json.loads((study / "initialization_profiles.json").read_text())
completion = json.loads((study / "completion.json").read_text())
assert completion["fully_measured"] and completion["trial_count"] == len(rows)
assert not profiles["skipped"] and not profiles["failures"]
assert len(profiles["completed"]) == len(rows), "Missing initialization profiles"
coverage = dict(trial_count=len(rows), paired_conditions=len(pairs), signatures_unique=True,
                modes_complete=True, primary_trials=sum(row["repetition"] == 0 for row in rows),
                outcomes=dict(Counter(row["outcome"] for row in rows)),
                primary_coverage=json.loads((study / "primary_coverage.json").read_text()),
                initialization_profiles=len(profiles["completed"]))
(study / "final_coverage.json").write_text(json.dumps(coverage, indent=2))
assert coverage["primary_trials"] == 260

status("regenerate_statistics")
reproduced = root / "statistics-reproduced"
assert run([sys.executable, "-m", "scripts.analyze_experiment_lifecycle_capacity",
            str(study), "--output", str(reproduced)], "final-regeneration.txt") == 0
original = root / "statistics"
names = sorted(p.name for p in original.iterdir() if p.is_file())
assert names == sorted(p.name for p in reproduced.iterdir() if p.is_file())
assert len(names) == 6
hashes = {}
for name in names:
    original_hash = hashlib.sha256((original / name).read_bytes()).hexdigest()
    assert original_hash == hashlib.sha256((reproduced / name).read_bytes()).hexdigest(), name
    hashes[name] = original_hash
(study / "final_regeneration.json").write_text(json.dumps(
    dict(source_commit=head, matched_files=hashes, all_byte_identical=True), indent=2))
status("complete", source_commit=head, tests=test_receipt, coverage=coverage,
       regenerated_files=hashes)
print(json.dumps(dict(source_commit=head, tests=summary[1], trials=len(rows),
                      matched_files=len(names))))

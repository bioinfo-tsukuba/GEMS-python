"""Session orchestration only; the repository CLIs implement each reproducible step."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
study = root / "capacity-a719506"
deadline = datetime.fromisoformat(
    json.loads((study / "study_budget.json").read_text())["deadline"]).timestamp()
commands = [
    ("confirm_boundaries", ["-m", "scripts.measure_experiment_lifecycle", "--resume",
                           "--config", str(study / "config.json"), "--output", str(study)]),
    ("profile_initialization", ["-m", "scripts.measure_experiment_lifecycle",
                               "--profile-all-initializations", str(study)]),
    ("verify_outputs", ["-m", "scripts.verify_experiment_lifecycle_capacity", str(study)]),
    ("generate_statistics", ["-m", "scripts.analyze_experiment_lifecycle_capacity",
                            str(study), "--output", str(root / "statistics")]),
]


def status(stage, **details):
    (root / "pipeline_status.json").write_text(json.dumps(dict(
        stage=stage, time=datetime.now(timezone.utc).isoformat(), **details), indent=2))


status("waiting_for_primary_measurement")
while not (study / "completion.json").exists():
    if time.time() >= deadline + 30:
        status("failed", error="Primary measurement did not produce completion by its deadline")
        raise RuntimeError("Missing primary completion")
    time.sleep(5)
for name, arguments in commands:
    status(name, command=[sys.executable, *arguments])
    with (root / f"postprocessing-{name}.txt").open("w") as log:
        result = subprocess.run([sys.executable, *arguments], stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        status("failed", step=name, returncode=result.returncode)
        raise RuntimeError(f"{name} failed")
status("complete")

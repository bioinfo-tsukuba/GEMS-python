"""Pin and audit ROUND3 snapshots without publishing private observations.

Download requires authenticated gh. Source files, normalized rows and detailed
provenance belong in an ignored/private output directory.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess


REPOSITORY = "CAB314/GEMS-python-ROUND3"
COMMIT = "26bd4552e08a3a3b8c1c654107d8b3198cb988f8"
RUN = "iPSsimulation/results/2024-11-13_real_round3"


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def git_blob_hash(content):
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def download(directory, workers=8):
    directory.mkdir(parents=True, exist_ok=True)
    tree_path = directory / "tree.json"
    if not tree_path.exists():
        tree_path.write_bytes(subprocess.check_output(
            ["gh", "api", f"repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1"]))
    tree = json.loads(tree_path.read_text())
    if tree.get("truncated"):
        raise ValueError("Truncated source tree")
    selected = []
    for entry in tree["tree"]:
        path = entry["path"]
        if entry["type"] != "blob":
            continue
        if (
            path.startswith(RUN + "/") and path.endswith("/experiments.json")
            or path.startswith(RUN + "/experimental_setting/") and path.endswith(".py")
            or path in {
                RUN + "/executed_schedule.csv",
                RUN + "/2024-11-13_mediator_log_tailed1000.txt",
                RUN + "/2024-11-13_result_tailed1000.txt",
                "2024-11-13_mediator_log.txt",
                "iPSsimulation/samples/experiments/hek293a.py",
                "iPSsimulation/samples/experiments/ips_253g1.py",
                "iPSsimulation/main.py",
            }
        ):
            selected.append(entry)

    def fetch(entry):
        path = directory / "files" / entry["path"]
        if not path.exists():
            content = subprocess.check_output([
                "gh", "api", "-H", "Accept: application/vnd.github.raw+json",
                f"repos/{REPOSITORY}/contents/{entry['path']}?ref={COMMIT}",
            ])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        content = path.read_bytes()
        if git_blob_hash(content) != entry["sha"]:
            raise ValueError(f"Source blob hash mismatch: {entry['path']}")
        return {"path": entry["path"], "bytes": len(content),
                "git_blob": entry["sha"], "sha256": sha256(content)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        manifest = list(pool.map(fetch, selected))
    write_json(directory / "input_manifest.json", {
        "repository": REPOSITORY, "commit": COMMIT, "files": manifest,
        "snapshot_count": sum(x["path"].endswith("/experiments.json") for x in manifest),
    })
    print(f"Verified {len(manifest)} source blobs at {COMMIT}")


def history_rows(history):
    """Decode the original one-machine history dictionary."""
    if isinstance(history, str):
        history = json.loads(history)
    if isinstance(history, dict) and "columns" in history and "data" not in history:
        columns = history["columns"]
        if not columns:
            return []
        if all(isinstance(column, dict) and "values" in column for column in columns):
            names = [column["name"] for column in columns]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate source column names")
            lengths = {len(column["values"]) for column in columns}
            if len(lengths) != 1:
                raise ValueError("Ragged source column values")
            return [dict(zip(names, row)) for row in
                    zip(*(column["values"] for column in columns))]
    if isinstance(history, dict) and "columns" in history and "data" in history:
        return [dict(zip(history["columns"], row)) for row in history["data"]]
    if isinstance(history, dict) and all(isinstance(x, list) for x in history.values()):
        lengths = {len(x) for x in history.values()}
        if len(lengths) > 1:
            raise ValueError("Ragged source history")
        return [dict(zip(history, row)) for row in zip(*history.values())]
    if isinstance(history, list):
        return history
    raise ValueError(f"Unknown history encoding: {type(history).__name__}")


def extract(directory):
    root = directory / "files" / RUN
    snapshots = sorted(root.glob("step_[0-9]*/experiments.json"))
    if not snapshots:
        raise ValueError("No numbered snapshots")
    ledger, revisions, appearances, seen_snapshots = {}, [], {}, {}
    for path in snapshots:
        snapshot = json.loads(path.read_text())
        step = int(path.parent.name.split("_")[-1])
        seen_snapshots[step] = {"reference_time": snapshot["reference_time"],
                                "sha256": sha256(path.read_bytes())}
        for exp in snapshot["experiments"]:
            identity = exp["experiment_uuid"]
            if not identity.startswith(("HEK_Experiment_", "iPS_Experiment_")):
                continue
            kind = "HEK" if identity.startswith("HEK") else "iPS"
            appearances.setdefault(identity, []).append({
                "step": step, "state": exp["current_state_name"],
                "reference_time": snapshot["reference_time"],
            })
            for row in history_rows(exp["shared_variable_history"]):
                if row.get("time") is None or not row.get("operation"):
                    raise ValueError(f"Missing operation/time in {path}")
                key = (identity, int(row["time"]), row["operation"])
                payload = {"culture_id": identity, "cell_type": kind, **row,
                           "density": row.get("density"), "memo": row.get("memo")}
                if key not in ledger:
                    ledger[key] = {"row": payload, "first_step": step, "last_step": step,
                                   "source": str(path.relative_to(directory))}
                elif ledger[key]["row"] != payload:
                    revisions.append({"key": list(key), "step": step,
                                      "before": ledger[key]["row"], "after": payload})
                    ledger[key]["row"] = payload
                    ledger[key]["last_step"] = step
                else:
                    ledger[key]["last_step"] = step
    rows = []
    for record in ledger.values():
        row = dict(record["row"])
        row.update({k: record[k] for k in ("first_step", "last_step", "source")})
        rows.append(row)
    rows.sort(key=lambda x: (x["cell_type"], x["culture_id"], int(x["time"]),
                             x["operation"]))
    columns = ["culture_id", "cell_type", "time", "operation", "density", "memo",
               "first_step", "last_step", "source"]
    with (directory / "normalized_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory / "audit.json", {
        "repository": REPOSITORY, "commit": COMMIT,
        "numbered_snapshot_count": len(snapshots),
        "snapshot_steps": seen_snapshots,
        "unique_operation_rows": len(rows),
        "density_rows": sum(row.get("density") is not None for row in rows),
        "culture_appearances": appearances, "revisions": revisions,
        "identity_key": ["culture_id", "time", "operation"],
        "deduplication": "Replayed cumulative rows once; latest revision retained with ledger",
        "units": {"time": "Unix epoch minutes", "density": "image-derived fraction"},
        "unit_status": "Source-code interpretation; not an independent image calibration",
    })
    print(f"Audited {len(snapshots)} snapshots, {len(rows)} unique rows, "
          f"{len(revisions)} revisions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.download:
        download(args.output, args.workers)
    extract(args.output)


if __name__ == "__main__":
    main()

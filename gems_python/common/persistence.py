"""Noninteractive loading of matching JSON/pickle experiment snapshots."""

from pathlib import Path
import hashlib
import json
import os
import shutil
from uuid import uuid4
import warnings

from gems_python.common.serialization import equivalent


def _write_synced(path, content):
    with path.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_snapshot(collection, directory):
    directory = Path(directory)
    # Serialize both before writing so unsupported values leave no partial pair.
    contents = {"json": collection.to_json().encode("utf-8"),
                "pickle": collection.to_pickle()}
    directory.mkdir(parents=True, exist_ok=True)
    generations = directory / ".snapshot_generations"
    generations.mkdir(exist_ok=True)
    generation = uuid4().hex
    staging = generations / f".pending-{generation}"
    staging.mkdir()
    names = {"json": "experiments.json", "pickle": "experiments.pkl"}
    pointer_tmp = directory / f".snapshot_manifest-{generation}.tmp"
    try:
        for kind, content in contents.items():
            _write_synced(staging / names[kind], content)
        _sync_directory(staging)
        os.replace(staging, generations / generation)
        _sync_directory(generations)
        manifest = dict(version=1, generation=generation,
                        sha256={kind: hashlib.sha256(content).hexdigest()
                                for kind, content in contents.items()})
        _write_synced(pointer_tmp, json.dumps(manifest).encode("utf-8"))
        # Linearization point: readers select one complete generation once.
        os.replace(pointer_tmp, directory / "snapshot_manifest.json")
        _sync_directory(directory)
        # Compatibility exports are not the source of truth for a versioned reader.
        for kind, content in contents.items():
            temporary = directory / f".{names[kind]}-{generation}.tmp"
            _write_synced(temporary, content)
            os.replace(temporary, directory / names[kind])
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        pointer_tmp.unlink(missing_ok=True)


def load_snapshot(directory, cls, *, source="auto"):
    if source not in ("auto", "json", "pickle"):
        raise ValueError("source must be 'auto', 'json', or 'pickle'")
    directory = Path(directory)
    selected = directory
    manifest = None
    pointer = directory / "snapshot_manifest.json"
    if pointer.exists():
        try:
            manifest = json.loads(pointer.read_text(encoding="utf-8"))
            generation = manifest["generation"]
            if (manifest["version"] != 1 or len(generation) != 32 or
                    any(c not in "0123456789abcdef" for c in generation)):
                raise ValueError("Invalid snapshot generation/version")
            selected = directory / ".snapshot_generations" / generation
        except Exception as err:
            raise RuntimeError(f"Cannot load snapshot manifest at {directory}: {err}") from err
    loaded, errors = {}, {}
    for format_name in (("json", "pickle") if source == "auto" else (source,)):
        try:
            if format_name == "json":
                path = selected / "experiments.json"
            else:
                path = selected / "experiments.pkl"
            content = path.read_bytes()
            if manifest is not None and hashlib.sha256(content).hexdigest() != manifest["sha256"][format_name]:
                raise ValueError(f"{format_name}: snapshot hash mismatch")
            loaded[format_name] = (cls.from_json(content.decode("utf-8")) if format_name == "json"
                                   else cls.from_pickle(dumped=content))
        except Exception as err:
            errors[format_name] = f"{type(err).__name__}: {err}"
    if not loaded:
        details = "; ".join(f"{name}: {error}" for name, error in errors.items())
        raise RuntimeError(f"Cannot load snapshot at {directory}: {details}")
    if len(loaded) == 2:
        if not equivalent(loaded["json"], loaded["pickle"]):
            raise ValueError("JSON and pickle snapshots differ; choose source='json' or source='pickle'.")
        return loaded["json"]
    if source == "auto":
        warnings.warn(f"Using {next(iter(loaded))} snapshot at {directory}; " +
                      "; ".join(f"{name}: {error}" for name, error in errors.items()), RuntimeWarning)
    return next(iter(loaded.values()))

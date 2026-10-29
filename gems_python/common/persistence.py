"""Noninteractive loading of matching JSON/pickle experiment snapshots."""

from pathlib import Path
import warnings

from gems_python.common.serialization import equivalent


def save_snapshot(collection, directory):
    directory = Path(directory)
    # Serialize both before writing so unsupported values leave no partial pair.
    json_text = collection.to_json()
    pickle_bytes = collection.to_pickle()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "experiments.json").write_text(json_text, encoding="utf-8")
    (directory / "experiments.pkl").write_bytes(pickle_bytes)


def load_snapshot(directory, cls, *, source="auto"):
    if source not in ("auto", "json", "pickle"):
        raise ValueError("source must be 'auto', 'json', or 'pickle'")
    directory = Path(directory)
    loaded, errors = {}, {}
    for format_name in (("json", "pickle") if source == "auto" else (source,)):
        try:
            if format_name == "json":
                loaded[format_name] = cls.from_json((directory / "experiments.json").read_text(encoding="utf-8"))
            else:
                loaded[format_name] = cls.from_pickle(directory / "experiments.pkl")
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

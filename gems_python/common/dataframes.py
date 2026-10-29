"""DataFrame boundaries shared by the experiment managers.

Histories are ordered rows with named columns; Pandas indexes are not data.
Arrow provides lossless numeric storage and a common schema for concatenation.
"""

import base64
import io
from typing import Literal

import pandas as pd
import polars as pl
import pyarrow as pa


DataFrame = pd.DataFrame | pl.DataFrame
Backend = Literal["auto", "pandas", "polars"]


def is_dataframe(value) -> bool:
    return isinstance(value, (pd.DataFrame, pl.DataFrame))


def backend_of(frame: DataFrame) -> str:
    if isinstance(frame, pd.DataFrame):
        return "pandas"
    if isinstance(frame, pl.DataFrame):
        return "polars"
    raise TypeError("Expected a pandas.DataFrame or polars.DataFrame")


def resolve_backend(backend: Backend = "auto", frame: DataFrame | None = None) -> str:
    if backend not in ("auto", "pandas", "polars"):
        raise ValueError(f"Unknown dataframe_backend: {backend!r}")
    if frame is not None:
        backend_of(frame)
    return (backend_of(frame) if frame is not None else "pandas") if backend == "auto" else backend


def convert(frame: DataFrame, backend: Backend = "auto") -> DataFrame:
    target = resolve_backend(backend, frame)
    if backend_of(frame) == target:
        return copy_frame(frame)
    if target == "polars":
        return pl.from_pandas(frame.reset_index(drop=True), include_index=False)
    return frame.to_pandas(use_pyarrow_extension_array=True)


def make_frame(data=None, backend: Backend = "auto") -> DataFrame:
    target = resolve_backend(backend)
    return pd.DataFrame(data) if target == "pandas" else pl.DataFrame(data)


def copy_frame(frame: DataFrame) -> DataFrame:
    return frame.copy(deep=True).reset_index(drop=True) if backend_of(frame) == "pandas" else frame.clone()


def _arrow(frame: DataFrame) -> pa.Table:
    backend_of(frame)
    if isinstance(frame, pd.DataFrame):
        if not frame.columns.is_unique or any(not isinstance(c, str) for c in frame.columns):
            raise ValueError("DataFrame columns must have unique string names")
        return pa.Table.from_pandas(frame, preserve_index=False)
    return frame.to_arrow()


def append(history: DataFrame, result: DataFrame) -> DataFrame:
    """Union columns, fill absent values, and promote compatible numeric types.

    Arrow rejects incompatible schemas instead of silently turning numbers into
    strings. Row order and first-seen column order are preserved.
    """
    target = backend_of(history)
    tables = [_arrow(history), _arrow(convert(result, target))]
    try:
        table = pa.concat_tables(tables, promote_options="permissive")
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as err:
        raise TypeError(f"Incompatible history/result column types: {err}") from err
    if target == "polars":
        return pl.from_arrow(table)
    return table.to_pandas(types_mapper=pd.ArrowDtype, ignore_metadata=True)


def update(history: DataFrame, result: DataFrame, update_type: str = "a") -> DataFrame:
    if update_type == "a":
        return append(history, result)
    if update_type == "o":
        return convert(result, backend_of(history))
    raise ValueError(f"Unexpected update_type: {update_type!r}")


def write_csv(frame: DataFrame, path) -> None:
    if backend_of(frame) == "pandas":
        frame.to_csv(path, index=False)
    else:
        frame.write_csv(path)


def read_csv(path, backend: Backend = "auto") -> DataFrame:
    # Use one inference path, including dates and empty files, for both backends.
    try:
        frame = pl.read_csv(path, try_parse_dates=True)
    except pl.exceptions.NoDataError:
        frame = pl.DataFrame()
    return convert(frame, resolve_backend(backend))


def serialize(frame: DataFrame) -> dict:
    table = _arrow(frame)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    envelope = {
        "_dataframe": 1,
        "backend": backend_of(frame),
        "format": "arrow-ipc",
        "data": base64.b64encode(sink.getvalue().to_pybytes()).decode("ascii"),
    }
    # Arrow has no column metadata for a table with zero columns.
    if isinstance(frame, pd.DataFrame) and isinstance(frame.columns, pd.RangeIndex) and not len(frame.columns):
        envelope["empty_range_columns"] = True
    return envelope


def deserialize(value) -> DataFrame:
    """Read the new envelope, legacy Polars JSON, or legacy column dictionaries."""
    if isinstance(value, str):
        return pl.DataFrame.deserialize(io.BytesIO(value.encode()), format="json")
    if isinstance(value, dict) and "_dataframe" in value:
        if value["_dataframe"] != 1 or value.get("format") != "arrow-ipc":
            raise ValueError("Unsupported DataFrame serialization format/version")
        backend = resolve_backend(value["backend"])
        if value["backend"] == "auto":
            raise ValueError("Serialized DataFrame must identify its backend")
        table = pa.ipc.open_stream(base64.b64decode(value["data"], validate=True)).read_all()
        restored = pl.from_arrow(table) if backend == "polars" else table.to_pandas()
        if backend == "pandas" and value.get("empty_range_columns") and not len(restored.columns):
            restored.columns = pd.RangeIndex(0)
        return restored
    if isinstance(value, dict):
        return pl.DataFrame(value)
    raise TypeError("Expected a serialized DataFrame")


def row_count(frame: DataFrame) -> int:
    return len(frame)


def last_value(frame: DataFrame, column: str):
    return frame[column].iloc[-1] if backend_of(frame) == "pandas" else frame[column][-1]


def records(frame: DataFrame) -> list[dict]:
    """Backend-independent rows, with NaN and null both represented as None."""
    table = _arrow(frame)
    return [
        {key: None if isinstance(value, float) and pd.isna(value) else value
         for key, value in row.items()}
        for row in table.to_pylist()
    ]

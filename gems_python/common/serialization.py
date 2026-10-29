"""Typed persistence of GEMS objects without rerunning their constructors.

Version 1 records carry concrete classes and container types. Classes must be
available to Python when reading. Cycles and unsupported resources fail with
an attribute path; no fields are silently discarded.
"""

import base64
import copy
from dataclasses import MISSING, fields, is_dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from importlib import import_module
import inspect
import json
import math
from pathlib import Path
import types
from typing import Any, Union, get_args, get_origin, get_type_hints
from uuid import UUID
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl

from gems_python.common import dataframes


MARKER = "_gems_serialization"
VERSION = 1


class SerializationError(ValueError):
    """A saved value cannot be represented or restored without information loss."""


def _record(kind, **payload):
    return {MARKER: VERSION, "kind": kind, **payload}


def _class_name(cls, path):
    if "<locals>" in cls.__qualname__:
        raise SerializationError(f"{path}: class must be importable: {cls.__module__}.{cls.__qualname__}")
    return f"{cls.__module__}:{cls.__qualname__}"


def _class_from_name(name, path):
    if not isinstance(name, str):
        raise SerializationError(f"{path}: missing concrete class")
    if ":" in name:
        module_name, qualified_name = name.split(":", 1)
    else:  # Historical State identifiers used module.Class.
        module_name, _, qualified_name = name.rpartition(".")
    try:
        cls = import_module(module_name)
        for part in qualified_name.split("."):
            cls = getattr(cls, part)
    except (ImportError, AttributeError, ValueError) as err:
        raise SerializationError(f"{path}: class {name!r} is unavailable") from err
    if not isinstance(cls, type):
        raise SerializationError(f"{path}: {name!r} is not a class")
    return cls


def _supported_object(value):
    return (is_dataclass(value) and not isinstance(value, type)) or getattr(type(value), "_gems_serializable", False)


def attributes(value):
    """Include dataclass fields, dynamic progress, and slots, without properties."""
    result = dict(getattr(value, "__dict__", {}))
    if is_dataclass(value):
        for field in fields(value):
            if hasattr(value, field.name):
                result.setdefault(field.name, getattr(value, field.name))
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in ("__dict__", "__weakref__"):
                continue
            if name.startswith("__") and not name.endswith("__"):
                name = f"_{cls.__name__.lstrip('_')}{name}"
            if hasattr(value, name):
                result.setdefault(name, getattr(value, name))
    return result


def encode(value):
    return _encode(value, "$", set(), {})


def _encode(value, path, active, seen):
    if isinstance(value, Enum):
        return _record("enum", **{"class": _class_name(type(value), path), "name": value.name})
    if isinstance(value, np.generic):
        if value.dtype.hasobject or value.dtype.fields:
            raise SerializationError(f"{path}: unsupported NumPy scalar dtype {value.dtype}")
        return _record("numpy_scalar", dtype=value.dtype.str, data=base64.b64encode(value.tobytes()).decode("ascii"))
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, UUID):
        return _record("uuid", value=str(value))
    if isinstance(value, Path):
        return _record("path", **{"class": _class_name(type(value), path), "value": str(value)})
    if isinstance(value, pd.Timestamp):
        return _record("timestamp", value=value.isoformat())
    if isinstance(value, datetime):
        return _record("datetime", value=value.isoformat(), fold=value.fold,
                       zone=getattr(value.tzinfo, "key", None), timezone_name=value.tzname())
    if isinstance(value, date):
        return _record("date", value=value.isoformat())
    if isinstance(value, timedelta):
        return _record("timedelta", days=value.days, seconds=value.seconds, microseconds=value.microseconds)
    if isinstance(value, bytes):
        return _record("bytes", data=base64.b64encode(value).decode("ascii"))
    if isinstance(value, type):
        return _record("class", **{"class": _class_name(value, path)})
    if id(value) in active:
        raise SerializationError(f"{path}: circular reference")
    if id(value) in seen:
        return _record("reference", target=seen[id(value)])
    object_id = len(seen)
    seen[id(value)] = object_id
    active.add(id(value))
    def record(kind, **payload):
        return _record(kind, object_id=object_id, **payload)
    try:
        if dataframes.is_dataframe(value):
            return record("dataframe", value=dataframes.serialize(value))
        if isinstance(value, list):
            return record("list", items=[_encode(v, f"{path}[{i}]", active, seen) for i, v in enumerate(value)])
        if isinstance(value, tuple):
            return record("tuple", items=[_encode(v, f"{path}[{i}]", active, seen) for i, v in enumerate(value)])
        if isinstance(value, (set, frozenset)):
            items = [_encode(v, f"{path}[]", active, seen) for v in value]
            items.sort(key=lambda v: json.dumps(v, sort_keys=True))
            return record("frozenset" if isinstance(value, frozenset) else "set", items=items)
        if isinstance(value, dict):
            return record("dict", items=[
                [_encode(k, f"{path}.<key>", active, seen), _encode(v, f"{path}[{k!r}]", active, seen)]
                for k, v in value.items()
            ])
        if _supported_object(value):
            return record("object", **{
                "class": _class_name(type(value), path),
                "attributes": {k: _encode(v, f"{path}.{k}", active, seen) for k, v in attributes(value).items()},
            })
        raise SerializationError(f"{path}: unsupported value type {type(value).__module__}.{type(value).__qualname__}")
    except SerializationError:
        raise
    except (TypeError, ValueError) as err:
        raise SerializationError(f"{path}: cannot serialize value: {err}") from err
    finally:
        active.remove(id(value))


def decode(value, expected=None):
    if isinstance(value, dict) and (MARKER in value or "_dataframe" in value):
        restored = _decode(value, "$")
    else:
        restored = _legacy(value, expected, "$")
    if isinstance(expected, type) and expected is not Any and not isinstance(restored, expected):
        raise SerializationError(f"$: expected {expected.__name__}, got {type(restored).__name__}")
    return restored


def _decode(value, path, context=None):
    if context is None:
        context = {"definitions": {}, "cache": {}, "active": set()}
        def collect(item):
            if isinstance(item, list):
                for child in item:
                    collect(child)
            elif isinstance(item, dict):
                if MARKER in item and "object_id" in item:
                    identifier = item["object_id"]
                    if type(identifier) is not int or identifier in context["definitions"]:
                        raise SerializationError(f"{path}: invalid or duplicate object identifier")
                    context["definitions"][identifier] = item
                for child in item.values():
                    collect(child)
        collect(value)
    identifier = value.get("object_id") if isinstance(value, dict) and MARKER in value else None
    if identifier in context["cache"]:
        return context["cache"][identifier]
    if identifier in context["active"]:
        raise SerializationError(f"{path}: circular saved reference")
    if identifier is not None:
        context["active"].add(identifier)
    try:
        result = _decode_value(value, path, context)
        if identifier is not None:
            context["cache"][identifier] = result
        return result
    except SerializationError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as err:
        raise SerializationError(f"{path}: invalid serialized value: {err}") from err
    finally:
        if identifier is not None:
            context["active"].remove(identifier)


def _decode_value(value, path, context):
    if isinstance(value, list):
        return [_decode(v, f"{path}[{i}]", context) for i, v in enumerate(value)]
    if not isinstance(value, dict):
        return value
    if "_dataframe" in value:
        return dataframes.deserialize(value)
    if MARKER not in value:
        return {k: _decode(v, f"{path}.{k}", context) for k, v in value.items()}
    if type(value[MARKER]) is not int or value[MARKER] != VERSION:
        raise SerializationError(f"{path}: unsupported serialization version {value[MARKER]!r}")
    kind = value.get("kind")
    if kind == "reference":
        target = value.get("target")
        if target not in context["definitions"]:
            raise SerializationError(f"{path}: unknown shared reference {target!r}")
        return _decode(context["definitions"][target], path, context)
    if kind == "dataframe":
        return dataframes.deserialize(value["value"])
    if kind == "object":
        cls = _class_from_name(value.get("class"), path)
        if not (is_dataclass(cls) or getattr(cls, "_gems_serializable", False)):
            raise SerializationError(f"{path}: unsupported object class {cls.__name__}")
        saved = value.get("attributes")
        if not isinstance(saved, dict):
            raise SerializationError(f"{path}: missing saved attributes")
        if is_dataclass(cls):
            missing = [f.name for f in fields(cls) if f.name not in saved]
            if missing:
                raise SerializationError(f"{path}: missing saved fields {', '.join(missing)}")
        return _restore(cls, {k: _decode(v, f"{path}.{k}", context) for k, v in saved.items()}, path)
    if kind == "dict":
        return {_decode(k, f"{path}.<key>", context): _decode(v, f"{path}[{i}]", context) for i, (k, v) in enumerate(value["items"])}
    if kind in ("list", "tuple", "set", "frozenset"):
        items = [_decode(v, f"{path}[{i}]", context) for i, v in enumerate(value["items"])]
        return {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[kind](items)
    if kind == "enum":
        cls = _class_from_name(value.get("class"), path)
        if not issubclass(cls, Enum):
            raise SerializationError(f"{path}: enum class required")
        return cls[value["name"]]
    if kind == "path":
        cls = _class_from_name(value.get("class"), path)
        if not issubclass(cls, Path):
            raise SerializationError(f"{path}: Path class required")
        return cls(value["value"])
    if kind == "uuid":
        return UUID(value["value"])
    if kind == "class":
        return _class_from_name(value.get("class"), path)
    if kind == "bytes":
        return base64.b64decode(value["data"], validate=True)
    if kind == "numpy_scalar":
        dtype = np.dtype(value["dtype"])
        raw = base64.b64decode(value["data"], validate=True)
        if dtype.hasobject or dtype.fields or len(raw) != dtype.itemsize:
            raise SerializationError(f"{path}: invalid NumPy scalar data")
        return np.frombuffer(raw, dtype=dtype)[0]
    if kind == "timestamp":
        return pd.Timestamp(value["value"])
    if kind == "datetime":
        result = datetime.fromisoformat(value["value"])
        if value.get("zone"):
            result = result.astimezone(ZoneInfo(value["zone"]))
        elif result.tzinfo is not None and value.get("timezone_name"):
            result = result.replace(tzinfo=timezone(result.utcoffset(), value["timezone_name"]))
        return result.replace(fold=value.get("fold", 0))
    if kind == "date":
        return date.fromisoformat(value["value"])
    if kind == "timedelta":
        return timedelta(days=value["days"], seconds=value["seconds"], microseconds=value["microseconds"])
    raise SerializationError(f"{path}: unknown serialized kind {kind!r}")


def _domain(value, name):
    return any(cls.__name__ == name and cls.__module__.startswith("gems_python.") for cls in type(value).__mro__)


def _restore(cls, saved, path):
    try:
        obj = object.__new__(cls)
        for key, value in saved.items():
            object.__setattr__(obj, key, value)
    except (TypeError, AttributeError) as err:
        raise SerializationError(f"{path}: cannot restore {cls.__name__}: {err}") from err
    if any(_domain(obj, name) for name in ("Experiment", "Task", "TaskGroup", "OneMachineTask", "Machine")):
        for name in ("experiment_uuid", "machine_id", "allocated_machine_id"):
            if isinstance(getattr(obj, name, None), UUID):
                object.__setattr__(obj, name, str(getattr(obj, name)))
    if _domain(obj, "Experiment"):
        names = [state.state_name for state in obj.states]
        index = obj.current_state_index
        if not isinstance(index, int) or not 0 <= index < len(names) or names[index] != obj.current_state_name:
            raise SerializationError(f"{path}.current_state_index: inconsistent with current_state_name")
        if obj.dataframe_backend != dataframes.backend_of(obj.shared_variable_history):
            raise SerializationError(f"{path}.dataframe_backend: inconsistent with history type")
    return obj


def _legacy(value, hint, path):
    if isinstance(value, dict) and (MARKER in value or "_dataframe" in value):
        return _decode(value, path)
    if value is None:
        return None
    origin, args = get_origin(hint), get_args(hint)
    if hint in (pd.DataFrame, pl.DataFrame) or pd.DataFrame in args or pl.DataFrame in args:
        return dataframes.deserialize(value)
    if origin in (Union, types.UnionType):
        candidates = [a for a in args if a is not type(None)]
        if len(candidates) == 1:
            return _legacy(value, candidates[0], path)
        matches = [a for a in candidates if isinstance(a, type) and isinstance(value, a)]
        if len(matches) == 1:
            return _legacy(value, matches[0], path)
        raise SerializationError(f"{path}: ambiguous legacy union type")
    if origin is type:
        hint = args[0]
    if isinstance(hint, type) and hasattr(hint, "calculate_penalty"):
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            raise SerializationError(f"{path}: invalid legacy penalty")
        if "penalty_type" in value:
            module = import_module(value.get("_penalty_module", hint.__module__))
            cls = getattr(module, value["penalty_type"], None)
            if cls is None or not isinstance(cls, type) or not hasattr(cls, "calculate_penalty"):
                raise SerializationError(f"{path}: unknown concrete penalty {value['penalty_type']!r}")
            value = {k: v for k, v in value.items() if k not in ("penalty_type", "_penalty_module")}
            return _legacy_object(cls, value, path)
        if inspect.isabstract(hint):
            raise SerializationError(f"{path}: missing concrete penalty type in legacy data")
    if isinstance(hint, type) and issubclass(hint, Enum):
        return hint[value]
    if isinstance(hint, type) and issubclass(hint, Path):
        return hint(value)
    if hint is UUID:
        return UUID(value)
    if isinstance(value, dict) and "_class" in value:
        cls = _class_from_name(value["_class"], path)
        saved = value.get("data", {k: v for k, v in value.items() if k != "_class"})
        return _legacy_object(cls, saved, path)
    if isinstance(hint, type) and getattr(hint, "_gems_serializable", False) and hint.__name__ == "State" and isinstance(value, str):
        pending, matches = list(hint.__subclasses__()), set()
        while pending:
            cls = pending.pop()
            pending.extend(cls.__subclasses__())
            if cls.__name__ == value:
                matches.add(cls)
        if len(matches) != 1:
            raise SerializationError(f"{path}: legacy State name {value!r} is unavailable or ambiguous")
        return _legacy_object(matches.pop(), {"state_name": value}, path)
    if isinstance(hint, type) and (is_dataclass(hint) or getattr(hint, "_gems_serializable", False)):
        return _legacy_object(hint, value, path)
    if isinstance(value, list):
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(_legacy(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
            if len(value) != len(args):
                raise SerializationError(f"{path}: wrong legacy tuple length")
            return tuple(_legacy(v, t, f"{path}[{i}]") for i, (v, t) in enumerate(zip(value, args)))
        item_hint = args[0] if origin is list and args else None
        return [_legacy(v, item_hint, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        key_hint, item_hint = args if origin is dict and len(args) == 2 else (None, None)
        return {_legacy(k, key_hint, f"{path}.<key>"): _legacy(v, item_hint, f"{path}.{k}") for k, v in value.items()}
    return value


_PROGRESS_FIELDS = {
    "tasks", "task_groups", "current_task", "current_task_group", "scheduled_time",
    "task_id", "task_group_id", "status", "task_status", "completed", "step",
    "reference_time", "experiment_uuid", "machine_id", "allocated_machine_id",
}


def _legacy_object(cls, saved, path):
    if not isinstance(saved, dict):
        raise SerializationError(f"{path}: legacy {cls.__name__} requires a dictionary")
    try:
        hints = get_type_hints(cls)
    except (NameError, TypeError) as err:
        raise SerializationError(f"{path}: cannot resolve legacy field types for {cls.__name__}") from err
    restored = {k: _legacy(v, hints.get(k), f"{path}.{k}") for k, v in saved.items()}
    if is_dataclass(cls):
        for field in fields(cls):
            if field.name in restored or field.name in ("current_state_index", "dataframe_backend"):
                continue
            if field.name == "state_name" and any(base.__name__ == "State" for base in cls.__mro__):
                restored[field.name] = cls.__name__
            elif field.name not in _PROGRESS_FIELDS and field.default is not MISSING:
                restored[field.name] = copy.deepcopy(field.default)
            else:
                raise SerializationError(f"{path}.{field.name}: missing legacy field; cannot reconstruct")
    if any(base.__name__ == "State" and base.__module__.startswith("gems_python.") for base in cls.__mro__):
        for name, parameter in inspect.signature(cls.__init__).parameters.items():
            if name != "self" and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD) and name not in restored:
                raise SerializationError(f"{path}.{name}: missing legacy State configuration")
    if any(base.__name__ == "Experiment" and base.__module__.startswith("gems_python.") for base in cls.__mro__):
        if "current_state_index" not in restored:
            indices = [i for i, s in enumerate(restored["states"]) if s.state_name == restored["current_state_name"]]
            if not indices:
                raise SerializationError(f"{path}.current_state_name: no matching State")
            if len(indices) != 1:
                raise SerializationError(f"{path}.current_state_name: ambiguous legacy State index")
            restored["current_state_index"] = indices[0]
        restored.setdefault("dataframe_backend", dataframes.backend_of(restored["shared_variable_history"]))
    return _restore(cls, restored, path)


def equivalent(left, right):
    """Compare persisted types, values and sharing without JSON byte equality."""
    return _equivalent(left, right, {}, {})


def _equivalent(left, right, left_seen, right_seen):
    if type(left) is not type(right):
        return False
    if _supported_object(left) or isinstance(left, (dict, list, tuple, set, frozenset, pd.DataFrame, pl.DataFrame)):
        left_id, right_id = id(left), id(right)
        if left_id in left_seen or right_id in right_seen:
            return left_seen.get(left_id) == right_id and right_seen.get(right_id) == left_id
        left_seen[left_id], right_seen[right_id] = right_id, left_id
    def same(a, b):
        return _equivalent(a, b, left_seen, right_seen)
    if isinstance(left, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(left, right, check_exact=True)
            return True
        except AssertionError:
            return False
    if isinstance(left, pl.DataFrame):
        return left.schema == right.schema and left.equals(right)
    if isinstance(left, np.generic):
        return left.dtype == right.dtype and left.tobytes() == right.tobytes()
    if _supported_object(left):
        a, b = attributes(left), attributes(right)
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and math.isnan(left):
        return math.isnan(right)
    return left == right

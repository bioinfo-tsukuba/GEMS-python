"""Public persistence helpers and the existing auto_dataclass decorator."""

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import pickle

from gems_python.common.serialization import decode, encode


def serialize_enum(value):
    return value.name if isinstance(value, Enum) else value


def deserialize_enum(enum_type, value):
    return enum_type[value] if issubclass(enum_type, Enum) else value


def recursive_to_dict(obj):
    return encode(obj)


def recursive_from_dict(cls, data):
    return decode(data, cls)


def recursive_to_json(obj):
    return json.dumps(encode(obj), ensure_ascii=False)


def recursive_from_json(cls, json_str):
    return decode(json.loads(json_str), cls)


class SerializationMixin:
    _gems_serializable = True

    def to_dict(self):
        return recursive_to_dict(self)

    @classmethod
    def from_dict(cls, data):
        return recursive_from_dict(cls, data)

    def to_json(self):
        return recursive_to_json(self)

    @classmethod
    def from_json(cls, json_str):
        return recursive_from_json(cls, json_str)

    def to_pickle(self, path: Path = None):
        encode(self)  # Apply the same supported-value/cycle contract as JSON.
        dumped = pickle.dumps(self)
        if path is not None:
            Path(path).write_bytes(dumped)
        return dumped

    @classmethod
    def from_pickle(cls, path: Path = None, dumped: bytes = None):
        if dumped is None:
            if path is None:
                raise ValueError("Either dumped or path must be specified.")
            dumped = Path(path).read_bytes()
        # Normalize legacy IDs and validate both persistence paths consistently.
        return decode(encode(pickle.loads(dumped)), cls)


def auto_dataclass(cls):
    for name in ("to_dict", "from_dict", "to_json", "from_json", "to_pickle", "from_pickle"):
        setattr(cls, name, SerializationMixin.__dict__[name])
    return dataclass(cls)

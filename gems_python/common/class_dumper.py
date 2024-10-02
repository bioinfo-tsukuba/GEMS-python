from enum import Enum
import json
from dataclasses import dataclass, asdict, fields, is_dataclass
from typing import List, Dict, get_type_hints


# Enumをシリアライズするためのヘルパー
def serialize_enum(value):
    if isinstance(value, Enum):
        return value.name  # Enumの名前を保存
    return value

# Enumをデシリアライズするためのヘルパー
def deserialize_enum(enum_type, value):
    if issubclass(enum_type, Enum):
        return enum_type[value]  # Enumの名前から元のEnumに復元
    return value


# 再帰的に辞書へ変換する関数
def recursive_to_dict(obj):
    if is_dataclass(obj):
        result = {}
        type_hints = get_type_hints(obj.__class__)  # 型ヒントを取得
        for field in fields(obj):
            key = field.name
            value = getattr(obj, key)
            field_type = type_hints.get(key)
            
            if isinstance(value, Enum):  # Enum型の場合、名前を保存
                result[key] = serialize_enum(value)
            elif is_dataclass(value):  # dataclassであれば再帰的に処理
                result[key] = recursive_to_dict(value)
            elif isinstance(value, list):  # リストの場合
                list_type = field_type.__args__[0]  # リストの中身の型を取得
                result[key] = [recursive_to_dict(v) if is_dataclass(v) else serialize_enum(v) for v in value]
            elif isinstance(value, dict):  # 辞書の場合
                dict_value_type = field_type.__args__[1]  # 辞書の値の型を取得
                result[key] = {k: recursive_to_dict(v) if is_dataclass(v) else serialize_enum(v) for k, v in value.items()}
            else:
                result[key] = value  # 単純な型の場合、そのまま格納
        return result
    return obj  # dataclass以外はそのまま返す

# 再帰的に辞書からオブジェクトを生成する関数
def recursive_from_dict(cls, data):
    if is_dataclass(cls):
        init_args = {}
        type_hints = get_type_hints(cls)
        for key, value in data.items():
            field_type = type_hints.get(key)
            if isinstance(field_type, type) and issubclass(field_type, Enum):  # Enum型の処理
                init_args[key] = deserialize_enum(field_type, value)
            elif is_dataclass(field_type):
                init_args[key] = recursive_from_dict(field_type, value)
            elif isinstance(value, list):
                list_type = field_type.__args__[0]
                init_args[key] = [recursive_from_dict(list_type, v) if is_dataclass(list_type) else deserialize_enum(list_type, v) for v in value]
            elif isinstance(value, dict):
                dict_value_type = field_type.__args__[1]
                init_args[key] = {k: recursive_from_dict(dict_value_type, v) if is_dataclass(dict_value_type) else deserialize_enum(dict_value_type, v) for k, v in value.items()}
            else:
                init_args[key] = value
        return cls(**init_args)
    return data

# 辞書をJSONに変換するメソッドも追加
def recursive_to_json(obj):
    return json.dumps(recursive_to_dict(obj))

# JSONから復元するメソッドも追加
def recursive_from_json(cls, json_str):
    data = json.loads(json_str)
    return recursive_from_dict(cls, data)


# カスタムデコレーター
def auto_dataclass(cls):
    # クラスに to_dict メソッドを追加
    def to_dict(self):
        return recursive_to_dict(self)

    # クラスに from_dict クラスメソッドを追加
    @classmethod
    def from_dict(cls, data: dict):
        return recursive_from_dict(cls, data)

    # クラスに to_json メソッドを追加
    def to_json(self):
        return recursive_to_json(self)

    # クラスに from_json クラスメソッドを追加
    @classmethod
    def from_json(cls, json_str: str):
        return recursive_from_json(cls, json_str)

    # メソッドをクラスに追加
    setattr(cls, 'to_dict', to_dict)
    setattr(cls, 'from_dict', from_dict)
    setattr(cls, 'to_json', to_json)
    setattr(cls, 'from_json', from_json)
    
    # 元のdataclassデコレーターを適用してクラスを返す
    return dataclass(cls)
from enum import Enum
import json
from dataclasses import dataclass, asdict, fields, is_dataclass
from pathlib import Path
from typing import List, Dict, get_type_hints
import polars as pl


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
        for field in fields(obj):
            key = field.name
            value = getattr(obj, key)
            if hasattr(value, 'to_dict'):  # to_dictメソッドがあればそれを使う
                result[key] = value.to_dict()
            
            elif isinstance(value, Enum):  # Enum型の場合、名前を保存
                result[key] = serialize_enum(value)
            elif is_dataclass(value):  # dataclassであれば再帰的に処理
                result[key] = recursive_to_dict(value)
            elif isinstance(value, list):  # リストの場合
                result[key] = [recursive_to_dict(v) if is_dataclass(v) else serialize_enum(v) for v in value]
            elif isinstance(value, dict):  # 辞書の場合
                result[key] = {k: recursive_to_dict(v) if is_dataclass(v) else serialize_enum(v) for k, v in value.items()}
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value  # 単純な型の場合、そのまま格納
        return result
    return obj  # dataclass以外はそのまま返す

# 再帰的に辞書からオブジェクトを生成する関数
def recursive_from_dict(cls, data):
    if is_dataclass(cls):
        init_args = {}
        type_hints = get_type_hints(cls)
        dataclass_fields = {f.name: f for f in fields(cls)}
        for key, value in data.items():
            if key not in type_hints or dataclass_fields[key].init is False:
                # 型ヒントに無いか、init=Falseのフィールドは無視
                continue

            field_type = type_hints.get(key)
            
            # field_typeがNoneの場合（型ヒントが指定されていない場合）はそのまま続行
            if field_type is None:
                init_args[key] = value
                continue
            
            # Enum 型の場合の処理
            if isinstance(field_type, type) and issubclass(field_type, Enum):
                init_args[key] = deserialize_enum(field_type, value)
            
            # dataclass の場合の再帰処理
            elif is_dataclass(field_type):
                init_args[key] = recursive_from_dict(field_type, value)
            
            # List の場合の処理
            elif isinstance(value, list):
                # 型ヒントに __args__ があるかを確認
                if hasattr(field_type, '__args__') and len(field_type.__args__) > 0:
                    list_type = field_type.__args__[0]  # リストの要素型を取得
                    init_args[key] = [recursive_from_dict(list_type, v) if is_dataclass(list_type) else v for v in value]
                else:
                    init_args[key] = value  # 型ヒントが無い場合、そのまま使う
            
            # Dict の場合の処理
            elif isinstance(value, dict):
                # 型ヒントに __args__ があり、辞書の値の型が指定されているかを確認
                if hasattr(field_type, '__args__') and len(field_type.__args__) > 1:
                    dict_value_type = field_type.__args__[1]
                    init_args[key] = {k: recursive_from_dict(dict_value_type, v) if is_dataclass(dict_value_type) else v for k, v in value.items()}
                else:
                    init_args[key] = value  # 型ヒントが無い場合、そのまま使う

            # Path の場合の処理
            elif isinstance(value, str) and field_type == Path:
                init_args[key] = Path(value)
            
            # その他の型（int, strなど）の場合
            else:
                init_args[key] = value

        return cls(**init_args)
    return data

# 辞書をJSONに変換するメソッドも追加
def recursive_to_json(obj):
    dic = recursive_to_dict(obj)
    print("WWWWWWWWWWWW")
    print(f"{dic=}")
    print("MMMMMMMMMMMM")
    return json.dumps(dic)

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
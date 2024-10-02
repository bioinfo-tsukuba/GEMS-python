import json
from dataclasses import dataclass, asdict, is_dataclass
from typing import List, Dict, get_type_hints

def recursive_to_dict(obj):
    if is_dataclass(obj):  # dataclassの場合
        result = {}
        for key, value in asdict(obj).items():
            if hasattr(value, 'to_dict'):  # to_dictがある場合
                result[key] = value.to_dict()
            elif isinstance(value, list):  # リストの場合、再帰的に処理
                result[key] = [recursive_to_dict(v) for v in value]
            elif isinstance(value, dict):  # 辞書の場合、再帰的に処理
                result[key] = {k: recursive_to_dict(v) for k, v in value.items()}
            else:
                result[key] = value  # 単純な型の場合、そのまま格納
        return result
    return obj  # dataclass以外はそのまま返す

def recursive_from_dict(cls, data):
    if is_dataclass(cls):  # dataclassの場合
        init_args = {}
        type_hints = get_type_hints(cls)  # クラスの型ヒントを取得
        for key, value in data.items():
            field_type = type_hints.get(key)  # 型ヒントに基づくクラスフィールドの型を取得
            if is_dataclass(field_type):  # dataclassであれば再帰的にfrom_dict
                init_args[key] = recursive_from_dict(field_type, value)
            elif isinstance(value, list) and hasattr(field_type.__args__[0], 'from_dict'):  # Listの場合
                init_args[key] = [recursive_from_dict(field_type.__args__[0], v) for v in value]
            elif isinstance(value, dict) and hasattr(field_type.__args__[1], 'from_dict'):  # Dictの場合
                init_args[key] = {k: recursive_from_dict(field_type.__args__[1], v) for k, v in value.items()}
            else:
                init_args[key] = value  # 単純な型の場合、そのまま格納
        return cls(**init_args)
    return data  # dataclass以外はそのまま返す

# 辞書をJSONに変換するメソッドも追加
def recursive_to_json(obj):
    return json.dumps(recursive_to_dict(obj))

# JSONから復元するメソッドも追加
def recursive_from_json(cls, json_str):
    data = json.loads(json_str)
    return recursive_from_dict(cls, data)
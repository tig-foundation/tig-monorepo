import json
from abc import ABC, abstractclassmethod, abstractmethod
from blake3 import blake3
from datetime import datetime
from dataclasses import dataclass, fields, is_dataclass
from hashlib import md5
from typing import TypeVar, Type, Dict, Any, List, Union, Optional, get_origin, get_args

T = TypeVar('T', bound='DataclassBase')

class FromStr(ABC):
    @abstractclassmethod
    def from_str(cls, s: str):
        raise NotImplementedError
        
    @abstractmethod
    def to_str(self) -> str:
        raise NotImplementedError

@dataclass
class FromDict:
    @classmethod
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        field_types = {f.name: f.type for f in fields(cls)}
        kwargs = {}

        for field in fields(cls):
            value = d.pop(field.name, None)
            field_type = field_types[field.name]
            origin_type = get_origin(field_type)
            
            is_optional = origin_type is Union and type(None) in get_args(field_type)
            
            if value is None:
                if not is_optional:
                    raise ValueError(f"Missing required field: {field.name}")
                kwargs[field.name] = None
                continue

            if is_optional:
                field_type = next(arg for arg in get_args(field_type) if arg is not type(None))

            kwargs[field.name] = cls._process_value(value, field_type)

        return cls(**kwargs)

    @classmethod
    def _process_value(cls, value: Any, field_type: Type) -> Any:
        if hasattr(field_type, 'from_dict') and isinstance(value, dict):
            return field_type.from_dict(value)
        elif hasattr(field_type, 'from_str') and isinstance(value, str):
            return field_type.from_str(value)
        elif get_origin(field_type) in (list, set):
            elem_type = get_args(field_type)[0]
            return get_origin(field_type)(cls._process_value(item, elem_type) for item in value)
        elif get_origin(field_type) is dict:
            key_type, val_type = get_args(field_type)
            return {k: cls._process_value(v, val_type) for k, v in value.items()}
        else:
            return value

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                if hasattr(value, 'to_dict'):
                    d[field.name] = value.to_dict()
                elif hasattr(value, 'to_str'):
                    d[field.name] = value.to_str()
                elif isinstance(value, (list, set)):
                    d[field.name] = [
                        item.to_dict() if hasattr(item, 'to_dict')
                        else item.to_str() if hasattr(item, 'to_str')
                        else item
                        for item in value
                    ]
                elif isinstance(value, dict):
                    d[field.name] = {
                        k: (v.to_dict() if hasattr(v, 'to_dict')
                           else v.to_str() if hasattr(v, 'to_str')
                           else v)
                        for k, v in value.items()
                    }
                elif is_dataclass(value):
                    d[field.name] = asdict(value)
                else:
                    d[field.name] = value
        return d

def now() -> int:
    return int(datetime.now().timestamp() * 1000)

def jsonify(obj: Any) -> str:
    if hasattr(obj, 'to_dict'):
        obj = obj.to_dict()
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))

def u8s_from_str(input: str) -> bytes:
    return blake3(input.encode()).digest()

def u64s_from_str(input: str) -> List[int]:
    u8s = u8s_from_str(input)
    return [
        int.from_bytes(
            u8s[i * 8:(i + 1) * 8],
            byteorder='little',
            signed=False
        )
        for i in range(4)
    ]

def u32_from_str(input_str: str) -> int:
    result = md5(input_str.encode('utf-8')).digest()
    return int.from_bytes(result[-4:], byteorder='little', signed=False)
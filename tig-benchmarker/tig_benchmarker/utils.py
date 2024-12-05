from __future__ import annotations
from abc import ABC, abstractclassmethod, abstractmethod
from blake3 import blake3
from dataclasses import dataclass, fields, is_dataclass, asdict
from typing import TypeVar, Type, Dict, Any, List, Union, Optional, get_origin, get_args
import json
import time
    
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
            
            if value is None:
                if cls._is_optional(field_type):
                    kwargs[field.name] = None
                else:
                    raise ValueError(f"Missing required field: {field.name}")
                continue

            kwargs[field.name] = cls._process_value(value, field_type)

        return cls(**kwargs)

    @classmethod
    def _process_value(cls, value: Any, field_type: Type) -> Any:
        origin_type = get_origin(field_type)
        
        if cls._is_optional(field_type):
            if value is None:
                return None
            non_none_type = next(arg for arg in get_args(field_type) if arg is not type(None))
            return cls._process_value(value, non_none_type)
        
        if hasattr(field_type, 'from_dict') and isinstance(value, dict):
            return field_type.from_dict(value)
        elif hasattr(field_type, 'from_str') and isinstance(value, str):
            return field_type.from_str(value)
        elif origin_type in (list, set, tuple):
            elem_type = get_args(field_type)[0]
            return origin_type(cls._process_value(item, elem_type) for item in value)
        elif origin_type is dict:
            key_type, val_type = get_args(field_type)
            return {cls._process_value(k, key_type): cls._process_value(v, val_type) for k, v in value.items()}
        else:
            return field_type(value)

    @staticmethod
    def _is_optional(field_type: Type) -> bool:
        return get_origin(field_type) is Union and type(None) in get_args(field_type)

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                if hasattr(value, 'to_dict'):
                    d[field.name] = value.to_dict()
                elif hasattr(value, 'to_str'):
                    d[field.name] = value.to_str()
                elif isinstance(value, (list, set, tuple)):
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


class PreciseNumber(FromStr):
    PRECISION = 10**18  # 18 decimal places of precision

    def __init__(self, value: Union[int, float, str, PreciseNumber]):
        if isinstance(value, PreciseNumber):
            self._value = value._value
        elif isinstance(value, int):
            self._value = value * self.PRECISION
        elif isinstance(value, float):
            self._value = int(value * self.PRECISION)
        elif isinstance(value, str):
            self._value = int(value)
        else:
            raise TypeError(f"Unsupported type for PreciseNumber: {type(value)}")

    @classmethod
    def from_str(cls, s: str) -> 'PreciseNumber':
        return cls(s)

    def to_str(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"PreciseNumber({self.to_float()})"

    def to_float(self) -> float:
        return self._value / self.PRECISION

    def __add__(self, other: Union[PreciseNumber, int, float]) -> PreciseNumber:
        if isinstance(other, (int, float)):
            other = PreciseNumber(other)
        return PreciseNumber(self._value + other._value)

    def __sub__(self, other: Union[PreciseNumber, int, float]) -> PreciseNumber:
        if isinstance(other, (int, float)):
            other = PreciseNumber(other)
        return PreciseNumber(self._value - other._value)

    def __mul__(self, other: Union[PreciseNumber, int, float]) -> PreciseNumber:
        if isinstance(other, (int, float)):
            other = PreciseNumber(other)
        return PreciseNumber((self._value * other._value) // self.PRECISION)

    def __truediv__(self, other: Union[PreciseNumber, int, float]) -> PreciseNumber:
        if isinstance(other, (int, float)):
            other = PreciseNumber(other)
        if other._value == 0:
            raise ZeroDivisionError
        return PreciseNumber((self._value * self.PRECISION) // other._value)

    def __floordiv__(self, other: Union[PreciseNumber, int, float]) -> PreciseNumber:
        if isinstance(other, (int, float)):
            other = PreciseNumber(other)
        if other._value == 0:
            raise ZeroDivisionError
        return PreciseNumber((self._value * self.PRECISION // other._value))

    def __eq__(self, other: Union[PreciseNumber, int, float]) -> bool:
        if isinstance(other, (int, float)):
            other = PreciseNumber(other)
        return self._value == other._value

    def __lt__(self, other: Union[PreciseNumber, int, float]) -> bool:
        if isinstance(other, (int, float)):
            other = PreciseNumber(other)
        return self._value < other._value

    def __le__(self, other: Union[PreciseNumber, int, float]) -> bool:
        if isinstance(other, (int, float)):
            other = PreciseNumber(other)
        return self._value <= other._value

    def __gt__(self, other: Union[PreciseNumber, int, float]) -> bool:
        if isinstance(other, (int, float)):
            other = PreciseNumber(other)
        return self._value > other._value

    def __ge__(self, other: Union[PreciseNumber, int, float]) -> bool:
        if isinstance(other, (int, float)):
            other = PreciseNumber(other)
        return self._value >= other._value

    def __radd__(self, other: Union[int, float]) -> PreciseNumber:
        return self + other

    def __rsub__(self, other: Union[int, float]) -> PreciseNumber:
        return PreciseNumber(other) - self

    def __rmul__(self, other: Union[int, float]) -> PreciseNumber:
        return self * other

    def __rtruediv__(self, other: Union[int, float]) -> PreciseNumber:
        return PreciseNumber(other) / self

    def __rfloordiv__(self, other: Union[int, float]) -> PreciseNumber:
        return PreciseNumber(other) // self

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

def now():
    return int(time.time() * 1000)


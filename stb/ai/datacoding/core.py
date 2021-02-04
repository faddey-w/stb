import operator
import numpy as np
from typing import Union, TypeVar, Generic, Iterable, List


T = TypeVar("T")


class Coder(Generic[T]):
    dim: int

    def encode(self, obj: T) -> np.ndarray:
        raise NotImplementedError

    def decode(self, vector: np.ndarray) -> T:
        raise NotImplementedError

    def batch_encode(self, objs: Iterable[T]) -> np.ndarray:
        encode = self.encode
        vectors = [encode(obj) for obj in objs]
        if not vectors:
            return np.zeros([0, self.dim], np.float64)
        return np.stack(vectors)

    def batch_decode(self, matrix: np.array) -> List[T]:
        decode = self.decode
        return [decode(vector) for vector in matrix]


class ScalarValue(Coder[Union[float, int]]):
    dim = 1

    def __init__(self, min_value=None, max_value=None, integer=False):
        self._min_value = min_value
        self._max_value = max_value
        self._integer = integer

    def encode(self, value):
        return np.array([value], np.float64)

    def decode(self, vector):
        value = vector[0].tolist()
        if self._min_value is not None:
            value = max(self._min_value, value)
        if self._max_value is not None:
            value = min(self._max_value, value)
        if self._integer:
            value = int(round(value))
        return value

    def __repr__(self):
        range_part = f"{self._min_value or '--'}..{self._max_value or '--'}"
        type_part = f"{'int' if self._integer else 'float'}"
        return f"scalar[{range_part}, {type_part}]"


class OneHotCoder(Coder[object]):
    def __init__(self, values):
        self._member_values = values
        self.dim = len(self._member_values)

    @classmethod
    def from_enum(cls, enum_class):
        return cls(list(enum_class.__members__.values()))

    def encode(self, obj):
        obj = getattr(obj, "value", obj)
        index = self._member_values.index(obj)
        vector = np.zeros([self.dim], np.float64)
        vector[index] = 1.0
        return vector

    def decode(self, vector):
        index = int(np.argmax(vector))
        return self._member_values[index]

    def __repr__(self):
        return f"onehot{self._member_values!r}"


class FieldCoder(Coder[dict]):
    fields = {}

    def __init__(self, fields=None, getter=operator.getitem, constructor=dict, read_only=()):
        if fields is not None:
            self.fields = fields
        self.getter = getter
        self.constructor = constructor
        self._read_only = set(read_only)

        self._field_dims = [field.dim for name, field in self.fields.items()]
        self._field_index = {name: i for i, name in enumerate(self.fields.keys())}
        self.dim = sum(self._field_dims)

        self._field_offsets = []
        curr_offs = 0
        for dim in self._field_dims:
            self._field_offsets.append(curr_offs)
            curr_offs += dim

    @classmethod
    def from_dataclass(cls, data_class, *, _exclude=(), _dict_mode=False, **fields):
        import dataclasses

        fields_result = {**fields}
        read_only = set(fields_result.keys())
        for dc_field in dataclasses.fields(data_class):
            name = dc_field.name
            if name in fields:
                field = fields[name]
            elif name in _exclude:
                field = None
            elif not dc_field.init:
                field = None
            else:
                type_ = dc_field.type
                if hasattr(type_, "__members__"):
                    field = OneHotCoder.from_enum(type_)
                elif issubclass(type_, (int, float, bool)):
                    field = ScalarValue(integer=issubclass(type_, (bool, int)))
                elif dataclasses.is_dataclass(type_):
                    field = cls.from_dataclass(type_)
                else:
                    raise TypeError(type_)
            if field is not None:
                read_only.discard(name)
                fields_result[name] = field

        if _dict_mode:
            constructor = dict
            getter = operator.getitem
        else:
            write_dummies = {
                dc_field.name
                for dc_field in dataclasses.fields(data_class)
                if dc_field.init and dc_field.name in _exclude
            }

            def constructor(dict_):
                dict_ = {**{name: None for name in write_dummies}, **dict_}
                return data_class(**dict_)

            getter = getattr

        return cls(fields_result, getter, constructor, read_only=read_only)

    def encode(self, obj):
        get_fn = self.getter
        return np.concatenate(
            [field.encode(get_fn(obj, name)) for name, field in self.fields.items()]
        )

    def decode(self, vector):
        result = {
            name: field.decode(vector[offs : offs + dim])
            for (name, field), offs, dim in zip(
                self.fields.items(), self._field_offsets, self._field_dims
            )
            if name not in self._read_only
        }
        if self.constructor is not dict:
            # noinspection PyArgumentList
            result = self.constructor(result)
        return result

    def get_slice(self, field_name):
        i = self._field_index[field_name]
        offset = self._field_offsets[i]
        dim = self._field_dims[i]
        return offset, offset + dim

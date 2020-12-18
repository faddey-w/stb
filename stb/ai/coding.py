import operator
import numpy as np
from dataclasses import dataclass
from typing import Union, TypeVar, Generic, Iterable, List, Dict, Tuple, Sequence
from stb import models, engine
from stb.ai import utils


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
    def from_dataclass(cls, data_class, *, _exclude=(), **fields):
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

        write_dummies = {
            dc_field.name
            for dc_field in dataclasses.fields(data_class)
            if dc_field.init and dc_field.name in _exclude
        }

        def constructor(dict_):
            dict_ = {**{name: None for name in write_dummies}, **dict_}
            return data_class(**dict_)

        return cls(fields_result, getattr, constructor, read_only=read_only)

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


bot_type_coder = FieldCoder(
    {"code": OneHotCoder([bt.code for bt in models.BotType.get_list()])},
    getter=getattr,
    constructor=lambda d: models.BotType.by_code(d["code"]),
)
bot_full_coder = FieldCoder.from_dataclass(
    models.BotModel,
    _exclude=["id"],
    team=OneHotCoder(engine.StbEngine.TEAMS),
    type=bot_type_coder,
    hp_ratio=ScalarValue(0, 1),
    shield_ratio=ScalarValue(0, 1),
    shot_ready=ScalarValue(0, 1, integer=True),
    has_shield=ScalarValue(0, 1, integer=True),
)
bot_visible_coder = FieldCoder.from_dataclass(
    models.BotModel,
    _exclude=["id", *models.BotModel.HIDDEN_FIELDS],
    team=OneHotCoder(engine.StbEngine.TEAMS),
    type=bot_type_coder,
    hp_ratio=ScalarValue(0, 1),
    shield_ratio=ScalarValue(0, 1),
    has_shield=ScalarValue(0, 1, integer=True),
)
bullet_coder = FieldCoder(
    {
        field: ScalarValue() if field != "type" else bot_type_coder
        for field in models.BulletModel.FIELDS
        if field != "origin_id"
    },
    getter=getattr,
    constructor=lambda d: models.BulletModel(origin_id=None, **d),
    read_only=["sin", "cos"],
)
ray_coder = bullet_coder
control_coder = FieldCoder.from_dataclass(
    models.BotControl,
    move=ScalarValue(min_value=-1, max_value=+1, integer=True),
    rotate=ScalarValue(min_value=-1, max_value=+1, integer=True),
    tower_rotate=ScalarValue(min_value=-1, max_value=+1, integer=True),
)


@dataclass
class WorldStateCodes:
    bots: "Union[Dict[object, '[ N BotDim ]'], '[ N BotDim ]']" = None
    bullets: "[ N BulletDim ]" = None
    rays: "[ N RayDim ]" = None
    controls: "[ N ControlDim ]" = None

    @classmethod
    def from_engine(
        cls, engine, with_controls=False, split_teams=False, bot_full_data=True
    ) -> "WorldStateCodes":
        self = cls()
        if split_teams:
            if bot_full_data is True:
                bot_full_data = engine.teams
            self.bots = {
                team: (bot_full_coder if team in bot_full_data else bot_visible_coder).batch_encode(
                    bot for bot in engine.iter_bots() if bot.team == team
                )
                for team in engine.teams
            }
        else:
            coder = bot_full_coder if bot_full_data else bot_visible_coder
            self.bots = coder.batch_encode(engine.iter_bots())

        self.bullets = bullet_coder.batch_encode(engine.iter_bullets())
        self.rays = ray_coder.batch_encode(engine.iter_rays())
        if with_controls:
            self.controls = control_coder.batch_encode(map(engine.get_control, engine.iter_bots()))
        return self

    def decode(self):
        result = {}

        coders = {
            "bots": {bot_full_coder.dim: bot_full_coder, bot_visible_coder.dim: bot_visible_coder},
            "bullets": {bullet_coder.dim: bullet_coder},
            "rays": {ray_coder.dim: ray_coder},
            "controls": {control_coder.dim: control_coder},
        }

        def _decode_and_set(_, attr, key=None):
            data = self[attr, key]
            coder = coders[attr][data.shape[1]]
            objects = coder.batch_decode(data)
            if key is None:
                result[attr] = objects
            else:
                result.setdefault(attr, {})[key] = objects

        self._map(_decode_and_set)

        return result

    @classmethod
    def to_batch(
        cls, states: Sequence["WorldStateCodes"], insert_batch_dim=0
    ) -> Tuple["WorldStateCodes", "WorldStateCodes"]:
        result = cls()
        pad_masks = cls()

        def _collate_and_set(_, attr, key=None):
            result[attr, key], pad_masks[attr, key] = utils.collate_sequences_with_padding(
                [s[attr, key] for s in states], insert_batch_dim=insert_batch_dim
            )

        states[0]._map(_collate_and_set)

        return result, pad_masks

    def get_batch_size(self, batch_axis=0) -> int:
        result = None

        def _get_batch_size(_, attr, key=None):
            nonlocal result
            thing = self[attr, key]
            if result is None and thing.ndim > 2:
                result = thing.shape[batch_axis]

        self._map(_get_batch_size)

        # noinspection PyTypeChecker
        return result

    def unbatch(self, pad_masks=None, batch_axis=0):
        batch_size = self.get_batch_size(batch_axis)
        if batch_size is None:
            raise ValueError(f"{self} is not batched")
        result = [self.__class__() for _ in range(batch_size)]

        def mapfunc(_, attr, key=None):
            data = self[attr, key]
            if pad_masks is not None:
                mask = pad_masks[attr, key]
            else:
                mask = None
            for i in range(batch_size):
                matrix = np.take(data, i, batch_axis)
                if mask is not None:
                    matrix = matrix[np.take(mask == 0, i, batch_axis)]
                result[i][attr, key] = matrix

        self._map(mapfunc)

    def numpy_to_torch(self):
        import torch

        copy = self.__class__()

        def to_torch(_, attr, key=None):
            copy[attr, key] = torch.tensor(self[attr, key])

        self._map(to_torch)

        return copy

    def torch_to_numpy(self):

        copy = self.__class__()

        def to_numpy(_, attr, key=None):
            copy[attr, key] = self[attr, key].numpy()

        self._map(to_numpy)

        return copy

    def _map(self, function):
        if isinstance(self.bots, dict):
            for team in self.bots.keys():
                function(self, "bots", team)
        elif self.bots is not None:
            function(self, "bots")
        if self.bullets is not None:
            function(self, "bullets")
        if self.rays is not None:
            function(self, "rays")
        if self.controls is not None:
            function(self, "controls")

    def __getitem__(self, item):
        if isinstance(item, tuple):
            attr, key = item
        else:
            attr = item
            key = None
        value = getattr(self, attr)
        if key is not None:
            value = value[key]
        return value

    def __setitem__(self, item, value):
        if isinstance(item, tuple):
            attr, key = item
        else:
            attr = item
            key = None
        if key is not None:
            attr_value = getattr(self, attr)
            if attr_value is None:
                attr_value = {}
                setattr(self, attr, attr_value)
            attr_value[key] = value
        else:
            setattr(self, attr, value)

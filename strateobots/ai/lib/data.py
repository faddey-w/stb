import numpy as np
from strateobots.engine import BotType, SHIELD_DURATION_SEC


class Mapper:

    def __init__(self, *fields):
        self._fields = []
        self._names = {}
        self._indices = {}
        for i, field in enumerate(fields):
            self._fields.append(field)
            if field.name is not None:
                self._names[i] = field.name
                self._indices[field.name] = i
        self.vector_length = len(self._fields)

    def map(self, obj):
        v = np.empty((self.vector_length, ), np.float32)
        for i, field in enumerate(self._fields):
            v[i] = field.convert(obj)
        return v

    def restore(self, vector, obj):
        for i, field in enumerate(self._fields):
            field.restore(obj, vector[i])

    def __call__(self, obj):
        return self.map(obj)

    def __getitem__(self, item):
        if isinstance(item, int):
            if item < 0:
                item += self.vector_length
            return self._names[item]
        elif isinstance(item, str):
            return self._indices[item]
        elif isinstance(item, (tuple, list)):
            return item.__class__(self[x] for x in item)
        else:
            raise TypeError(type(item))

    def get(self, vector, field_name):
        idx = self._indices[field_name]
        return vector[idx]


class CombinedMapper:

    def __init__(self, *mappers):
        self.mappers = mappers
        self.vector_length = sum(m.vector_length for m in mappers)

    def map(self, obj_tuple):
        if len(obj_tuple) != len(self.mappers):
            raise ValueError
        outs = [
            mapper.map(obj)
            for obj, mapper in zip(obj_tuple, self.mappers)
        ]
        return np.concatenate(outs, 0)

    def restore(self, vector, obj_tuple):
        if len(obj_tuple) != len(self.mappers):
            raise ValueError
        offs = 0
        for obj, mapper in zip(obj_tuple, self.mappers):
            vlen = mapper.vector_length
            mapper.restore(obj, vector[offs:offs+vlen])
            offs += vlen

    def __call__(self, obj_tuple):
        return self.map(obj_tuple)

    def __getitem__(self, item):
        idx, item = item
        return self.mappers[idx][item]

    def get(self, vector, *field_path):
        idx, *field_path = field_path
        offs = 0
        for mapper in self.mappers[:idx]:
            offs += mapper.vector_length
        mapper = self.mappers[idx]
        return vector[offs:offs+mapper.vector_length]


class MappedVector:

    def __init__(self, vector, mapper):
        self.vector = vector
        self.mapper = mapper

    def __getattr__(self, item):
        idx = self.mapper[item]
        return self.vector[..., idx]


class Field:

    def __init__(self, name, convert=None, restore=None, categorical=None, attr=None):
        self.name = name
        attr = attr or name
        if categorical is None:
            self.convert = convert or self._converter_as_is(attr)
            self.restore = restore or self._restorer_as_is(attr)
        else:
            self.convert = convert or self._converter_categorical(attr, categorical)
            self.restore = restore or self._restorer_categorical(attr, categorical)

    def _converter_as_is(self, name):
        def convert_as_is(obj):
            return getattr(obj, name)
        return convert_as_is

    def _restorer_as_is(self, name):
        def restore_as_is(obj, value):
            setattr(obj, name, value)
        return restore_as_is

    def _converter_categorical(self, name, category):
        def convert_categorical(obj):
            return int(getattr(obj, name) == category)
        return convert_categorical

    def _restorer_categorical(self, name, category):
        def restore_categorical(obj, value):
            if value > 0.5:
                setattr(obj, name, category)
        return restore_categorical


bot2vec = Mapper(
    Field('x'),
    Field('y'),
    Field('vx'),
    Field('vy'),
    Field('is_raider', categorical=BotType.Raider, attr='type'),
    Field('is_heavy', categorical=BotType.Heavy, attr='type'),
    Field('is_sniper', categorical=BotType.Sniper, attr='type'),
    Field('hp_ratio'),
    Field('load'),
    Field('shield', lambda bot: max(0, bot.shield_remaining / SHIELD_DURATION_SEC)),
    Field('is_firing'),
    Field('orientation'),
    Field('tower_orientation'),
)


action2vec = Mapper(
    Field('move_ahead', categorical=+1, attr='move'),
    Field('move_no', categorical=0, attr='move'),
    Field('move_back', categorical=-1, attr='move'),

    Field('rotate_left', categorical=-1, attr='rotate'),
    Field('rotate_no', categorical=0, attr='rotate'),
    Field('rotate_right', categorical=+1, attr='rotate'),

    Field('tower_rotate_left', categorical=-1, attr='tower_rotate'),
    Field('tower_rotate_no', categorical=0, attr='tower_rotate'),
    Field('tower_rotate_right', categorical=+1, attr='tower_rotate'),

    Field('fire'),
    Field('shield'),
)


state2vec = CombinedMapper(bot2vec, bot2vec)

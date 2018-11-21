import random


class _GeneratorBase:

    def __call__(self):
        raise NotImplementedError


class value(_GeneratorBase):

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.range = max_value - min_value

    def __call__(self):
        return self.min_value + random.random() * self.range


class categorical(_GeneratorBase):

    def __init__(self, *categories):
        self.categories = categories

    def __call__(self):
        return random.choice(self.categories)


class structure(_GeneratorBase):

    def __init__(self, struct):
        self.struct = struct

    def __call__(self):
        return self._generate(self.struct)

    def _generate(self, struct):
        if struct is None:
            return None
        elif isinstance(struct, _GeneratorBase):
            return struct()
        elif isinstance(struct, (set, tuple, list, frozenset)):
            typ = struct.__class__
            return typ(self._generate(val) for val in struct)
        elif isinstance(struct, dict):
            typ = struct.__class__
            return typ((self._generate(key), self._generate(val))
                       for key, val in struct.items())
        elif isinstance(struct, str):
            return struct
        else:
            raise TypeError(type(struct))

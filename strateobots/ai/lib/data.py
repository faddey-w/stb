import numpy as np


def _try_int(x):
    try:
        x = int(x)
    except:
        pass
    return x


class Feature:

    dtype = np.float32

    def __init__(self, path, converter=None):
        if isinstance(path, str):
            path = map(_try_int, path.split('.'))
        self.path = tuple(path)
        self.dimension = 1
        self.converter = converter

    def _get_value(self, value, variable_keys):
        for item in self.path:
            if item.startswith('$'):
                item = variable_keys[item[1:]]
            value = value[item]
        return value

    def _process_value(self, value):
        return np.asarray([value], dtype=self.dtype)

    def __call__(self, value, **variable_keys):
        value = self._get_value(value, variable_keys)
        if self.converter:
            value = self.converter
        return self._process_value(value)


class CategoricalFeature(Feature):

    def __init__(self, path, categories, converter=None):
        super().__init__(path, converter)
        self.categories = tuple(categories)
        self.dimension = len(self.categories)

    def _process_value(self, value):
        result = np.zeros([self.dimension], dtype=self.dtype)
        result[self.categories.index(value)] = 1
        return result

    def decode(self, array):
        cat_idx = np.argmax(array, -1)
        return self.categories[cat_idx]


class IntervalFeature(Feature):

    def __init__(self, path, boundaries, converter=None):
        super().__init__(path, converter)
        self.boundaries = tuple(sorted(boundaries))
        self.dimension = len(self.boundaries) + 1

    def _process_value(self, value):
        bin_ = self.dimension - 1
        for i, b in enumerate(self.boundaries):
            if value < b:
                bin_ = i
            else:
                break
        result = np.zeros([self.dimension], dtype=self.dtype)
        result[bin_] = 1
        return result


class FeatureSet:
    def __init__(self, *features):
        self.features = features
        self.dimension = sum(f.dimension for f in features)

    def __call__(self, value, **variable_keys):
        return np.concatenate([f(value, **variable_keys) for f in self.features])


bot_visible_fields = FeatureSet(
    Feature(['x']),
    Feature(['y']),
    Feature(['hp']),
    Feature(['orientation']),
    Feature(['tower_orientation']),
    Feature(['shield']),
    Feature(['has_shield']),
    Feature(['is_firing']),
)
bot_private_fields = FeatureSet(
    Feature(['vx']),
    Feature(['vy']),
    Feature(['load']),
    Feature(['shot_ready']),
    Feature(['shield_warmup']),
)
coordinates_fields = FeatureSet(
    Feature(['x']),
    Feature(['y']),
)
bullet_fields = FeatureSet(
    Feature(['present']),
    Feature(['x']),
    Feature(['y']),
    Feature(['orientation']),
)

ALL_CONTROLS = 'move', 'rotate', 'tower_rotate', 'fire', 'shield'

ctl_move = CategoricalFeature(['move'], [-1, 0, +1])
ctl_rotate = CategoricalFeature(['rotate'], [-1, 0, +1])
ctl_tower_rotate = CategoricalFeature(['tower_rotate'], [-1, 0, +1])
ctl_fire = CategoricalFeature(['fire'], [False, True])
ctl_shield = CategoricalFeature(['shield'], [False, True])

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
            value = self.converter(value)
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


class RangeSensorFeature(Feature):

    def __init__(self, path, min_value, max_value, n_sensors, converter=None):
        super().__init__(path, converter)
        self.min_value = min_value
        self.max_value = max_value
        self.dimension = n_sensors
        self._step = (max_value - min_value) / n_sensors

    def _process_value(self, value):
        result = np.zeros([self.dimension], dtype=self.dtype)
        if value < self.min_value:
            result[0] = 1
        elif value >= self.max_value:
            result[-1] = 1
        else:
            index = (value - self.min_value) / self._step
            lower = int(index)
            lower_prop = index - lower
            higher_prop = 1 - lower_prop
            result[lower] = lower_prop
            result[lower+1] = higher_prop
        return result


class FeatureSet:
    def __init__(self, features):
        self.features = list(features)
        self.dimension = sum(f.dimension for f in self.features)

    def __call__(self, value, **variable_keys):
        if self.features:
            return np.concatenate([f(value, **variable_keys) for f in self.features])
        else:
            return np.array([], dtype=np.float)


ALL_CONTROLS = 'move', 'rotate', 'tower_rotate', 'fire', 'shield'

ctl_move = CategoricalFeature(['move'], [-1, 0, +1])
ctl_rotate = CategoricalFeature(['rotate'], [-1, 0, +1])
ctl_tower_rotate = CategoricalFeature(['tower_rotate'], [-1, 0, +1])
ctl_fire = CategoricalFeature(['fire'], [False, True])
ctl_shield = CategoricalFeature(['shield'], [False, True])


BOT_VISIBLE_FIELDS = 'x', 'y', 'hp', 'orientation', 'tower_orientation', 'shield', 'has_shield', 'is_firing'
BOT_PRIVATE_FIELDS = 'vx', 'vy', 'load', 'shot_ready', 'shield_warmup'
BULLET_FIELDS = 'present', 'x', 'y', 'orientation'

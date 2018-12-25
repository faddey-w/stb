import tensorflow as tf
import numpy as np
from math import pi
from strateobots.ai.lib import data
from strateobots.ai.simple_duel import norm_angle


IMAGE_DIM = 120
POINT_DIM = 1 + 2 * 2  # twice - one for us, one for enemy
# enemy direction, (orientation, tower_orientation) x 2


def _angle_feature(path):
    return data.RangeSensorFeature(path, 0, 2*pi, IMAGE_DIM, norm_angle)


def _scale(scale):
    return lambda x: scale * x


_angle_readings = data.FeatureSet([
    _angle_feature(['bot', 'orientation']),
    _angle_feature(['bot', 'tower_orientation']),
    _angle_feature(['enemy', 'orientation']),
    _angle_feature(['enemy', 'tower_orientation']),
    # _angle_feature(['engineered', 'bot', 'velocity_orientation']),
    # _angle_feature(['engineered', 'enemy', 'velocity_orientation']),
    _angle_feature(['engineered', 'angle_to_enemy']),
])

# _scalar_readings
COORDINATE_SCALE = _scale(0.01)
VELOCITY_SCALE = _scale(0.1)
_range_readings = data.FeatureSet([
    data.Feature(['bot', 'x'], COORDINATE_SCALE),
    data.Feature(['bot', 'y'], COORDINATE_SCALE),
    data.Feature(['bot', 'vx'], VELOCITY_SCALE),
    data.Feature(['bot', 'vy'], VELOCITY_SCALE),
    data.Feature(['enemy', 'x'], COORDINATE_SCALE),
    data.Feature(['enemy', 'y'], COORDINATE_SCALE),
    # data.Feature(['bot_bullet', 'x'], COORDINATE_SCALE),
    # data.Feature(['bot_bullet', 'y'], COORDINATE_SCALE),
    # data.Feature(['enemy_bullet', 'x'], COORDINATE_SCALE),
    # data.Feature(['enemy_bullet', 'y'], COORDINATE_SCALE),
    data.Feature(['engineered', 'distance_to_enemy'], COORDINATE_SCALE),
])

_scalar_readings = data.FeatureSet([
    data.Feature(['bot', 'hp']),
    data.Feature(['bot', 'shield']),
    data.Feature(['bot', 'has_shield']),
    data.Feature(['bot', 'is_firing']),
    data.Feature(['bot', 'load']),
    data.Feature(['bot', 'shot_ready']),
    data.Feature(['bot', 'shield_warmup']),
    data.Feature(['enemy', 'hp']),
    data.Feature(['enemy', 'shield']),
    data.Feature(['enemy', 'has_shield']),
    data.Feature(['enemy', 'is_firing']),
    # data.Feature(['bot_bullet', 'present']),
    # data.Feature(['enemy_bullet', 'present']),
])


def _encode_one(bot, enemy):
    dx = enemy['x'] - bot['x']
    dy = enemy['y'] - bot['y']
    extra = {
        'angle_to_enemy': np.arctan2(dy, dx),
        'distance_to_enemy': np.sqrt(dx * dx + dy * dy),
    }

    # build complete data structure
    all_state = dict(
        bot=bot,
        enemy=enemy,
        engineered=extra
    )

    # do encoding
    angles = _angle_readings(all_state)
    ranges = _range_readings(all_state)
    others = _scalar_readings(all_state)
    return np.concatenate([angles, ranges, others])


class Model:

    state_dimension =_angle_readings.dimension + _range_readings.dimension + _scalar_readings.dimension
    name = 'RadarModel'

    def __init__(self, name=None):
        if name is not None:
            self.name = name

        with tf.variable_scope(self.name):
            pass
        self.var_list = sum([
        ], [])
        self.init_op = tf.variables_initializer(self.var_list)

    @staticmethod
    @data.generator_encoder
    def data_encoder():
        state_vector = None
        while True:
            state = yield state_vector
            state_vector = _encode_one(state[0], state[1])

    def apply(self, state_vector_array):
        sva = state_vector_array
        import pdb; pdb.set_trace()

    class _Apply:

        def __init__(self, state, main, move, rotate, tower_rotate, fire, shield):
            self.state = state
            self.main = main
            self.move = move
            self.rotate = rotate
            self.tower_rotate = tower_rotate
            self.fire = fire
            self.shield = shield
            self.controls = {
                'move': self.move[-1].out,
                'rotate': self.rotate[-1].out,
                'tower_rotate': self.tower_rotate[-1].out,
                'fire': self.fire[-1].out,
                'shield': self.shield[-1].out,
            }


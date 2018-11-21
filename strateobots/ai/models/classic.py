import tensorflow as tf
import numpy as np
from math import pi
from collections import defaultdict

from strateobots.ai.lib import data, nn
from strateobots.ai.simple_duel import norm_angle


def _angle_feature(path):
    return data.RangeSensorFeature(path, 0, 2*pi, 30, norm_angle)


def _scale(scale):
    return lambda x: scale * x


angle_features = data.FeatureSet([
    _angle_feature(['bot', 'orientation']),
    _angle_feature(['bot', 'tower_orientation']),
    _angle_feature(['enemy', 'orientation']),
    _angle_feature(['enemy', 'tower_orientation']),
    _angle_feature(['bot_bullet', 'orientation']),
    _angle_feature(['enemy_bullet', 'orientation']),
    _angle_feature(['engineered', 'angle_to_enemy']),
])

COORDINATE_SCALE = _scale(0.01)
VELOCITY_SCALE = _scale(0.1)
range_features = data.FeatureSet([
    data.Feature(['bot', 'x'], COORDINATE_SCALE),
    data.Feature(['bot', 'y'], COORDINATE_SCALE),
    data.Feature(['bot', 'vx'], VELOCITY_SCALE),
    data.Feature(['bot', 'vy'], VELOCITY_SCALE),
    data.Feature(['enemy', 'x'], COORDINATE_SCALE),
    data.Feature(['enemy', 'y'], COORDINATE_SCALE),
    data.Feature(['bot_bullet', 'x'], COORDINATE_SCALE),
    data.Feature(['bot_bullet', 'y'], COORDINATE_SCALE),
    data.Feature(['enemy_bullet', 'x'], COORDINATE_SCALE),
    data.Feature(['enemy_bullet', 'y'], COORDINATE_SCALE),
    data.Feature(['engineered', 'distance_to_enemy'], COORDINATE_SCALE),
])

other_features = data.FeatureSet([
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
    data.Feature(['bot_bullet', 'present']),
    data.Feature(['enemy_bullet', 'present']),
])
coordinates_fields = data.FeatureSet([
    data.Feature(['x'], COORDINATE_SCALE),
    data.Feature(['y'], COORDINATE_SCALE),
])


class Model:

    _prev_state_dimension = coordinates_fields.dimension
    _current_state_dimension = sum((
        angle_features.dimension,
        range_features.dimension,
        other_features.dimension,
    ))
    state_dimension = _prev_state_dimension + _current_state_dimension
    name = 'ClassicNNModel'

    def __init__(self, name=None):
        if name is not None:
            self.name = name

        with tf.variable_scope(self.name):
            self.move_block = nn.LayerChain(
                nn.Linear.chain_factory(self.state_dimension, 'MoveBlock'),
                (50, tf.nn.relu),
                (data.ctl_move.dimension, tf.identity),
            )
            self.rotate_block = nn.LayerChain(
                nn.Residual.chain_factory(self.state_dimension, 'RotateBlock', allow_skip_transform=True),
                (20, tf.square),
                (20, tf.nn.leaky_relu),
                (20, tf.nn.leaky_relu),
                (20, tf.nn.leaky_relu),
                (20, tf.nn.leaky_relu),
                (20, tf.nn.leaky_relu),
                (20, tf.nn.leaky_relu),
                (20, tf.nn.leaky_relu),
                (20, tf.nn.leaky_relu),
                # (30, tf.nn.relu),
                (data.ctl_rotate.dimension, tf.identity),
            )
            self.tower_rotate_block = nn.LayerChain(
                nn.Linear.chain_factory(self.state_dimension, 'TowerBlock'),
                (30, tf.nn.relu),
                (30, tf.nn.relu),
                (30, tf.nn.relu),
                (30, tf.nn.relu),
                (data.ctl_tower_rotate.dimension, tf.identity),
            )
            self.fire_block = nn.LayerChain(
                nn.Linear.chain_factory(self.state_dimension, 'FireBlock'),
                (10, tf.sigmoid),
                (data.ctl_fire.dimension, tf.identity),
            )
            self.shield_block = nn.LayerChain(
                nn.Linear.chain_factory(self.state_dimension, 'ShieldBlock'),
                (10, tf.sigmoid),
                (data.ctl_shield.dimension, tf.identity),
            )
        self.var_list = sum([
            self.move_block.var_list,
            self.rotate_block.var_list,
            self.tower_rotate_block.var_list,
            self.fire_block.var_list,
            self.shield_block.var_list,
        ], [])
        self.init_op = tf.variables_initializer(self.var_list)

    def encode_prev_state(self, bot, enemy, bot_bullet, enemy_bullet):
        return COORDINATE_SCALE(data.coordinates_fields(enemy))

    def encode_state(self, bot, enemy, bot_bullet, enemy_bullet):
        # prepare data of optional objects
        if bot_bullet is None:
            bot_bullet = defaultdict(float, present=False)
        else:
            bot_bullet = {'present': True, **bot_bullet}
        if enemy_bullet is None:
            enemy_bullet = defaultdict(float, present=False)
        else:
            enemy_bullet = {'present': True, **enemy_bullet}

        # prepare engineered features
        dx = enemy['x'] - bot['x']
        dy = enemy['y'] - bot['y']
        extra = {
            'angle_to_enemy': np.arctan2(dy, dx),
            'distance_to_enemy': np.sqrt(dx*dx + dy*dy),
        }

        # build complete data structure
        all_state = dict(
            bot=bot,
            enemy=enemy,
            bot_bullet=bot_bullet,
            enemy_bullet=enemy_bullet,
            engineered=extra
        )

        # do encoding
        angles = angle_features(all_state)
        ranges = range_features(all_state)
        others = other_features(all_state)
        return np.concatenate([angles, ranges, others])

    def apply(self, state_vector_array):
        move = self.move_block.apply(state_vector_array)
        rotate = self.rotate_block.apply(state_vector_array)
        tower_rotate = self.tower_rotate_block.apply(state_vector_array)
        fire = self.fire_block.apply(state_vector_array)
        shield = self.shield_block.apply(state_vector_array)
        return self._Apply(state_vector_array,
                           move, rotate, tower_rotate, fire, shield)

    class _Apply:

        def __init__(self, state, move, rotate, tower_rotate, fire, shield):
            self.state = state
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

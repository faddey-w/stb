import tensorflow as tf
import numpy as np
from collections import defaultdict
from strateobots.ai.lib import data, model_function
from strateobots.ai.simple_duel import norm_angle
from tensorflow.python.keras.layers import Dense, InputLayer, Input, Lambda
from tensorflow.python.keras.models import Sequential, Model as KerasModel


coord_scale = lambda x: x * 0.001
velocity_scale = lambda x: x * 0.01

angle_base_fields = data.FeatureSet([
    data.Feature(['orientation']),
    data.Feature(['tower_orientation']),
])
bot_visible_fields = data.FeatureSet([
    data.Feature(['x'], coord_scale),
    data.Feature(['y'], coord_scale),
    data.Feature(['hp']),
    data.Feature(['orientation'], norm_angle),
    data.Feature(['orientation'], np.sin),
    data.Feature(['orientation'], np.cos),
    data.Feature(['tower_orientation'], norm_angle),
    data.Feature(['tower_orientation'], np.sin),
    data.Feature(['tower_orientation'], np.cos),
    data.Feature(['shield']),
    data.Feature(['has_shield']),
    data.Feature(['is_firing']),
])
bot_private_fields = data.FeatureSet([
    data.Feature(['vx'], velocity_scale),
    data.Feature(['vy'], velocity_scale),
    data.Feature(['load']),
    data.Feature(['shot_ready']),
    data.Feature(['shield_warmup']),
])
coordinates_fields = data.FeatureSet([
    data.Feature(['x'], coord_scale),
    data.Feature(['y'], coord_scale),
])
bullet_fields = data.FeatureSet([
    data.Feature(['present']),
    data.Feature(['x'], coord_scale),
    data.Feature(['y'], coord_scale),
    data.Feature(['orientation'], norm_angle),
    data.Feature(['orientation'], np.sin),
    data.Feature(['orientation'], np.cos),
])
extra_fields = data.FeatureSet([
    data.Feature(['bot_gun_orientation'], norm_angle),
    data.Feature(['bot_gun_orientation'], np.sin),
    data.Feature(['bot_gun_orientation'], np.cos),
    data.Feature(['enemy_gun_orientation'], norm_angle),
    data.Feature(['enemy_gun_orientation'], np.sin),
    data.Feature(['enemy_gun_orientation'], np.cos),
    data.Feature(['angle_to_enemy'], norm_angle),
    data.Feature(['angle_to_enemy'], np.sin),
    data.Feature(['angle_to_enemy'], np.cos),
    data.Feature(['distance_to_enemy'], coord_scale),
    data.Feature(['recipr_distance_to_enemy']),
])


class Model(model_function.TwoStepDataEncoderMixin):

    _prev_state_dimension = coordinates_fields.dimension
    _current_state_dimension = sum((
        bot_private_fields.dimension,
        2 * bot_visible_fields.dimension,
        2 * bullet_fields.dimension,
        extra_fields.dimension,
    ))
    _real_state_dim = _prev_state_dimension + _current_state_dimension
    state_dimension = angle_base_fields.dimension + _real_state_dim
    name = 'NNModel'

    def __init__(self, name=None):
        if name is not None:
            self.name = name

        self.main_block = Sequential([
            Dense(100, 'relu', input_shape=(self._real_state_dim,)),
            # Dense(20, 'relu'),
        ], name='MainBlock')
        self.spatial_block = Sequential([
            self.main_block,
            Dense(100, tf.nn.leaky_relu),
            Dense(2, tf.atan),
            # Dense(2, tf.identity),
        ], name='SpatialBlock')

        self.output_blocks = dict(
            move=Sequential([
                self.main_block,
                # Dense(20, 'sigmoid'),
                # Dense(10, 'relu'),
                Dense(data.ctl_move.dimension),
            ], name='MainBlock'),
            orientation=IndexModel(0, self.spatial_block, 'MoveAimXBlock'),
            gun_orientation=IndexModel(1, self.spatial_block, 'MoveAimYBlock'),
            action=Sequential([
                self.main_block,
                Dense(25, 'sigmoid'),
                Dense(data.ctl_action.dimension),
            ], name='ActionBlock'),
        )

        var_set = set()
        for seq_m in [self.main_block,
                      self.spatial_block,
                      *self.output_blocks.values()]:
            seq_m.build()
            var_set.update(seq_m.variables)

        self.var_list = list(var_set)
        self.init_op = tf.variables_initializer(self.var_list)

    @property
    def control_set(self):
        return tuple(sorted(self.output_blocks.keys()))

    @staticmethod
    def _encode_prev_state(bot, enemy, bot_bullet, enemy_bullet):
        return coordinates_fields(enemy)

    @staticmethod
    def _encode_state(bot, enemy, bot_bullet, enemy_bullet):
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
        distance_to_enemy = np.sqrt(dx*dx + dy*dy)
        angle_to_enemy = np.arctan2(dy, dx)
        bot_gun_orientation = bot['orientation'] + bot['tower_orientation']
        enemy_gun_orientation = enemy['orientation'] + enemy['tower_orientation']
        extra = {
            'bot_gun_orientation': bot_gun_orientation,
            'enemy_gun_orientation': enemy_gun_orientation,
            'angle_to_enemy': angle_to_enemy,
            'distance_to_enemy': distance_to_enemy,
            'recipr_distance_to_enemy': 1 / coord_scale(distance_to_enemy),
        }

        # do encoding
        angle_bases = angle_base_fields(bot)
        bot_vector = np.concatenate([
             bot_visible_fields(bot),
             bot_private_fields(bot),
        ])
        enemy_vector = bot_visible_fields(enemy)
        bot_bullet_vector = bullet_fields(bot_bullet)
        enemy_bullet_vector = bullet_fields(enemy_bullet)
        extra_vector = extra_fields(extra)

        return np.concatenate([
            angle_bases,
            bot_vector,
            enemy_vector,
            bot_bullet_vector,
            enemy_bullet_vector,
            extra_vector,
        ])

    def apply(self, state_vector_array):
        assert angle_base_fields.dimension == 2

        orientation = state_vector_array[..., 0]
        gun_orientation = state_vector_array[..., 1]
        state_vector_array = state_vector_array[..., angle_base_fields.dimension:]

        inputs = Input(tensor=state_vector_array)
        model = KerasModel(
            inputs=[inputs],
            outputs=[
                self.output_blocks[key](state_vector_array)
                for key in self.control_set
            ],
            name=self.name
        )
        kwargs = {
            key: model.output[i]
            for i, key in enumerate(self.control_set)
        }

        kwargs['orientation'] = orientation + 2.01 * kwargs['orientation']
        kwargs['gun_orientation'] = gun_orientation + 2.01 * kwargs['gun_orientation']

        model.controls = kwargs
        return model


def Index(i, name=None):
    return Lambda(lambda x: x[..., i], output_shape=(), name=name)


def IndexModel(i, input, name=None):
    return Sequential([input, Index(i)], name)

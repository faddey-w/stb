import tensorflow as tf
import numpy as np
from collections import defaultdict
from strateobots.ai.lib import data, nn, model_function
from strateobots.ai.simple_duel import norm_angle


coord_scale = lambda x: x * 0.001
velocity_scale = lambda x: x * 0.01

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


class Model(model_function.VectorGeneratorDataEncoderMixin):

    _prev_state_dimension = coordinates_fields.dimension
    _current_state_dimension = sum((
        bot_private_fields.dimension,
        2 * bot_visible_fields.dimension,
        2 * bullet_fields.dimension,
        extra_fields.dimension,
    ))
    state_dimension = _prev_state_dimension + _current_state_dimension
    name = 'NNModel'

    def __init__(self, name=None):
        if name is not None:
            self.name = name

        internal_repr_size = 50
        with tf.variable_scope(self.name):
            self.main_block = nn.LayerChain(
                nn.Linear.chain_factory(self.state_dimension, 'MainBlock'),
                (20, tf.nn.relu),
                (20, tf.nn.relu),
                (internal_repr_size, tf.nn.relu),
            )
            self.move_block = nn.LayerChain(
                nn.Linear.chain_factory(internal_repr_size, 'MoveBlock'),
                (20, tf.sigmoid),
                (10, tf.nn.relu),
                (data.ctl_move.dimension, tf.identity),
            )
            self.rotate_block = nn.LayerChain(
                nn.Residual.chain_factory(internal_repr_size, 'RotateBlock', allow_skip_transform=True),
                (20, tf.tanh),
                (20, tf.nn.relu),
                (data.ctl_rotate.dimension, tf.identity),
            )
            self.tower_rotate_block = nn.LayerChain(
                nn.Linear.chain_factory(self.state_dimension, 'TowerBlock'),
                (20, tf.tanh),
                (20, tf.nn.relu),
                (data.ctl_tower_rotate.dimension, tf.identity),
                # (data.ctl_tower_rotate.dimension, tf.sin),
            )
            self.fire_block = nn.LayerChain(
                nn.Linear.chain_factory(internal_repr_size, 'FireBlock'),
                (25, tf.sigmoid),
                (data.ctl_fire.dimension, tf.identity),
            )
            self.shield_block = nn.LayerChain(
                nn.Linear.chain_factory(internal_repr_size, 'ShieldBlock'),
                (25, tf.sigmoid),
                (data.ctl_shield.dimension, tf.identity),
            )
        self.var_list = sum([
            self.main_block.var_list,
            self.move_block.var_list,
            self.rotate_block.var_list,
            self.tower_rotate_block.var_list,
            self.fire_block.var_list,
            self.shield_block.var_list,
        ], [])
        self.init_op = tf.variables_initializer(self.var_list)

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
        # tower_rot_solution = np.sin(angle_to_enemy - bot['orientation'] - bot['tower_orientation'])
        extra = {
            'bot_gun_orientation': bot_gun_orientation,
            'enemy_gun_orientation': enemy_gun_orientation,
            'angle_to_enemy': angle_to_enemy,
            'distance_to_enemy': distance_to_enemy,
            'recipr_distance_to_enemy': 1 / coord_scale(distance_to_enemy),
            # 'torotsol': tower_rot_solution,
        }

        # do encoding
        bot_vector = np.concatenate([
             bot_visible_fields(bot),
             bot_private_fields(bot),
        ])
        enemy_vector = bot_visible_fields(enemy)
        bot_bullet_vector = bullet_fields(bot_bullet)
        enemy_bullet_vector = bullet_fields(enemy_bullet)
        extra_vector = extra_fields(extra)

        return np.concatenate([
            bot_vector,
            enemy_vector,
            bot_bullet_vector,
            enemy_bullet_vector,
            extra_vector,
        ])

    def apply(self, state_vector_array):
        main = self.main_block.apply(state_vector_array)
        internal_repr = main[-1].out
        move = self.move_block.apply(internal_repr)
        rotate = self.rotate_block.apply(internal_repr)
        tower_rotate = self.tower_rotate_block.apply(state_vector_array)
        fire = self.fire_block.apply(internal_repr)
        shield = self.shield_block.apply(internal_repr)
        return self._Apply(state_vector_array, main,
                           move, rotate, tower_rotate, fire, shield)

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

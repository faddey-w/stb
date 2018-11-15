import tensorflow as tf
import numpy as np
from collections import defaultdict
from strateobots.ai.lib import data, nn


extra_fields = data.FeatureSet([
    data.Feature(['angle_to_enemy']),
    data.Feature(['distance_to_enemy']),
    data.Feature(['recipr_distance_to_enemy']),
])
COORDINATE_SCALE = 0.01


class Model:

    _prev_state_dimension = data.coordinates_fields.dimension
    _current_state_dimension = sum((
        data.bot_private_fields.dimension,
        2 * data.bot_visible_fields.dimension,
        2 * data.bullet_fields.dimension,
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
                (50, tf.nn.relu),
                (internal_repr_size, tf.nn.relu),
            )
            self.move_block = nn.LayerChain(
                nn.Linear.chain_factory(internal_repr_size, 'MoveBlock'),
                (200, tf.sigmoid),
                (10, tf.nn.relu),
                (data.ctl_move.dimension, tf.identity),
            )
            self.rotate_block = nn.LayerChain(
                nn.Residual.chain_factory(internal_repr_size, 'RotateBlock', allow_skip_transform=True),
                (20, tf.square),
                (20, tf.sin),
                (20, tf.nn.relu),
                (20, tf.nn.relu),
                (20, tf.nn.relu),
                (data.ctl_rotate.dimension, tf.identity),
            )
            self.tower_rotate_block = nn.LayerChain(
                nn.Linear.chain_factory(self.state_dimension, 'TowerBlock'),
                (20, tf.sin),
                (data.ctl_tower_rotate.dimension, tf.identity),
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

    def encode_prev_state(self, bot, enemy, bot_bullet, enemy_bullet):
        return data.coordinates_fields(enemy) * COORDINATE_SCALE

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

        # data normalization
        bot = bot.copy()
        enemy = enemy.copy()
        bot['x'] *= COORDINATE_SCALE
        bot['y'] *= COORDINATE_SCALE
        enemy['x'] *= COORDINATE_SCALE
        enemy['y'] *= COORDINATE_SCALE
        bot_bullet['x'] *= COORDINATE_SCALE
        bot_bullet['y'] *= COORDINATE_SCALE
        enemy_bullet['x'] *= COORDINATE_SCALE
        enemy_bullet['y'] *= COORDINATE_SCALE

        # prepare engineered features
        dx = enemy['x'] - bot['x']
        dy = enemy['y'] - bot['y']
        distance_to_enemy = np.sqrt(dx*dx + dy*dy)
        extra = {
            'angle_to_enemy': np.arctan2(dy, dx),
            'distance_to_enemy': distance_to_enemy,
            'recipr_distance_to_enemy': 1 / distance_to_enemy,
        }

        # do encoding
        bot = np.concatenate([
             data.bot_visible_fields(bot),
             data.bot_private_fields(bot),
        ])
        enemy = data.bot_visible_fields(enemy)
        bot_bullet = data.bullet_fields(bot_bullet)
        enemy_bullet = data.bullet_fields(enemy_bullet)
        extra = extra_fields(extra)

        return np.concatenate([bot, enemy, bot_bullet, enemy_bullet, extra])

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

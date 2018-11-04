import tensorflow as tf
import numpy as np
from collections import defaultdict
from strateobots.ai.lib import data, nn


class Model:

    prev_state_dimension = data.coordinates_fields.dimension
    current_state_dimension = sum((
        data.bot_private_fields.dimension,
        data.bot_visible_fields.dimension,
        2 * data.bullet_fields.dimension,
    ))
    state_dimension = prev_state_dimension + current_state_dimension
    name = 'NNModel'

    def __init__(self, name=None):
        if name is not None:
            self.name = name

        internal_repr_size = 30
        self.main_block = nn.LayerChain(
            nn.Linear.chain_factory(self.state_dimension, 'MainBlock'),
            (50, tf.nn.relu),
            (40, tf.sigmoid),
            (internal_repr_size, tf.sigmoid),
        )
        self.move_block = nn.LayerChain(
            nn.Linear.chain_factory(internal_repr_size, 'MoveBlock'),
            (25, tf.sigmoid),
            (data.ctl_move.dimension, tf.identity),
        )
        self.rotate_block = nn.LayerChain(
            nn.Linear.chain_factory(internal_repr_size, 'RotateBlock'),
            (25, tf.sigmoid),
            (data.ctl_rotate.dimension, tf.identity),
        )
        self.tower_rotate_block = nn.LayerChain(
            nn.Linear.chain_factory(internal_repr_size, 'TowerBlock'),
            (25, tf.sigmoid),
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
        return data.coordinates_fields(enemy)

    def encode_state(self, bot, enemy, bot_bullet, enemy_bullet):
        bot = data.bot_private_fields(bot)
        enemy = data.bot_visible_fields(enemy)

        if bot_bullet is None:
            bot_bullet = defaultdict(float, present=False)
        else:
            bot_bullet = {'present': True, **bot_bullet}
        bot_bullet = data.bullet_fields(bot_bullet)

        if enemy_bullet is None:
            enemy_bullet = defaultdict(float, present=False)
        else:
            enemy_bullet = {'present': True, **enemy_bullet}
        enemy_bullet = data.bullet_fields(enemy_bullet)

        return np.concatenate([bot, enemy, bot_bullet, enemy_bullet])

    def apply(self, state_vector_array):
        main = self.main_block.apply(state_vector_array)
        internal_repr = main[-1].out
        move = self.move_block.apply(internal_repr)
        rotate = self.rotate_block.apply(internal_repr)
        tower_rotate = self.tower_rotate_block.apply(internal_repr)
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

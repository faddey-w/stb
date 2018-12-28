import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np
from collections import namedtuple
from math import pi
from strateobots.ai.lib import data, nn
from strateobots.ai.simple_duel import norm_angle


IMAGE_DIM = 120
POINT_DIM = 1 + 2 * 2
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


class Model:

    state_dimension = _angle_readings.dimension + _range_readings.dimension + _scalar_readings.dimension
    name = 'RadarModel'

    def __init__(self, name=None):
        if name is not None:
            self.name = name

        with tf.variable_scope(self.name):
            self.conv_block = [
                Conv1d('C1', 7, POINT_DIM, 9, activation=tf.nn.leaky_relu),
                Pool1d('MAX', 5),
                Conv1d('C2', 5, 9, 12, activation=tf.nn.leaky_relu),
                Pool1d('MAX', 4),
            ]
            out_im_dim = IMAGE_DIM // 5 // 4
            scalars_dim = _range_readings.dimension + _scalar_readings.dimension
            internal_repr_size = 50
            self.ff_block = nn.LayerChain(
                nn.Linear.chain_factory(
                    out_im_dim * 12 + scalars_dim,
                    'FFBlock'
                ),
                # (50, tf.sigmoid),
                (internal_repr_size, tf.sigmoid),
            )
            self.move_block = nn.LayerChain(
                nn.Linear.chain_factory(internal_repr_size, 'MoveBlock'),
                (20, tf.sigmoid),
                (10, tf.nn.relu),
                (data.ctl_move.dimension, tf.identity),
            )
            self.rotate_block = nn.LayerChain(
                nn.Residual.chain_factory(internal_repr_size, 'RotateBlock',
                                          allow_skip_transform=True),
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
            self.action_block = nn.LayerChain(
                nn.Linear.chain_factory(internal_repr_size, 'ActionBlock'),
                (25, tf.sigmoid),
                (data.ctl_action.dimension, tf.identity),
            )

        self.var_list = nest.flatten([
            [b.var_list for b in self.conv_block],
            self.ff_block.var_list,
            self.move_block.var_list,
            self.rotate_block.var_list,
            self.tower_rotate_block.var_list,
            self.action_block.var_list,
        ])
        self.init_op = tf.variables_initializer(self.var_list)

    @staticmethod
    @data.function_encoder
    def data_encoder(bot, enemy, bot_bullet, enemy_bullet):
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

    def apply(self, state_vector_array):

        angle_readings = state_vector_array[..., :IMAGE_DIM*POINT_DIM]
        angle_readings = tf.reshape(
            angle_readings,
            [tf.shape(angle_readings)[0], POINT_DIM, IMAGE_DIM],
        )
        angle_readings = tf.transpose(angle_readings, [0, 2, 1])
        scalars = state_vector_array[..., IMAGE_DIM*POINT_DIM:]

        scan_nodes = [angle_readings]
        img = angle_readings
        for layer in self.conv_block:
            node = layer.apply(img)
            scan_nodes.append(node)
            img = node.out

        out_img_shape = tf.shape(img)
        out_img_dim = out_img_shape[-2]
        out_img_channels = out_img_shape[-1]
        batch_shape = out_img_shape[:-2]
        flat_img_shape = tf.concat([batch_shape, [out_img_dim * out_img_channels]], 0)
        out_img_flat = tf.reshape(img, flat_img_shape)

        ff_input = tf.concat([scalars, out_img_flat], 1)

        ffmain = self.ff_block.apply(ff_input)
        internal_repr = ffmain[-1].out
        move = self.move_block.apply(internal_repr)
        rotate = self.rotate_block.apply(internal_repr)
        tower_rotate = self.tower_rotate_block.apply(state_vector_array)
        action = self.action_block.apply(internal_repr)

        return self._Apply(state_vector_array, scan_nodes, ffmain,
                           move, rotate, tower_rotate, action)

    class _Apply:

        def __init__(self, state, scan, ffmain, move, rotate, tower_rotate, action):
            self.state = state
            self.scan = scan
            self.ffmain = ffmain
            self.move = move
            self.rotate = rotate
            self.tower_rotate = tower_rotate
            self.action = action
            self.controls = {
                'move': self.move[-1].out,
                'rotate': self.rotate[-1].out,
                'tower_rotate': self.tower_rotate[-1].out,
                'action': self.action[-1].out,
            }


class Conv1d:
    def __init__(self, name, filter_size, in_dim, out_dim, paddind='CIRCULAR', activation=None):
        assert filter_size % 2 == 1
        self.name = name
        self.filter_size = filter_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.paddind = paddind
        self.activation = activation

        with tf.variable_scope(self.name):
            self.kernel = tf.get_variable(
                'Kernel',
                shape=[filter_size, in_dim+1, out_dim],
            )
        self.var_list = [self.kernel]

    def apply(self, x_nwc, activation=None):
        activation = activation or self.activation
        if self.paddind == 'CIRCULAR':
            x_pad, padding = _prepare_circular_pad(x_nwc, self.filter_size)
        else:
            x_pad = x_nwc
            padding = self.paddind
        x_pad = tf.pad(x_pad, [[0, 0], [0, 0], [0, 1]], constant_values=1)

        y_noact = tf.nn.conv1d(x_pad, self.kernel, 1, padding, data_format='NHWC')
        out = activation(y_noact)
        return self._Apply(x_nwc, x_pad, y_noact, out)

    _Apply = namedtuple('Conv1d', 'x x_pad y_noact out')


class Pool1d:

    def __init__(self, kind, size, stride=None, padding='CIRCULAR'):
        self.kind = kind
        self.size = size
        self.stride = stride or size
        self.padding = padding
        self.var_list = []

    def apply(self, x_nwc):
        if self.padding == 'CIRCULAR':
            x_pad, padding = _prepare_circular_pad(x_nwc, self.size)
        else:
            x_pad = x_nwc
            padding = self.padding
        out = tf.nn.pool(x_pad, [self.size], self.kind, padding, strides=[self.stride])
        return self._Apply(x_nwc, x_pad, out)

    _Apply = namedtuple('Pool1d', 'x x_pad out')


def _prepare_circular_pad(x_nwc, window_size):
    left_pad_size = window_size // 2
    right_pad_size = window_size - left_pad_size - 1
    leftpad = x_nwc[:, -left_pad_size:, :]
    rightpad = x_nwc[:, :right_pad_size, :]
    x_pad = tf.concat([leftpad, x_nwc, rightpad], 1)
    padding = 'VALID'
    return x_pad, padding


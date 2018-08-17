import tensorflow as tf
import numpy as np
import contextlib

from strateobots.ai.lib import layers
from strateobots.ai.lib.data import action2vec, state2vec


class BaseActionInferenceModel:

    def __new__(cls, **kwargs):
        self = super().__new__(cls)
        self.construct_params = kwargs
        return self

    def __init__(self, **kwargs):

        self.name = 'AIN'

        with tf.variable_scope(self.name):
            last_dim, units = self._create_layers(**kwargs)
            self.out_layer = layers.Linear('Out', last_dim, action2vec.vector_length)

        self.var_list = []
        for lr in units:
            self.var_list.extend(lr.var_list)
        self.var_list.extend(self.out_layer.var_list)

    def _create_layers(self, **kwargs):
        raise NotImplementedError

    def create_inference(self, inference, state):
        raise NotImplementedError

    def apply(self, state):
        return ActionInference(self, state, self.create_inference)


class ActionInference:
    def __init__(self, model, state, create_inference):
        """
        :type model: ActionInferenceModel
        :param state: [..., state_vector_len]
        """
        state = normalize_state(state)

        self.model = model  # type: BaseActionInferenceModel
        self.state = state  # type: tf.Tensor

        self.features = create_inference(self, state)
        with finite_assert(self.features, model.var_list):
            self.action_evidences = model.out_layer.apply(self.features, tf.identity)
            vec = self.action_evidences.out
            move_prediction = tf.nn.softmax(vec[..., 0:3], -1)
            rotate_prediction = tf.nn.softmax(vec[..., 3:6], -1)
            tower_rotate_prediction = tf.nn.softmax(vec[..., 6:9], -1)
            fire_prediction = tf.sigmoid(vec[..., 9:10])
            shield_prediction = tf.sigmoid(vec[..., 10:11])

            self.action_prediction = combine_predictions(
                move=move_prediction,
                rotate=rotate_prediction,
                tower_rotate=tower_rotate_prediction,
                fire=fire_prediction,
                shield=shield_prediction,
            )


def normalize_state(state_t):
    normalizer = tf.one_hot([
        state2vec[0, 'x'],
        state2vec[0, 'y'],
        state2vec[1, 'x'],
        state2vec[1, 'y'],
        state2vec[2, 'x'],
        state2vec[2, 'y'],
        state2vec[3, 'x'],
        state2vec[3, 'y'],
    ], depth=state2vec.vector_length, on_value=1.0 / 1000, off_value=1.0)
    normalizer = tf.reduce_min(normalizer, 0)
    state_t *= normalizer

    normalizer = tf.one_hot([
        state2vec[0, 'vx'],
        state2vec[0, 'vy'],
        state2vec[1, 'vx'],
        state2vec[1, 'vy'],
    ], depth=state2vec.vector_length, on_value=1.0 / 10, off_value=1.0)
    normalizer = tf.reduce_min(normalizer, 0)
    state_t *= normalizer
    return state_t


@contextlib.contextmanager
def finite_assert(tensor, var_list):
    assert_op = tf.Assert(
        tf.reduce_all(tf.is_finite(tensor)),
        [
            op
            for v in var_list
            for op in [v.name, tf.reduce_all(tf.is_finite(v))]
        ],
    )
    with tf.control_dependencies([assert_op]) as ctx:
        yield ctx


def combine_predictions(*, move, rotate, tower_rotate, fire, shield):
    return tf.concat([move, rotate, tower_rotate, fire, shield], -1)


class BaseActionInferenceModelV2:

    def __new__(cls, **kwargs):
        self = super().__new__(cls)
        self.construct_params = kwargs
        return self

    def __init__(self,
                 move,
                 rotate,
                 tower_rotate,
                 fire,
                 shield,
                 common=None):
        self.name = 'AIN'

        with tf.variable_scope(self.name):
            if hasattr(self, '_create_common_net'):
                self.has_common = True
                self.common_net = self._create_common_net(**(common or {}))
            else:
                self.has_common = False
            self.move_net = self._create_move_net(**move)
            self.rotate_net = self._create_rotate_net(**rotate)
            self.tower_rotate_net = self._create_tower_rotate_net(**tower_rotate)
            self.fire_net = self._create_fire_net(**fire)
            self.shield_net = self._create_shield_net(**shield)

            self.move_last = layers.Linear('move_last', self.move_net.n_features, 3)
            self.rotate_last = layers.Linear('rotate_last', self.rotate_net.n_features, 3)
            self.tower_rotate_last = layers.Linear('tower_rotate_last', self.tower_rotate_net.n_features, 3)
            self.fire_last = layers.Linear('fire_last', self.fire_net.n_features, 2)
            self.shield_last = layers.Linear('shield_last', self.shield_net.n_features, 2)

        self.var_list = [
            *(self.common_net.var_list if self.has_common else []),

            *self.move_net.var_list,
            *self.rotate_net.var_list,
            *self.tower_rotate_net.var_list,
            *self.fire_net.var_list,
            *self.shield_net.var_list,

            *self.move_last.var_list,
            *self.rotate_last.var_list,
            *self.tower_rotate_last.var_list,
            *self.fire_last.var_list,
            *self.shield_last.var_list,
        ]

    def _create_move_net(self, **kwargs):
        raise NotImplementedError

    def _create_rotate_net(self, **kwargs):
        raise NotImplementedError

    def _create_tower_rotate_net(self, **kwargs):
        raise NotImplementedError

    def _create_fire_net(self, **kwargs):
        raise NotImplementedError

    def _create_shield_net(self, **kwargs):
        raise NotImplementedError

    def apply(self, state, with_exploration=False):
        state = normalize_state(state)
        classify = lambda x: tf.nn.softmax(x, -1)
        inference = ActionInferenceV2()
        inference.model = self
        inference.state = state

        if self.has_common:
            inference.common = self.common_net.apply(state)
            state = inference.common

        inference.move = self.move_net.apply(state)
        inference.rotate = self.rotate_net.apply(state)
        inference.tower_rotate = self.tower_rotate_net.apply(state)
        inference.fire = self.fire_net.apply(state)
        inference.shield = self.shield_net.apply(state)

        if with_exploration:
            inference.explore_move = tf.placeholder(tf.float32, [self.move_net.n_features] * 2)
            inference.explore_rotate = tf.placeholder(tf.float32, [self.rotate_net.n_features] * 2)
            inference.explore_tower_rotate = tf.placeholder(tf.float32, [self.tower_rotate_net.n_features] * 2)
            inference.explore_fire = tf.placeholder(tf.float32, [self.fire_net.n_features] * 2)
            inference.explore_shield = tf.placeholder(tf.float32, [self.shield_net.n_features] * 2)

            def apply_explore(control_name):
                control_node = getattr(inference, control_name)
                control_net = getattr(self, control_name + '_net')
                explore_mat = getattr(inference, 'explore_' + control_name)
                explore_mat += tf.eye(control_net.n_features)
                control_node.optimal_features = control_node.features
                control_node.features = layers.batch_matmul(control_node.features, explore_mat)
            apply_explore('move')
            apply_explore('rotate')
            apply_explore('tower_rotate')
            apply_explore('fire')
            apply_explore('shield')

        inference.classify_nodes = dict(
            move=self.move_last.apply(inference.move.features, classify),
            rotate=self.rotate_last.apply(inference.rotate.features, classify),
            tower_rotate=self.tower_rotate_last.apply(inference.tower_rotate.features, classify),
            fire=self.fire_last.apply(inference.fire.features, classify),
            shield=self.shield_last.apply(inference.shield.features, classify),
        )
        action_prediction = combine_predictions(
            move=inference.classify_nodes['move'].out,
            rotate=inference.classify_nodes['rotate'].out,
            tower_rotate=inference.classify_nodes['tower_rotate'].out,
            fire=inference.classify_nodes['fire'].out,
            shield=inference.classify_nodes['shield'].out,
        )
        with finite_assert(action_prediction, self.var_list):
            inference.action_prediction = tf.identity(action_prediction)
        return inference

    def generate_exploration_feed(self, inference, strength):
        return {
            getattr(inference, 'explore_' + name): strength * np.random.standard_normal([
                getattr(self, name + '_net').n_features
            ] * 2)
            for name in ['move', 'rotate', 'tower_rotate', 'shield', 'fire']
        }


class ActionInferenceV2:
    pass

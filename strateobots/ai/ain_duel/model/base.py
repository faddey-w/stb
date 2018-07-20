import tensorflow as tf

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
        state *= normalizer

        normalizer = tf.one_hot([
            state2vec[0, 'vx'],
            state2vec[0, 'vy'],
            state2vec[1, 'vx'],
            state2vec[1, 'vy'],
        ], depth=state2vec.vector_length, on_value=1.0 / 10, off_value=1.0)
        normalizer = tf.reduce_min(normalizer, 0)
        state *= normalizer

        self.model = model  # type: BaseActionInferenceModel
        self.state = state  # type: tf.Tensor

        self.features = create_inference(self, state)
        finite_assert = tf.Assert(
            tf.reduce_all(tf.is_finite(self.features)),
            [tf.reduce_all(tf.is_finite(v)) for v in model.var_list],
        )
        with tf.control_dependencies([finite_assert]):
            self.action_evidences = model.out_layer.apply(self.features, tf.identity)
            vec = self.action_evidences.out
            move_prediction = tf.nn.softmax(vec[..., 0:3], -1)
            rotate_prediction = tf.nn.softmax(vec[..., 3:6], -1)
            tower_rotate_prediction = tf.nn.softmax(vec[..., 6:9], -1)
            fire_prediction = tf.sigmoid(vec[..., 9:10])
            shield_prediction = tf.sigmoid(vec[..., 10:11])

            self.action_prediction = tf.concat([
                move_prediction,
                rotate_prediction,
                tower_rotate_prediction,
                fire_prediction,
                shield_prediction,
            ], -1)

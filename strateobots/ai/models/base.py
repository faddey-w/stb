import tensorflow as tf
import numpy as np
import contextlib
import copy

from strateobots.ai.lib import nn, data


class BaseModel:

    name = "BaseModel"
    n_actions = None  # type: int

    def __new__(cls, **kwargs):
        self = super().__new__(cls)
        self.construct_params = copy.deepcopy(kwargs)
        if "_n_actions" in kwargs:
            self.n_actions = kwargs.pop("_n_actions")
        return self

    def __init__(self, **kwargs):

        with tf.variable_scope(self.name):
            last_dim, units = self._create_layers(**kwargs)
            self.out_layer = layers.Linear("Out", last_dim, self.n_actions)

        self.var_list = []
        for lr in tf.nest.flatten(units):
            self.var_list.extend(lr.var_list)
        self.var_list.extend(self.out_layer.var_list)

    def _create_layers(self, **kwargs):
        raise NotImplementedError

    def create_inference(self, inference, state):
        raise NotImplementedError

    def apply(self, state):
        return ModelComputation(self, state, self.create_inference)


class ModelComputation:
    def __init__(self, model, state, create_inference):
        """
        :type model: BaseModel
        :param state: [..., state_vector_len]
        """
        state = normalize_state(state)

        self.model = model  # type: BaseModel
        self.state = state  # type: tf.Tensor

        self.features = create_inference(self, state)
        with finite_assert(self.features, model.var_list):
            self.action_quality = model.out_layer.apply(self.features, tf.identity)


def normalize_state(state_t):
    normalizer = tf.one_hot(
        [
            state2vec[0, "x"],
            state2vec[0, "y"],
            state2vec[1, "x"],
            state2vec[1, "y"],
            state2vec[2, "x"],
            state2vec[2, "y"],
            state2vec[3, "x"],
            state2vec[3, "y"],
        ],
        depth=state2vec.vector_length,
        on_value=1.0 / 1000,
        off_value=1.0,
    )
    normalizer = tf.reduce_min(normalizer, 0)
    state_t *= normalizer

    normalizer = tf.one_hot(
        [
            state2vec[0, "vx"],
            state2vec[0, "vy"],
            state2vec[1, "vx"],
            state2vec[1, "vy"],
        ],
        depth=state2vec.vector_length,
        on_value=1.0 / 10,
        off_value=1.0,
    )
    normalizer = tf.reduce_min(normalizer, 0)
    state_t *= normalizer
    return state_t


@contextlib.contextmanager
def finite_assert(tensor, var_list):
    assert_op = tf.Assert(
        tf.reduce_all(tf.is_finite(tensor)),
        [op for v in var_list for op in [v.name, tf.reduce_all(tf.is_finite(v))]],
    )
    with tf.control_dependencies([assert_op]) as ctx:
        yield ctx


def combine_predictions(*, move, rotate, tower_rotate, fire, shield):
    return tf.concat([move, rotate, tower_rotate, fire, shield], -1)


class CombinedModel:

    name = "CombinedModel"

    def __new__(cls, **kwargs):
        self = super().__new__(cls)
        self.construct_params = copy.deepcopy(kwargs)
        return self

    def __init__(self, default=None, common=None, **per_action_kwargs):

        self.action_names = tuple(sorted(per_action_kwargs))

        with tf.variable_scope(self.name):
            if hasattr(self, "_create_common_net"):
                self.has_common = True
                self.common_net = self._create_common_net(**(common or {}))
            else:
                self.has_common = False

            default = default or {}
            default_constructor = self._create_default_action_net

            for action_name, kwargs in per_action_kwargs.items():
                if isinstance(kwargs, int):
                    kwargs = {"_n_actions": kwargs}
                kwargs = dict(default, **kwargs)
                attrname = action_name + "_net"
                constructor_method = getattr(
                    self, "_create_" + attrname, default_constructor
                )
                submodel = constructor_method(**kwargs)
                setattr(self, attrname, submodel)

        self.var_list = [
            *(self.common_net.var_list if self.has_common else []),
            *(
                getattr(self, action_name + "_net").var_list
                for action_name in self.action_names
            ),
        ]

    def _create_default_action_net(self, **kwargs):
        raise NotImplementedError

    def apply(self, state):
        return CombinedModelComputation(self, state)


class CombinedModelComputation:
    def __init__(self, model, state):
        """
        :type model: CombinedModel
        :type state: tf.Tensor
        """
        state = normalize_state(state)
        self.model = model
        self.state = state

        if model.has_common:
            self.common = model.common_net.apply(state)
            state = self.common.features

        for action_name in model.action_names:
            setattr(
                self, action_name, getattr(model, action_name + "_net").apply(state)
            )

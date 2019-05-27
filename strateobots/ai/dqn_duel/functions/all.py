import tensorflow as tf

from strateobots.ai.lib.data import state2vec, action2vec
from .util import SelectOneAction


class QualityFunction:
    def __init__(self, move, rotate, tower_rotate, fire, shield):
        self.move = move
        self.rotate = rotate
        self.tower_rotate = tower_rotate
        self.fire = fire
        self.shield = shield
        self.quality = (
            tf.add_n(
                [
                    self.move.get_quality(),
                    self.rotate.get_quality(),
                    self.tower_rotate.get_quality(),
                    self.fire.get_quality(),
                    self.shield.get_quality(),
                ]
            )
            / 5.0
        )

    def get_quality(self):
        return self.quality


class QualityFunctionModelset:

    node_cls = None
    name = None

    def __new__(cls, **kwargs):
        self = super().__new__(cls)
        self.construct_params = kwargs
        return self

    def __init__(self, move, rotate, tower_rotate, fire, shield):
        with tf.variable_scope(self.name):
            with tf.variable_scope("move"):
                self.move = self.node_cls(3, **move)
            with tf.variable_scope("rotate"):
                self.rotate = self.node_cls(3, **rotate)
            with tf.variable_scope("tower_rotate"):
                self.tower_rotate = self.node_cls(3, **tower_rotate)
            with tf.variable_scope("fire"):
                self.fire = self.node_cls(2, **fire)
            with tf.variable_scope("shield"):
                self.shield = self.node_cls(2, **shield)

    @classmethod
    def AllTheSame(cls, **kwargs):
        return cls(
            move=kwargs, rotate=kwargs, tower_rotate=kwargs, fire=kwargs, shield=kwargs
        )

    def apply(self, state, action):
        move_action = action[..., 0:3]
        rotate_action = action[..., 3:6]
        tower_rotate_action = action[..., 6:9]
        fire_action = action[..., 9:11]
        shield_action = action[..., 11:13]
        return QualityFunction(
            self.move.apply(state, move_action),
            self.rotate.apply(state, rotate_action),
            self.tower_rotate.apply(state, tower_rotate_action),
            self.fire.apply(state, fire_action),
            self.shield.apply(state, shield_action),
        )

    @property
    def var_list(self):
        return [
            *self.move.var_list,
            *self.rotate.var_list,
            *self.tower_rotate.var_list,
            *self.fire.var_list,
            *self.shield.var_list,
        ]

    def make_selector(self, state):
        return SelectAction(self, state)

    def make_function(self, session):
        return ModelbasedFunction(self, session)


class SelectAction:
    def __init__(self, qfunc_modelset, state):
        """
        :type qfunc_modelset: QualityFunctionModelset
        :type state:
        """
        self.modelset = qfunc_modelset
        self.state = state
        self.select_move = SelectOneAction(qfunc_modelset.move, state)
        self.select_rotate = SelectOneAction(qfunc_modelset.rotate, state)
        self.select_tower_rotate = SelectOneAction(qfunc_modelset.tower_rotate, state)
        self.select_shield = SelectOneAction(qfunc_modelset.shield, state)
        self.select_fire = SelectOneAction(qfunc_modelset.fire, state)

        self.action = tf.concat(
            [
                self.select_move.action,
                self.select_rotate.action,
                self.select_tower_rotate.action,
                self.select_fire.action,
                self.select_shield.action,
            ],
            -1,
        )
        self.max_q = (
            tf.add_n(
                [
                    self.select_move.max_q,
                    self.select_rotate.max_q,
                    self.select_tower_rotate.max_q,
                    self.select_shield.max_q,
                    self.select_fire.max_q,
                ]
            )
            / 5.0
        )


class ModelbasedFunction:
    def __init__(self, modelset, session):
        self.modelset = modelset
        self.state_ph = tf.placeholder(tf.float32, [1, state2vec.vector_length])
        self.selector = SelectAction(self.modelset, self.state_ph)
        self.state_vector = None
        self.action = None
        self.max_q = None
        self.session = session

    def set_state_vector(self, state_vector):
        self.state_vector = state_vector

    def __call__(self, bot, enemy, control, engine):
        assert self.state_vector is not None
        max_qs, actions = self.session.run(
            [self.selector.max_q, self.selector.action],
            {self.state_ph: [self.state_vector]},
        )
        self.max_q = max_qs[0]
        self.action = actions[0]
        action2vec.restore(self.action, control)

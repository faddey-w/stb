import random

import numpy as np
import tensorflow as tf

from ..lib import layers
from ..lib.data import state2vec, action2vec
from ..lib.util import add_batch_shape


class QualityFunction:

    def __init__(self, move, rotate, tower_rotate, fire, shield):
        self.move = move
        self.rotate = rotate
        self.tower_rotate = tower_rotate
        self.fire = fire
        self.shield = shield
        self.quality = tf.add_n([
            self.move.get_quality(),
            self.rotate.get_quality(),
            self.tower_rotate.get_quality(),
            self.fire.get_quality(),
            self.shield.get_quality(),
        ]) / 5.0

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
            with tf.variable_scope('move'):
                self.move = self.node_cls(3, **move)
            with tf.variable_scope('rotate'):
                self.rotate = self.node_cls(3, **rotate)
            with tf.variable_scope('tower_rotate'):
                self.tower_rotate = self.node_cls(3, **tower_rotate)
            with tf.variable_scope('fire'):
                self.fire = self.node_cls(2, **fire)
            with tf.variable_scope('shield'):
                self.shield = self.node_cls(2, **shield)

    @classmethod
    def AllTheSame(cls, **kwargs):
        return cls(move=kwargs,
                   rotate=kwargs,
                   tower_rotate=kwargs,
                   fire=kwargs,
                   shield=kwargs)

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


class SelectOneAction:

    def __init__(self, qfunc_model, state):
        n_actions = qfunc_model.n_actions
        all_actions = [
            [1.0 if i == j else 0.0 for j in range(n_actions)]
            for i in range(n_actions)
        ]

        batch_shape = layers.shape_to_list(state.shape[:-1])
        batched_all_actions = np.reshape(
            all_actions,
            [n_actions] + [1] * len(batch_shape) + [n_actions]
        ) + np.zeros([n_actions, *batch_shape, n_actions])

        self.all_actions = tf.constant(all_actions, dtype=tf.float32)
        self.batched_all_actions = tf.constant(batched_all_actions, dtype=tf.float32)
        self.state = add_batch_shape(state, [1])

        self.qfunc = qfunc_model.apply(self.state, self.batched_all_actions)
        self.max_idx = tf.argmax(self.qfunc.get_quality(), 0)
        self.max_q = tf.reduce_max(self.qfunc.get_quality(), 0)

        self.action = tf.gather_nd(
            self.all_actions,
            tf.expand_dims(self.max_idx, -1),
        )


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

        self.action = tf.concat([
            self.select_move.action,
            self.select_rotate.action,
            self.select_tower_rotate.action,
            self.select_fire.action,
            self.select_shield.action,
        ], -1)
        self.max_q = tf.add_n([
            self.select_move.max_q,
            self.select_rotate.max_q,
            self.select_tower_rotate.max_q,
            self.select_shield.max_q,
            self.select_fire.max_q,
        ]) / 5.0


class ModelbasedFunction:

    def __init__(self, modelset, session):
        self.modelset = modelset
        self.state_ph = tf.placeholder(tf.float32, [1, state2vec.vector_length])
        self.selector = SelectAction(self.modelset, self.state_ph)
        self._state_vector = None
        self.session = session

    def set_state_vector(self, state_vector):
        self._state_vector = state_vector

    def __call__(self, bot, enemy, control, engine):
        assert self._state_vector is not None
        actions = self.session.run(self.selector.action,
                                   {self.state_ph: [self._state_vector]})
        action2vec.restore(actions[0], control)


class ExplorationFunction:

    def __init__(self, replay_memory):
        self._states_tree = None
        self._actions = None
        self._mem = replay_memory
        self._state_vector = None
        self.update_index()

    def update_index(self):
        states, actions, _ = self._mem.get_last_entries(self._mem.used_size)
        self._actions = actions
        raise NotImplementedError

    def find_action_to_explore(self):
        raise NotImplementedError

    def set_state_vector(self, state_vector):
        self._state_vector = state_vector

    def __call__(self, bot, enemy, control, engine):
        assert self._state_vector is not None
        _, action = self.find_action_to_explore()
        action2vec.restore(action, control)


class ReinforcementLearning:

    def __init__(self, modelset, batch_size=10, reward_prediction=0.97,
                 regularization_weight=None):
        self.modelset = modelset
        self.batch_size = batch_size

        self.state_ph = tf.placeholder(tf.float32, [batch_size, state2vec.vector_length])
        self.next_state_ph = tf.placeholder(tf.float32, [batch_size, state2vec.vector_length])
        self.action_ph = tf.placeholder(tf.float32, [batch_size, action2vec.vector_length])
        self.reward_ph = tf.placeholder(tf.float32, [batch_size])
        self.selector = SelectAction(self.modelset, self.next_state_ph)
        self.qfunc = self.modelset.apply(self.state_ph, self.action_ph)

        # self.optimizer = tf.train.RMSPropOptimizer(0.001)
        self.optimizer = tf.train.AdamOptimizer()
        # self.optimizer = tf.train.GradientDescentOptimizer(0.001)

        self.is_terminal_ph = tf.placeholder(tf.bool, [batch_size])
        self.y_predict_part = tf.where(
            self.is_terminal_ph,
            tf.zeros_like(self.selector.max_q),
            reward_prediction * self.selector.max_q,
        )
        self.y = self.reward_ph + self.y_predict_part
        self.loss_vector = (self.y - self.qfunc.get_quality()) ** 2
        self.loss = tf.reduce_mean(self.loss_vector)

        if regularization_weight is not None:
            regularization_losses = [
                tf.reduce_mean(tf.square(v))
                for v in modelset.var_list
            ]
            self.regularization_loss = tf.add_n(regularization_losses) / len(regularization_losses)
            self.total_loss = self.loss + 0.001 * self.regularization_loss
        else:
            self.regularization_loss = None
            self.total_loss = self.loss

        self.train_step = self.optimizer.minimize(self.total_loss, var_list=modelset.var_list)

        self.init_op = tf.variables_initializer(self.optimizer.variables())

        self.summaries = tf.summary.merge([
            tf.summary.scalar('loss', self.total_loss),
        ])

    def do_train_step(self, session, replay_memory, extra_tensors=(),
                      **sampling_kwargs):
        _, extra_results = self.compute_on_sample(
            session,
            replay_memory,
            [self.train_step, extra_tensors],
            **sampling_kwargs
        )
        return extra_results

    def compute_on_sample(self, session, replay_memory, tensors,
                          n_seq_samples=0, seq_sample_size=0,
                          n_rnd_entries=0, n_last_entries=0):
        total = n_seq_samples * seq_sample_size + n_rnd_entries + n_last_entries
        if total != self.batch_size:
            raise ValueError("incorrect batch size: {}".format(total))

        states_before, actions, states_after = [], [], []

        for _ in range(n_seq_samples):
            st_before, act, st_after = replay_memory.get_random_slice(seq_sample_size)
            states_before.append(st_before)
            actions.append(act)
            states_after.append(st_after)

        if n_rnd_entries > 0:
            st_before, act, st_after = replay_memory.get_random_sample(n_rnd_entries)
            states_before.append(st_before)
            actions.append(act)
            states_after.append(st_after)

        if n_last_entries > 0:
            st_before, act, st_after = replay_memory.get_last_entries(n_last_entries)
            states_before.append(st_before)
            actions.append(act)
            states_after.append(st_after)

        states_before_sample = np.concatenate(states_before, axis=0)
        actions_sample = np.concatenate(actions, axis=0)
        states_after_sample = np.concatenate(states_after, axis=0)
        reward_sample = self.compute_reward(
            states_before_sample,
            actions_sample,
            states_after_sample
        )

        return session.run(tensors, {
            self.state_ph: states_before_sample,
            self.next_state_ph: states_after_sample,
            self.action_ph: actions_sample,
            self.reward_ph: reward_sample,
            self.is_terminal_ph: self.compute_is_terminal(states_after_sample)
        })

    def compute_reward(self, state_before, action, state_after):
        b_hp_idx = state2vec[0, 'hp_ratio']
        e_hp_idx = state2vec[1, 'hp_ratio']
        b_hp_delta = state_after[..., b_hp_idx] - state_before[..., b_hp_idx]
        e_hp_delta = state_after[..., e_hp_idx] - state_before[..., e_hp_idx]
        # activity_punishment = 0.001 * (
        #     action[..., action2vec['tower_rotate_left']]
        #     + action[..., action2vec['tower_rotate_right']]
        # )
        # return 100 * (b_hp_delta - e_hp_delta) - activity_punishment
        return 100 * (b_hp_delta - e_hp_delta)
        # return 10 * (1 - state_after[..., e_hp_idx])
        # e_hp_before = state_before[..., e_hp_idx]
        # e_hp_after = state_after[..., e_hp_idx]
        # return 100 * (e_hp_before - e_hp_after)
        # return 100 * (1 - e_hp_after)
        # reward_flag = e_hp_after + 0.01 < e_hp_before
        # return np.cast[np.float32](reward_flag)

    def compute_is_terminal(self, state):
        b_hp = state[..., state2vec[0, 'hp_ratio']]
        e_hp = state[..., state2vec[1, 'hp_ratio']]
        return (b_hp <= 0) | (e_hp <= 0)
        # return e_hp <= 0


def control_noise(ctl, noise_prob):
    if random.random() < noise_prob:
        ctl.move = random.choice([-1, 0, +1])
    if random.random() < noise_prob:
        ctl.rotate = random.choice([-1, 0, +1])
    if random.random() < noise_prob:
        ctl.tower_rotate = random.choice([-1, 0, +1])
    if random.random() < noise_prob:
        ctl.fire = random.choice([False, True])
    if random.random() < noise_prob:
        ctl.shield = random.choice([False, True])


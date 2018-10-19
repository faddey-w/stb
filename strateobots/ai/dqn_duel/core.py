import random
import logging

import numpy as np
import tensorflow as tf

from strateobots.ai.lib.data import state2vec, action2vec


log = logging.getLogger(__name__)


class ReinforcementLearning:

    def __init__(self, modelset, batch_size=10, reward_prediction=0.97,
                 regularization_weight=None):
        self.modelset = modelset
        self.batch_size = batch_size

        self.state_ph = tf.placeholder(tf.float32, [batch_size, state2vec.vector_length])
        self.next_state_ph = tf.placeholder(tf.float32, [batch_size, state2vec.vector_length])
        self.action_ph = tf.placeholder(tf.float32, [batch_size, action2vec.vector_length])
        self.reward_ph = tf.placeholder(tf.float32, [batch_size])
        self.selector = self.modelset.make_selector(self.next_state_ph)
        self.qfunc = self.modelset.apply(self.state_ph, self.action_ph)

        # self.optimizer = tf.train.RMSPropOptimizer(0.001)
        # self.optimizer = tf.train.AdamOptimizer()
        self.optimizer = tf.train.GradientDescentOptimizer(0.001)

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
        reward_sample = compute_reward_from_vectors(
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

    def compute_is_terminal(self, state):
        b_hp = state[..., state2vec[0, 'hp_ratio']]
        e_hp = state[..., state2vec[1, 'hp_ratio']]
        return (b_hp <= 0) | (e_hp <= 0)
        # return e_hp <= 0


def _compute_reward__core(hp1_before, hp1_after, hp2_before, hp2_after):
    # activity_punishment = 0.001 * (
    #     action[..., action2vec['tower_rotate_left']]
    #     + action[..., action2vec['tower_rotate_right']]
    # )
    # return 100 * (b_hp_delta - e_hp_delta) - activity_punishment

    # return 10 * (1 - state_after[..., e_hp_idx])
    # e_hp_before = state_before[..., e_hp_idx]
    # e_hp_after = state_after[..., e_hp_idx]
    # return 100 * (e_hp_before - e_hp_after)
    # return 100 * (1 - e_hp_after)
    # reward_flag = e_hp_after + 0.01 < e_hp_before
    # return np.cast[np.float32](reward_flag)
    return 100 * ((hp1_after-hp1_before) - (hp2_after-hp2_before))


def compute_reward_from_vectors(state_before, action, state_after):
    b_hp_idx = state2vec[0, 'hp_ratio']
    e_hp_idx = state2vec[1, 'hp_ratio']
    return _compute_reward__core(
        hp1_before=state_before[..., b_hp_idx],
        hp1_after=state_after[..., b_hp_idx],
        hp2_before=state_before[..., e_hp_idx],
        hp2_after=state_after[..., e_hp_idx]
    )


class compute_reward_from_engine:

    def __init__(self, engine):
        self.hp1b = engine.ai1.bot.hp_ratio
        self.hp2b = engine.ai2.bot.hp_ratio

    def get_next(self, engine):
        hp1a = engine.ai1.bot.hp_ratio
        hp2a = engine.ai2.bot.hp_ratio
        rew = _compute_reward__core(self.hp1b, hp1a, self.hp2b, hp2a)
        self.hp1b = hp1a
        self.hp2b = hp2a
        return rew


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


class noised_ai_func:
    def __init__(self, function, noise_prob):
        self.wrapped_function = function
        self.noise_prob = noise_prob

    def __call__(self, bot, enemy, control, engine):
        self.wrapped_function(bot, enemy, control, engine)
        if log.isEnabledFor(logging.DEBUG):
            before = str(control)
            control_noise(control, self.noise_prob)
            after = str(control)
            if before != after:
                log.debug('Noise was added: {} -> {}'.format(before, after))
        else:
            control_noise(control, self.noise_prob)

    def __getattr__(self, item):
        return getattr(self.wrapped_function, item)

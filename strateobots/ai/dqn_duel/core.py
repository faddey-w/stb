import collections
import itertools
import os
import random
import weakref
from math import pi

import numpy as np
import tensorflow as tf

from strateobots.engine import StbEngine, dist_points, BotControl
from ..lib import layers
from ..lib.data import state2vec, action2vec
from ..lib.handcrafted import get_angle


class QualityFunction:

    def __init__(self, model, state, action):
        self.model = model
        self.state = state
        self.action = action

    def get_quality(self):
        raise NotImplementedError


class SelectAction:

    def __init__(self, qfunc_model, qfunc_class, state):
        n_all_actions = all_actions.shape[0]
        batch_shape = layers.shape_to_list(state.shape[:-1])

        batched_all_actions = np.reshape(
            all_actions,
            [n_all_actions] + [1] * len(batch_shape) + [action2vec.vector_length]
        ) + np.zeros([n_all_actions, *batch_shape, action2vec.vector_length])

        self.all_actions = tf.constant(all_actions, dtype=tf.float32)
        self.batched_all_actions = tf.constant(batched_all_actions, dtype=tf.float32)
        self.state = add_batch_shape(state, [1])

        self.qfunc = qfunc_class(qfunc_model, self.state, self.batched_all_actions)
        self.max_idx = tf.argmax(self.qfunc.get_quality(), 0)
        self.max_q = tf.reduce_max(self.qfunc.get_quality(), 0)

        self.action = tf.gather_nd(
            self.all_actions,
            tf.expand_dims(self.max_idx, -1),
        )

    def call(self, feed_dict, session=None):
        session = session or get_session()
        return session.run(self.action, feed_dict=feed_dict)


class ReinforcementLearning:

    def __init__(self, model, qfunc_class, batch_size=10, n_games=10,
                 reward_prediction=0.97, self_play=True, select_random_prob_decrease=0.05,
                 select_random_min_prob=0.1):
        self.model = model
        self.batch_size = batch_size
        self.n_games = n_games
        self.self_play = self_play

        emu_items = 2 if self_play else 1
        self.select_random_prob_decrease = select_random_prob_decrease
        self.select_random_min_prob = select_random_min_prob
        self.emu_state_ph = tf.placeholder(tf.float32, [emu_items, state2vec.vector_length])
        self.emu_selector = SelectAction(self.model, qfunc_class, self.emu_state_ph)

        self.train_state_ph = tf.placeholder(tf.float32, [batch_size, state2vec.vector_length])
        self.train_next_state_ph = tf.placeholder(tf.float32, [batch_size, state2vec.vector_length])
        self.train_action_ph = tf.placeholder(tf.float32, [batch_size, action2vec.vector_length])
        self.train_reward_ph = tf.placeholder(tf.float32, [batch_size])
        self.train_selector = SelectAction(self.model, qfunc_class, self.train_next_state_ph)
        self.train_qfunc = qfunc_class(self.model, self.train_state_ph, self.train_action_ph)

        self.optimizer = tf.train.AdamOptimizer()

        self.is_terminal_ph = tf.placeholder(tf.bool, [batch_size])
        self.y_predict_part = tf.where(
            self.is_terminal_ph,
            tf.zeros_like(self.train_selector.max_q),
            reward_prediction * self.train_selector.max_q,
        )
        self.y = self.train_reward_ph + self.y_predict_part
        self.loss_vector = (self.y - self.train_qfunc.get_quality()) ** 2
        self.loss = tf.reduce_mean(self.loss_vector)

        regularization_losses = [
            tf.reduce_mean(tf.square(v))
            for v in model.var_list
        ]
        self.regularization_loss = tf.add_n(regularization_losses)
        self.total_loss = self.loss  # + 0.001 * self.regularization_loss

        self.train_step = self.optimizer.minimize(self.total_loss, var_list=model.var_list)

        self.init_op = tf.variables_initializer(self.optimizer.variables())

        state = self.emu_state_ph
        b_x = state[0, state2vec[0, 'x']]
        b_y = state[0, state2vec[0, 'y']]
        e_x = state[0, state2vec[1, 'x']]
        e_y = state[0, state2vec[1, 'y']]
        gun_orientation = (
            state[0, state2vec[0, 'orientation']]
            + state[0, state2vec[0, 'tower_orientation']]
            - tf.atan2(e_x - b_x, e_y - b_y)
        )
        self.emu_summaries = tf.summary.merge([
            tf.summary.scalar('hp1', state[0, state2vec[0, 'hp_ratio']]),
            tf.summary.scalar('hp2', state[0, state2vec[1, 'hp_ratio']]),
            tf.summary.scalar('gun', gun_orientation),
            tf.summary.scalar('gun_norm', gun_orientation % (2*pi)),

        ])
        self.train_summaries = tf.summary.merge([
            tf.summary.scalar('loss', self.total_loss),
        ])

    def run(self, replay_memory, frameskip=0, max_ticks=1000, world_size=300,
            log_root_dir=None, *, step_counter, ai1_cls, ai2_cls):
        sess = get_session()
        emu_writer = train_writer = None
        if log_root_dir is not None:
            logdir = os.path.join(log_root_dir, str(step_counter))
            os.makedirs(logdir, exist_ok=False)
            emu_writer = tf.summary.FileWriter(
                os.path.join(logdir, 'emu'),
                sess.graph
            )
            train_writer = tf.summary.FileWriter(
                os.path.join(logdir, 'train'),
                sess.graph
            )

        games = {
            i: StbEngine(
                world_size, world_size,
                ai1_cls, ai2_cls,
                max_ticks,
                wait_after_win=0,
            )
            for i in range(self.n_games)
        }

        stats = collections.defaultdict(float)

        step_base = int(step_counter * max_ticks / (1 + frameskip))

        iteration = 0
        while games:
            iteration += 1
            # summary_step = step_base + iteration
            summary_step = iteration

            # do step of games
            for g, engine in list(games.items()):
                state1_before = state2vec((engine.ai1.bot, engine.ai2.bot))
                state2_before = state2vec((engine.ai2.bot, engine.ai1.bot))

                emu_stats_t = self.get_emu_stats()
                if self.self_play:
                    two_states_before = np.stack([state1_before, state2_before], 0)
                    actions, emu_stats, sumry = sess.run([
                        self.emu_selector.action,
                        emu_stats_t,
                        self.emu_summaries
                    ], {
                        self.emu_state_ph: two_states_before,
                    })
                else:
                    actions, emu_stats, sumry = sess.run([
                        self.emu_selector.action,
                        emu_stats_t,
                        self.emu_summaries,
                    ], {
                        self.emu_state_ph: [state1_before],
                    })
                if emu_writer:
                    emu_writer.add_summary(sumry, summary_step)

                select_random_prob = max(
                    self.select_random_min_prob,
                    1 - self.select_random_prob_decrease * step_counter
                )
                action2vec.restore(actions[0], engine.ai1.ctl)
                control_noise(engine.ai1.ctl, select_random_prob)
                if self.self_play:
                    action2vec.restore(actions[1], engine.ai2.ctl)
                    control_noise(engine.ai2.ctl, select_random_prob)

                # do game ticks
                for _ in range(1 + frameskip):
                    engine.tick()

                action1 = action2vec(engine.ai1.ctl)
                action2 = action2vec(engine.ai2.ctl)
                state1_after = state2vec((engine.ai1.bot, engine.ai2.bot))
                state2_after = state2vec((engine.ai2.bot, engine.ai1.bot))
                reward1 = self.compute_reward(state1_before, action1, state1_after)

                self.add_emu_stats(stats, emu_stats, reward1)
                replay_memory.put_entry(state1_before, action1, state1_after)
                replay_memory.put_entry(state2_before, action2, state2_after)

                if engine.is_finished:
                    games.pop(g)

            # do GD step
            if replay_memory.used_size >= self.batch_size:
                random_part = 1  # self.batch_size // 5
                states_before_1, actions_1, states_after_1 = \
                    replay_memory.get_random_sample(random_part)
                states_before_2, actions_2, states_after_2 = \
                    replay_memory.get_last_entries(self.batch_size - random_part)

                states_before_sample = np.concatenate([states_before_1, states_before_2], axis=0)
                states_after_sample = np.concatenate([states_after_1, states_after_2], axis=0)
                actions_sample = np.concatenate([actions_1, actions_2], axis=0)
                reward_sample = self.compute_reward(
                    states_before_sample,
                    actions_sample,
                    states_after_sample
                )

                _, train_stats, sumry = sess.run([
                    self.train_step,
                    self.get_train_stats(),
                    self.train_summaries
                ], {
                    self.train_state_ph: states_before_sample,
                    self.train_next_state_ph: states_after_sample,
                    self.train_action_ph: actions_sample,
                    self.train_reward_ph: reward_sample,
                    self.is_terminal_ph: self.compute_is_terminal(states_after_sample)
                })
                self.add_train_stats(stats, reward_sample, train_stats)
                if train_writer:
                    train_writer.add_summary(sumry, summary_step)

            # report on games
            if iteration % 11 == 1 and games:
                averages = collections.defaultdict(float)
                for g, engine in games.items():
                    print('#{}: {}/{} t={} hp={:.2f}/{:.2f} {}'.format(
                        g,
                        engine.ai1.bot.type.name[:1],
                        engine.ai2.bot.type.name[:1],
                        engine.nticks,
                        engine.ai1.bot.hp_ratio,
                        engine.ai2.bot.hp_ratio,
                        self.report_on_game(engine, averages, stats)
                    ))
                if self.n_games > 1:
                    for key, val in averages.items():
                        print('{} = {:.3f}'.format(key, val / len(games)))
                    print()

        if emu_writer:
            emu_writer.flush()
        if train_writer:
            train_writer.flush()

    def get_emu_stats(self):
        return self.emu_selector.max_q

    def compute_reward(self, state_before, action, state_after):
        # b_hp_idx = state2vec[0, 'hp_ratio']
        e_hp_idx = state2vec[1, 'hp_ratio']
        # return 10 * (state_after[..., b_hp_idx] - state_after[..., e_hp_idx])
        return 10 * (1 - state_after[..., e_hp_idx])
        # e_hp_before = state_before[..., e_hp_idx]
        # e_hp_after = state_after[..., e_hp_idx]
        # reward_flag = e_hp_after + 0.01 < e_hp_before
        # return np.cast[np.float32](reward_flag)

    def compute_is_terminal(self, state):
        b_hp = state[..., state2vec[0, 'hp_ratio']]
        e_hp = state[..., state2vec[1, 'hp_ratio']]
        return (b_hp <= 0) | (e_hp <= 0)

    def add_emu_stats(self, stat_store, stat_values, reward):
        max_q = stat_values
        stat_store['reward'] += reward
        stat_store['emu_max_q'] += np.sum(max_q) / np.size(max_q)
        stat_store['n_emu'] += 1

    def get_train_stats(self):
        return [
            self.loss,
            self.y,
            self.train_qfunc.get_quality(),
        ]

    def add_train_stats(self, stat_store, reward_sample, stat_values):
        loss, y, t_q = stat_values
        stat_store['y'] += np.sum(y) / np.size(y)
        stat_store['t_q'] += np.sum(t_q) / np.size(t_q)
        stat_store['reward_sample'] += np.sum(reward_sample) / np.size(reward_sample)
        stat_store['n_train'] += 1

    def report_on_game(self, engine, averages, stats):
        d_center = dist_points(
            engine.ai1.bot.x,
            engine.ai1.bot.y,
            engine.world_width / 2,
            engine.world_height / 2,
        )
        d_enemy = dist_points(
            engine.ai1.bot.x,
            engine.ai1.bot.y,
            engine.ai2.bot.x,
            engine.ai2.bot.y,
        )
        gun_ori = engine.ai1.bot.orientation + engine.ai1.bot.tower_orientation
        init_gun_ori = stats.setdefault('init_gun_ori', gun_ori)
        bot_ori = get_angle(engine.ai1.bot, engine.ai2.bot)
        averages['dist_sum'] += d_center
        report = (
            'a={:5.3f}   {:5.3f}   '
            'b1=({:5.1f},{:5.1f})   b2=({:5.1f}, {:5.1f})    '
            'r={:5.3f}    Q={:5.3f}    '
            'r\'={:5.3f}    y={:5.3f}    Q\'={:5.3f}'.format(
                ((gun_ori - bot_ori) / pi + 1) % 2 - 1,
                (gun_ori-init_gun_ori) / pi,
                engine.ai1.bot.x,
                engine.ai1.bot.y,
                engine.ai2.bot.x,
                engine.ai2.bot.y,
                stats['reward'] / max(1, stats['n_emu']),
                stats['emu_max_q'] / max(1, stats['n_emu']),
                stats['reward_sample'] / max(1, stats['n_train']),
                stats['y'] / max(1, stats['n_train']),
                stats['t_q'] / max(1, stats['n_train']),
            ))
        stats.clear()
        stats['init_gun_ori'] = init_gun_ori

        return report


def __generate_all_actions():
    opts = [[-1, 0, +1], [-1, 0, +1], [-1, 0, +1], [0, 1], [0, 1]]
    for mv, rt, trt, fr, sh in itertools.product(*opts):
        ctl = BotControl(mv, rt, trt, fr, sh)
        yield action2vec(ctl)
all_actions = np.array(list(__generate_all_actions()))


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


def shape_to_list(shape):
    if hasattr(shape, 'as_list'):
        return shape.as_list()
    else:
        return list(shape)


def add_batch_shape(x, batch_shape):
    if hasattr(batch_shape, 'as_list'):
        batch_shape = batch_shape.as_list()
    tail_shape = x.get_shape().as_list()
    newshape = [1] * len(batch_shape) + tail_shape
    x = tf.reshape(x, newshape)
    newshape = batch_shape + tail_shape
    return x * tf.ones(newshape)


_global_session_ref = None


def get_session():
    global _global_session_ref
    sess = _global_session_ref() if _global_session_ref else None
    if sess is None:
        sess = tf.Session()
        _global_session_ref = weakref.ref(sess, sess.close)
    return sess


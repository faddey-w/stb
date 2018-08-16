import collections
import itertools
import os
import random
import weakref
from math import pi

import numpy as np
import tensorflow as tf

from strateobots.engine import StbEngine, dist_points, BotControl, BulletModel
from ..lib import layers
from ..lib.data import state2vec, action2vec
from ..lib.handcrafted import get_angle
from .._base import BaseAI


class QualityFunction:

    def __init__(self, move, rotate, tower_rotate,
                 fire, shield):
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
        ])

    def get_quality(self):
        raise NotImplementedError


class QualityFunctionModelset:

    def __init__(self, move, rotate, tower_rotate,
                 fire, shield):
        self.move = move
        self.rotate = rotate
        self.tower_rotate = tower_rotate
        self.fire = fire
        self.shield = shield

    def apply(self, state, action):
        move_action = action[..., 0:3]
        rotate_action = action[..., 3:6]
        tower_rotate_action = action[..., 6:9]
        fire_action = action[..., 9:11]
        shield_action = action[..., 11:13]


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
        ]) / 5


class ReinforcementLearning:

    def __init__(self, modelset, batch_size=10,
                 reward_prediction=0.97, self_play=True):
        self.modelset = modelset
        self.batch_size = batch_size
        self.self_play = self_play

        emu_items = 2 if self_play else 1
        self.emu_state_ph = tf.placeholder(tf.float32, [emu_items, state2vec.vector_length])
        self.emu_selector = SelectAction(self.modelset, self.emu_state_ph)

        self.train_state_ph = tf.placeholder(tf.float32, [batch_size, state2vec.vector_length])
        self.train_next_state_ph = tf.placeholder(tf.float32, [batch_size, state2vec.vector_length])
        self.train_action_ph = tf.placeholder(tf.float32, [batch_size, action2vec.vector_length])
        self.train_reward_ph = tf.placeholder(tf.float32, [batch_size])
        self.train_selector = SelectAction(self.modelset, self.train_next_state_ph)
        self.train_qfunc = self.modelset.apply(self.train_state_ph, self.train_action_ph)

        # self.optimizer = tf.train.RMSPropOptimizer(0.001)
        self.optimizer = tf.train.AdamOptimizer()
        # self.optimizer = tf.train.GradientDescentOptimizer(0.001)

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

    def run(self, replay_memory, frameskip=0, max_ticks=1000, world_size=300, *,
            ai1_cls, ai2_cls, n_games, select_random_prob, do_train=True,
            emu_writer=None, train_writer=None,
            **sampling_kwargs):
        sess = get_session()

        if not self.self_play:
            ai2_cls = ProxyTrainerAI.wrap(ai2_cls)

        games = {
            i: StbEngine(
                world_size, world_size,
                ai1_cls, ai2_cls,
                max_ticks,
                wait_after_win=0,
            )
            for i in range(n_games)
        }

        stats = collections.defaultdict(float)

        iteration = 0
        while games:
            iteration += 1
            summary_step = iteration

            # do step of games
            for g, engine in list(games.items()):
                emu_stats_t = self.get_emu_stats()
                transition1, transition2, (emu_stats, sumry) = self.do_emulate_step(
                    sess, engine, select_random_prob, frameskip,
                    [emu_stats_t, self.emu_summaries],
                )
                if emu_writer:
                    emu_writer.add_summary(sumry, summary_step)

                reward1 = self.compute_reward(*transition1)

                self.add_emu_stats(stats, emu_stats, reward1)
                replay_memory.put_entry(*transition1)
                if self.self_play:
                    replay_memory.put_entry(*transition2)

                if engine.is_finished:
                    games.pop(g)

            # do GD step
            if do_train and replay_memory.used_size >= self.batch_size:
                train_stats, sumry, reward_sample = self.do_train_step(
                    sess, replay_memory,
                    extra_tensors=[
                        self.get_train_stats(),
                        self.train_summaries,
                        self.train_reward_ph,
                    ],
                    **sampling_kwargs
                )
                self.add_train_stats(stats, reward_sample, train_stats)
                if train_writer:
                    train_writer.add_summary(sumry, summary_step)

            # report on games
            if iteration % 11 == 1 and games:
                self.print_games_report(games, stats)

        if emu_writer:
            emu_writer.flush()
        if train_writer:
            train_writer.flush()

    def do_emulate_step(self, sess, engine, select_random_prob, frameskip, extra_tensors=()):
        if not self.self_play and not isinstance(engine.ai2, ProxyTrainerAI):
            raise TypeError("In non-self-play mode second AI must be "
                            "wrapped into ProxyTrainerAI")

        bot1, bot2 = engine.ai1.bot, engine.ai2.bot
        bullet1, bullet2 = find_bullets(engine, [bot1, bot2])
        state1_before = state2vec((bot1, bot2, bullet1, bullet2))
        state2_before = state2vec((bot2, bot1, bullet2, bullet1))

        if self.self_play:
            two_states_before = np.stack([state1_before, state2_before], 0)
            actions, extra_values = sess.run([
                self.emu_selector.action,
                extra_tensors
            ], {
                self.emu_state_ph: two_states_before,
            })
        else:
            actions, extra_values = sess.run([
                self.emu_selector.action,
                extra_tensors
            ], {
                self.emu_state_ph: [state1_before],
            })

        action2vec.restore(actions[0], engine.ai1.ctl)
        control_noise(engine.ai1.ctl, select_random_prob)
        if self.self_play:
            action2vec.restore(actions[1], engine.ai2.ctl)
            control_noise(engine.ai2.ctl, select_random_prob)
        else:
            engine.ai2.update_action()

        # do game ticks
        for _ in range(1 + frameskip):
            engine.tick()

        bullet1, bullet2 = find_bullets(engine, [bot1, bot2])
        action1 = action2vec(engine.ai1.ctl)
        action2 = action2vec(engine.ai2.ctl)
        state1_after = state2vec((bot1, bot2, bullet1, bullet2))
        state2_after = state2vec((bot2, bot1, bullet2, bullet1))

        transition1 = (state1_before, action1, state1_after)
        transition2 = (state2_before, action2, state2_after)
        return transition1, transition2, extra_values

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
            self.train_state_ph: states_before_sample,
            self.train_next_state_ph: states_after_sample,
            self.train_action_ph: actions_sample,
            self.train_reward_ph: reward_sample,
            self.is_terminal_ph: self.compute_is_terminal(states_after_sample)
        })

    def get_emu_stats(self):
        return self.emu_selector.max_q, self.emu_selector.action

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

    def add_emu_stats(self, stat_store, stat_values, reward):
        max_q, action = stat_values
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
        stat_store['loss'] += np.sum(loss) / np.size(loss)
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
            'a={:6.3f}   {:6.3f}   '
            'b1=({:5.1f},{:5.1f})   b2=({:5.1f}, {:5.1f})    '
            'r={:5.3f}    Q={:5.3f}    '
            'loss={:7.5f}'.format(
                ((gun_ori - bot_ori) / pi + 1) % 2 - 1,
                (gun_ori-init_gun_ori) / pi,
                engine.ai1.bot.x,
                engine.ai1.bot.y,
                engine.ai2.bot.x,
                engine.ai2.bot.y,
                stats['reward'] / max(1, stats['n_emu']),
                stats['emu_max_q'] / max(1, stats['n_emu']),
                # stats['reward_sample'] / max(1, stats['n_train']),
                # stats['y'] / max(1, stats['n_train']),
                # stats['t_q'] / max(1, stats['n_train']),
                stats['loss'] / max(1, stats['n_train']),
            ))
        stats.clear()
        stats['init_gun_ori'] = init_gun_ori

        return report

    def print_games_report(self, games_dict, stats_dict):
        averages = collections.defaultdict(float)
        for g, engine in games_dict.items():
            print('#{}: {}/{} t={} hp={:.2f}/{:.2f} {}'.format(
                g,
                engine.ai1.bot.type.name[:1],
                engine.ai2.bot.type.name[:1],
                engine.nticks,
                engine.ai1.bot.hp_ratio,
                engine.ai2.bot.hp_ratio,
                self.report_on_game(engine, averages, stats_dict)
            ))
        if len(games_dict) > 1:
            for key, val in averages.items():
                print('{} = {:.3f}'.format(key, val / len(games_dict)))
            print()


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


class ProxyTrainerAI(BaseAI):

    wrapped_cls = None
    wrapped_ai = None
    noise = None

    @classmethod
    def wrap(cls, ai_cls):
        return cls.parametrize(wrapped_cls=ai_cls)

    def initialize(self):
        self.wrapped_ai = self.wrapped_cls(self.team, self.engine)
        self.wrapped_ai.initialize()

    def update_action(self):
        self.wrapped_ai.tick()

    def tick(self):
        pass

    def __getattr__(self, item):
        return getattr(self.wrapped_ai, item)


def shape_to_list(shape):
    if hasattr(shape, 'as_list'):
        return shape.as_list()
    else:
        return list(shape)


def find_bullets(engine, bots):
    bullets = {
        bullet.origin_id: bullet
        for bullet in engine.iter_bullets()
    }
    return [
        bullets.get(bot.id, BulletModel(None, None, 0, bot.x, bot.y, 0))
        for bot in bots
    ]


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


def reset_session():
    global _global_session_ref
    _global_session_ref = None

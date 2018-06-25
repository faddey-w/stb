import argparse
import itertools
import traceback
import collections
import random
import numpy as np
import tensorflow as tf
import os
import weakref
import logging
from math import pi, sin, cos
from strateobots.engine import BotType, StbEngine, dist_points, BotControl
from ._base import DuelAI
from .simple_duel import RaiderVsSniper, SniperVsRaider
from .lib.data import state2vec, action2vec, bot2vec
from .lib import layers, stable, replay
from .lib.handcrafted import get_angle, StatefulChaotic


tf.reset_default_graph()
log = logging.getLogger(__name__)


class BaseDQNDualAI(DuelAI):

    bot_type = None

    def create_bot(self, teamize_x=True):
        x = random.random()
        if teamize_x:
            x = self.x_to_team_field(x)
        else:
            x *= self.engine.world_width
        bot_type = self.bot_type or random_bot_type()
        return self.engine.add_bot(
            bottype=bot_type,
            team=self.team,
            x=x,
            y=self.engine.world_height * random.random(),
            orientation=random.random() * 2 * pi,
            tower_orientation=random.random() * 2 * pi,
        )


def AI(team, engine):
    if RunAI.Shared.instance is None:
        RunAI.Shared.instance = RunAI.Shared(2 * pi * random.random())
    if team == engine.teams[0] or True:
        return RunAI(team, engine, RunAI.Shared.instance)
    else:
        # return PassiveAI.parametrize(side=+1)(team, engine)
        # return TrainerAI(team, engine)
        return ChaoticAI.parametrize(orientation=RunAI.Shared.instance.orientation)(team, engine)


class RunAI(BaseDQNDualAI):

    class Shared:
        instance = None

        def __init__(self, orientation):
            self.state_ph = tf.placeholder(tf.float32, [1, state2vec.vector_length])
            self.model = QualityFunction.Model()
            self.model.load_vars()
            self.selector = SelectAction(self.model, self.state_ph)
            self.orientation = orientation

    def __init__(self, team, engine, ai_shared):
        super(RunAI, self).__init__(team, engine)
        self.shared = ai_shared  # type: RunAI.Shared

    def initialize(self):
        # self.create_bot()
        self.orientation = self.shared.orientation
        PassiveAI.initialize(self)

    def tick(self):
        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return
        st = state2vec([bot, enemy])
        action = self.shared.selector.call({self.shared.state_ph: [st]})
        action2vec.restore(action[0], ctl)
        self.ctl.move = 0
        self.ctl.shield = False


class TrainerRaiderAI(RaiderVsSniper):

    def initialize(self):
        self.bot_type = random_bot_type()
        super(TrainerRaiderAI, self).initialize()
        _randomize_position(self.bot, self.engine)


class TrainerSniperAI(SniperVsRaider):

    def initialize(self):
        self.bot_type = random_bot_type()
        super(TrainerSniperAI, self).initialize()
        _randomize_position(self.bot, self.engine)


class ChaoticAI(BaseDQNDualAI):

    algo = None  # type: StatefulChaotic

    def initialize(self):
        PassiveAI.initialize(self)
        self.algo = StatefulChaotic(self.bot, self.ctl, self.engine,
                                    shield_period=(1000, 0))

    def tick(self):
        self.algo.run()
        self.ctl.move = 0
        self.ctl.shield = 0


def TrainerAI(team, engine):
    if random.random() > 0.5:
        log.info('initializing short-range-attack trainer')
        return TrainerRaiderAI(team, engine)
    else:
        log.info('initializing distant-attack trainer')
        return TrainerSniperAI(team, engine)


def _randomize_position(bot, engine):
    bot.x = random.random() * engine.world_width
    bot.y = random.random() * engine.world_height
    bot.orientation = random.random() * 2 * pi - pi
    bot.tower_orientation = random.random() * 2 * pi - pi


class PassiveAI(BaseDQNDualAI):

    action = None
    bot = None
    ctl = None
    orientation = 0

    def initialize(self):
        bot = self.create_bot(teamize_x=False)
        _randomize_position(bot, self.engine)
        self.bot = bot
        self.ctl = self.engine.get_control(self.bot)

        ori = self.orientation
        if self.team != self.engine.teams[0]:
            ori += pi
        bot.x = (0.5 + cos(ori) * 0.05) * self.engine.world_width
        bot.y = (0.5 + sin(ori) * 0.05) * self.engine.world_height

    def tick(self):
        self.ctl.move = 0
        self.ctl.shield = False


class QualityFunction:
    
    class Model:
        _cnt = 0
        save_path = os.path.join(os.path.dirname(__file__), 'dqn_duel_modeldata/')

        def __init__(self, levels=None):
            if levels is None:
                levels = DEFAULT_QFUNC_MODEL
            self.levels_cfg = tuple(levels)
            self.__class__._cnt += 1
            self.name = 'Q{}'.format(self._cnt)
            self.var_list = []
            self.layers = []
            with tf.variable_scope(self.name):
                # l0x = layers.Linear.Model('L0x', 4, self.levels_cfg[0])
                # l0y = layers.Linear.Model('L0y', 4, self.levels_cfg[0], l0x.weight)
                # l0a = layers.Linear.Model('L0a', 29, self.levels_cfg[0])
                l0x = layers.Linear.Model('L0x', 6, self.levels_cfg[0])
                l0y = layers.Linear.Model('L0y', 6, self.levels_cfg[0], l0x.weight)
                l0a = layers.Linear.Model('L0a', 8, self.levels_cfg[0])
                self.layers.append((l0x, l0y, l0a))
                d_in = levels[0]
                for i, d_out in enumerate(self.levels_cfg[1:], 1):
                    lx = layers.Linear.Model('L{}x'.format(i), d_in, d_out)
                    ly = layers.Linear.Model('L{}y'.format(i), d_in, d_out, lx.weight)
                    la = layers.Linear.Model('L{}a'.format(i), 2 * d_in, d_out)
                    self.layers.append((lx, ly, la))
                    d_in = d_out

                for lx, ly, la in self.layers:
                    self.var_list.extend([*lx.var_list, *ly.var_list, *la.var_list])

                self.qw = tf.get_variable('QW', [d_in, action2vec.vector_length-4])
                self.var_list.append(self.qw)

            self.initializer = tf.variables_initializer(self.var_list)
            self.saver = tf.train.Saver(
                self.var_list,
                pad_step_number=True,
                save_relative_paths=True,
            )
            self.step_counter = 0

        def init_vars(self, session=None):
            session = session or get_session()
            session.run(self.initializer)
            session.run(self.qw + 1)

        def save_vars(self, session=None):
            session = session or get_session()
            self.step_counter += 1
            self.saver.save(session, self.save_path, self.step_counter)
            with open(self.save_path + 'step', 'w') as f:
                f.write(str(self.step_counter))
            log.info('saved model "%s" at step=%s to %s',
                     self.name, self.step_counter, self.save_path)

        def load_vars(self, session=None):
            session = session or get_session()
            with open(self.save_path + 'step') as f:
                self.step_counter = int(f.read().strip())
            ckpt = tf.train.get_checkpoint_state(self.save_path)
            log.info('loading model "%s" from %s at step=%s',
                     self.name, ckpt.model_checkpoint_path, self.step_counter)
            self.saver.restore(session, ckpt.model_checkpoint_path)

    def __init__(self, model, state, action):
        """
        :param model: QualityFunction.Model
        :param state: [..., aug_state_vector_len]
        :param action: [..., action_vector_len]
        ellipsis shapes should be the same
        """
        self.model = model  # type: QualityFunction.Model
        self.state = state  # type: tf.Tensor
        self.action = action[..., 3:-1]  # type: tf.Tensor

        bvl = bot2vec.vector_length
        # self.angles0 = tf.concat([state[..., bvl-2:bvl], state[..., -2:]], -1)  # 4
        # self.cos0 = tf.concat([state[..., 4:bvl-2], state[..., bvl+4:-2], action], -1)  # 25
        # self.x0 = tf.concat([state[..., :4:2], state[..., bvl:bvl+4:2]], -1)  # 4
        # self.y0 = tf.concat([state[..., 1:4:2], state[..., bvl+1:bvl+4:2]], -1)  # 4
        # self.a0 = tf.concat([self.angles0, tf.acos(self.cos0)], -1)  # 29
        self.angles0 = tf.concat([state[..., bvl-2:bvl], state[..., -2:]], -1)  # 4
        self.cos0 = tf.concat([state[..., 7:9], state[..., bvl+7:bvl+9]], -1)  # 4
        self.sin0 = tf.sqrt(1 - tf.square(self.cos0))
        self.x0 = tf.concat([state[..., :1], state[..., bvl:bvl+1], self.cos0], -1)  # 6
        self.y0 = tf.concat([state[..., 1:2], state[..., bvl+1:bvl+2], self.sin0], -1)  # 6
        self.a0 = tf.concat([self.angles0, tf.acos(self.cos0)], -1)  # 8

        def make_activation(dim):
            def activation(vec):
                half = dim // 2
                vec1 = tf.nn.relu(vec[..., :half])
                vec2 = tf.identity(vec[..., half:])
                return tf.concat([vec1, vec2], -1)
            return activation

        self.levels = []
        vectors = (self.x0, self.y0, self.a0)
        for i, (mx, my, ma) in enumerate(self.model.layers):
            x, y, a = vectors
            lx = layers.Linear(mx.name, x, mx, make_activation(mx.out_dim))
            ly = layers.Linear(my.name, y, my, make_activation(my.out_dim))
            la = layers.Linear(ma.name, a, ma, make_activation(ma.out_dim))
            a_cos = tf.cos(la.out)
            a_sin = tf.sin(la.out)
            new_x = lx.out * a_cos - ly.out * a_sin
            new_y = lx.out * a_sin + ly.out * a_cos
            # add_a = tf.atan2(
            #     lx.out + tf.stop_gradient(10 * tf.sign(lx.out)),
            #     ly.out,# + tf.stop_gradient(10 * tf.sign(ly.out))
            # )
            # add_a = tf.acos(0.98 * lx.out / norm(tf.stack([lx.out, ly.out], -1))) * tf.sign(ly.out)
            add_a = stable.atan2(ly.out, lx.out)
            new_a = tf.concat([la.out, add_a], -1)
            self.levels.append((lx, ly, la, (new_x, new_y)))
            vectors = (new_x, new_y, new_a)

        self.features = layers.batch_matmul(vectors[0], self.model.qw, )
        finite_assert = tf.Assert(
            tf.reduce_all(tf.is_finite(self.features)),
            [tf.reduce_all(tf.is_finite(v)) for v in model.var_list],
        )
        masked_features = self.action * self.features
        with tf.control_dependencies([finite_assert]):
            self.quality = tf.reduce_mean(masked_features, axis=-1)

    def call(self, state, action, session=None):
        session = session or get_session()
        return session.run(self.quality, feed_dict={
            self.state: state,
            self.action: action,
        })


class SelectAction:

    def __init__(self, qfunc_model, state):
        n_all_actions = all_actions.shape[0]
        batch_shape = layers.shape_to_list(state.shape[:-1])

        batched_all_actions = np.reshape(
            all_actions,
            [n_all_actions] + [1] * len(batch_shape) + [action2vec.vector_length]
        ) + np.zeros([n_all_actions, *batch_shape, action2vec.vector_length])

        self.all_actions = tf.constant(all_actions, dtype=tf.float32)
        self.batched_all_actions = tf.constant(batched_all_actions, dtype=tf.float32)
        self.state = add_batch_shape(state, [n_all_actions])

        self.qfunc = QualityFunction(qfunc_model, self.state, self.batched_all_actions)
        self.max_idx = tf.argmax(self.qfunc.quality, 0)
        self.max_q = tf.reduce_max(self.qfunc.quality, 0)

        self.action = tf.gather_nd(
            self.all_actions,
            tf.expand_dims(self.max_idx, -1),
        )

    def call(self, feed_dict, session=None):
        session = session or get_session()
        return session.run(self.action, feed_dict=feed_dict)


class ReinforcementLearning:

    def __init__(self, model, batch_size=10, n_games=10,
                 reward_prediction=0.97, self_play=True, select_random_prob_decrease=0.05,
                 select_random_min_prob=0.1):
        self.model = model  # type: QualityFunction.Model
        self.batch_size = batch_size
        self.n_games = n_games
        self.self_play = self_play

        emu_items = 2 if self_play else 1
        self.select_random_prob_decrease = select_random_prob_decrease
        self.select_random_min_prob = select_random_min_prob
        self.emu_state_ph = tf.placeholder(tf.float32, [emu_items, state2vec.vector_length])
        self.emu_selector = SelectAction(self.model, self.emu_state_ph)

        self.train_state_ph = tf.placeholder(tf.float32, [batch_size, state2vec.vector_length])
        self.train_next_state_ph = tf.placeholder(tf.float32, [batch_size, state2vec.vector_length])
        self.train_action_ph = tf.placeholder(tf.float32, [batch_size, action2vec.vector_length])
        self.train_reward_ph = tf.placeholder(tf.float32, [batch_size])
        self.train_selector = SelectAction(self.model, self.train_next_state_ph)
        self.train_qfunc = QualityFunction(self.model, self.train_state_ph, self.train_action_ph)

        self.optimizer = tf.train.AdamOptimizer()

        self.is_terminal_ph = tf.placeholder(tf.bool, [batch_size])
        self.y_predict_part = tf.where(
            self.is_terminal_ph,
            tf.zeros_like(self.train_selector.max_q),
            reward_prediction * self.train_selector.max_q,
        )
        self.y = self.train_reward_ph + self.y_predict_part
        self.loss_vector = (self.y - self.train_qfunc.quality) ** 2
        self.loss = tf.reduce_mean(self.loss_vector)

        regularization_losses = [
            tf.reduce_mean(tf.square(v))
            for v in model.var_list
        ]
        self.regularization_loss = tf.add_n(regularization_losses)
        self.total_loss = self.loss  # + 0.001 * self.regularization_loss

        self.train_step = self.optimizer.minimize(self.total_loss, var_list=model.var_list)

        self.init_op = tf.variables_initializer(self.optimizer.variables())

    def run(self, replay_memory, frameskip=0, max_ticks=1000, world_size=300):
        sess = get_session()

        init_orientation = random.random() * 2 * pi
        ai1_cls = PassiveAI.parametrize(
            orientation=init_orientation,
            bot_type=BotType.Heavy,
        )
        ai2_cls = ai1_cls
        # ai2_cls = ChaoticAI.parametrize(
        #     orientation=init_orientation,
        #     bot_type=BotType.Sniper,
        # )
        # ai2_cls = ai1_cls if self.self_play else TrainerAI

        games = {
            i: StbEngine(
                world_size, world_size,
                ai1_cls, ai2_cls,
                max_ticks, wait_after_win=0)
            for i in range(self.n_games)
        }

        stats = collections.defaultdict(float)

        iteration = 0
        while games:
            iteration += 1

            # do step of games
            for g, engine in list(games.items()):
                state1_before = state2vec((engine.ai1.bot, engine.ai2.bot))
                state2_before = state2vec((engine.ai2.bot, engine.ai1.bot))

                emu_stats_t = self.get_emu_stats()
                if self.self_play:
                    two_states_before = np.stack([state1_before, state2_before], 0)
                    actions, emu_stats = sess.run([
                        self.emu_selector.action,
                        emu_stats_t
                    ], {
                        self.emu_state_ph: two_states_before,
                    })
                else:
                    # try:
                    actions, emu_stats = sess.run([
                        self.emu_selector.action,
                        emu_stats_t
                    ], {
                        self.emu_state_ph: [state1_before],
                    })
                    # except:
                    #     import pdb; pdb.set_trace()

                select_random_prob = max(
                    self.select_random_min_prob,
                    1 - self.select_random_prob_decrease * self.model.step_counter
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
                random_part = self.batch_size // 5
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

                _, train_stats = sess.run([
                    self.train_step,
                    self.get_train_stats()
                ], {
                    self.train_state_ph: states_before_sample,
                    self.train_next_state_ph: states_after_sample,
                    self.train_action_ph: actions_sample,
                    self.train_reward_ph: reward_sample,
                    self.is_terminal_ph: self.compute_is_terminal(states_after_sample)
                })
                self.add_train_stats(stats, reward_sample, train_stats)

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

    def get_emu_stats(self):
        return self.emu_selector.max_q

    def compute_reward(self, state_before, action, state_after):
        # import pdb; pdb.set_trace()
        b_hp_idx = state2vec[0, 'hp_ratio']
        e_hp_idx = state2vec[1, 'hp_ratio']
        return 10 * (state_after[..., b_hp_idx] - state_after[..., e_hp_idx])

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
            self.train_qfunc.quality,
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


def random_bot_type():
    return random.choice([
        BotType.Raider,
        BotType.Heavy,
        BotType.Sniper,
    ])


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


DEFAULT_QFUNC_MODEL = (10, 8, 8, 8)


def main():
    this_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'play'], default='train')
    parser.add_argument('--no-save', action='store_false', dest='save')
    parser.add_argument('--replay-dir', default=os.path.join(this_dir, '_replay_memory'))
    opts = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(opts.replay_dir):
        os.mkdir(opts.replay_dir)

    model = QualityFunction.Model()
    sess = get_session()
    try:
        model.load_vars()
    except:
        model.init_vars()
    replay_memory = replay.ReplayMemory(
        capacity=2000,
        state_size=state2vec.vector_length,
        action_size=action2vec.vector_length
    )
    rl = ReinforcementLearning(
        model,
        batch_size=80,
        n_games=1,
        reward_prediction=0.95,
        select_random_prob_decrease=0.01,
        select_random_min_prob=0.1,
        self_play=True,
    )
    sess.run(rl.init_op)
    try:
        replay_memory.load(opts.replay_dir)
        log.info('replay memory buffer loaded from %s', opts.replay_dir)
    except FileNotFoundError:
        log.info('collecting new replay memory buffer')

    if opts.action == 'play':
        import code
        with sess.as_default():
            code.interact(local=dict(globals(), **locals()))
    if opts.action == 'train':
        try:
            while True:
                rl.run(
                    frameskip=2,
                    max_ticks=2000,
                    world_size=1000,
                    replay_memory=replay_memory,
                )
                if opts.save:
                    model.save_vars()
                    replay_memory.save(opts.replay_dir)
        except KeyboardInterrupt:
            if opts.save:
                model.save_vars()
                replay_memory.save(opts.replay_dir)


if __name__ == '__main__':
    main()

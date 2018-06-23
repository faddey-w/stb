import argparse
import itertools
import functools
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
from .lib import layers


tf.reset_default_graph()
log = logging.getLogger(__name__)


class BaseDQNDualAI(DuelAI):

    def create_bot(self, teamize_x=True):
        x = random.random()
        if teamize_x:
            x = self.x_to_team_field(x)
        else:
            x *= self.engine.world_width
        return self.engine.add_bot(
            bottype=random_bot_type(),
            team=self.team,
            x=x,
            y=self.engine.world_height * random.random(),
            orientation=random.random() * 2 * pi,
            tower_orientation=random.random() * 2 * pi,
        )


def AI(team, engine):
    if team == engine.teams[0]:
        if RunAI.Shared.instance is None:
            RunAI.Shared.instance = RunAI.Shared()
        return RunAI(team, engine, RunAI.Shared.instance)
    else:
        return PassiveAI(team, engine)
        # return TrainerAI(team, engine)


class RunAI(BaseDQNDualAI):

    class Shared:
        instance = None

        def __init__(self):
            self.state_ph = tf.placeholder(tf.float32, [1, state2vec.vector_length])
            self.model = QualityFunction.Model()
            self.model.load_vars()
            self.selector = SelectAction(self.model, self.state_ph)

    def __init__(self, team, engine, ai_shared):
        super(RunAI, self).__init__(team, engine)
        self.shared = ai_shared  # type: RunAI.Shared

    def initialize(self):
        self.create_bot()

    def tick(self):
        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return
        st = state2vec([bot, enemy])
        action = self.shared.selector.call({self.shared.state_ph: [st]})
        action2vec.restore(action[0], ctl)


class TrainableAI(BaseDQNDualAI):

    action = None
    bot = None
    ctl = None

    def initialize(self):
        self.bot = self.create_bot(teamize_x=False)
        self.ctl = self.engine.get_control(self.bot)

    def tick(self):
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
    bot.orientation = random.random() * 2 * pi
    bot.tower_orientation = random.random() * 2 * pi


class PassiveAI(BaseDQNDualAI):

    def initialize(self):
        bot = self.create_bot()
        _randomize_position(bot, self.engine)
        self.bot = bot

    def tick(self):
        pass


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
                l0x = layers.Linear.Model('L0x', 4, self.levels_cfg[0])
                l0y = layers.Linear.Model('L0y', 4, self.levels_cfg[0], l0x.weight)
                l0a = layers.Linear.Model('L0a', 29, self.levels_cfg[0])
                self.layers.append((l0x, l0y, l0a))
                d_in = levels[0]
                for i, d_out in enumerate(self.levels_cfg[1:], 1):
                    lx = layers.Linear.Model('L{}x'.format(i), d_in, d_out)
                    ly = layers.Linear.Model('L{}y'.format(i), d_in, d_out, lx.weight)
                    la = layers.Linear.Model('L{}a'.format(i), d_in, d_out)
                    self.layers.append((lx, ly, la))
                    d_in = d_out

                for lx, ly, la in self.layers:
                    self.var_list.extend([*lx.var_list, *ly.var_list, *la.var_list])

                self.qw = tf.get_variable('QW', [d_in, 1])
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
        self.action = action  # type: tf.Tensor

        bvl = bot2vec.vector_length
        self.angles0 = tf.concat([state[..., bvl-2:bvl], state[..., -2:]], -1)  # 4
        self.cos0 = tf.concat([state[..., 4:bvl-2], state[..., bvl+4:-2], action], -1)  # 25
        self.x0 = tf.concat([state[..., :4:2], state[..., bvl:bvl+4:2]], -1)  # 4
        self.y0 = tf.concat([state[..., 1:4:2], state[..., bvl+1:bvl+4:2]], -1)  # 4
        self.a0 = tf.concat([self.angles0, tf.acos(self.cos0)], -1)  # 29

        self.levels = []
        vectors = (self.x0, self.y0, self.a0)
        for i, (mx, my, ma) in enumerate(self.model.layers):
            x, y, a = vectors
            lx = layers.Linear(mx.name, x, mx, tf.nn.relu)
            ly = layers.Linear(my.name, y, my, tf.nn.relu)
            la = layers.Linear(ma.name, a, ma, tf.nn.relu)
            a_cos = tf.cos(la.out)
            a_sin = tf.sin(la.out)
            new_x = lx.out * a_cos - ly.out * a_sin
            new_y = lx.out * a_sin + ly.out * a_cos
            self.levels.append((lx, ly, la, (new_x, new_y)))
            vectors = (new_x, new_y, la.out)

        quality = layers.batch_matmul(vectors[0], self.model.qw, )
        finite_assert = tf.Assert(
            tf.reduce_all(tf.is_finite(quality)),
            [tf.reduce_all(tf.is_finite(v)) for v in model.var_list],
        )
        with tf.control_dependencies([finite_assert]):
            self.quality = tf.squeeze(quality, [-1])

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

    def __init__(self, model, batch_size=10, n_games=10, memory_cap=200,
                 reward_decay=0.03, self_play=True, select_random_prob_decrease=0.05):
        self.memory_cap = memory_cap
        self.replay_memory = None

        self.model = model  # type: QualityFunction.Model
        self.batch_size = batch_size
        self.n_games = n_games
        self.self_play = self_play

        emu_items = 2 if self_play else 1
        self.select_random_prob_decrease = select_random_prob_decrease
        self.emu_state_ph = tf.placeholder(tf.float32, [emu_items, state2vec.vector_length])
        self.emu_selector = SelectAction(self.model, self.emu_state_ph)

        self.train_state_ph = tf.placeholder(tf.float32, [batch_size, state2vec.vector_length])
        self.train_next_state_ph = tf.placeholder(tf.float32, [batch_size, state2vec.vector_length])
        self.train_action_ph = tf.placeholder(tf.float32, [batch_size, action2vec.vector_length])
        self.train_reward_ph = tf.placeholder(tf.float32, [batch_size])
        self.train_selector = SelectAction(self.model, self.train_next_state_ph)
        self.train_qfunc = QualityFunction(self.model, self.train_state_ph, self.train_action_ph)

        self.optimizer = tf.train.AdamOptimizer()

        self.y = self.train_reward_ph + (1 - reward_decay) * self.train_selector.max_q
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

    def run(self, frameskip=0, max_ticks=1000, world_size=300):
        sess = get_session()
        ai2_cls = TrainableAI if self.self_play else PassiveAI
        # ai2_cls = TrainableAI if self.self_play else TrainerAI
        games = {
            i: StbEngine(
                world_size, world_size,
                TrainableAI, ai2_cls,
                max_ticks)
            for i in range(self.n_games)
        }
        memcap = self.memory_cap
        self.init_replay_memory()
        replay_state, replay_action, replay_reward, replay_indices = self.replay_memory
        replay_idx = {i: i for i in range(self.n_games)}
        last_replay_idx = {}

        stats = collections.defaultdict(float)

        for engine in games.values():
            side = random.choice([-1, +1])
            engine.ai1.bot.x = (0.5 + side * 0.05) * engine.world_width
            engine.ai2.bot.x = (0.5 - side * 0.05) * engine.world_width
            engine.ai1.bot.y = 0.50 * engine.world_height
            engine.ai2.bot.y = 0.50 * engine.world_height

        iteration = 0
        while games:
            iteration += 1

            # do step of games
            for g, engine in games.items():
                idx = replay_idx[g]
                replay_state[idx, 0, 0] = state2vec((engine.ai1.bot, engine.ai2.bot))
                replay_state[idx, 0, 1] = state2vec((engine.ai2.bot, engine.ai1.bot))

                emu_stats_t = self.get_emu_stats()
                if self.self_play:
                    actions, emu_stats = sess.run([
                        self.emu_selector.action,
                        emu_stats_t
                    ], {
                        self.emu_state_ph: replay_state[idx, 0],
                    })
                else:
                    actions, emu_stats = sess.run([
                        self.emu_selector.action,
                        emu_stats_t
                    ], {
                        self.emu_state_ph: replay_state[idx, 0, :1],
                    })
                replay_action[idx] = actions[0]

                select_random_prob = max(
                    0.1,
                    1 - self.select_random_prob_decrease * self.model.step_counter
                )
                action2vec.restore(actions[0], engine.ai1.ctl)
                control_noise(engine.ai1.ctl, select_random_prob)
                if self.self_play:
                    action2vec.restore(actions[1], engine.ai2.ctl)
                    control_noise(engine.ai2.ctl, select_random_prob)
                for _ in range(1 + frameskip):
                    engine.tick()

                reward = engine.ai1.bot.hp_ratio - engine.ai2.bot.hp_ratio
                reward *= 10  # make reward scale larger to stabilize learning
                replay_reward[idx] = reward

                self.add_emu_stats(stats, emu_stats, reward)

                replay_state[idx, 1, 0] = state2vec((engine.ai1.bot, engine.ai2.bot))
                replay_state[idx, 1, 1] = state2vec((engine.ai2.bot, engine.ai1.bot))
                replay_indices.add(idx)

                if engine.is_finished:
                    games.pop(g)
                    replay_idx.pop(g)
                    break

                next_idx = idx+1
                if len(replay_indices) < memcap:
                    while next_idx in replay_indices or next_idx in replay_idx.values():
                        next_idx += 1
                    next_idx %= memcap
                else:
                    for _ in range(memcap):
                        next_idx += 1
                        next_idx %= memcap
                        if next_idx not in replay_idx.values():
                            break
                    else:
                        next_idx = idx

                last_replay_idx[g] = replay_idx[g]
                replay_idx[g] = next_idx

            # do GD step
            if len(replay_indices) >= self.batch_size:
                replay_sample = tuple(random.sample(
                    replay_indices - set(replay_idx.values()),
                    self.batch_size-len(last_replay_idx),
                )) + tuple(last_replay_idx.values())
                bots_sample = replay_state[replay_sample, ]
                action_sample = replay_action[replay_sample, ]
                reward_sample = replay_reward[replay_sample, ]
                _, train_stats = sess.run([
                    self.train_step,
                    self.get_train_stats()
                ], {
                    self.train_state_ph: bots_sample[:, 0, 0],
                    self.train_next_state_ph: bots_sample[:, 1, 0],
                    self.train_action_ph: action_sample,
                    self.train_reward_ph: reward_sample,
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
        averages['dist_sum'] += d_center
        report = (
            'dc={:5.1f} de={:5.1f}   '
            'b1=({:5.1f},{:5.1f})   b2=({:5.1f}, {:5.1f})    '
            'r={:5.3f}    Q={:5.3f}    '
            'r\'={:5.3f}    y={:5.3f}    Q\'={:5.3f}'.format(
                d_center,
                d_enemy,
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
        return report

    def init_replay_memory(self):
        if self.replay_memory is None:
            memcap = self.memory_cap
            replay_state = np.empty((memcap, 2, 2, state2vec.vector_length), np.float32)
            replay_action = np.empty((memcap, action2vec.vector_length), np.float32)
            replay_reward = np.empty((memcap,), np.float32)
            replay_indices = set()
            self.replay_memory = replay_state, replay_action, replay_reward, replay_indices

    def save_replay_memory(self, directory):
        if self.replay_memory is None:
            return
        replay_state, replay_action, replay_reward, replay_indices = self.replay_memory
        idx = tuple(replay_indices)
        np.save(os.path.join(directory, 'state.npy'), replay_state[idx, ])
        np.save(os.path.join(directory, 'action.npy'), replay_action[idx, ])
        np.save(os.path.join(directory, 'reward.npy'), replay_reward[idx, ])

    def load_replay_memory(self, directory):
        try:
            state = np.load(os.path.join(directory, 'state.npy'))
            action = np.load(os.path.join(directory, 'action.npy'))
            reward = np.load(os.path.join(directory, 'reward.npy'))
        except:
            traceback.print_exc()
            return
        else:
            if not (state.shape[0] == action.shape[0] == reward.shape[0]):
                raise AssertionError('saved replay memory is inconsistent')
            data_size = min(self.memory_cap, state.shape[0])
            self.init_replay_memory()
            self.replay_memory[0][:data_size] = state[:data_size]
            self.replay_memory[1][:data_size] = action[:data_size]
            self.replay_memory[2][:data_size] = reward[:data_size]
            self.replay_memory[3].clear()
            self.replay_memory[3].update(range(data_size))


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


DEFAULT_QFUNC_MODEL = (25, 15, 10)


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
    rl = ReinforcementLearning(
        model,
        batch_size=80,
        n_games=1,
        memory_cap=200000,
        reward_decay=0.95,
        select_random_prob_decrease=0.03,
        self_play=False,
    )
    sess.run(rl.init_op)
    rl.load_replay_memory(opts.replay_dir)

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
                )
                if opts.save:
                    model.save_vars()
                    rl.save_replay_memory(opts.replay_dir)
        except KeyboardInterrupt:
            pass
        finally:
            if opts.save:
                model.save_vars()
                rl.save_replay_memory(opts.replay_dir)


if __name__ == '__main__':
    main()

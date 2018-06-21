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
        # return PassiveAI(team, engine)
        return TrainerAI(team, engine)


class RunAI(BaseDQNDualAI):

    class Shared:
        instance = None

        def __init__(self):
            self.state_ph = tf.placeholder(tf.float32, [1, aug_state_vector_len])
            self.model = QualityFunction.Model()
            self.model.load_vars()
            self.selector = SelectAction(self.model, self.state_ph, -1)

    def __init__(self, team, engine, ai_shared):
        super(RunAI, self).__init__(team, engine)
        self.shared = ai_shared  # type: RunAI.Shared

    def initialize(self):
        self.create_bot()

    def tick(self):
        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return
        action = self.shared.selector.call({
            self.shared.state_ph: [make_state_vector(bot, enemy, self.engine)],
        })
        decode_action(action[0], ctl)


class TrainableAI(BaseDQNDualAI):

    action = None
    bot = None
    ctl = None

    def initialize(self):
        self.bot = self.create_bot(teamize_x=False)
        self.ctl = self.engine.get_control(self.bot)

    def tick(self):
        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return
        decode_action(self.action, ctl)


class TrainerRaiderAI(RaiderVsSniper):

    def initialize(self):
        self.bot_type = random_bot_type()
        super(TrainerRaiderAI, self).initialize()


class TrainerSniperAI(SniperVsRaider):

    def initialize(self):
        self.bot_type = random_bot_type()
        super(TrainerSniperAI, self).initialize()


def TrainerAI(team, engine):
    if random.random() > 0.5:
        log.info('initializing short-range-attack trainer')
        return TrainerRaiderAI(team, engine)
    else:
        log.info('initializing distant-attack trainer')
        return TrainerSniperAI(team, engine)


class PassiveAI(BaseDQNDualAI):

    def initialize(self):
        bot = self.create_bot()
        bot.x = self.engine.world_width * 0.6
        bot.y = self.engine.world_height * 0.5
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
            self.levels = tuple(levels)
            self.__class__._cnt += 1
            self.name = 'Q{}'.format(self._cnt)
            with tf.variable_scope(self.name):
                d_in = aug_state_vector_len + action_vector_len
                self.transforms = []
                self.weights = []
                self.biases = []
                self.var_list = []
                for i, d_out in enumerate(self.levels):
                    t = tf.get_variable('T{}'.format(i), [d_out, d_in])
                    w = tf.get_variable('W{}'.format(i), [d_out, d_in])
                    b = tf.get_variable('B{}'.format(i), [d_out, 1])
                    self.transforms.append(t)
                    self.weights.append(w)
                    self.biases.append(b)
                    self.var_list.extend((w, b, t))
                    d_in = d_out

                self.qw = tf.get_variable('QW', [1, d_in])
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

        def augment_state(self, vector):
            return augment_state_vector(vector)

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
        self.args = tf.concat([state, action], -1)  # type: tf.Tensor

        syncshape = functools.partial(
            add_batch_shape,
            batch_shape=self.args.get_shape()[:-1],
        )

        self.levels = []
        vector = tf.expand_dims(self.args, -1)
        for w, b, tr in zip(model.weights, model.biases, model.transforms):
            transformed = tf.matmul(syncshape(tr), vector)
            nobias = tf.matmul(syncshape(w), vector)
            noact = tf.add(nobias, syncshape(b))
            d = transformed.shape.as_list()[-1]
            dhalf = d // 2
            act1 = tf.nn.relu(noact[..., :dhalf])
            act2 = tf.sigmoid(noact[..., dhalf:])
            out1 = transformed[..., :dhalf] + act1
            out2 = transformed[..., dhalf:] * act2
            out = tf.concat([out1, out2], -1)
            self.levels.append(dict(
                transf=transformed,
                nobias=nobias,
                noact=noact,
                act1=act1,
                act2=act2,
                out=out,
            ))
            vector = out

        quality = tf.matmul(syncshape(model.qw), vector)
        finite_assert = tf.Assert(
            tf.reduce_all(tf.is_finite(quality)),
            [tf.reduce_all(tf.is_finite(v)) for v in model.var_list],
        )
        with tf.control_dependencies([finite_assert]):
            self.quality = tf.squeeze(quality, [-2, -1])

    def call(self, state, action, session=None):
        session = session or get_session()
        return session.run(self.quality, feed_dict={
            self.state: state,
            self.action: action,
        })


class SelectAction:

    def __init__(self, qfunc_model, state, random_prob=0.05):
        n_all_actions = all_actions.shape[0]
        batch_shape = shape_to_list(state.shape[:-1])

        batched_all_actions = np.reshape(
            all_actions,
            [n_all_actions] + [1] * len(batch_shape) + [action_vector_len]
        ) + np.zeros([n_all_actions, *batch_shape, action_vector_len])

        self.all_actions = tf.constant(all_actions, dtype=tf.float32)
        self.batched_all_actions = tf.constant(batched_all_actions, dtype=tf.float32)
        self.state = add_batch_shape(state, [n_all_actions])

        self.qfunc = QualityFunction(qfunc_model, self.state, self.batched_all_actions)
        self.max_idx = tf.argmax(self.qfunc.quality, 0)
        self.max_q = tf.reduce_max(self.qfunc.quality, 0)

        self.random_prob = random_prob
        self.random_prob_sample = tf.random_uniform(batch_shape, 0, 1)
        self.random_mask = tf.less(self.random_prob_sample, random_prob)

        self.random_indices = tf.random_uniform(batch_shape, 0, n_all_actions, tf.int64)

        self.selected_idx = tf.where(self.random_mask, self.random_indices, self.max_idx)

        self.action = tf.gather_nd(
            self.all_actions,
            tf.expand_dims(self.selected_idx, -1),
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
        self.select_random_prob_ph = tf.placeholder(tf.float32, [])
        self.emu_state_ph = tf.placeholder(tf.float32, [emu_items, aug_state_vector_len])
        self.emu_selector = SelectAction(self.model, self.emu_state_ph, self.select_random_prob_ph)

        self.train_state_ph = tf.placeholder(tf.float32, [batch_size, aug_state_vector_len])
        self.train_next_state_ph = tf.placeholder(tf.float32, [batch_size, aug_state_vector_len])
        self.train_action_ph = tf.placeholder(tf.float32, [batch_size, action_vector_len])
        self.train_reward_ph = tf.placeholder(tf.float32, [batch_size])
        self.train_selector = SelectAction(self.model, self.train_next_state_ph, -1)
        self.train_qfunc = QualityFunction(self.model, self.train_state_ph, self.train_action_ph)

        self.optimizer = tf.train.AdamOptimizer()

        self.y = self.train_reward_ph + (1 - reward_decay) * self.train_selector.max_q
        self.loss_vector = (self.y - self.train_qfunc.quality) ** 2
        self.loss = tf.reduce_mean(self.loss_vector)
        self.train_step = self.optimizer.minimize(self.loss, var_list=model.var_list)

        self.init_op = tf.variables_initializer(self.optimizer.variables())

    def run(self, frameskip=0, max_ticks=1000, world_size=300):
        sess = get_session()
        # ai2_cls = TrainableAI if self.self_play else PassiveAI
        ai2_cls = TrainableAI if self.self_play else TrainerAI
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
        replay_state_aug = np.empty((*replay_state.shape[:-1], aug_state_vector_len))
        for i in replay_indices:
            for j, k in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                s_aug = self.model.augment_state(replay_state[i, j, k])
                replay_state_aug[i, j, k] = s_aug
        replay_idx = {i: i for i in range(self.n_games)}
        last_replay_idx = {}

        stats = collections.defaultdict(float)

        iteration = 0
        while games:
            iteration += 1

            # do step of games
            for g, engine in games.items():
                idx = replay_idx[g]
                replay_state[idx, 0, 0] = s1 = make_state_vector(engine.ai1.bot, engine.ai2.bot, engine)
                replay_state[idx, 0, 1] = s2 = make_state_vector(engine.ai2.bot, engine.ai1.bot, engine)
                replay_state_aug[idx, 0, 0] = self.model.augment_state(s1)
                replay_state_aug[idx, 0, 1] = self.model.augment_state(s2)

                select_random_prob = max(
                    0.1,
                    1 - self.select_random_prob_decrease * self.model.step_counter
                )
                emu_stats_t = self.get_emu_stats()
                if self.self_play:
                    actions, emu_stats = sess.run([
                        self.emu_selector.action,
                        emu_stats_t
                    ], {
                        self.emu_state_ph: replay_state_aug[idx, 0],
                        self.select_random_prob_ph: select_random_prob,
                    })
                else:
                    actions, emu_stats = sess.run([
                        self.emu_selector.action,
                        emu_stats_t
                    ], {
                        self.emu_state_ph: replay_state_aug[idx, 0, :1],
                        self.select_random_prob_ph: select_random_prob,
                    })
                replay_action[idx] = actions[0]

                engine.ai1.action = actions[0]
                if self.self_play:
                    engine.ai2.action = actions[1]
                for _ in range(1 + frameskip):
                    engine.tick()

                reward = engine.ai1.bot.hp_ratio - engine.ai2.bot.hp_ratio
                reward *= 10  # make reward scale larger to stabilize learning
                replay_reward[idx] = reward

                self.add_emu_stats(stats, emu_stats, reward)

                replay_state[idx, 1, 0] = s1 = make_state_vector(engine.ai1.bot, engine.ai2.bot, engine)
                replay_state[idx, 1, 1] = s2 = make_state_vector(engine.ai2.bot, engine.ai1.bot, engine)
                replay_state_aug[idx, 1, 0] = self.model.augment_state(s1)
                replay_state_aug[idx, 1, 1] = self.model.augment_state(s2)
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
                bots_sample = replay_state_aug[replay_sample, ]
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
                self.add_train_stats(stats, train_stats)

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

    def add_train_stats(self, stat_store, stat_values):
        loss, y, t_q = stat_values
        stat_store['y'] += np.sum(y) / np.size(y)
        stat_store['t_q'] += np.sum(t_q) / np.size(t_q)
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
            'y={:5.3f}    Q\'={:5.3f}'.format(
                d_center,
                d_enemy,
                engine.ai1.bot.x,
                engine.ai1.bot.y,
                engine.ai2.bot.x,
                engine.ai2.bot.y,
                stats['reward'] / max(1, stats['n_emu']),
                stats['emu_max_q'] / max(1, stats['n_emu']),
                stats['y'] / max(1, stats['n_train']),
                stats['t_q'] / max(1, stats['n_train']),
            ))
        stats.clear()
        return report

    def init_replay_memory(self):
        if self.replay_memory is None:
            memcap = self.memory_cap
            replay_state = np.empty((memcap, 2, 2, state_vector_len), np.float32)
            replay_action = np.empty((memcap, action_vector_len), np.float32)
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


def make_state_vector(bot, enemy, engine):
    b = bot2vec(bot, engine)
    e = bot2vec(enemy, engine)
    return b + e


def augment_state_vector(vector):
    sz = len(vector)
    b = bot_vector(*vector[:sz//2])
    e = bot_vector(*vector[sz//2:])
    aug = tuple(
        getattr(b1, elem1) * getattr(b2, elem2)
        for elem1 in _aug_elements
        for elem2 in _aug_elements
        if (elem1, elem2) not in _no_aug
        for b1, b2 in itertools.product([b, e], [b, e])
    )
    return np.concatenate([vector, aug])

_aug_elements = ['x', 'y', 'vx', 'vy', 'o_cos', 'o_sin', 'g_cos', 'g_sin']
_no_aug = [
    ('x', 'vx'),
    ('x', 'vy'),
    ('y', 'vx'),
    ('y', 'vy'),
    ('o_cos', 'g_cos'),
    ('o_cos', 'g_sin'),
    ('o_sin', 'g_cos'),
    ('o_sin', 'g_sin'),
]
_no_aug += [tup[::-1] for tup in _no_aug]


bot_vector = collections.namedtuple('bot_vector', [
    'type1', 'type2', 'type3', 'hp', 'load', 'shield',
    'x', 'y', 'vx', 'vy', 'o_cos', 'o_sin', 'g_cos', 'g_sin',
    # 'ori', 'gun_ori',
])


def bot2vec(bot, engine):
    return bot_vector(
        int(bot.type == BotType.Raider),
        int(bot.type == BotType.Heavy),
        int(bot.type == BotType.Sniper),
        bot.hp_ratio,
        bot.load,
        max(0, bot.shield_remaining / engine.ticks_per_sec),
        bot.x,
        bot.y,
        bot.vx,
        bot.vy,
        cos(bot.orientation),
        sin(bot.orientation),
        cos(bot.orientation + bot.tower_orientation),
        sin(bot.orientation + bot.tower_orientation),
        # bot.orientation % (2*pi),
        # bot.tower_orientation % (2*pi),
    )


def action2vec(ctl):
    return [
        ctl.move,
        ctl.rotate,
        ctl.tower_rotate,
        int(ctl.fire),
        int(ctl.shield),
    ]


def decode_action(vec, ctl):
    ctl.move = +1 if vec[0] > 0.5 else 0 if vec[0] > -0.5 else -1
    ctl.rotate = +1 if vec[1] > 0.5 else 0 if vec[1] > -0.5 else -1
    ctl.tower_rotate = +1 if vec[2] > 0.5 else 0 if vec[2] > -0.5 else -1
    ctl.fire = vec[3] > 0.5
    ctl.shield = vec[4] > 0.5


all_actions = np.array(list(itertools.product(*[
    [-1, 0, +1],
    [-1, 0, +1],
    [-1, 0, +1],
    [0, 1],
    [0, 1]
])))


def random_bot_type():
    return random.choice([
        BotType.Raider,
        BotType.Heavy,
        BotType.Sniper,
    ])


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


def __get_state_vector_len():
    engine = StbEngine(1, 1, PassiveAI, PassiveAI)
    st = make_state_vector(engine.ai1.bot, engine.ai2.bot, engine)
    st_aug = augment_state_vector(st)
    return len(st), len(st_aug)
state_vector_len, aug_state_vector_len = __get_state_vector_len()
action_vector_len = len(action2vec(BotControl()))


_global_session_ref = None


def get_session():
    global _global_session_ref
    sess = _global_session_ref() if _global_session_ref else None
    if sess is None:
        sess = tf.Session()
        _global_session_ref = weakref.ref(sess, sess.close)
    return sess


DEFAULT_QFUNC_MODEL = (15, 10, 5)


def main():
    this_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'play'], default='train')
    parser.add_argument('--no-save', action='store_false', dest='save')
    parser.add_argument('--replay-dir', default=os.path.join(this_dir, '_replay_memory'))
    opts = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

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
        memory_cap=50000,
        reward_decay=0.95,
        select_random_prob_decrease=0.01,
        self_play=False,
    )
    sess.run(rl.init_op)
    rl.load_replay_memory(opts.replay_dir)

    if opts.action == 'play':
        import code
        code.interact(local=dict(globals(), **locals()))
    if opts.action == 'train':
        try:
            while True:
                rl.run(
                    frameskip=2,
                    max_ticks=8000,
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

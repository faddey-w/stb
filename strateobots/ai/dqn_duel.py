import itertools
import functools
import random
import numpy as np
import tensorflow as tf
import os
import weakref
import logging
from math import pi, sin, cos
from strateobots.engine import BotType, StbEngine, dist_points, BotControl
from ._base import DuelAI
from .simple_duel import RaiderVsSniper


tf.reset_default_graph()
log = logging.getLogger(__name__)


class BaseDQNDualAI(DuelAI):

    def create_bot(self):
        return self.engine.add_bot(
            bottype=random_bot_type(),
            team=self.team,
            x=self.x_to_team_field(random.random()),
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
        ai = RaiderVsSniper(team, engine)
        ai.bot_type = random_bot_type()
        return ai


class RunAI(BaseDQNDualAI):

    class Shared:
        instance = None

        def __init__(self):
            self.bot_ph = tf.placeholder(tf.float32, [1, bot_vector_len])
            self.enemy_ph = tf.placeholder(tf.float32, [1, bot_vector_len])
            self.model = QualityFunction.Model()  # type: QualityFunction.Model
            self.model.load_vars()
            self.selector = SelectAction(self.model, self.bot_ph, self.enemy_ph)  # type: SelectAction

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
            self.shared.bot_ph: [bot2vec(bot, self.engine)],
            self.shared.enemy_ph: [bot2vec(enemy, self.engine)],
        })
        decode_action(action[0], ctl)


class TrainableAI(BaseDQNDualAI):

    action = None
    bot = None
    ctl = None

    def initialize(self):
        self.bot = self.create_bot()
        self.ctl = self.engine.get_control(self.bot)

    def tick(self):
        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return
        decode_action(self.action, ctl)


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

        def __init__(self, levels=(30, 20, 15, 10, 10)):
            self.levels = tuple(levels)
            self.__class__._cnt += 1
            self.name = 'Q{}'.format(self._cnt)
            with tf.variable_scope(self.name):
                d_in = input_vector_len * (input_vector_len + 1)
                self.weights = []
                self.biases = []
                self.var_list = []
                for i, d_out in enumerate(self.levels):
                    w = tf.get_variable('W{}'.format(i), [d_out, d_in])
                    b = tf.get_variable('B{}'.format(i), [d_out, 1])
                    self.weights.append(w)
                    self.biases.append(b)
                    self.var_list.extend((w, b))
                    d_in = d_out

                self.qw = tf.get_variable('QW', [1, d_in])
                self.var_list.append(self.qw)

            self.initializer = tf.variables_initializer(self.var_list)
            self.saver = tf.train.Saver(self.var_list)
            self.step_counter = 0

        def init_vars(self, session=None):
            session = session or get_session(self)
            session.run(self.initializer)
            session.run(self.qw + 1)

        def save_vars(self, session=None):
            session = session or get_session(self)
            self.step_counter += 1
            self.saver.save(session, self.save_path, self.step_counter)
            with open(self.save_path + 'step', 'w') as f:
                f.write(str(self.step_counter))
            log.info('saved model "%s" at step=%s to %s',
                     self.name, self.step_counter, self.save_path)

        def load_vars(self, session=None):
            session = session or get_session(self)
            with open(self.save_path + 'step') as f:
                self.step_counter = int(f.read().strip())
            ckpt = tf.train.get_checkpoint_state(self.save_path)
            log.info('loading model "%s" from %s at step=%s',
                     self.name, ckpt.model_checkpoint_path, self.step_counter)
            self.saver.restore(session, ckpt.model_checkpoint_path)

    def __init__(self, model, bot, enemy, action):
        """
        :param model: QualityFunction.Model
        :param bot: [..., bot_vector_len]
        :param enemy: [..., bot_vector_len]
        :param action: [..., action_vector_len]
        ellipsis shapes should be the same
        """
        self.model = model  # type: QualityFunction.Model
        self.bot = bot  # type: tf.Tensor
        self.enemy = enemy  # type: tf.Tensor
        self.action = action  # type: tf.Tensor
        self.args = tf.concat([bot, enemy, action], -1)  # type: tf.Tensor

        syncshape = functools.partial(
            add_batch_shape,
            batch_shape=self.args.get_shape()[:-1],
        )
        args_ext = tf.concat([self.args, syncshape(tf.constant([1.0]))], -1)
        premul_args = tf.multiply(
            tf.expand_dims(self.args, -1),
            tf.expand_dims(args_ext, -2),
        )
        premul_shape = premul_args.shape.as_list()
        self.features = tf.reshape(
            premul_args,
            premul_shape[:-2] + [premul_shape[-2] * premul_shape[-1]]
        )

        self.levels = []
        vector = tf.expand_dims(self.features, -1)
        for w, b in zip(self.model.weights, self.model.biases):
            nobias = tf.matmul(syncshape(w), vector)
            noact = tf.add(nobias, syncshape(b))
            if w is self.model.weights[-1]:  # if is last level
                out = tf.sigmoid(noact)
            else:
                out = tf.nn.relu(noact)
            self.levels.append(dict(nobias=nobias, noact=noact, out=out))
            vector = out

        quality = tf.matmul(syncshape(model.qw), vector)
        self.quality = tf.squeeze(quality, [-2, -1])

    def call(self, bot, enemy, action, session=None):
        session = session or get_session(self)
        return session.run(self.quality, feed_dict={
            self.bot: bot,
            self.enemy: enemy,
            self.action: action,
        })


class SelectAction:

    def __init__(self, qfunc_model, bot, enemy, random_prob=0.05):
        n_all_actions = all_actions.shape[0]
        batch_shape = shape_to_list(bot.shape[:-1])

        batched_all_actions = np.reshape(
            all_actions,
            [n_all_actions] + [1] * len(batch_shape) + [action_vector_len]
        ) + np.zeros([n_all_actions, *batch_shape, action_vector_len])

        self.all_actions = tf.constant(all_actions, dtype=tf.float32)
        self.batched_all_actions = tf.constant(batched_all_actions, dtype=tf.float32)
        self.bot = add_batch_shape(bot, [n_all_actions])
        self.enemy = add_batch_shape(enemy, [n_all_actions])

        self.qfunc = QualityFunction(qfunc_model, self.bot, self.enemy, self.batched_all_actions)
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
        session = session or get_session(self)
        return session.run(self.action, feed_dict=feed_dict)


class ReinforcementLearning:

    def __init__(self, model, batch_size=10, n_games=10, memory_cap=200,
                 reward_decay=0.03, self_play=True):
        self.memory_cap = memory_cap

        self.model = model  # type: QualityFunction.Model
        self.batch_size = batch_size
        self.n_games = n_games
        self.self_play = self_play

        emu_items = 2 if self_play else 1
        self.emu_bot_ph = tf.placeholder(tf.float32, [emu_items, bot_vector_len])
        self.emu_enemy_ph = tf.placeholder(tf.float32, [emu_items, bot_vector_len])
        self.emu_selector = SelectAction(self.model, self.emu_bot_ph, self.emu_enemy_ph)

        self.train_bot_ph = tf.placeholder(tf.float32, [batch_size, bot_vector_len])
        self.train_enemy_ph = tf.placeholder(tf.float32, [batch_size, bot_vector_len])
        self.train_next_bot_ph = tf.placeholder(tf.float32, [batch_size, bot_vector_len])
        self.train_next_enemy_ph = tf.placeholder(tf.float32, [batch_size, bot_vector_len])
        self.train_action_ph = tf.placeholder(tf.float32, [batch_size, action_vector_len])
        self.train_reward_ph = tf.placeholder(tf.float32, [batch_size])
        self.optimizer = tf.train.GradientDescentOptimizer(0.01)
        self.train_selector = SelectAction(self.model, self.train_next_bot_ph, self.train_next_enemy_ph)
        self.train_qfunc = QualityFunction(self.model, self.train_bot_ph, self.train_bot_ph, self.train_action_ph)

        self.y = self.train_reward_ph + (1 - reward_decay) * self.train_selector.max_q
        self.loss_vector = (self.y - self.train_qfunc.quality) ** 2
        self.loss = tf.reduce_max(self.loss_vector)
        self.train_step = self.optimizer.minimize(self.loss, var_list=model.var_list)

    def run(self, frameskip=0, max_ticks=1000, world_size=300):
        ctl = BotControl()

        def trainer_ai(team, engine):
            ai = RaiderVsSniper(team, engine)
            ai.bot_type = random_bot_type()
            return ai

        sess = get_session(self)
        ai2_cls = TrainableAI if self.self_play else PassiveAI
        games = {
            i: StbEngine(
                world_size, world_size,
                TrainableAI, ai2_cls,
                max_ticks)
            for i in range(self.n_games)
        }
        memcap = self.memory_cap
        replay_bots = np.empty((memcap, 2, 2, bot_vector_len), np.float32)
        replay_action = np.empty((memcap, action_vector_len), np.float32)
        replay_reward = np.empty((memcap,), np.float32)
        replay_indices = set()
        replay_idx = {i: i for i in range(self.n_games)}

        iteration = 0
        while games:
            iteration += 1

            # do step of games
            for g, engine in games.items():
                idx = replay_idx[g]
                replay_bots[idx, 0, 0] = bot2vec(engine.ai1.bot, engine)
                replay_bots[idx, 0, 1] = bot2vec(engine.ai2.bot, engine)
                # score_before = engine.ai1.bot.hp_ratio - engine.ai2.bot.hp_ratio
                if self.self_play:
                    actions = sess.run(self.emu_selector.action, {
                        self.emu_bot_ph: replay_bots[idx, 0, ::+1],
                        self.emu_enemy_ph: replay_bots[idx, 0, ::-1],
                    })
                else:
                    actions = sess.run(self.emu_selector.action, {
                        self.emu_bot_ph: replay_bots[idx, 0, :1],
                        self.emu_enemy_ph: replay_bots[idx, 0, 1:],
                    })
                replay_action[idx] = actions[0]

                engine.ai1.action = actions[0]
                if self.self_play:
                    engine.ai2.action = actions[1]
                for _ in range(1 + frameskip):
                    engine.tick()

                score_after = engine.ai1.bot.hp_ratio - engine.ai2.bot.hp_ratio
                # decode_action(actions[0], ctl)
                # activity_penalty = ctl.fire + (ctl.move != 0) + (ctl.rotate != 0) + (ctl.tower_rotate != 0)
                # reward = score_after - score_before  # - 0.025 * activity_penalty
                reward = score_after
                reward -= dist_points(
                    engine.ai1.bot.x,
                    engine.ai1.bot.y,
                    world_size / 2,
                    world_size / 2,
                ) / world_size
                replay_reward[idx] = reward

                replay_bots[idx, 1, 0] = bot2vec(engine.ai1.bot, engine)
                replay_bots[idx, 1, 1] = bot2vec(engine.ai2.bot, engine)
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

                replay_idx[g] = next_idx

            # do GD step
            if len(replay_indices) >= self.batch_size:
                replay_sample = tuple(random.sample(
                    replay_indices - set(replay_idx.values()),
                    self.batch_size,
                ))
                bots_sample = replay_bots[replay_sample, ]
                action_sample = replay_action[replay_sample, ]
                reward_sample = replay_reward[replay_sample, ]
                sess.run(self.train_step, {
                    self.train_bot_ph: bots_sample[:, 0, 0],
                    self.train_enemy_ph: bots_sample[:, 0, 1],
                    self.train_next_bot_ph: bots_sample[:, 1, 0],
                    self.train_next_enemy_ph: bots_sample[:, 1, 1],
                    self.train_action_ph: action_sample,
                    self.train_reward_ph: reward_sample,
                })

            # report on games
            if iteration % 10 == 1 and games:
                dist_sum = 0
                for g, engine in games.items():
                    d = dist_points(
                        engine.ai1.bot.x,
                        engine.ai1.bot.y,
                        engine.ai2.bot.x,
                        engine.ai2.bot.y,
                    )
                    dist_sum += d
                    print('#{}: {}/{} t={} vs={:.2f}/{:.2f} d={:.2f}'.format(
                        g,
                        engine.ai1.bot.type.name[:1],
                        engine.ai2.bot.type.name[:1],
                        engine.nticks,
                        engine.ai1.bot.hp_ratio,
                        engine.ai2.bot.hp_ratio,
                        d
                    ))
                print('avg dist = {:.3f}'.format(dist_sum / len(games)))
                print()

            # report on memory
            # if iteration % 100 == 1 and len()


def bot2vec(bot, engine):
    return [
        int(bot.type == BotType.Raider),
        int(bot.type == BotType.Heavy),
        int(bot.type == BotType.Sniper),
        bot.hp_ratio,
        bot.load,
        bot.x,
        bot.y,
        bot.vx,
        bot.vy,
        cos(bot.orientation),
        sin(bot.orientation),
        cos(bot.tower_orientation),
        sin(bot.tower_orientation),
        bot.shield_remaining,
    ]


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


all_actions = np.array(list(itertools.product(
    [-1, 0, +1],
    [-1, 0, +1],
    [-1, 0, +1],
    [0, 1],
    [0, 1]
)))
bot_vector_len = (
    + 3  # type
    + 1  # hp_ratio
    + 1  # load
    + 1  # x normed
    + 1  # y normed
    + 1  # vx normed
    + 1  # vy normed
    + 2  # orientation sin and cos
    + 2  # tower orientation sin and cos
    + 1  # shield remaining
)
action_vector_len = (
    + 1  # move
    + 1  # rotate
    + 1  # tower_rotate
    + 1  # fire
    + 1  # shield
)
input_vector_len = 2 * bot_vector_len + action_vector_len


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


_session = None


def get_session(owner):
    global _session
    if not _session:
        _session = tf.Session()
    weakref.finalize(owner, _clear_session)
    return _session


def _clear_session():
    global _session
    if _session:
        _session.close()
    _session = None


def main():
    logging.basicConfig(level=logging.INFO)

    model = QualityFunction.Model([4, 3, 2])
    try:
        model.load_vars()
    except:
        model.init_vars()
    rl = ReinforcementLearning(
        model,
        batch_size=50,
        n_games=10,
        memory_cap=500,
        reward_decay=0.01,
        self_play=False,
    )

    try:
        while True:
            rl.run(frameskip=22, max_ticks=10000, world_size=1000)
            model.save_vars()
    except KeyboardInterrupt:
        pass
    finally:
        pass
        model.save_vars()


if __name__ == '__main__':
    main()

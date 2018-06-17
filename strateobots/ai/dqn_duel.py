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

    def create_bot(self, teamize_x=True):
        x = random.random()
        if teamize_x:
            x = self.x_to_team_field(x)
        return self.engine.add_bot(
            bottype=random_bot_type(),
            team=self.team,
            x=x,
            y=self.engine.world_height * random.random(),
            orientation=random.random() * 2 * pi,
            tower_orientation=random.random() * 2 * pi,
        )


def AI(team, engine):
    if team == engine.teams[0] or True:
        if RunAI.Shared.instance is None:
            RunAI.Shared.instance = RunAI.Shared()
        return RunAI(team, engine, RunAI.Shared.instance)
    else:
        return PassiveAI(team, engine)
        ai = RaiderVsSniper(team, engine)
        ai.bot_type = random_bot_type()
        return ai


class RunAI(BaseDQNDualAI):

    class Shared:
        instance = None

        def __init__(self):
            self.state_ph = tf.placeholder(tf.float32, [1, state_vector_len])
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
                d_in = state_vector_len + action_vector_len
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
            self.saver = tf.train.Saver(self.var_list)
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
        :param state: [..., state_vector_len]
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
            if w is self.model.weights[-1]:  # if is last level
                resid = tf.sigmoid(noact)
            else:
                resid = tf.nn.relu(noact)
            d = transformed.shape.as_list()[-1]
            dhalf = d // 2
            out1 = transformed[..., :dhalf] + resid[..., :dhalf]
            out2 = transformed[..., dhalf:] * resid[..., dhalf:]
            out = tf.concat([out1, out2], -1)
            self.levels.append(dict(
                transf=transformed,
                nobias=nobias,
                noact=noact,
                out=out,
                resid=resid,
            ))
            vector = out

        quality = tf.matmul(syncshape(model.qw), vector)
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
                 reward_decay=0.03, self_play=True):
        self.memory_cap = memory_cap
        self.replay_memory = None

        self.model = model  # type: QualityFunction.Model
        self.batch_size = batch_size
        self.n_games = n_games
        self.self_play = self_play

        emu_items = 2 if self_play else 1
        self.emu_state_ph = tf.placeholder(tf.float32, [emu_items, state_vector_len])
        self.emu_selector = SelectAction(self.model, self.emu_state_ph)

        self.train_state_ph = tf.placeholder(tf.float32, [batch_size, state_vector_len])
        self.train_next_state_ph = tf.placeholder(tf.float32, [batch_size, state_vector_len])
        self.train_action_ph = tf.placeholder(tf.float32, [batch_size, action_vector_len])
        self.train_reward_ph = tf.placeholder(tf.float32, [batch_size])
        self.optimizer = tf.train.GradientDescentOptimizer(0.01)
        self.train_selector = SelectAction(self.model, self.train_next_state_ph)
        self.train_qfunc = QualityFunction(self.model, self.train_state_ph, self.train_action_ph)

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

        sess = get_session()
        # ai2_cls = TrainableAI if self.self_play else PassiveAI
        ai2_cls = TrainableAI if self.self_play else trainer_ai
        games = {
            i: StbEngine(
                world_size, world_size,
                TrainableAI, ai2_cls,
                max_ticks)
            for i in range(self.n_games)
        }
        memcap = self.memory_cap
        if self.replay_memory is None:
            replay_state = np.empty((memcap, 2, 2, state_vector_len), np.float32)
            replay_action = np.empty((memcap, action_vector_len), np.float32)
            replay_reward = np.empty((memcap,), np.float32)
            replay_indices = set()
            self.replay_memory = replay_state, replay_action, replay_reward, replay_indices
        else:
            replay_state, replay_action, replay_reward, replay_indices = self.replay_memory
        replay_idx = {i: i for i in range(self.n_games)}

        iteration = 0
        while games:
            iteration += 1

            # do step of games
            for g, engine in games.items():
                idx = replay_idx[g]
                replay_state[idx, 0, 0] = make_state_vector(engine.ai1.bot, engine.ai2.bot, engine)
                replay_state[idx, 0, 1] = make_state_vector(engine.ai2.bot, engine.ai1.bot, engine)
                if self.self_play:
                    actions = sess.run(self.emu_selector.action, {
                        self.emu_state_ph: replay_state[idx, 0],
                    })
                else:
                    actions = sess.run(self.emu_selector.action, {
                        self.emu_state_ph: replay_state[idx, 0, :1],
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
                reward -= 0.1 * dist_points(
                    engine.ai1.bot.x,
                    engine.ai1.bot.y,
                    engine.world_width / 2,
                    engine.world_height / 2,
                    # engine.ai2.bot.x,
                    # engine.ai2.bot.y,
                ) / world_size
                replay_reward[idx] = reward

                replay_state[idx, 1, 0] = make_state_vector(engine.ai1.bot, engine.ai2.bot, engine)
                replay_state[idx, 1, 1] = make_state_vector(engine.ai2.bot, engine.ai1.bot, engine)
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
                    # if engine.nticks > 5000:
                    #     return replay_state, replay_action, replay_reward
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
                    self.batch_size-len(replay_idx),
                )) + tuple(replay_idx.values())
                bots_sample = replay_state[replay_sample, ]
                action_sample = replay_action[replay_sample, ]
                reward_sample = replay_reward[replay_sample, ]
                sess.run(self.train_step, {
                    self.train_state_ph: bots_sample[:, 0, 0],
                    self.train_next_state_ph: bots_sample[:, 1, 0],
                    self.train_action_ph: action_sample,
                    self.train_reward_ph: reward_sample,
                })

            # report on games
            if iteration % 7 == 1 and games:
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


def make_state_vector(bot, enemy, engine):
    return bot2vec(bot, engine) + bot2vec(enemy, engine)


def bot2vec(bot, engine):
    return [
        int(bot.type == BotType.Raider),
        int(bot.type == BotType.Heavy),
        int(bot.type == BotType.Sniper),
        bot.hp_ratio,
        bot.load,
        bot.x / engine.world_width,
        bot.y / engine.world_height,
        bot.vx / bot.type.max_ahead_speed,
        bot.vy / bot.type.max_ahead_speed,
        cos(bot.orientation),
        sin(bot.orientation),
        cos(bot.orientation + bot.tower_orientation),
        sin(bot.orientation + bot.tower_orientation),
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
    return len(make_state_vector(engine.ai1.bot, engine.ai1.bot, engine))
state_vector_len = __get_state_vector_len()
action_vector_len = len(action2vec(BotControl()))


_global_session_ref = None


def get_session():
    global _global_session_ref
    sess = _global_session_ref() if _global_session_ref else None
    if sess is None:
        sess = tf.Session()
        _global_session_ref = weakref.ref(sess, sess.close)
    return sess


DEFAULT_QFUNC_MODEL = (20, 15, 10)


def main():
    logging.basicConfig(level=logging.INFO)

    model = QualityFunction.Model()
    try:
        model.load_vars()
    except:
        model.init_vars()
    rl = ReinforcementLearning(
        model,
        batch_size=80,
        n_games=10,
        memory_cap=10000,
        reward_decay=0.02,
        self_play=True,
    )

    try:
        while True:
            rl.run(
                frameskip=22,
                max_ticks=10000,
                world_size=1000,
            )
            # break
            model.save_vars()
    except KeyboardInterrupt:
        pass
    finally:
        pass
        model.save_vars()

    # import code; code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    main()

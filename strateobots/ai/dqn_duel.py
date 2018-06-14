import itertools
import functools
import random
import numpy as np
import tensorflow as tf
import weakref
from math import pi
from strateobots.engine import BotType
from ._base import DuelAI


tf.reset_default_graph()


class AI(DuelAI):

    selector = None  # type: SelectAction
    model = None  # type: QualityFunctionModel
    enemy_ph = None
    bot_ph = None

    def initialize(self):
        all_types = [
            BotType.Raider,
            BotType.Heavy,
            BotType.Sniper,
        ]
        self.engine.add_bot(
            bottype=random.choice(all_types),
            team=self.team,
            x=self.x_to_team_field(random.random()),
            y=self.engine.world_height * random.random(),
            orientation=random.random() * 2 * pi,
            tower_orientation=random.random() * 2 * pi,
        )
        self.bot_ph = tf.placeholder(tf.float32, [1, bot_vector_len])
        self.enemy_ph = tf.placeholder(tf.float32, [1, bot_vector_len])
        self.model = QualityFunctionModel()
        self.selector = SelectAction(self.model, self.bot_ph, self.enemy_ph)
        self.model.init_vars()

    def tick(self):
        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return
        action = self.selector.call({
            self.bot_ph: [bot2vec(bot, self.engine)],
            self.enemy_ph: [bot2vec(enemy, self.engine)],
        })
        decode_action(action[0], ctl)


class QualityFunctionModel:

    _cnt = 0

    def __init__(self, d0=30, d1=15):
        self.d0 = d0
        self.d1 = d1
        self.__class__._cnt += 1
        name = 'Q{}'.format(self._cnt)
        with tf.variable_scope(name):
            self.w0 = tf.get_variable(
                'W0', [self.d0, input_vector_len]
            )
            self.b0 = tf.get_variable('B0', [self.d0, 1])
            self.w1 = tf.get_variable(
                'W1', [self.d1, self.d0]
            )
            self.b1 = tf.get_variable('B1', [self.d1, 1])
            self.qw = tf.get_variable('QW', [1, self.d1])

        self.initializer = tf.variables_initializer([
            self.w0, self.b0,
            self.w1, self.b1,
            self.qw,
        ])

    def init_vars(self, session=None):
        session = session or get_session(self)
        session.run(self.initializer)
        session.run(self.qw + 1)


class QualityFunction:

    def __init__(self, model, bot, enemy, action):
        """
        :param model: QualityFunctionModel
        :param bot: [..., bot_vector_len]
        :param enemy: [..., bot_vector_len]
        :param action: [..., action_vector_len]
        ellipsis shapes should be the same
        """
        self.model = model  # type: QualityFunctionModel
        self.bot = bot  # type: tf.Tensor
        self.enemy = enemy  # type: tf.Tensor
        self.action = action  # type: tf.Tensor
        self.args = tf.concat([bot, enemy, action], -1)  # type: tf.Tensor

        syncshape = functools.partial(
            add_batch_shape,
            batch_shape=self.args.get_shape()[:-1],
        )

        self.l1_nobias = tf.matmul(syncshape(model.w0), tf.expand_dims(self.args, -1))
        self.l1_noact = tf.add(self.l1_nobias, syncshape(model.b0))
        self.l1 = tf.nn.relu(self.l1_noact)

        self.l2_nobias = tf.matmul(syncshape(model.w1), self.l1)
        self.l2_noact = tf.add(self.l2_nobias, syncshape(model.b1))
        self.l2 = tf.sigmoid(self.l2_noact)

        quality = tf.matmul(syncshape(model.qw), self.l2)
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

    def __init__(self, model, batch_size=10):
        pass


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
        bot.orientation % (2 * pi),
        bot.tower_orientation % (2 * pi),
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
    + 1  # orientation
    + 1  # tower orientation
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
    batchshape = [17]
    bot_ph = tf.placeholder(tf.float32, [*batchshape, bot_vector_len])
    enemy_ph = tf.placeholder(tf.float32, [*batchshape, bot_vector_len])
    action_ph = tf.placeholder(tf.float32, [*batchshape, action_vector_len])

    model = QualityFunctionModel()
    qfunc = QualityFunction(
        model=model,
        bot=bot_ph,
        enemy=enemy_ph,
        action=action_ph,
    )
    selact = SelectAction(model, bot_ph, enemy_ph)

    import code
    code.interact(local=locals())


if __name__ == '__main__':
    main()

import argparse
import random
import shutil
import time
import numpy as np
import tensorflow as tf
from math import pi
from strateobots.engine import StbEngine, BotType, BulletModel, BotControl
from .._base import DuelAI
from ..lib import replay, model_saving, handcrafted, data, util
from . import model as modellib


class AINDuelAI(DuelAI):

    bot_type = None
    function = None
    enabled = True

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

    def initialize(self):
        self.bot = bot = self.create_bot(False)
        self.ctl = self.engine.get_control(bot)

    def tick(self):
        if not self.enabled:
            return
        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return
        if callable(self.function):
            self.function(self.engine, bot, enemy, ctl)


def random_bot_type():
    return random.choice([
        BotType.Raider,
        BotType.Heavy,
        BotType.Sniper,
    ])


def decode_prediction(prediction, ctl=None, ctl_list=None):
    assert prediction.ndim <= 2
    was_1d = prediction.ndim == 1
    if was_1d:
        ctl_list = [ctl or BotControl()]
        prediction = np.reshape(prediction, (1, prediction.shape[-1]))
    elif ctl_list is None:
        ctl_list = [BotControl() for _ in range(prediction.shape[0])]
    move = -(np.argmax(prediction[..., 0:3], -1) - 1)
    rotate = np.argmax(prediction[..., 3:6], -1) - 1
    tower_rotate = np.argmax(prediction[..., 6:9], -1) - 1
    fire = prediction[..., 9] > 0.5
    shield = prediction[..., 10] > 0.5
    for i in range(prediction.shape[0]):
        ctl = ctl_list[i]
        ctl.move = move[i]
        ctl.rotate = rotate[i]
        ctl.tower_rotate = tower_rotate[i]
        ctl.fire = fire[i]
        ctl.shield = shield[i]
    if was_1d:
        return ctl_list[0]
    else:
        return ctl_list


def model_based_function(model, session):
    state_ph = tf.placeholder(tf.float32, [1, data.state2vec.vector_length])
    inference = model.apply(state_ph)

    def function(engine, bot, enemy, ctl):
        bullet_b, bullet_e = find_bullets(engine, [bot, enemy])
        state = data.state2vec((bot, enemy, bullet_b, bullet_e))
        prediction = session.run(inference.action_prediction, {state_ph: [state]})[0]
        decode_prediction(prediction, ctl)

    return function


def adopt_handcrafted_function(ai_function):
    def function(engine, bot, enemy, ctl):
        ai_function(bot, enemy, ctl)
    return function


def run_one_game(function1, function2, winner_memory, loser_memory, frequency=1,
                 save_win=True, **params):
    ai1_factory = AINDuelAI.parametrize(function=function1, **params)
    ai2_factory = AINDuelAI.parametrize(function=function2, **params)

    mem1 = replay.ReplayMemory(
        int(2000 / frequency),
        data.state2vec.vector_length,
        data.action2vec.vector_length,
    )
    mem2 = replay.ReplayMemory(
        int(2000 / frequency),
        data.state2vec.vector_length,
        data.action2vec.vector_length,
    )

    engine = StbEngine(1000, 1000, ai1_factory, ai2_factory, 2500, 0)

    while not engine.is_finished:
        state1, state2 = make_states(engine)
        action1 = make_action(engine.ai1)
        action2 = make_action(engine.ai2)

        mem1.put_entry(state1, action1)
        mem2.put_entry(state2, action2)

        for _ in range(frequency):
            engine.tick()

    if engine.ai1.bot.hp_ratio != engine.ai2.bot.hp_ratio:
        if engine.ai1.bot.hp_ratio > engine.ai2.bot.hp_ratio:
            # winner_memory.update(mem1)
            loser_memory.update(mem2)
            win = True
            if save_win and engine.ai2.bot.hp_ratio <= 0.2:
                winner_memory.update(mem1)
            elif engine.ai2.bot.hp_ratio > 0.99:
                loser_memory.update(mem1)
        else:
            loser_memory.update(mem1)
            win = False
            if save_win and engine.ai1.bot.hp_ratio <= 0.2:
                winner_memory.update(mem1)
            elif engine.ai1.bot.hp_ratio > 0.99:
                loser_memory.update(mem2)
    else:
        win = False
    return win, engine.ai1.bot.hp_ratio, engine.ai2.bot.hp_ratio


class AccuracyMetrics:

    def __init__(self, symbols):
        self.symbols = symbols
        self.matches = {v: 0 for v in symbols}
        self.predicted_amounts = {v: 0 for v in symbols}
        self.amounts = {v: 0 for v in symbols}
        self.total_amount = 0

    def add(self, correct, predicted):
        self.amounts[correct] += 1
        self.predicted_amounts[predicted] += 1
        if correct == predicted:
            self.matches[correct] += 1
        self.total_amount += 1

    def get_accuracy(self, symbol):
        return self.matches[symbol] / max(1, self.amounts[symbol])

    def get_overall_accuracy(self):
        return sum(self.matches.values()) / max(1, self.total_amount)

    def get_jaccard(self, symbol):
        m = self.matches[symbol]
        a = self.amounts[symbol]
        h = self.predicted_amounts[symbol]
        return m / max(1, a + h - m)

    def get_overall_jaccard(self):
        m = sum(self.matches.values())
        return m / (2 * self.total_amount - m)

    def __str__(self):
        return '{:.1f}%\t{:.2f}\t{}'.format(
            100 * self.get_overall_accuracy(),
            self.get_overall_jaccard(),
            '\t'.join(
                # '{}={:.2f}'.format(s, self.get_jaccard(s))
                '{}={:.2f}'.format(s, self.get_accuracy(s))
                for s in self.symbols
            )
        )


class AINTraining:

    def __init__(self, model, batch_size):
        self.batch_size = batch_size
        self.win_state_ph = tf.placeholder(tf.float32, [batch_size, data.state2vec.vector_length])
        self.win_action_ph = tf.placeholder(tf.float32, [batch_size, data.action2vec.vector_length])
        self.lose_state_ph = tf.placeholder(tf.float32, [batch_size, data.state2vec.vector_length])
        self.lose_action_ph = tf.placeholder(tf.float32, [batch_size, data.action2vec.vector_length])

        self.model = model
        self.win_inference = model.apply(self.win_state_ph)
        self.lose_inference = model.apply(self.lose_state_ph)

        self.cross_entropy = -(
            self.win_action_ph * tf.log(0.001 + 0.999 * self.win_inference.action_prediction)
            +
            (1 - self.win_action_ph) * tf.log(1 - 0.999 * self.win_inference.action_prediction)
        )
        self.win_loss_vector = self.cross_entropy
        self.win_loss = tf.reduce_mean(self.win_loss_vector)

        delta = self.lose_inference.action_prediction - self.lose_action_ph
        self.lose_loss_vector = tf.square(1 - tf.abs(delta))
        self.lose_loss = tf.reduce_mean(self.lose_loss_vector)

        # w = 0.2
        # self.loss = w * self.lose_loss + (1 - w) * self.win_loss
        self.loss = self.lose_loss

        self.optimizer = tf.train.AdamOptimizer(0.001)
        # self.optimizer = tf.train.RMSPropOptimizer(0.0001)
        self.train_op = self.optimizer.minimize(self.loss, var_list=model.var_list)

        self.init_op = tf.variables_initializer(self.optimizer.variables())

    def train_n_steps(self, session, win_memory, lose_memory, n_steps, print_each_step=10):
        win_loss_avg = util.Average()
        lose_loss_avg = util.Average()
        started = time.time()
        for i in range(1, 1+n_steps):
            (
                _,
                # wloss,
                lloss,
            ) = self.run_on_sample(
                [
                    self.train_op,
                    # self.win_loss,
                    self.lose_loss,
                ],
                win_memory, lose_memory, session,
            )
            # win_loss_avg.add(wloss)
            lose_loss_avg.add(lloss)
            if i % print_each_step == 0:
                # print('#{}: win_loss={:.4f}; lose_loss={:.4f}; time={:.2f}sec'.format(
                #     i, win_loss_avg.get(), lose_loss_avg.get(), time.time()-started))
                print('#{}: loss={:.4f}; time={:.2f}sec'.format(
                    i, lose_loss_avg.get(), time.time()-started))
                started = time.time()
                win_loss_avg.reset()
                lose_loss_avg.reset()

    def run_on_sample(self, tensors, win_memory, lose_memory, session):
        # win_states, win_actions = win_memory.get_random_sample(self.batch_size)
        lose_states, lose_actions = lose_memory.get_random_sample(self.batch_size)
        return session.run(tensors, {
            # self.win_state_ph: win_states,
            # self.win_action_ph: win_actions,
            self.lose_state_ph: lose_states,
            self.lose_action_ph: lose_actions,
        })

    def evaluate_accuracy(self, session, winner_memory, max_batches=50):
        bsize = self.batch_size
        n_batches = min(max_batches, int(winner_memory.used_size / bsize))
        states, actions = winner_memory.get_random_sample(bsize * n_batches)
        assert states.shape[0] == n_batches * bsize
        metrics = dict(
            move=AccuracyMetrics([-1, 0, +1]),
            rotate=AccuracyMetrics([-1, 0, +1]),
            tower_rotate=AccuracyMetrics([-1, 0, +1]),
            fire=AccuracyMetrics([0, 1]),
            shield=AccuracyMetrics([0, 1]),
        )
        example_ctls = [BotControl() for _ in range(bsize)]
        prediction_ctls = [BotControl() for _ in range(bsize)]
        for i in range(n_batches):
            slc = slice(i*bsize, (i+1)*bsize)
            predictions = session.run(
                self.win_inference.action_prediction,
                {self.win_state_ph: states[slc]},
            )
            decode_prediction(predictions, ctl_list=prediction_ctls)
            decode_prediction(actions[slc], ctl_list=example_ctls)
            for ctl_e, ctl_p in zip(example_ctls, prediction_ctls):
                for prop, metric in metrics.items():
                    metric.add(getattr(ctl_e, prop), getattr(ctl_p, prop))
        return metrics


def find_bullets(engine, bots):
    bullets = {
        bullet.origin_id: bullet
        for bullet in engine.iter_bullets()
    }
    return [
        bullets.get(bot.id, BulletModel(None, None, 0, bot.x, bot.y, 0))
        for bot in bots
    ]


def make_states(engine):
    bot1, bot2 = engine.ai1.bot, engine.ai2.bot
    bullet1, bullet2 = find_bullets(engine, [bot1, bot2])
    state1 = data.state2vec((bot1, bot2, bullet1, bullet2))
    state2 = data.state2vec((bot2, bot1, bullet2, bullet1))
    return state1, state2


def make_action(ai):
    ai.enabled = True
    ai.tick()
    action = data.action2vec(ai.ctl)
    ai.enabled = False
    return action


def noised(function, noise_prob):
    def _function(engine, bot, enemy, ctl):
        function(engine, bot, enemy, ctl)
        control_noise(ctl, noise_prob)
    return _function


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


def memory_keyfunc(state, action):
    """state and action are 1d vectors"""
    x0 = state[data.state2vec[0, 'x']]
    y0 = state[data.state2vec[0, 'y']]
    x1 = state[data.state2vec[1, 'x']]
    y1 = state[data.state2vec[1, 'y']]
    x_flag = x0 > x1
    y_flag = y0 > y1
    d_flag = ((x1-x0)**2 + (y1-y0)**2) > 200 ** 2
    return (*map(int, action), x_flag, y_flag, d_flag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-save', action='store_false', dest='save')
    parser.add_argument('--restart', action='store_true', dest='restart')
    opts = parser.parse_args()

    winner_memory = replay.BalancedMemory(
        keyfunc=memory_keyfunc,
        cap_per_class=1000,
        vector_sizes=[data.state2vec.vector_length, data.action2vec.vector_length]
    )
    winner_data_path = '_data/AIN-data-win'
    try:
        winner_memory.load(winner_data_path, eval)
    except:
        pass

    loser_memory = replay.ReplayMemory(
        500000,
        data.state2vec.vector_length, data.action2vec.vector_length
    )
    loser_data_path = '_data/AIN-data-lose'
    try:
        loser_memory.load(loser_data_path)
    except:
        pass

    try:
        if opts.restart:
            shutil.rmtree('_data/AIN')
        mgr = model_saving.ModelManager.load_existing_model('_data/AIN')
        session = tf.Session()
        mgr.load_vars(session)
        model = mgr.model
        print("model loaded at step ", mgr.step_counter)
    except:
        print("NEW model")
        tf.reset_default_graph()
        model = modellib.vec2d_fc_v2.ActionInferenceModel(
            vec_cfg=[(50, 50)] * 5,
            fc_cfg=[30] * 3,
        )
        # model = modellib.handcrafted.ActionInferenceModel()
        # model = modellib.simple.ActionInferenceModel(
        #     cfg=[60] * 6,
        # )
        # model = modellib.residual.ActionInferenceModel(
        #     res_cfg=[(50, 100)] * 6,
        #     sigm_cfg=[150] * 6,
        # )
        mgr = model_saving.ModelManager(model, '_data/AIN', opts.save)
        session = tf.Session()
        mgr.init_vars(session)
    # import code; code.interact(local=dict(**globals(), **locals()))
    # session.run(tf.assign(model.var_list[0], [
    #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0, 0.0],
    # ]))
    # session.run(tf.assign(model.var_list[1], [[0.0] * 11]))

    training = AINTraining(model, batch_size=500)
    session.run(training.init_op)

    ai_func = model_based_function(model, session)
    train_func_shortrange = adopt_handcrafted_function(handcrafted.short_range_attack)
    train_func_longrange = adopt_handcrafted_function(handcrafted.distance_attack)
    ai2_funcs = {
        'NN': noised(ai_func, 0.01),
        # 'MA': train_func_shortrange,
        # 'DA': train_func_longrange,
    }

    N_GAMES = 1000
    # N_GAMES = 200
    statstr = ''
    for i in range(N_GAMES):
        # ai1_func_name = random.choice(list(ai2_funcs.keys()))
        ai1_func_name = 'NN'
        # ai2_func_name = random.choice(list(ai2_funcs.keys()))
        ai2_func_name = 'NN'
        win, hp1, hp2 = run_one_game(
            ai2_funcs[ai1_func_name],
            ai2_funcs[ai2_func_name],
            winner_memory,
            loser_memory,
            frequency=4,
            bot_type=BotType.Raider,
            save_win=(ai2_func_name == 'MA'),
        )
        print('GAME#{}: {} {}-vs-{}, hp1={:.2f} hp2={:.2f}'.format(
            i, 'win' if win else 'lost',
            ai1_func_name,
            ai2_func_name,
            hp1, hp2,
        ))
        # if winner_memory.used_size >= training.batch_size and loser_memory.used_size >= training.batch_size:
        if loser_memory.used_size >= training.batch_size:
            # import pdb; pdb.set_trace()
            training.train_n_steps(session, winner_memory, loser_memory, 99, print_each_step=33)
        if i != 0 and i % 10 == 0 or i+1 == N_GAMES:
            if opts.save:
                mgr.save_vars(session)
                # winner_memory.save(winner_data_path)
                loser_memory.save(loser_data_path)
                print("Save model at step", mgr.step_counter)
    #     if True:
    #         metrics = training.evaluate_accuracy(session, winner_memory, max_batches=10)
    #         width = max(map(len, metrics.keys()))
    #         for prop in sorted(metrics.keys()):
    #             print('  {} : {}'.format(prop.ljust(width), metrics[prop]))
    #         statstr = '-'.join(
    #             '{:.0f}'.format(100 * metrics[p].get_overall_accuracy())
    #             for p in sorted(metrics.keys())
    #         )
    # print('STATS:', statstr)


if __name__ == '__main__':
    main()

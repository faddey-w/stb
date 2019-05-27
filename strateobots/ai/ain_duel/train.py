import argparse
import random
import shutil
import time
import contextlib
import numpy as np
import tensorflow as tf
import os
import itertools
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
        x = 0.2
        # x = random.random()
        orientation = 0.0
        # orientation = random.random() * 2 * pi
        if teamize_x:
            x = self.x_to_team_field(x)
            orientation += pi
        else:
            x *= self.engine.world_width
        bot_type = self.bot_type or random_bot_type()
        return self.engine.add_bot(
            bottype=bot_type,
            team=self.team,
            x=x,
            # y=self.engine.world_height * random.random(),
            y=self.engine.get_constants().world_height * 0.5,
            orientation=orientation,
            # tower_orientation=random.random() * 2 * pi,
            tower_orientation=0.0,
        )

    def initialize(self):
        self.bot = bot = self.create_bot(True)
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
    return random.choice([BotType.Raider, BotType.Heavy, BotType.Sniper])


def decode_prediction(
    prediction,
    ctl=None,
    ctl_list=None,
    move_alt=0,
    rotate_alt=0,
    tower_rotate_alt=0,
    shield_alt=0,
    fire_alt=0,
):
    assert prediction.ndim <= 2
    was_1d = prediction.ndim == 1
    if was_1d:
        ctl_list = [ctl or BotControl()]
        prediction = np.reshape(prediction, (1, prediction.shape[-1]))
    elif ctl_list is None:
        ctl_list = [BotControl() for _ in range(prediction.shape[0])]

    def get_action(pred_entry, actions, alternative):
        idx = sorted(range(len(actions)), key=pred_entry.__getitem__, reverse=True)[
            alternative
        ]
        return actions[idx]

    for i in range(prediction.shape[0]):
        ctl = ctl_list[i]
        pred = prediction[i]
        ctl.move = get_action(pred[0:3], (+1, 0, -1), move_alt)
        ctl.rotate = get_action(pred[3:6], (-1, 0, +1), rotate_alt)
        ctl.tower_rotate = get_action(pred[6:9], (-1, 0, +1), tower_rotate_alt)
        ctl.fire = get_action(pred[9:11], (True, False), fire_alt)
        ctl.shield = get_action(pred[11:13], (True, False), shield_alt)
    if was_1d:
        return ctl_list[0]
    else:
        return ctl_list


def model_based_function(model, session, alt_dict=None, exploration_feed=None):
    state_ph = tf.placeholder(tf.float32, [1, data.state2vec.vector_length])
    inference = model.apply(state_ph, with_exploration=exploration_feed is not None)
    if alt_dict is None:
        alt_dict = {}

    def function(engine, bot, enemy, ctl):
        bullet_b, bullet_e = find_bullets(engine, [bot, enemy])
        state = data.state2vec((bot, enemy, bullet_b, bullet_e))
        feeds = {state_ph: [state]}
        if exploration_feed is not None:
            feeds.update(exploration_feed)
        prediction = session.run(inference.action_prediction, feeds)[0]
        decode_prediction(prediction, ctl, **alt_dict)

    return function, inference


def adopt_handcrafted_function(ai_function):
    def function(engine, bot, enemy, ctl):
        ai_function(bot, enemy, ctl)

    return function


def run_one_game(
    function1,
    function2,
    memories,
    reward_func,
    reward_decay,
    frequency=1,
    self_play=False,
    data_dilution=1,
    **params
):
    ai1_factory = AINDuelAI.parametrize(function=function1, **params)
    ai2_factory = AINDuelAI.parametrize(function=function2, **params)

    max_ticks = 2500
    max_entries = int(max_ticks / frequency) + 1

    mem1 = replay.ReplayMemory(
        max_entries, data.state2vec.vector_length, data.action2vec.vector_length, 1
    )
    mem2 = replay.ReplayMemory(
        max_entries, data.state2vec.vector_length, data.action2vec.vector_length, 1
    )

    engine = StbEngine(ai1_factory, ai2_factory, max_ticks, 0)

    state1bef, state2bef = make_states(engine)
    while not engine.is_finished:

        action1 = make_action(engine.ai1)
        action2 = make_action(engine.ai2)

        for _ in range(frequency):
            engine.tick()

        state1aft, state2aft = make_states(engine)

        rew1 = reward_func(state1bef, action1, state1aft)
        rew2 = reward_func(state2bef, action2, state2aft)

        mem1.put_entry(state1bef, action1, [rew1])
        mem2.put_entry(state2bef, action2, [rew2])

        state1bef = state1aft
        state2bef = state2aft

    def put_memory_with_reward(mem):
        states, actions, rewards = mem.get_last_entries(mem.used_size)
        rew_cum = np.zeros_like(rewards)
        # total_damage = 1 - states[-1, data.state2vec[1, 'hp_ratio']]
        # rew_cum[-1] = rewards[-1] + total_damage
        rew_cum[-1] = rewards[-1]
        for i in range(1, rewards.shape[0]):
            rew_cum[-i - 1] = rewards[-i - 1] + reward_decay * rew_cum[-i]
        if data_dilution > 1:
            dilution_offset = random.choice(range(data_dilution))
            states = states[dilution_offset::data_dilution]
            actions = actions[dilution_offset::data_dilution]
            rew_cum = rew_cum[dilution_offset::data_dilution]
        for memory in memories:
            memory.put_many(states, actions, rew_cum)
        return rew_cum

    rwrd = put_memory_with_reward(mem1)
    if self_play and memories:
        put_memory_with_reward(mem2)

    return GameStats(
        engine.ai1.bot.hp_ratio, engine.ai2.bot.hp_ratio, np.sum(rwrd) / np.size(rwrd)
    )


class GameStats:
    def __init__(self, hp1, hp2, average_reward):
        self.hp1 = hp1
        self.hp2 = hp2
        self.win = hp1 > hp2
        self.average_reward = average_reward

    @classmethod
    def SummaryTensors(cls, session, logs_location):
        hp1 = tf.placeholder(tf.float32, [])
        hp2 = tf.placeholder(tf.float32, [])
        avg_rew = tf.placeholder(tf.float32, [])
        summaries = tf.summary.merge(
            [
                tf.summary.scalar("hp1", hp1),
                tf.summary.scalar("hp2", hp2),
                tf.summary.scalar("avg_rew", avg_rew),
            ]
        )
        writer = tf.summary.FileWriter(logs_location, session.graph)
        self = cls(hp1, hp2, avg_rew)
        self.summaries = summaries
        self.session = session
        self.summary_writer = writer
        return self

    def write_summaries(self, step, stats):
        session = getattr(self, "session")
        summaries = getattr(self, "summaries")
        summary_writer = getattr(self, "summary_writer")
        sumry = session.run(
            summaries,
            {
                self.hp1: stats.hp1,
                self.hp2: stats.hp2,
                self.average_reward: stats.average_reward,
            },
        )
        summary_writer.add_summary(sumry, step)


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
        return "{:.1f}%\t{:.2f}\t{}".format(
            100 * self.get_overall_accuracy(),
            self.get_overall_jaccard(),
            "\t".join(
                # '{}={:.2f}'.format(s, self.get_jaccard(s))
                "{}={:.2f}".format(s, self.get_accuracy(s))
                for s in self.symbols
            ),
        )


class AINTraining:
    def __init__(self, model, batch_size):
        self.batch_size = batch_size
        self.state_ph = tf.placeholder(
            tf.float32, [batch_size, data.state2vec.vector_length]
        )
        self.action_ph = tf.placeholder(
            tf.float32, [batch_size, data.action2vec.vector_length]
        )
        self.reward_ph = tf.placeholder(tf.float32, [batch_size, 1])
        self.reward_threshold_ph = tf.placeholder(tf.float32, [])

        self.model = model
        self.inference = model.apply(self.state_ph)

        act = self.inference.action_prediction
        act_diff = tf.abs(act - self.action_ph)
        self.act_same = 1 - act_diff
        eps = 0.001
        clip = 0.15
        # self.entropy = - self.act_same * (
        # act *= act_diff
        # punishment = tf.reduce_max(self.reward_ph) - self.reward_ph
        # self.entropy = - punishment * tf.log(eps + (1-eps) * act_diff)
        # self.entropy = - (
        #     tf.maximum(0.0, self.reward_ph) * tf.log(eps + (1-eps) * self.act_same)
        #     +
        #     tf.maximum(0.0, -self.reward_ph) * tf.log(eps + (1-eps) * act_diff)
        # )
        # self.entropy = - self.reward_ph * tf.log(eps + (1-eps) * act) * self.act_same
        # self.entropy = - self.reward_ph * tf.log(eps + (1-eps) * act_diff)

        # reward is negative!
        # so training will lead to avoiding too negative rewards
        reward = self.reward_ph - self.reward_threshold_ph
        clipped_act_same = tf.clip_by_value(self.act_same, clip, 1 - clip)
        # self.entropy = - reward * tf.log(eps + (1-eps) * self.act_same)
        self.entropy = -reward * tf.log(clipped_act_same)

        self.loss_vector = self.entropy
        # self.loss_vector = self.entropy * self.act_same
        self.loss = tf.reduce_mean(self.loss_vector)

        self.optimizer = tf.train.GradientDescentOptimizer(0.003)
        # self.optimizer = tf.train.AdamOptimizer(0.001)
        # self.optimizer = tf.train.RMSPropOptimizer(0.0001)
        self.train_op = self.optimizer.minimize(self.loss, var_list=model.var_list)

        self.avg_reward = tf.reduce_mean(self.reward_ph)

        self.init_op = tf.variables_initializer(self.optimizer.variables())

    def train_n_steps(
        self, session, memory, n_steps, print_each_step=10, reward_threshold=0
    ):
        loss_avg = util.Average()
        rew_avg = util.Average()
        started = time.time()
        for i in range(1, 1 + n_steps):
            (_, loss, rew) = self.run_on_sample(
                [self.train_op, self.loss, self.avg_reward],
                memory,
                session,
                reward_threshold=reward_threshold,
            )
            loss_avg.add(loss)
            rew_avg.add(rew)
        log_entry = "loss={:.4f}; rew={:.4f}; time={:.2f}sec".format(
            loss_avg.get(), rew_avg.get(), time.time() - started
        )
        return log_entry

    def run_on_sample(self, tensors, memory, session, reward_threshold=0):
        states, actions, cum_rewards = memory.get_random_sample(self.batch_size)
        return session.run(
            tensors,
            {
                self.state_ph: states,
                self.reward_ph: cum_rewards,
                self.action_ph: actions,
                self.reward_threshold_ph: reward_threshold,
            },
        )

    # def evaluate_accuracy(self, session, winner_memory, max_batches=50):
    #     bsize = self.batch_size
    #     n_batches = min(max_batches, int(winner_memory.used_size / bsize))
    #     states, actions = winner_memory.get_random_sample(bsize * n_batches)
    #     assert states.shape[0] == n_batches * bsize
    #     metrics = dict(
    #         move=AccuracyMetrics([-1, 0, +1]),
    #         rotate=AccuracyMetrics([-1, 0, +1]),
    #         tower_rotate=AccuracyMetrics([-1, 0, +1]),
    #         fire=AccuracyMetrics([0, 1]),
    #         shield=AccuracyMetrics([0, 1]),
    #     )
    #     example_ctls = [BotControl() for _ in range(bsize)]
    #     prediction_ctls = [BotControl() for _ in range(bsize)]
    #     for i in range(n_batches):
    #         slc = slice(i*bsize, (i+1)*bsize)
    #         predictions = session.run(
    #             self.win_inference.action_prediction,
    #             {self.win_state_ph: states[slc]},
    #         )
    #         decode_prediction(predictions, ctl_list=prediction_ctls)
    #         decode_prediction(actions[slc], ctl_list=example_ctls)
    #         for ctl_e, ctl_p in zip(example_ctls, prediction_ctls):
    #             for prop, metric in metrics.items():
    #                 metric.add(getattr(ctl_e, prop), getattr(ctl_p, prop))
    #     return metrics

    @contextlib.contextmanager
    def evaluate_policy_change(self, session, memory, n_batches=10):
        bs = self.batch_size
        n_batches = min(n_batches, memory.used_size // bs)
        if n_batches > 0:
            holder = [None, None]
            state, action, reward = memory.get_random_sample(n_batches * bs)
            before = self.compute_dispersion(session, state, action)
            yield holder
            after = self.compute_dispersion(session, state, action)
            holder[:] = before, after
        else:
            yield [0.0, 0.0]

    def compute_dispersion(self, session, states, actions):
        changes_sum = 0
        bs = self.batch_size
        n_batches = states.shape[0] // bs
        for i in range(n_batches):
            action_same = session.run(
                self.act_same,
                {
                    self.state_ph: states[i * bs : (i + 1) * bs],
                    self.action_ph: actions[i * bs : (i + 1) * bs],
                },
            )
            changes_sum += 1 - np.sum(action_same) / np.size(action_same)
        return changes_sum / n_batches


def find_bullets(engine, bots):
    bullets = {bullet.origin_id: bullet for bullet in engine.iter_bullets()}
    return [
        bullets.get(bot.id, BulletModel(None, None, 0, bot.x, bot.y, 0)) for bot in bots
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


def memory_keyfunc(state, action, reward):
    """state and action are 1d vectors"""
    # x0 = state[data.state2vec[0, 'x']]
    # y0 = state[data.state2vec[0, 'y']]
    # x1 = state[data.state2vec[1, 'x']]
    # y1 = state[data.state2vec[1, 'y']]
    # x_flag = x0 > x1
    # y_flag = y0 > y1
    # d_flag = ((x1-x0)**2 + (y1-y0)**2) > 200 ** 2
    # return tuple([*map(int, action), x_flag, y_flag, d_flag])
    # return tuple([*map(int, action), d_flag])
    e_hp_idx = data.state2vec[1, "hp_ratio"]
    e_hp = state[e_hp_idx]
    damage_key = int(np.sqrt(e_hp) * 10)
    reward_key = int(reward)
    return damage_key, reward_key


def reward_function(state_before, action, state_after):
    b_hp_idx = data.state2vec[0, "hp_ratio"]
    e_hp_idx = data.state2vec[1, "hp_ratio"]
    # return state_after[..., b_hp_idx] - state_after[..., e_hp_idx]
    b_hp_delta = state_after[..., b_hp_idx] - state_before[..., b_hp_idx]
    e_hp_delta = state_after[..., e_hp_idx] - state_before[..., e_hp_idx]
    return 100 * (b_hp_delta - e_hp_delta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--no-save", action="store_false", dest="save")
    parser.add_argument("--restart", action="store_true", dest="restart")
    parser.add_argument("--debug", action="store_true", dest="debug")
    parser.add_argument("--explore-strength", type=float, required=True)
    parser.add_argument("--alt-prob", type=float, default=-1)
    opts = parser.parse_args()

    # memory = replay.BalancedMemory(
    #     keyfunc=memory_keyfunc,
    #     cap_per_class=1000,
    #     vector_sizes=[data.state2vec.vector_length, data.action2vec.vector_length, 1]
    # )
    memory = replay.ReplayMemory(
        100000,
        # 10000,
        # 500,
        data.state2vec.vector_length,
        data.action2vec.vector_length,
        1,
    )
    eval_memory = replay.ReplayMemory(
        100000, data.state2vec.vector_length, data.action2vec.vector_length, 1
    )
    data_path = "_data/AIN-data/" + opts.name
    eval_data_path = data_path + "/eval"
    model_path = "_data/AIN/" + opts.name
    logs_path = "_data/AIN-logs/{}/".format(opts.name)
    if opts.restart:
        shutil.rmtree(model_path)
        shutil.rmtree(data_path)
        shutil.rmtree(logs_path)
    try:
        if isinstance(memory, replay.BalancedMemory):
            memory.load(data_path, eval)
        else:
            memory.load(data_path)
        eval_memory.load(eval_data_path)
    except:
        pass

    REWARD_THRESHOLD_SIGMA = 0
    enable_reward_threshold = True

    try:
        mgr = model_saving.ModelManager.load_existing_model(model_path)
        session = tf.Session()
        mgr.load_vars(session)
        model = mgr.model
        print("model loaded at step ", mgr.step_counter)
    except:
        print("NEW model")
        os.makedirs(data_path, exist_ok=False)
        os.makedirs(eval_data_path, exist_ok=False)
        os.makedirs(model_path, exist_ok=False)
        tf.reset_default_graph()
        # model = modellib.vec2d_fc_v2.ActionInferenceModel(
        #     move_vec=[(5, 5)] * 3,
        #     move_fc=[],
        #     rotate_vec=[(5, 5)] * 3,
        #     rotate_fc=[],
        #     tower_rotate_vec=[(5, 5)] * 3,
        #     tower_rotate_fc=[],
        #     shield_vec=[(5, 5)] * 3,
        #     shield_fc=[10] * 1,
        #     fire_vec=[(5, 5)] * 3,
        #     fire_fc=[10] * 2,
        # )
        # model = modellib.handcrafted.ActionInferenceModel()
        # model = modellib.simple.ActionInferenceModel(
        #     cfg=[60] * 6,
        # )
        # model = modellib.classic.ActionInferenceModel(
        #     layer_sizes=[50, 30],
        #     angle_sections=10,
        # )
        model = modellib.classic_v2.ActionInferenceModel(
            move=dict(layer_sizes=[15, 5], n_angles=30),
            rotate=dict(layer_sizes=[15], n_angles=30),
            tower_rotate=dict(layer_sizes=[15], n_angles=30),
            fire=dict(layer_sizes=[15, 5], n_angles=30),
            shield=dict(layer_sizes=[5, 5], n_angles=30),
        )
        # model = modellib.residual.ActionInferenceModel(
        #     res_cfg=[(50, 100)] * 6,
        #     sigm_cfg=[150] * 6,
        # )
        mgr = model_saving.ModelManager(model, model_path, opts.save)
        session = tf.Session()
        mgr.init_vars(session)

    os.makedirs(logs_path, exist_ok=True)
    stats_writer = GameStats.SummaryTensors(session, logs_path)

    training = AINTraining(model, batch_size=1000)
    session.run(training.init_op)

    train_func_shortrange = adopt_handcrafted_function(handcrafted.short_range_attack)
    train_func_longrange = adopt_handcrafted_function(handcrafted.distance_attack)
    ai2_funcs = {
        # 'NN': noised(ai_func, 0.01),
        "NN": model_based_function(model, session)[0],
        "MA": train_func_shortrange,
        "DA": train_func_longrange,
    }

    _alt_keys = [slot + "_alt" for slot in BotControl.__slots__]
    alternatives = [
        dict(zip(_alt_keys, alt_indices))
        for alt_indices in itertools.product(*map(range, [3, 3, 3, 2, 2]))
    ]
    normal_ai_func = model_based_function(model, session)[0]
    altdict = {}
    alt_ai_func = model_based_function(model, session, altdict)[0]
    explore_feed = {}
    explore_ai_func, explore_inference = model_based_function(
        model, session, exploration_feed=explore_feed
    )

    N_GAMES = 30000
    # N_GAMES = 200
    statstr = ""
    avg_dmg = 0.0
    reward_threshold = 0

    for i in range(N_GAMES):
        if opts.debug:
            memory.trunc(0)

        if i % 5 > 0:
            if random.random() < opts.alt_prob:
                altdict.clear()
                idx = random.choice(range(len(alternatives)))
                altdict.update(alternatives[idx])
                ai_func = alt_ai_func
                mode_name = "alt_{:02d}".format(idx)
            else:
                explore_feed.clear()
                explore_feed.update(
                    model.generate_exploration_feed(
                        explore_inference, opts.explore_strength
                    )
                )
                ai_func = explore_ai_func
                mode_name = "explor"
            evaluation = False
        else:
            evaluation = True
            mode_name = " EVAL "
            ai_func = normal_ai_func

        stats = run_one_game(
            ai_func,
            ai2_funcs["MA"],
            [memory, eval_memory] if evaluation else [memory],
            reward_decay=0.97,
            reward_func=reward_function,
            frequency=2,
            bot_type=BotType.Raider,
            self_play=False,
            data_dilution=1,
        )
        mgr.step_counter += 1
        if evaluation:
            stats_writer.write_summaries(mgr.step_counter, stats)
            avg_dmg = 0.9 * avg_dmg + 0.1 * (1 - stats.hp2)
        logentry = "GAME#{}: {}{}, hp1={:.2f} hp2={:.2f}".format(
            mgr.step_counter,
            " win" if stats.win else "lost",
            " ({})".format(mode_name),
            stats.hp1,
            stats.hp2,
        )

        if enable_reward_threshold and i % 20 == 0:
            _, _, all_rewards = memory.get_last_entries(memory.used_size)
            n_entries = np.size(all_rewards)
            mean = np.sum(all_rewards) / n_entries
            square_mean = np.sum(np.square(all_rewards)) / n_entries
            sigma = np.sqrt(square_mean - np.square(mean))
            reward_threshold = mean + REWARD_THRESHOLD_SIGMA * sigma
            higher_ratio = np.sum(all_rewards > reward_threshold) / n_entries
            print(
                "Set reward threshold: {} ({:.1f}%)".format(
                    reward_threshold, higher_ratio * 100
                )
            )

        if opts.debug and stats.hp2 < 0.80:
            import code

            code.interact(local=dict(**globals(), **locals()))

        can_train = memory.used_size >= 5 * training.batch_size

        if can_train:
            # import pdb; pdb.set_trace()
            with training.evaluate_policy_change(session, eval_memory) as change:
                train_log = training.train_n_steps(
                    session, memory, 25, reward_threshold=reward_threshold
                )
            before, after = change
            sign = ">" if before > after else "<"
            dispersion_log = "D: {:.5f} -{} {:.5f}".format(before, sign, after)
            logentry = "{}   {}   {}".format(logentry, train_log, dispersion_log)
        if evaluation:
            logentry += ",  avg_dmg={:.3f}".format(avg_dmg)

        print(logentry)

        if can_train and i != 0 and i % 10 == 0 or i + 1 == N_GAMES:
            if opts.save:
                mgr.save_vars(session, inc_step=False)
                memory.save(data_path)
                eval_memory.save(eval_data_path)
                print("Save model at step", mgr.step_counter)


if __name__ == "__main__":
    main()

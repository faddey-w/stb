import argparse
import itertools
import datetime
import collections
import random
import numpy as np
import tensorflow as tf
import os
import weakref
import logging
from math import pi, sin, cos
from strateobots.engine import BotType, StbEngine, dist_points, BotControl
from .._base import DuelAI
from ..simple_duel import RaiderVsSniper, SniperVsRaider
from ..lib.data import state2vec, action2vec, bot2vec
from ..lib import layers, stable, replay, handcrafted
from ..lib.handcrafted import get_angle, StatefulChaotic


tf.reset_default_graph()
log = logging.getLogger(__name__)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


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
    if team == engine.teams[0]:
        ai = RunAI(team, engine, RunAI.Shared.instance)
        ai.bot_type = BotType.Raider
        return ai
    else:
        # return PassiveAI.parametrize(side=+1)(team, engine)
        # return TrainerAI(team, engine)
        return ShootTrainerAI.parametrize(
            orientation=RunAI.Shared.instance.orientation,
            bot_type=BotType.Raider
        )(team, engine)


class RunAI(BaseDQNDualAI):

    class Shared:
        instance = None

        def __init__(self, orientation):
            self.state_ph = tf.placeholder(tf.float32, [1, state2vec.vector_length])
            self.model = QualityFunction.Model(
                os.path.join(REPO_ROOT, '_data/2018-06-26_01-29-44/model/')
            )
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


class ShootTrainerAI(PassiveAI):

    def tick(self):
        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return

        angl = get_angle(bot, enemy)
        rot = handcrafted.navigate_shortest(bot, angl, with_gun=True)
        ctl.tower_rotate = rot
        ctl.fire = handcrafted.should_fire(bot, enemy)
        if ctl.fire:
            ctl.rotate = handcrafted.navigate_shortest(bot, angl, with_gun=False)
        else:
            ctl.rotate = rot



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'play'], default='train')
    parser.add_argument('--no-save', action='store_false', dest='save')
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--max-games', default=None, type=int)
    opts = parser.parse_args()

    if opts.save_dir is None:
        run_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        opts.save_dir = os.path.join(REPO_ROOT, '_data', run_name)
    logs_dir = os.path.join(opts.save_dir, 'logs')
    model_dir = os.path.join(opts.save_dir, 'model', '')
    replay_dir = os.path.join(opts.save_dir, 'replay')

    logging.basicConfig(level=logging.INFO)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)

    model = QualityFunction.Model(model_dir)
    sess = get_session()
    try:
        model.load_vars()
    except:
        model.init_vars()
    replay_memory = replay.ReplayMemory(
        capacity=20000,
        state_size=state2vec.vector_length,
        action_size=action2vec.vector_length
    )
    rl = ReinforcementLearning(
        model,
        batch_size=80,
        n_games=1,
        reward_prediction=0.05,
        select_random_prob_decrease=0.005,
        select_random_min_prob=0.03,
        self_play=False,
    )
    sess.run(rl.init_op)
    try:
        replay_memory.load(replay_dir)
        log.info('replay memory buffer loaded from %s', replay_dir)
    except FileNotFoundError:
        log.info('collecting new replay memory buffer')

    if opts.action == 'play':
        import code
        with sess.as_default():
            code.interact(local=dict(globals(), **locals()))
    if opts.action == 'train':
        try:
            i = 0
            while True:
                i += 1
                rl.run(
                    frameskip=2,
                    max_ticks=2000,
                    world_size=1000,
                    replay_memory=replay_memory,
                    log_root_dir=logs_dir if opts.save else None,
                )
                if opts.save:
                    model.save_vars()
                    replay_memory.save(replay_dir)
                if opts.max_games is not None and i >= opts.max_games:
                    break
        except KeyboardInterrupt:
            if opts.save:
                model.save_vars()
                replay_memory.save(replay_dir)


if __name__ == '__main__':
    main()

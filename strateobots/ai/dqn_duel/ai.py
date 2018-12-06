import logging
import os
import random
from math import pi, sin, cos, atan2

import tensorflow as tf

from strateobots import REPO_ROOT
from strateobots.engine import BotType, BulletModel
from .._base import DuelAI
from ..lib import model_saving
from ..lib.data import state2vec, action2vec
from ..lib.handcrafted import StatefulChaotic
from ..lib.handcrafted import short_range_attack, distance_attack, turret_behavior

log = logging.getLogger(__name__)


class DQNDuelAI(DuelAI):

    bot_type = None
    modes = ()
    trainer_function = None

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
            y=self.engine.get_constants().world_height * random.random(),
            orientation=random.random() * 2 * pi,
            tower_orientation=random.random() * 2 * pi,
        )

    def initialize(self):
        self.bot = bot = self.create_bot(False)
        self.ctl = self.engine.get_control(bot)
        for mode in self.modes:
            mode.on_init(bot, self.team, self.engine)

    def tick(self):
        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return
        if self.trainer_function is not None:
            self.trainer_function(bot, enemy, ctl)
        for mode in self.modes:
            mode.on_runtime(bot, enemy, ctl, self.engine)


class Mode:

    def reset(self):
        pass

    def on_init(self, bot, team, engine):
        pass

    def on_runtime(self, bot, enemy, ctl, engine):
        pass


class PassiveMode(Mode):

    pass


class ChaoticMode(Mode):

    chaos = None

    def reset(self):
        self.chaos = None

    def on_runtime(self, bot, enemy, ctl, engine):
        if self.chaos is None:
            self.chaos = StatefulChaotic(bot, ctl, engine,
                                         shield_period=(1000, 0))
        self.chaos.run()


class NotMovingMode(Mode):

    def on_runtime(self, bot, enemy, ctl, engine):
        ctl.move = 0


class NotBodyRotatingMode(Mode):

    def on_runtime(self, bot, enemy, ctl, engine):
        ctl.rotate = 0


class NotTowerRotatingMode(Mode):

    def on_runtime(self, bot, enemy, ctl, engine):
        ctl.tower_rotate = 0


class NoShieldMode(Mode):

    def on_runtime(self, bot, enemy, ctl, engine):
        ctl.shield = False


class LocateAtCircleMode(Mode):

    def __init__(self, orientation=None, radius_ratio=0.05):
        self.configured_orientation = orientation
        self.orientation = orientation
        self.radius_ratio = radius_ratio

    def reset(self):
        if self.configured_orientation is None:
            self.orientation = random.random() * 2 * pi
        else:
            self.orientation = self.configured_orientation

    def on_init(self, bot, team, engine):
        _randomize_position(bot, engine)

        ori = self.orientation
        if team != engine.teams[0]:
            ori += pi
        r = self.radius_ratio
        bot.x = (0.5 + cos(ori) * r) * engine.get_constants().world_width
        bot.y = (0.5 + sin(ori) * r) * engine.get_constants().world_height


class BackToCenter(Mode):

    def on_init(self, bot, team, engine):
        x = bot.x
        y = bot.y
        cx = engine.get_constants().world_width / 2
        cy = engine.get_constants().world_height / 2
        angle = atan2(y-cy, x-cx)
        bot.orientation = angle


def AI(team, engine):
    if RunAI.Shared.instance is None:
        RunAI.Shared.instance = RunAI.Shared()
    if team == engine.teams[0]:
        return RunAI(team, engine, RunAI.Shared.instance)
    else:
        return DQNDuelAI.parametrize(
            modes=RunAI.Shared.instance.modes,
            bot_type=BotType.Raider,
            trainer_function=turret_behavior,
        )(team, engine)


class RunAI(DQNDuelAI):

    class Shared:
        instance = None  # type: RunAI.Shared
        model_path = os.path.join(REPO_ROOT, '_data/DQN/2018-09-01_00-56-31/model/')
        self_play = False
        bot_type = BotType.Raider
        modes = [
            NotMovingMode(),
            LocateAtCircleMode(),
            NoShieldMode(),
            NotBodyRotatingMode(),
            BackToCenter(),
        ]

        def __init__(self):
            self.state_ph = tf.placeholder(tf.float32, [1, state2vec.vector_length])
            self.model_mgr = model_saving.ModelManager.load_existing_model(self.model_path)
            self.session = tf.Session()
            self.model_mgr.load_vars(self.session)
            self.selector = self.model_mgr.model.make_selector(
                self.state_ph,
            )
            for mode in self.modes:
                mode.reset()

    def __init__(self, team, engine, ai_shared):
        super(RunAI, self).__init__(team, engine)
        self.shared = ai_shared  # type: RunAI.Shared
        self.modes = self.shared.modes
        self.bot_type = self.shared.bot_type

    def tick(self):
        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return
        b1, b2 = find_bullets(self.engine, [bot, enemy])
        st = state2vec([bot, enemy, b1, b2])
        action = self.shared.selector.call({self.shared.state_ph: [st]})
        action2vec.restore(action[0], ctl)
        super(RunAI, self).tick()


def _randomize_position(bot, engine):
    bot.x = random.random() * engine.get_constants().world_width
    bot.y = random.random() * engine.get_constants().world_height
    bot.orientation = random.random() * 2 * pi - pi
    bot.tower_orientation = random.random() * 2 * pi - pi


def random_bot_type():
    return random.choice([
        BotType.Raider,
        BotType.Heavy,
        BotType.Sniper,
    ])


def find_bullets(engine, bots):
    bullets = {
        bullet.origin_id: bullet
        for bullet in engine.iter_bullets()
    }
    return [
        bullets.get(bot.id, BulletModel(None, None, 0, bot.x, bot.y, 0))
        for bot in bots
    ]


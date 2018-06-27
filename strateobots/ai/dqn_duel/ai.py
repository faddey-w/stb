import logging
import os
import random
from math import pi, sin, cos

import tensorflow as tf

from strateobots.engine import BotType
from .core import SelectAction, get_session
from .train import REPO_ROOT
from .._base import DuelAI
from ..lib import model_saving
from ..lib.data import state2vec, action2vec
from ..lib.handcrafted import StatefulChaotic
from ..lib.handcrafted import short_range_attack, distance_attack, turret_behavior

log = logging.getLogger(__name__)


class DQNDuelAI(DuelAI):

    bot_type = None
    modes = ()

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
        for mode in self.modes:
            mode.on_init(bot, self.team, self.engine)

    def tick(self):
        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return
        for mode in self.modes:
            mode.on_runtime(bot, enemy, ctl, self.engine)


class Mode:

    def on_init(self, bot, team, engine):
        pass

    def on_runtime(self, bot, enemy, ctl, engine):
        pass


class PassiveMode(Mode):

    pass


class TrainerMode(Mode):

    def __init__(self, algorithms=(distance_attack, short_range_attack)):
        self.algorithms = list(algorithms)
        self.selected_algorithm = None

    def on_init(self, bot, team, engine):
        self.selected_algorithm = random.choice(self.algorithms)
        log.info("selected trainer algorithm: %s", self.selected_algorithm)

    def on_runtime(self, bot, enemy, ctl, engine):
        self.selected_algorithm(bot, enemy, ctl)


class ChaoticMode(Mode):

    chaos = None

    def on_runtime(self, bot, enemy, ctl, engine):
        if self.chaos is None:
            self.chaos = StatefulChaotic(bot, ctl, engine,
                                         shield_period=(1000, 0))
        self.chaos.run()


class NotMovingMode(Mode):

    def on_runtime(self, bot, enemy, ctl, engine):
        ctl.move = 0


class NotRotatingMode(Mode):

    def on_runtime(self, bot, enemy, ctl, engine):
        ctl.rotate = 0
        ctl.tower_rotate = 0


class NoShieldMode(Mode):

    def on_runtime(self, bot, enemy, ctl, engine):
        ctl.shield = False


class LocateAtCircleMode(Mode):

    def __init__(self, orientation=None, radius_ratio=0.05):
        if orientation is None:
            orientation = random.random() * 2 * pi
        self.orientation = orientation
        self.radius_ratio = radius_ratio

    def on_init(self, bot, team, engine):
        _randomize_position(bot, engine)

        ori = self.orientation
        if team != engine.teams[0]:
            ori += pi
        r = self.radius_ratio
        bot.x = (0.5 + cos(ori) * r) * engine.world_width
        bot.y = (0.5 + sin(ori) * r) * engine.world_height


def AI(team, engine):
    if RunAI.Shared.instance is None:
        RunAI.Shared.instance = RunAI.Shared()
    if team == engine.teams[0]:
        return RunAI(team, engine, RunAI.Shared.instance)
    else:
        all_modes = [RunAI.Shared.instance.ai2_mode, *RunAI.Shared.instance.modes]
        return DQNDuelAI.parametrize(
            modes=all_modes,
            bot_type=BotType.Raider
        )(team, engine)


class RunAI(DQNDuelAI):

    class Shared:
        instance = None  # type: RunAI.Shared
        model_path = os.path.join(REPO_ROOT, '_data/2018-06-27_21-18-54/model/')
        self_play = False
        bot_type = BotType.Raider
        ai2_mode = TrainerMode([turret_behavior])

        @staticmethod
        def _modes():
            return [
                NotMovingMode(),
                LocateAtCircleMode(),
                NoShieldMode(),
            ]

        def __init__(self):
            self.state_ph = tf.placeholder(tf.float32, [1, state2vec.vector_length])
            self.model_mgr = model_saving.ModelManager.load_existing_model(self.model_path)
            self.model_mgr.load_vars(get_session())
            self.selector = SelectAction(
                self.model_mgr.model,
                self.model_mgr.model.QualityFunction,
                self.state_ph,
            )
            self.modes = self._modes()

    def __init__(self, team, engine, ai_shared):
        super(RunAI, self).__init__(team, engine)
        self.shared = ai_shared  # type: RunAI.Shared
        self.modes = self.shared.modes
        self.bot_type = self.shared.bot_type

    def tick(self):
        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return
        st = state2vec([bot, enemy])
        action = self.shared.selector.call({self.shared.state_ph: [st]})
        action2vec.restore(action[0], ctl)
        super(RunAI, self).tick()


def _randomize_position(bot, engine):
    bot.x = random.random() * engine.world_width
    bot.y = random.random() * engine.world_height
    bot.orientation = random.random() * 2 * pi - pi
    bot.tower_orientation = random.random() * 2 * pi - pi


def random_bot_type():
    return random.choice([
        BotType.Raider,
        BotType.Heavy,
        BotType.Sniper,
    ])


from strateobots.engine import BotType
from math import pi
from ._base import DuelAI
from .lib import handcrafted


def AI(team, engine):
    if team == engine.teams[0]:
        return SniperVsRaider(team, engine)
        # return RaiderVsSniper(team, engine)
    else:
        return SniperVsRaider(team, engine)


class SimpleDuelAI(DuelAI):

    bot_type = None  # type: BotType

    def _initialize(self, dx):
        if self.team == self.engine.teams[0]:
            x = self.engine.world_width * dx
            a = 0
        else:
            x = self.engine.world_width * (1 - dx)
            a = pi
        self.bot = self.engine.add_bot(
            bottype=self.bot_type,
            team=self.team,
            x=x,
            y=self.engine.world_height * 0.5,
            orientation=a,
            tower_orientation=0
        )


class RaiderVsSniper(SimpleDuelAI):

    bot_type = BotType.Raider

    def initialize(self):
        self._initialize(0.05)

    def tick(self):
        bot, enemy, ctl = self._get_bots()
        handcrafted.short_range_attack(bot, enemy, ctl)


class SniperVsRaider(SimpleDuelAI):

    bot_type = BotType.Sniper

    def initialize(self):
        self._initialize(0.3)

    def tick(self):
        bot, enemy, ctl = self._get_bots()
        handcrafted.distance_attack(bot, enemy, ctl)

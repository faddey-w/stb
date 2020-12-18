import math
import random
from dataclasses import dataclass
from typing import Collection
from stb.engine import StbEngine
from stb.models import BotType


@dataclass
class DuelInitializer:
    type1: BotType
    type2: BotType
    distance_ratio: float = 0.7

    def __call__(self, engine: StbEngine):
        y = engine.get_constants().world_height / 2
        x1 = (1 - self.distance_ratio) * engine.get_constants().world_width / 2
        x2 = (1 + self.distance_ratio) * engine.get_constants().world_width / 2
        engine.add_bot(
            bottype=self.type1,
            team=engine.team1,
            x=x1,
            y=y,
            orientation=0,
            tower_orientation=0,
        )
        engine.add_bot(
            bottype=self.type2,
            team=engine.team2,
            x=x2,
            y=y,
            orientation=math.pi,
            tower_orientation=0,
        )


@dataclass
class RandomInitializer:
    team1_types: Collection[BotType]
    team2_types: Collection[BotType]

    def __call__(self, engine: StbEngine):
        teamtypes = []
        teamtypes.extend((engine.team1, typ) for typ in self.team1_types)
        teamtypes.extend((engine.team2, typ) for typ in self.team2_types)

        for team, bottype in teamtypes:
            engine.add_bot(
                bottype=bottype,
                team=team,
                x=random.random() * engine.get_constants().world_width,
                y=random.random() * engine.get_constants().world_height,
                orientation=random.random() * 2 * math.pi,
                tower_orientation=random.random() * 2 * math.pi,
            )


@dataclass
class RandomSidedInitializer:
    team1_types: Collection[BotType]
    team2_types: Collection[BotType]

    def __call__(self, engine: StbEngine):
        for bottype in self.team1_types:
            engine.add_bot(
                bottype=bottype,
                team=engine.team1,
                x=(0.4 * random.random()) * engine.get_constants().world_width,
                y=random.random() * engine.get_constants().world_height,
                orientation=0,
                tower_orientation=0,
            )

        for bottype in self.team2_types:
            engine.add_bot(
                bottype=bottype,
                team=engine.team2,
                x=(1 - 0.4 * random.random()) * engine.get_constants().world_width,
                y=random.random() * engine.get_constants().world_height,
                orientation=-math.pi,
                tower_orientation=0,
            )

import math
import random


def duel_bot_initializer(type1, type2, distance_ratio):
    def initialize_bots_for_duel(engine):
        y = engine.world_height / 2
        x1 = (1 - distance_ratio) * engine.world_width / 2
        x2 = (1 + distance_ratio) * engine.world_width / 2
        engine.add_bot(
            bottype=type1,
            team=engine.team1,
            x=x1,
            y=y,
            orientation=0,
            tower_orientation=0
        )
        engine.add_bot(
            bottype=type2,
            team=engine.team2,
            x=x2,
            y=y,
            orientation=math.pi,
            tower_orientation=0
        )
    return initialize_bots_for_duel


def random_bot_initializer(team1_types, team2_types):
    def initialize_bots_randomly(engine):
        teamtypes = []
        teamtypes.extend((engine.team1, typ) for typ in team1_types)
        teamtypes.extend((engine.team2, typ) for typ in team2_types)

        for team, bottype in teamtypes:
            engine.add_bot(
                bottype=bottype,
                team=team,
                x=random.random() * engine.world_width,
                y=random.random() * engine.world_height,
                orientation=random.random() * 2 * math.pi,
                tower_orientation=random.random() * 2 * math.pi,
            )
    return initialize_bots_randomly

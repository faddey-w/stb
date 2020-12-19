from stb.game import StbGame
from stb.engine import StbEngine
from stb.models import BotType
from stb.bot_initializers import RandomInitializer
from stb.ai.simple_team import TeamCoordinatorV1


def generate_replay():
    lineup = ([BotType.Raider] * 5) + [BotType.Heavy] + ([BotType.Sniper] * 2)
    game = StbGame(
        StbEngine(2000, wait_after_win_ticks=1),
        TeamCoordinatorV1(),
        TeamCoordinatorV1(),
        RandomInitializer(lineup, lineup),
    )
    game.play()
    return game.replay

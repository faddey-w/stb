from stb.game import StbGame
from stb.engine import StbEngine
from stb.models import BotType
from stb.bot_initializers import RandomInitializer, RandomSidedInitializer
from stb.ai.simple_team import TeamCoordinatorV1


class ReplayGenerator:
    def __init__(self, n_raiders=1, n_heavy=0, n_snipers=0, max_ticks=2000, init_mode="random"):
        lineup = (
            ([BotType.Raider] * n_raiders)
            + ([BotType.Heavy] * n_heavy)
            + ([BotType.Sniper] * n_snipers)
        )
        self.max_ticks = max_ticks
        if init_mode == "random":
            self.initializer = RandomInitializer(lineup, lineup)
        elif init_mode == "random-sided":
            self.initializer = RandomSidedInitializer(lineup, lineup)
        else:
            raise ValueError(init_mode)

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate()

    def generate(self):
        game = StbGame(
            StbEngine(self.max_ticks, wait_after_win_ticks=1),
            TeamCoordinatorV1(),
            TeamCoordinatorV1(),
            self.initializer,
        )
        game.play()
        return game.replay


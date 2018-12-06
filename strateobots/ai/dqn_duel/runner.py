import random
import math
from .._base import DuelAI
from strateobots.ai.lib.data import state2vec, action2vec
from strateobots.ai.lib.util import find_bullets
from strateobots.engine import BotType, StbEngine, BotControl


def run_one_game(replay_memory, ai1_func, ai2_func, frames_per_action=1,
                 max_ticks=1000, world_size=300, report=None, remember_for_2=False):

    ai1_cls = DQNTrainingAI.parametrize(bot_type=BotType.Raider, function=ai1_func)
    ai2_cls = DQNTrainingAI.parametrize(bot_type=BotType.Raider, function=ai2_func)

    engine = StbEngine(
        ai1_cls, ai2_cls,
        max_ticks,
        wait_after_win=0,
    )
    bot1, bot2 = engine.ai1.bot, engine.ai2.bot
    bullet1, bullet2 = find_bullets(engine, [bot1, bot2])
    state1_before = state2vec((bot1, bot2, bullet1, bullet2))
    state2_before = state2vec((bot2, bot1, bullet2, bullet1))

    while not engine.is_finished:
        engine.ai1.update_action(state1_before)
        engine.ai2.update_action(state2_before)

        for _ in range(frames_per_action):
            engine.tick()

        bullet1, bullet2 = find_bullets(engine, [bot1, bot2])
        action1 = action2vec(engine.ai1.ctl)
        action2 = action2vec(engine.ai2.ctl)
        state1_after = state2vec((bot1, bot2, bullet1, bullet2))
        state2_after = state2vec((bot2, bot1, bullet2, bullet1))

        replay_memory.put_entry(state1_before, action1, state1_after)
        if remember_for_2:
            replay_memory.put_entry(state2_before, action2, state2_after)

        state1_before = state1_after
        state2_before = state2_after

        if report:
            report(engine)


class DQNTrainingAI(DuelAI):

    bot_type = None
    function = None

    def create_bot(self, teamize_x=True):
        x = 0.2
        # x = random.random()
        orientation = 0.0
        # orientation = random.random() * 2 * pi
        if teamize_x:
            x = self.x_to_team_field(x)
            orientation += math.pi
        else:
            x *= self.engine.get_constants().world_width
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
        self._ctl = BotControl()

    def update_action(self, state_vector):
        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return
        if hasattr(self.function, 'set_state_vector'):
            self.function.set_state_vector(state_vector)
        self.function(bot, enemy, self._ctl, self.engine)

    def tick(self):
        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return
        ctl.move = self._ctl.move
        ctl.rotate = self._ctl.rotate
        ctl.tower_rotate = self._ctl.tower_rotate
        ctl.fire = self._ctl.fire
        ctl.shield = self._ctl.shield


def random_bot_type():
    return random.choice([
        BotType.Raider,
        BotType.Heavy,
        BotType.Sniper,
    ])

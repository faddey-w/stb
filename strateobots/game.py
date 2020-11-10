import inspect
from strateobots.engine import StbEngine
from strateobots.models import Action, BotControl


class StbGame:
    def __init__(self, engine: StbEngine, ai1: callable, ai2: callable, initializer: callable):
        self.engine = engine
        self.ai1 = ai1
        self.ai2 = ai2
        self.initializer = initializer
        self.replay = []
        self.initializer(self.engine)

    def play_one_tick(self):
        game_state = self.engine.serialize_state()
        game_state["controls"] = self.communicate_with_ais(game_state)
        self.apply_controls(game_state["controls"])
        self.replay.append(game_state)
        self.engine.tick()

    @property
    def is_finished(self):
        return self.engine.is_finished

    def play(self):
        while not self.is_finished:
            self.play_one_tick()

    def communicate_with_ais(self, game_state):
        visible_bots = self.engine.serialize_bots_visible_state()
        if not self.engine.win_condition_reached:
            control1_data = self._communicate_with_ai(
                self.ai1, game_state, visible_bots, self.engine.team1, self.engine.team2
            )
            control2_data = self._communicate_with_ai(
                self.ai2, game_state, visible_bots, self.engine.team2, self.engine.team1
            )
        else:
            control1_data = control2_data = []

        return {
            self.engine.team1: control1_data,
            self.engine.team2: control2_data,
        }

    def _communicate_with_ai(self, ai, game_state, visible_bots, own_team, opp_team):
        return ai(
            {
                "tick": self.engine.nticks,
                "friendly_bots": game_state["bots"][own_team],
                "enemy_bots": visible_bots[opp_team],
                "bullets": game_state["bullets"],
                "rays": game_state["rays"],
            }
        )

    def apply_controls(self, ai_controls):

        # this validates that AIs won't set controls other that for own alive bots:
        controls_by_team_bot_id = {
            (team, ctl_data["id"]): ctl_data
            for team, controls in ai_controls.items()
            for ctl_data in controls
        }
        controls_by_bot = {
            bot: controls_by_team_bot_id.get((bot.team, bot.id)) for bot in self.engine.iter_bots()
        }

        for bot, ctl_data in controls_by_bot.items():
            ctl = self.engine.get_control(bot)
            if ctl_data is not None:
                for attr in BotControl.FIELDS:
                    value = ctl_data.get(attr)
                    if value is not None:
                        setattr(ctl, attr, value)
            else:
                ctl.action = Action.IDLE
                ctl.move = 0
                ctl.rotate = 0
                ctl.tower_rotate = 0

    def get_metadata(self):
        ai1_type = self.ai1 if inspect.isfunction(self.ai1) else self.ai1.__class__
        ai2_type = self.ai2 if inspect.isfunction(self.ai2) else self.ai2.__class__
        initer = self.initializer
        init_type = initer if inspect.isfunction(initer) else initer.__class__
        metadata = dict(
            init_name=f"{init_type.__module__}.{init_type.__name__}",
            ai1_module=ai1_type.__module__,
            ai1_name=ai1_type.__name__,
            ai2_module=ai2_type.__module__,
            ai2_name=ai2_type.__name__,
            team1=str(self.engine.team1),
            team2=str(self.engine.team2),
        )
        if self.is_finished:
            metadata["nticks"] = self.engine.nticks
            if self.engine.win_condition_reached:
                metadata["winner"] = str(self.engine.get_any_nonloser_team())
            else:
                metadata["winner"] = None
        return metadata

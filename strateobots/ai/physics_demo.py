from math import pi, atan
from functools import partial
from strateobots.engine import BotType, Action
from . import base


class AIModule(base.AIModule):

    def __init__(self):
        self.controller = None
        self.simple_controller = None

    def list_ai_function_descriptions(self):
        return [
            ('Physics demo AI', 'full'),
            # ('Simple demo AI', 'simple'),
        ]

    def list_bot_initializers(self):
        return [
            ('Physics demo', self._bot_initializer),
            # ('Simple demo', self._bot_initializer_simple),
        ]

    def construct_ai_function(self, team, parameters):
        func = self._ai_function if parameters == 'full' else self._ai_function_simple
        return partial(func, team)

    def _bot_initializer(self, engine):
        self.controller = PhysicsDemoController(engine)
        self.controller.initialize()

    def _bot_initializer_simple(self, engine):
        self.simple_controller = SimpleDemoController(engine)
        self.simple_controller.initialize()

    def _ai_function(self, team, state):
        return self.controller.do_control(team, state['tick'])

    def _ai_function_simple(self, team, state):
        return self.simple_controller.do_control(team, state['tick'])


class _BaseSceneController:

    def __init__(self, engine):
        """
        :type engine: strateobots.engine.StbEngine
        """
        self.engine = engine
        self._triggers = {}

    def mkbot(self, bottype, team, x, y, orientation, tower_orientation=0, hp=1.0):
        # x = 100*x - 0
        # y = 100*y - 600
        x *= self.engine.get_constants().world_width / 10
        y *= self.engine.get_constants().world_height / 10
        return self.engine.add_bot(
            bottype=bottype,
            team=team,
            x=x,
            y=y,
            orientation=orientation,
            tower_orientation=tower_orientation,
            hp=bottype.max_hp * hp
        )

    def trigger(self, bot, sec, **attrs):
        tick = int(sec * self.engine.get_constants().ticks_per_sec)
        triglist = self._triggers.setdefault(bot.id, [])
        triglist.append((tick, dict(attrs), bot.team))

    def do_control(self, control_team, tick):
        result = []
        for bot_id, triglist in self._triggers.items():
            for trigger in triglist:
                t, attrs, team = trigger
                if t == tick and control_team == team:
                    result.append({'id': bot_id, **attrs})
        return result


class PhysicsDemoController(_BaseSceneController):

    def initialize(self):
        team1, team2 = self.engine.teams
        ahead, left, back, right = 0, pi/2, pi, -pi/2
        east, north, west, south = 0, pi/2, pi, -pi/2

        mkbot = self.mkbot
        trig = self.trigger

        # head-on collisions
        b1 = mkbot(BotType.Raider, team1, 1, 1, east)
        b2 = mkbot(BotType.Raider, team2, 5, 1, west, hp=0.2)
        trig(b1, 0, move=1)
        trig(b2, 0, move=1)

        # lateral collisions
        b1 = mkbot(BotType.Raider, team2, 1, 2, east, hp=0.25)
        mkbot(BotType.Heavy, team1, 5, 2, north)
        trig(b1, 0, move=1)

        # move by circle and rotate tower
        b1 = mkbot(BotType.Sniper, team1, 7, 1, east)
        trig(b1, 0, move=1, rotate=1, tower_rotate=-1, action=Action.FIRE)

        # heavy tank duel
        b1 = mkbot(BotType.Heavy, team1, 5, 3, east)
        b2 = mkbot(BotType.Heavy, team2, 7, 3, south, right)
        trig(b1, 0, action=Action.FIRE)
        trig(b2, 0, action=Action.FIRE)

        # laser mass kill
        mkbot(BotType.Raider, team2, 3.5, 4.0, west, hp=0.1)
        mkbot(BotType.Raider, team2, 4.0, 4.5, west, hp=0.1)
        mkbot(BotType.Raider, team2, 4.5, 5.0, west, hp=0.1)
        mkbot(BotType.Raider, team2, 3.5, 4.5, west, hp=0.1)
        mkbot(BotType.Raider, team2, 3.5, 5.0, west, hp=0.1)
        b = mkbot(BotType.Raider, team2, 4.0, 5.0, west, hp=0.1)
        trig(b, 0, action=Action.SHIELD_WARMUP)

        b1 = mkbot(BotType.Sniper, team1, 1, 4, east)
        scene_delay = 1
        rot_delay = 0.07
        when = atan(1 / 2) / BotType.Sniper.gun_rot_speed
        trig(b1, scene_delay, action=Action.FIRE)
        trig(b1, scene_delay + rot_delay, tower_rotate=1)
        trig(b1, scene_delay + rot_delay + 1 * when, tower_rotate=-1)
        trig(b1, scene_delay + rot_delay + 2 * when, tower_rotate=0)
        trig(b1, scene_delay + rot_delay + 2 * when + 0.1, action=Action.FIRE)
        trig(b, scene_delay + rot_delay + 3 * when, action=Action.IDLE)

        # raider firing
        mkbot(BotType.Sniper, team1, 2, 7, north)
        mkbot(BotType.Sniper, team1, 3, 7, north)
        mkbot(BotType.Sniper, team1, 4, 7, north)
        mkbot(BotType.Sniper, team1, 5, 7, north)
        mkbot(BotType.Sniper, team1, 6, 7, north)
        b1 = mkbot(BotType.Raider, team2, 1.0, 6, east, (ahead + left) / 2, hp=0.1)
        trig(b1, 0.0, move=1, action=Action.FIRE)
        b1 = mkbot(BotType.Raider, team2, 1.5, 6, east, (ahead + left) / 2, hp=0.1)
        trig(b1, 0.0, move=1, action=Action.FIRE)
        b1 = mkbot(BotType.Raider, team2, 2.0, 6, east, (ahead + left) / 2, hp=0.1)
        trig(b1, 0.0, move=1, action=Action.FIRE)
        b1 = mkbot(BotType.Heavy, team1, 7, 6, east)
        trig(b1, 0, action=Action.SHIELD_WARMUP)
        trig(b1, 6, action=Action.IDLE)

        # drifting
        b1 = mkbot(BotType.Raider, team1, 5, 8, west)
        b1.vx = b1.type.max_ahead_speed / 2
        b1.vy = b1.type.max_ahead_speed / 2
        trig(b1, 0, move=1)


class SimpleDemoController(_BaseSceneController):

    def initialize(self):
        team1, team2 = self.engine.teams

        mkbot = self.mkbot
        trig = self.trigger

        r = mkbot(BotType.Raider, team1, 1, 9, 7*pi/4)
        trig(r, 0, action=Action.SHIELD_WARMUP)

        l = mkbot(BotType.Sniper, team2, 3, 8, pi - atan(1/2))
        trig(l, BotType.Raider.shield_warmup_period, action=Action.FIRE)

        t = mkbot(BotType.Heavy, team1, 1, 7.5, pi/2 - pi/10, -pi/6)
        trig(t, BotType.Raider.shield_warmup_period + 0.1, action=Action.FIRE)

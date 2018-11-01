from math import pi, atan
from functools import partial
from strateobots.engine import BotType
from . import base


class AIModule(base.AIModule):

    def __init__(self):
        self.controller = None

    def list_ai_function_descriptions(self):
        return [
            ('Physics demo AI', None),
        ]

    def list_bot_initializers(self):
        return [
            ('Physics demo', self._bot_initializer),
        ]

    def construct_ai_function(self, team, parameters):
        return partial(self._ai_function, team)

    def _bot_initializer(self, engine):
        self.controller = PhysicsDemoController(engine)
        self.controller.initialize()

    def _ai_function(self, team, state):
        return self.controller.do_control(team)


class PhysicsDemoController:

    def __init__(self, engine):
        """
        :type team: int
        :type engine: strateobots.engine.StbEngine
        """
        self.engine = engine
        self._triggers = {}

    def initialize(self):
        team1, team2 = self.engine.teams
        ahead, left, back, right = 0, pi/2, pi, -pi/2
        east, north, west, south = 0, pi/2, pi, -pi/2

        def mkbot(bottype, team, x, y, orientation, tower_orientation=ahead, hp=1.0):
            x *= self.engine.world_width / 10
            y *= self.engine.world_height / 10
            return self.engine.add_bot(
                bottype=bottype,
                team=team,
                x=x,
                y=y,
                orientation=orientation,
                tower_orientation=tower_orientation,
                hp=bottype.max_hp * hp
            )

        def trig(bot, sec, **attrs):
            tick = int(sec * self.engine.ticks_per_sec)
            triglist = self._triggers.setdefault(bot.id, [])
            triglist.append((tick, dict(attrs), bot.team))

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
        trig(b1, 0, move=1, rotate=1, tower_rotate=-1, fire=True)

        # heavy tank duel
        b1 = mkbot(BotType.Heavy, team1, 5, 3, east)
        b2 = mkbot(BotType.Heavy, team2, 7, 3, south, right)
        trig(b1, 0, fire=True)
        trig(b2, 0, fire=True)

        # laser mass kill
        mkbot(BotType.Raider, team2, 3.5, 4.0, west, hp=0.1)
        mkbot(BotType.Raider, team2, 4.0, 4.5, west, hp=0.1)
        mkbot(BotType.Raider, team2, 4.5, 5.0, west, hp=0.1)
        mkbot(BotType.Raider, team2, 3.5, 4.5, west, hp=0.1)
        mkbot(BotType.Raider, team2, 3.5, 5.0, west, hp=0.1)
        b = mkbot(BotType.Raider, team2, 4.0, 5.0, west, hp=0.1)
        trig(b, 0, shield=True)

        b1 = mkbot(BotType.Sniper, team1, 1, 4, east)
        scene_delay = 1
        rot_delay = 0.07
        when = atan(1 / 2) / BotType.Sniper.gun_rot_speed
        trig(b1, scene_delay, fire=True)
        trig(b1, scene_delay + rot_delay, tower_rotate=1)
        trig(b1, scene_delay + rot_delay + 1 * when, tower_rotate=-1)
        trig(b1, scene_delay + rot_delay + 2 * when, tower_rotate=0)
        trig(b1, scene_delay + rot_delay + 2 * when + 0.1, fire=False)
        trig(b, scene_delay + rot_delay + 3 * when, shield=False)

        # raider firing
        mkbot(BotType.Sniper, team1, 2, 7, north)
        mkbot(BotType.Sniper, team1, 3, 7, north)
        mkbot(BotType.Sniper, team1, 4, 7, north)
        mkbot(BotType.Sniper, team1, 5, 7, north)
        mkbot(BotType.Sniper, team1, 6, 7, north)
        b1 = mkbot(BotType.Raider, team2, 1.0, 6, east, (ahead + left) / 2, hp=0.1)
        trig(b1, 0.0, move=1, fire=True)
        b1 = mkbot(BotType.Raider, team2, 1.5, 6, east, (ahead + left) / 2, hp=0.1)
        trig(b1, 0.0, move=1, fire=True)
        b1 = mkbot(BotType.Raider, team2, 2.0, 6, east, (ahead + left) / 2, hp=0.1)
        trig(b1, 0.0, move=1, fire=True)
        b1 = mkbot(BotType.Heavy, team1, 7, 6, east)
        trig(b1, 0, shield=True)
        trig(b1, 6, shield=False)

        # drifting
        b1 = mkbot(BotType.Raider, team1, 5, 8, west)
        b1.vx = b1.type.max_ahead_speed / 2
        b1.vy = b1.type.max_ahead_speed / 2
        trig(b1, 0, move=1)

        # sort by tick
        for triglist in self._triggers.values():
            triglist.sort(key=lambda tick_and_etc: tick_and_etc[0])

    def do_control(self, control_team):
        next_triggers = {}
        result = []
        for bot_id, triglist in self._triggers.items():
            new_triglist = []
            for trigger in triglist:
                tick, attrs, team = trigger
                if tick <= self.engine.nticks and control_team == team:
                    result.append({'id': bot_id, **attrs})
                else:
                    new_triglist.append(trigger)
            next_triggers[bot_id] = new_triglist
        self._triggers = next_triggers
        return result

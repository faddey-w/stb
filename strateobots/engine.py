import enum
import random
from math import pi, sin, cos, sqrt, atan
import collections


class StbEngine:

    def __init__(self, world_width, world_height):
        self.world_width = world_width
        self.world_height = world_height
        self._bots = {}
        self._rays = []
        self._bullets = []

        self._controls = {}
        self._triggers = []

        self._n_bots = {}

        self.nticks = 1
        self.ticks_per_sec = 50

        self._initialize()

    def iter_bots(self):
        return self._bots.values()

    def iter_rays(self):
        return self._rays

    def iter_bullets(self):
        return self._bullets

    def _initialize(self):
        red, blue = 0xFF0000, 0x0000FF
        self._n_bots.update({red: 0, blue: 0})
        ahead, left, back, right = 0, pi/2, pi, -pi/2
        east, north, west, south = 0, pi/2, pi, -pi/2

        def mkbot(bottype, team, x, y, orientation, tower_orientation=ahead, hp=1.0):
            x *= self.world_width / 10
            y *= self.world_height / 10
            bot = BotModel(
                type=bottype,
                id=len(self._bots),
                orientation=orientation,
                team=team,
                x=x, y=y,
            )
            bot.tower_orientation = tower_orientation
            bot.hp *= hp
            self._bots[bot.id] = bot
            self._controls[bot.id] = BotControl()
            self._n_bots[team] += 1
            return bot

        def trig(bot, sec, **attrs):
            tick = int(sec * self.ticks_per_sec)
            self._triggers.extend(
                (tick, bot.id, attr, val)
                for attr, val in attrs.items()
            )

        # head-on collisions
        b1 = mkbot(BotType.Raider, red, 1, 1, east)
        b2 = mkbot(BotType.Raider, blue, 5, 1, west, hp=0.2)
        trig(b1, 0, move=1)
        trig(b2, 0, move=1)

        # lateral collisions
        b1 = mkbot(BotType.Raider, blue, 1, 2, east, hp=0.5)
        mkbot(BotType.Heavy, red, 5, 2, north)
        trig(b1, 0, move=1)

        # move by circle and rotate tower
        b1 = mkbot(BotType.Sniper, red, 7, 1, east)
        trig(b1, 0, move=1, rotate=1, tower_rotate=-1)

        # heavy tank duel
        b1 = mkbot(BotType.Heavy, red, 3, 3, east)
        b2 = mkbot(BotType.Heavy, blue, 7, 3, south, right)
        trig(b1, 0, fire=True)
        trig(b2, 0, fire=True)

        # laser mass kill
        mkbot(BotType.Raider, blue, 2.0, 4.0, west, hp=0.3)
        mkbot(BotType.Raider, blue, 2.5, 4.5, west, hp=0.3)
        mkbot(BotType.Raider, blue, 3.0, 5.0, west, hp=0.3)
        mkbot(BotType.Raider, blue, 2.0, 4.5, west, hp=0.3)
        mkbot(BotType.Raider, blue, 2.0, 5.0, west, hp=0.3)
        mkbot(BotType.Raider, blue, 2.5, 5.0, west, hp=0.3)

        b1 = mkbot(BotType.Sniper, red, 1, 4, east)
        trig(b1, 0, fire=True, tower_rotate=1)
        when = atan(1/2) / BotType.Sniper.gun_rot_speed
        trig(b1, when, tower_rotate=0)
        trig(b1, when+0.1, fire=False)

        # raider firing
        mkbot(BotType.Sniper, red, 2, 7, north)
        mkbot(BotType.Sniper, red, 3, 7, north)
        mkbot(BotType.Sniper, red, 4, 7, north)
        mkbot(BotType.Sniper, red, 5, 7, north)
        mkbot(BotType.Sniper, red, 6, 7, north)
        b1 = mkbot(BotType.Raider, blue, 1, 6, east, (ahead+left)/2, hp=0.1)
        mkbot(BotType.Heavy, red, 7, 6, east)
        trig(b1, 0.0, move=1, fire=True)

    def tick(self):
        nbots = len(self._bots)
        global_time = self.nticks / self.ticks_per_sec
        radius = 0.3 * min(self.world_width, self.world_height)
        speed_factor = 1/4
        for i, bot in enumerate(self._bots.values()):
            time = speed_factor * global_time + i * 2 * pi / nbots
            bot.x = self.world_width / 2 + radius * cos(time)
            bot.y = self.world_height / 2 + radius * sin(time)
            bot.orientation = time
            bot.tower_orientation = 2.5 * time
        for ray in self._rays:
            bot = self._bots[ray.origin_id]
            angle = bot.orientation + bot.tower_orientation
            tower_shift = vec_rotate(0, -12, bot.orientation)
            ray_start_shift = vec_rotate(53, 0, angle)
            x, y = vec_sum((bot.x, bot.y), tower_shift, ray_start_shift)
            ray.x = x
            ray.y = y
            ray.orientation = angle
        for bullet in self._bullets:
            bullet.y += 1000 / self.ticks_per_sec
            if bullet.y > self.world_height:
                bullet.y -= self.world_height
        self.nticks += 1

    @property
    def is_finished(self):
        return sum(1 for n in self._n_bots.values() if n > 0) <= 1 \
               or self.nticks > 5000


BotTypeProperties = collections.namedtuple(
    'BotTypeProperties',
    [
        'code',
        'max_hp',
        'cd_period',  # sec
        'move_ahead_speed',  # points / sec
        'move_back_speed',  # points / sec
        'rot_speed',  # radian / sec
        'gun_rot_speed',  # radian / sec
        'shots_ray',  # boolean; all rays beam during 1 sec
        'shot_range',
        'damage',  # per-shot or per-second if ray
    ]
)


class BotType(BotTypeProperties, enum.Enum):

    Heavy = BotTypeProperties(
        code=1,
        max_hp=1000,
        cd_period=5,
        move_ahead_speed=70,
        move_back_speed=60,
        rot_speed=pi / 3,
        gun_rot_speed=2 * pi / 3,
        shots_ray=False,
        shot_range=250,
        damage=120,
    )
    Raider = BotTypeProperties(
        code=2,
        max_hp=400,
        cd_period=1,
        move_ahead_speed=160,
        move_back_speed=30,
        rot_speed=2 * pi / 3,
        gun_rot_speed=2 * pi,
        shots_ray=False,
        shot_range=200,
        damage=45,
    )
    Sniper = BotTypeProperties(
        code=3,
        max_hp=250,
        cd_period=10,
        move_ahead_speed=80,
        move_back_speed=40,
        rot_speed=pi / 6,
        gun_rot_speed=pi / 3,
        shots_ray=True,
        shot_range=400,
        damage=350,
    )


class BotModel:

    __slots__ = ['id', 'team', 'type', 'hp', 'load', 'x', 'y',
                 'orientation', 'tower_orientation',
                 'move_ahead', 'move_back', 'shoot',
                 'rotation', 'tower_rotation']

    def __init__(self, id, team, type, x, y, orientation):
        self.id = id
        self.team = team
        self.type = type  # type: BotTypeProperties
        self.hp = type.max_hp
        self.load = 1.0
        self.x = x
        self.y = y
        self.orientation = orientation
        self.tower_orientation = 0.0
        self.move_ahead = False
        self.move_back = False
        self.rotation = 0
        self.tower_rotation = 0
        self.shoot = False

    @property
    def hp_ratio(self):
        return self.hp / self.type.max_hp

    @property
    def shot_ready(self):
        return self.load > 0.999


class BulletModel:

    __slots__ = ['type', 'origin_id', 'orientation', 'x', 'y', 'range', '_cos', '_sin']

    def __init__(self, type, origin_id, orientation, x, y, range):
        self.origin_id = origin_id
        self.type = type
        self.orientation = orientation
        self.x = x
        self.y = y
        self.range = range
        self._cos = cos(orientation)
        self._sin = sin(orientation)


def vec_rotate(x, y, angle):
    sin_a = sin(angle)
    cos_a = cos(angle)
    nx = x * cos_a - y * sin_a
    ny = x * sin_a + y * cos_a
    return nx, ny


def vec_sum(vec, *vecs):
    rx, ry = vec
    for x, y in vecs:
        rx += x
        ry += y
    return rx, ry


class BotControl:

    __slots__ = ['move', 'rotate', 'tower_rotate', 'fire']

    def __init__(self, move=0, rotate=0, tower_rotate=0, fire=False):
        self.move = move
        self.rotate = rotate
        self.tower_rotate = tower_rotate
        self.fire = fire


def make_ray(bot, ray=None):
    angle = bot.orientation + bot.tower_orientation
    tower_shift = vec_rotate(0, -12, bot.orientation)
    ray_start_shift = vec_rotate(53, 0, angle)
    x, y = vec_sum((bot.x, bot.y), tower_shift, ray_start_shift)
    if ray is None:
        ray = BulletModel(
            type=BotType.Sniper,
            origin_id=bot.id,
            orientation=angle,
            x=x,
            y=y,
            range=3000
        )
    else:
        ray.orientation = angle
        ray.x = x
        ray.y = y
    return ray

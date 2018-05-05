import enum
from math import pi, sin, cos, sqrt
import collections


class StbEngine:

    def __init__(self, world_width, world_height):
        self.world_width = world_width
        self.world_height = world_height
        self.tanks = {}
        self.rays = []
        self.bullets = []
        self.nticks = 0
        self.ticks_per_sec = 50

        import random
        for team in [0xFF0000, 0x00FF00, 0x0000FF]:
            for ttype in TankType.__members__.values():
                for i in range(3):
                    tank = TankModel(
                        id=len(self.tanks),
                        type=ttype,
                        team=team,
                        x=0, y=0,
                        orientation=0,
                    )
                    tank.hp = random.random() * tank.type.max_hp
                    tank.load = random.random()
                    self.tanks[tank.id] = tank
                    if ttype == TankType.Sniper:
                        self.rays.append(BulletModel(
                            type=TankType.Sniper,
                            origin_id=tank.id,
                            x=0, y=0, orientation=0,
                            range=3000
                        ))
        for i in range(75):
            self.bullets.append(BulletModel(
                origin_id=None,
                type=TankType.Heavy if i % 2 == 0 else TankType.Raider,
                orientation=0,
                x=random.random()*world_width,
                y=random.random()*world_height,
                range=3000,
            ))

    def tick(self):
        ntanks = len(self.tanks)
        global_time = self.nticks / self.ticks_per_sec
        radius = 0.3 * min(self.world_width, self.world_height)
        for i, tank in enumerate(self.tanks.values()):
            time = global_time + i * 2 * pi / ntanks
            tank.x = radius * cos(time)
            tank.y = radius * sin(time)
            tank.orientation = time
            tank.tower_orientation = 2.5 * time
        for ray in self.rays:
            tank = self.tanks[ray.origin_id]
            angle = tank.orientation + tank.tower_orientation
            tower_shift = vec_rotate(0, -12, tank.orientation)
            ray_start_shift = vec_rotate(53, 0, angle)
            x, y = vec_sum((tank.x, tank.y), tower_shift, ray_start_shift)
            ray.x = x
            ray.y = y
            ray.orientation = angle
        for bullet in self.bullets:
            bullet.y += 1000 / self.ticks_per_sec
            if bullet.y > self.world_height:
                bullet.y -= self.world_height
        self.nticks += 1

    @property
    def is_finished(self):
        return self.nticks >= 3000


TankTypeProperties = collections.namedtuple(
    'TankTypeProperties',
    [
        'code',
        'max_hp',
        'cd_period',  # sec
        'move_ahead_speed',  # points / sec
        'move_back_speed',  # points / sec
        'rot_speed',  # radian / sec
        'tower_rot_speed',  # radian / sec
        'shots_ray',  # boolean; all rays beam during 1 sec
        'shot_range',
        'damage',  # per-shot or per-second if ray
    ]
)


class TankType(enum.Enum):

    Heavy = TankTypeProperties(
        code=1,
        max_hp=1000,
        cd_period=5,
        move_ahead_speed=70,
        move_back_speed=60,
        rot_speed=pi / 3,
        tower_rot_speed=2 * pi / 3,
        shots_ray=False,
        shot_range=250,
        damage=120,
    )
    Raider = TankTypeProperties(
        code=2,
        max_hp=400,
        cd_period=1,
        move_ahead_speed=160,
        move_back_speed=30,
        rot_speed=2 * pi / 3,
        tower_rot_speed=2 * pi,
        shots_ray=False,
        shot_range=200,
        damage=45,
    )
    Sniper = TankTypeProperties(
        code=3,
        max_hp=250,
        cd_period=10,
        move_ahead_speed=80,
        move_back_speed=40,
        rot_speed=pi / 6,
        tower_rot_speed=pi / 3,
        shots_ray=True,
        shot_range=400,
        damage=350,
    )


class TankModel:

    __slots__ = ['id', 'team', 'type', 'hp', 'load', 'x', 'y',
                 'orientation', 'tower_orientation',
                 'move_ahead', 'move_back', 'shoot',
                 'rotation', 'tower_rotation']

    def __init__(self, id, team, type, x, y, orientation):
        self.id = id
        self.team = team
        self.type = type  # type: TankTypeProperties
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

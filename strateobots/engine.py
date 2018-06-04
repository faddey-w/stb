import enum
import random
from math import pi, sin, cos, sqrt, atan
import collections


class StbEngine:

    def __init__(self, world_width, world_height):
        self.world_width = world_width
        self.world_height = world_height
        self._bots = {}
        self._rays = {}
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
        return self._rays.values()

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
        b1 = mkbot(BotType.Heavy, red, 5, 3, east)
        b2 = mkbot(BotType.Heavy, blue, 7, 3, south, right)
        trig(b1, 0, fire=True)
        trig(b2, 0, fire=True)

        # laser mass kill
        mkbot(BotType.Raider, blue, 3.5, 4.0, west, hp=0.1)
        mkbot(BotType.Raider, blue, 4.0, 4.5, west, hp=0.1)
        mkbot(BotType.Raider, blue, 4.5, 5.0, west, hp=0.1)
        mkbot(BotType.Raider, blue, 3.5, 4.5, west, hp=0.1)
        mkbot(BotType.Raider, blue, 3.5, 5.0, west, hp=0.1)
        mkbot(BotType.Raider, blue, 4.0, 5.0, west, hp=0.1)

        b1 = mkbot(BotType.Sniper, red, 1, 4, east)
        when = atan(1/2) / BotType.Sniper.gun_rot_speed
        trig(b1, 0, fire=True, tower_rotate=1)
        trig(b1, 1*when, tower_rotate=-1)
        trig(b1, 2*when, tower_rotate=0)
        trig(b1, 2*when+0.1, fire=False)

        # raider firing
        mkbot(BotType.Sniper, red, 2, 7, north)
        mkbot(BotType.Sniper, red, 3, 7, north)
        mkbot(BotType.Sniper, red, 4, 7, north)
        mkbot(BotType.Sniper, red, 5, 7, north)
        mkbot(BotType.Sniper, red, 6, 7, north)
        b1 = mkbot(BotType.Raider, blue, 1, 6, east, (ahead+left)/2, hp=0.1)
        mkbot(BotType.Heavy, red, 7, 6, east)
        trig(b1, 0.0, move=1, fire=True)

        self._triggers.sort()

    def tick(self):

        next_bullets = []
        next_rays = {}
        tps = float(self.ticks_per_sec)
        bullet_speed = 1000 / tps
        ray_charge_per_tick = 1 / tps
        bot_radius = 25

        # process control triggers
        next_triggers = []
        for trig in self._triggers:
            tick, b_id, attr, val = trig
            if tick <= self.nticks:
                setattr(self._controls[b_id], attr, val)
            else:
                next_triggers.append(trig)
        self._triggers = next_triggers

        # move bullets
        for bullet in self._bullets:
            bullet.x += bullet_speed * bullet.cos
            bullet.y += bullet_speed * bullet.sin
            bullet.remaining_range -= bullet_speed
            if 0 <= bullet.x <= self.world_width \
                    and 0 <= bullet.y <= self.world_height \
                    and bullet.remaining_range > 0:
                next_bullets.append(bullet)

        # move bots
        for b_id, bot in self._bots.items():
            ctl = self._controls[bot.id]  # type: BotControl
            typ = bot.type  # type: BotTypeProperties
            ori_change = ctl.rotate * typ.rot_speed / tps
            move_ori = bot.orientation + ori_change / 2
            _sin = sin(move_ori)
            _cos = cos(move_ori)
            if ctl.move == 1:
                speed = typ.move_ahead_speed
            elif ctl.move == -1:
                speed = typ.move_back_speed
            else:
                speed = 0
            speed /= tps
            bot.x += _cos * speed
            bot.y += _sin * speed
            bot.orientation += ori_change

            bot.tower_orientation += ctl.tower_rotate * typ.gun_rot_speed / tps

            if ctl.fire and not typ.shots_ray and bot.shot_ready:
                bullet = BulletModel(
                    typ, b_id, bot.orientation + bot.tower_orientation,
                    bot.x, bot.y, typ.shot_range
                )
                next_bullets.append(bullet)
                bot.load = 0
            elif ctl.fire and typ.shots_ray and bot.load > ray_charge_per_tick:
                if bot.id not in self._rays:
                    bullet = BulletModel(
                        typ, b_id, bot.orientation + bot.tower_orientation,
                        bot.x, bot.y, typ.shot_range
                    )
                    self._rays[bot.id] = bullet
            else:
                if bot.load < 1:
                    bot.load += 1 / (typ.cd_period * tps)

        # update rays
        for ray in self._rays.values():
            bot = self._bots.get(ray.origin_id)
            if bot is None or bot.load < 0 or not self._controls[bot.id].fire:
                continue
            bot.load -= ray_charge_per_tick
            next_rays[bot.id] = ray
            position_ray(bot, ray)

        # make bullet damage
        next_bullets_after_damage = []
        for bullet in next_bullets:
            for bot in self._bots.values():
                if bullet.origin_id == bot.id:
                    continue
                d = dist_points(bullet.x, bullet.y, bot.x, bot.y)
                if d > bot_radius:
                    continue
                h = dist_line(bot.x, bot.y, bullet.cos, bullet.sin, bullet.x, bullet.y)
                dx = bullet.x - bot.x
                dy = bullet.y - bot.y
                bsin = sin(bot.orientation)
                bcos = cos(bot.orientation)

                hit_factor = half_chord_len(bot_radius, h) / bot_radius
                armor_factor = vec_dot(dx, dy, bcos, bsin) / vec_len(dx, dy)

                damage = bullet.type.damage * hit_factor / (2 + armor_factor)
                bot.hp -= damage
                break
            else:
                next_bullets_after_damage.append(bullet)
        next_bullets = next_bullets_after_damage

        # make ray damage
        for ray in next_rays.values():
            base_dmg = ray.type.damage / tps
            damaged = []
            for bot in self._bots.values():
                if ray.origin_id == bot.id:
                    continue
                d = dist_line(bot.x, bot.y, ray.cos, ray.sin, ray.x, ray.y)
                if d > bot_radius:
                    continue
                dx = bot.x - ray.x
                dy = bot.y - ray.y
                t = sqrt(dx*dx + dy*dy - d*d)
                if vec_dot(ray.cos, ray.sin, dx, dy) < 0:
                    t = -t
                if not (0 <= t <= ray.range):
                    continue

                hit_factor = half_chord_len(bot_radius, d) / bot_radius
                damaged.append((t, base_dmg * hit_factor, bot))
            damaged.sort(key=lambda item: item[0])
            decay_factor = 1.0
            for _, dmg, bot in damaged:
                bot.hp -= dmg * decay_factor
                decay_factor /= 2

        # make collisions damage, fix coordinates

        # remove killed bots
        self._bots = {bot.id: bot for bot in self._bots.values() if bot.hp > 0}

        self._bullets = next_bullets
        self._rays = next_rays
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

    __slots__ = [
        'type', 'origin_id', 'orientation', 'x', 'y',
        'range', 'remaining_range', 'cos', 'sin']

    def __init__(self, type, origin_id, orientation, x, y, range):
        self.origin_id = origin_id
        self.type = type
        self.orientation = orientation
        self.x = x
        self.y = y
        self.range = range
        self.remaining_range = range
        self.cos = cos(orientation)
        self.sin = sin(orientation)


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


def vec_dot(x1, y1, x2, y2):
    return x1*x2 + y1*y2


def half_chord_len(radius, distance):
    return sqrt(radius*radius - distance*distance)


def dist_line(point_x, point_y, line_cos, line_sin, line_x, line_y):
    q = line_sin * (point_x - line_x) - line_cos * (point_y - line_y)
    return abs(q)


def dist_points(x1, y1, x2, y2):
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))


def vec_len(x, y):
    return sqrt(x*x + y*y)


class BotControl:

    __slots__ = ['move', 'rotate', 'tower_rotate', 'fire']

    def __init__(self, move=0, rotate=0, tower_rotate=0, fire=False):
        self.move = move
        self.rotate = rotate
        self.tower_rotate = tower_rotate
        self.fire = fire


def position_ray(bot, ray):
    angle = bot.orientation + bot.tower_orientation
    tower_shift = vec_rotate(-12, 0, bot.orientation)
    ray_start_shift = vec_rotate(53, 0, angle)
    x, y = vec_sum((bot.x, bot.y), tower_shift, ray_start_shift)
    ray.orientation = angle
    ray.x = x
    ray.y = y
    ray.cos = cos(angle)
    ray.sin = sin(angle)

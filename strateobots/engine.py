import enum
import random
import logging
import collections
from math import pi, sin, cos, sqrt


EPS = 0.000001
BOT_RADIUS = 25
SHIELD_DAMAGE_FACTOR = 1/5
SHIELD_DURATION_SEC = 0.75
SHIELD_LOAD_REQUIRED = 0.5


log = logging.getLogger(__name__)


class StbEngine:

    def __init__(self, world_width, world_height, ai1_cls, ai2_cls, max_ticks=1000):
        self.world_width = world_width
        self.world_height = world_height
        self.teams = 0x00DD00, 0x0000FF
        self._bots = {}
        self._rays = {}
        self._bullets = []
        self._explosions = []

        self._controls = {}

        team1, team2 = self.teams
        self._n_bots = {team1: 0, team2: 0}

        self.error = False
        self.nticks = 1
        self.max_ticks = max_ticks
        self._win_reached_at = None
        self.ticks_per_sec = 50

        self.ai1 = ai1_cls(team1, self)
        self.ai2 = ai2_cls(team2, self)
        self.ai1.initialize()
        self.ai2.initialize()

    def iter_bots(self):
        return self._bots.values()

    def iter_rays(self):
        return self._rays.values()

    def iter_bullets(self):
        return self._bullets

    def iter_explosions(self):
        return self._explosions

    def get_control(self, bot):
        return self._controls[bot.id]

    def add_bot(self, bottype, team, x, y, orientation, tower_orientation, hp=None):
        bot = BotModel(
            type=bottype,
            id=len(self._bots),
            orientation=orientation,
            team=team,
            x=x, y=y,
        )
        bot.tower_orientation = tower_orientation
        if hp is not None:
            bot.hp = hp
        self._bots[bot.id] = bot
        self._controls[bot.id] = BotControl()
        self._n_bots[team] += 1
        return bot

    def tick(self):

        next_bullets = []
        next_rays = {}
        tps = float(self.ticks_per_sec)
        bullet_speed = 1000 / tps
        ray_charge_per_tick = 1 / tps
        bot_radius = BOT_RADIUS
        friction_factor = 175
        collision_factor = 0.0002

        # process AI
        try:
            self.ai1.tick()
            self.ai2.tick()
        except KeyboardInterrupt:
            raise
        except:
            log.exception('ERROR while processing AI')
            self.error = True
            return

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
            bot.rot_speed = (bot.rot_speed + ctl.rotate * typ.rot_speed) / 2
            ori_change = little_noise(bot.rot_speed) / tps

            a_angle = bot.orientation + ori_change / 2
            a_sin = sin(a_angle)
            a_cos = cos(a_angle)

            v = vec_len(bot.vx, bot.vy)
            v_cos = bot.vx / v if v else a_cos
            v_sin = bot.vy / v if v else a_sin

            # acceleration
            f_cos = f_sin = None
            if ctl.move != 0:
                # if we move in positive direction to acceleration
                # then engine accelerates in needed direction
                # and friction reduces other part of velocity vector
                a_cos *= ctl.move
                a_sin *= ctl.move
                if vec_dot(a_cos, a_sin, v_cos, v_sin) > 0:
                    av = bot.vx * a_cos + bot.vy * a_sin
                    fvx = bot.vx - av * a_cos
                    fvy = bot.vy - av * a_sin
                    acc = typ.acc / tps
                    bot.vx -= fvx
                    bot.vy -= fvy
                else:
                    max_speed = typ.max_ahead_speed if ctl.move == 1 else typ.max_back_speed
                    fvx = bot.vx - max_speed * a_cos
                    fvy = bot.vy - max_speed * a_sin
                    fv = vec_len(fvx, fvy) or 1
                    f_cos = fvx / fv
                    f_sin = fvy / fv
                    fvx = bot.vx
                    fvy = bot.vy
                    acc = 0
                    bot.vx = bot.vy = 0
            else:
                acc = 0
                fvx = bot.vx
                fvy = bot.vy
                bot.vx = bot.vy = 0

            # friction
            fv = vec_len(fvx, fvy) or 1
            f_cos = f_cos if f_cos is not None else fvx / fv
            f_sin = f_sin if f_sin is not None else fvy / fv
            fax = friction_factor * f_cos / tps
            fay = friction_factor * f_sin / tps
            if abs(fvx) <= abs(fax):
                fvx = 0
            else:
                fvx -= fax
            if abs(fvy) <= abs(fay):
                fvy = 0
            else:
                fvy -= fay

            # apply acceleration and friction
            bot.vx += a_cos * acc + fvx
            bot.vy += a_sin * acc + fvy

            # maximum speed
            v = sqrt(bot.vx*bot.vx + bot.vy*bot.vy)
            if ctl.move == 1:
                v_coeff = max(1, v / typ.max_ahead_speed)
            else:
                v_coeff = max(1, v / typ.max_back_speed)
            bot.vx /= v_coeff
            bot.vy /= v_coeff

            # change position and orientation
            bot.x += bot.vx / tps
            bot.y += bot.vy / tps
            bot.orientation += ori_change
            if bot.x < bot_radius:
                bot.x = bot_radius
            elif bot.x > self.world_width-bot_radius:
                bot.x = self.world_width-bot_radius
            if bot.y < bot_radius:
                bot.y = bot_radius
            elif bot.y > self.world_height-bot_radius:
                bot.y = self.world_height-bot_radius

            bot.tower_rot_speed = (bot.tower_rot_speed + ctl.tower_rotate * typ.gun_rot_speed) / 2
            bot.tower_orientation += little_noise(bot.tower_rot_speed) / tps

        # firing
        for b_id, bot in self._bots.items():
            ctl = self._controls[bot.id]  # type: BotControl
            typ = bot.type  # type: BotTypeProperties
            if ctl.shield and bot.load >= SHIELD_LOAD_REQUIRED and not bot.has_shield:
                bot.shield_remaining = SHIELD_DURATION_SEC * tps
                bot.load -= SHIELD_LOAD_REQUIRED
            if bot.has_shield:
                bot.shield_remaining -= 1
            else:
                if ctl.fire and not typ.shots_ray and bot.shot_ready:
                    angle = random.gauss(
                        mu=bot.orientation + bot.tower_orientation,
                        sigma=typ.fire_scatter
                    )
                    bullet = BulletModel(
                        typ, b_id, angle,
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
                armor_factor = vec_dot(dx, dy, bcos, bsin) / (vec_len(dx, dy) + EPS)

                damage = bullet.type.damage * hit_factor / (2 + armor_factor)
                if bot.has_shield:
                    damage *= SHIELD_DAMAGE_FACTOR
                bot.hp -= damage
                self._explosions.append(
                    ExplosionModel(bullet.x, bullet.y, 0.75 * tps, 0.5*bot_radius))
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
                if bot.has_shield:
                    hit_factor *= SHIELD_DAMAGE_FACTOR
                damaged.append((t, base_dmg * hit_factor, bot))

                dt = sqrt(bot_radius * bot_radius - d * d)
                for t_i in range(int(t-dt), int(t+dt+1), 2):
                    self._explosions.append(ExplosionModel(
                        x=ray.x + t_i * ray.cos,
                        y=ray.y + t_i * ray.sin,
                        size=8,
                        duration=2,
                    ))
            damaged.sort(key=lambda item: item[0])
            decay_factor = 1.0
            for _, dmg, bot in damaged:
                bot.hp -= dmg * decay_factor
                decay_factor /= 2

        # make collisions damage, fix coordinates
        all_bots = list(self._bots.values())
        for i1, b1 in enumerate(all_bots):
            m1 = b1.type.mass
            dmg1_coeff = SHIELD_DAMAGE_FACTOR if b1.has_shield else 1
            for i2 in range(i1+1, len(all_bots)):
                b2 = all_bots[i2]
                d = dist_points(b1.x, b1.y, b2.x, b2.y)
                if d >= 2*bot_radius:
                    continue
                d = max(d, EPS)

                dmg2_coeff = SHIELD_DAMAGE_FACTOR if b1.has_shield else 1
                m2 = b2.type.mass

                v1x, v1y = b1.vx, b1.vy
                v2x, v2y = b2.vx, b2.vy
                e1b = vec_len2(v1x, v1y)
                e2b = vec_len2(v2x, v2y)

                c_cos = (b2.x-b1.x) / d
                c_sin = (b2.y-b1.y) / d

                v1r = v1x*c_cos + v1y*c_sin
                v1t = -v1x*c_sin + v1y*c_cos

                v2r = v2x*c_cos + v2y*c_sin
                v2t = -v2x*c_sin + v2y*c_cos

                vr = (v1r*m1 + v2r*m2) / (m1 + m2)

                b1.vx = vr * c_cos - v1t * c_sin
                b1.vy = vr * c_sin + v1t * c_cos
                b2.vx = vr * c_cos - v2t * c_sin
                b2.vy = vr * c_sin + v2t * c_cos

                mx = (b1.x + b2.x) / 2
                my = (b1.y + b2.y) / 2
                b1.x = mx - (bot_radius + EPS) * c_cos
                b1.y = my - (bot_radius + EPS) * c_sin
                b2.x = mx + (bot_radius + EPS) * c_cos
                b2.y = my + (bot_radius + EPS) * c_sin

                # make damage
                e1a = vec_len2(b1.vx, b1.vy)
                e2a = vec_len2(b2.vx, b2.vy)

                h = max(0, m1 * (e1b - e1a) + m2 * (e2b - e2a))
                cf1 = 2 - vec_dot(cos(b1.orientation), sin(b1.orientation), c_cos, c_sin)
                cf2 = 2 + vec_dot(cos(b2.orientation), sin(b2.orientation), c_cos, c_sin)

                b1.hp -= dmg1_coeff * collision_factor * cf1 * h * m2 / (m1+m2)
                b2.hp -= dmg2_coeff * collision_factor * cf2 * h * m1 / (m1+m2)

        # remove killed bots
        next_bots = {}
        for bot in self._bots.values():
            if bot.hp > 0:
                next_bots[bot.id] = bot
            else:
                self._explosions.append(
                    ExplosionModel(bot.x, bot.y, tps, 2*bot_radius))
                self._n_bots[bot.team] -= 1
        self._bots = next_bots

        # update explosions
        for expl in self._explosions:
            expl.t += 1
        self._explosions = [e for e in self._explosions if not e.is_ended]

        self._bullets = next_bullets
        self._rays = next_rays
        self.nticks += 1

    @property
    def is_finished(self):
        if self._win_reached_at is None:
            win_reached = sum(1 for n in self._n_bots.values() if n > 0) <= 1
            if win_reached:
                self._win_reached_at = self.nticks
            return self.nticks >= self.max_ticks or self.error
        else:
            return self.nticks >= self._win_reached_at + self.ticks_per_sec \
                   or self.nticks >= self.max_ticks or self.error


BotTypeProperties = collections.namedtuple(
    'BotTypeProperties',
    [
        'code',
        'max_hp',
        'mass',
        'cd_period',  # sec
        'acc',
        'max_ahead_speed',  # points / sec
        'max_back_speed',  # points / sec
        'rot_speed',  # radian / sec
        'gun_rot_speed',  # radian / sec
        'shots_ray',  # boolean; all rays beam during 1 sec
        'shot_range',
        'fire_scatter',  # sigma of bullet direction distribution
        'damage',  # per-shot or per-second if ray
    ]
)


class BotType(BotTypeProperties, enum.Enum):

    Heavy = BotTypeProperties(
        code=1,
        max_hp=1000,
        mass=100,
        cd_period=5,
        acc=17,
        max_ahead_speed=55,
        max_back_speed=50,
        rot_speed=pi / 4,
        gun_rot_speed=2 * pi / 3,
        shots_ray=False,
        shot_range=250,
        fire_scatter=2 * pi / 180,
        damage=120,
    )
    Raider = BotTypeProperties(
        code=2,
        max_hp=400,
        mass=50,
        cd_period=1,
        acc=75,
        max_ahead_speed=160,
        max_back_speed=30,
        rot_speed=pi,
        gun_rot_speed=2 * pi,
        shots_ray=False,
        shot_range=200,
        fire_scatter=4 * pi / 180,
        damage=45,
    )
    Sniper = BotTypeProperties(
        code=3,
        max_hp=250,
        mass=80,
        cd_period=10,
        acc=15,
        max_ahead_speed=80,
        max_back_speed=40,
        rot_speed=pi / 8,
        gun_rot_speed=pi / 6,
        shots_ray=True,
        shot_range=400,
        fire_scatter=0,
        damage=500,
    )


class BotModel:

    __slots__ = ['id', 'team', 'type', 'hp', 'load', 'x', 'y', 'vx', 'vy',
                 'rot_speed', 'orientation', 'tower_orientation',
                 'move_ahead', 'move_back', 'shoot',
                 'shield_remaining', 'rotation', 'tower_rot_speed']

    def __init__(self, id, team, type, x, y, orientation):
        self.id = id
        self.team = team
        self.type = type  # type: BotTypeProperties
        self.hp = type.max_hp
        self.load = 1.0
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.rot_speed = 0
        self.tower_rot_speed = 0
        self.orientation = orientation
        self.tower_orientation = 0.0
        self.move_ahead = False
        self.move_back = False
        self.rotation = 0
        self.shield_remaining = 0
        self.shoot = False

    @property
    def hp_ratio(self):
        return self.hp / self.type.max_hp

    @property
    def shot_ready(self):
        return self.load > 0.999

    @property
    def has_shield(self):
        return self.shield_remaining > 0

    @property
    def can_enable_shield(self):
        return self.load >= SHIELD_LOAD_REQUIRED


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


class ExplosionModel:

    __slots__ = ['x', 'y', 'duration', 'size', 't']

    def __init__(self, x, y, duration, size):
        self.x = x
        self.y = y
        self.duration = duration
        self.size = size
        self.t = 0

    @property
    def t_ratio(self):
        return self.t / self.duration

    @property
    def is_ended(self):
        return self.t >= self.duration


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


def vec_len2(x, y):
    return x*x + y*y


def quantize(x, q):
    r = x % q
    if r >= q/2:
        r -= q
    return x - r


def little_noise(x):
    if abs(x) < EPS:
        return x
    return random.gauss(x, x/10)


class BotControl:

    __slots__ = ['move', 'rotate', 'tower_rotate', 'fire', 'shield']

    def __init__(self, move=0, rotate=0, tower_rotate=0, fire=False, shield=False):
        self.move = move
        self.rotate = rotate
        self.tower_rotate = tower_rotate
        self.fire = fire
        self.shield = shield

    def __repr__(self):
        return 'BotControl(move={}, rotate={}, tower_rotate={}, fire={}, shield={})'.format(
            self.move, self.rotate, self.tower_rotate, self.fire, self.shield
        )


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

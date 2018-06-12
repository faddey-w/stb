from strateobots.engine import BotType, dist_points, vec_len, dist_line, BOT_RADIUS, vec_dot
from math import pi, acos, sqrt, asin, copysign, cos, sin
from ._base import BaseAI


def AI(team, engine):
    if team == engine.teams[0]:
        return RaiderVsSniper(team, engine)
    else:
        return SniperVsRaider(team, engine)


class SimpleDuelAI(BaseAI):

    def _initialize(self, bottype, dx):
        if self.team == self.engine.teams[0]:
            x = self.engine.world_width * dx
            a = 0
        else:
            x = self.engine.world_width * (1 - dx)
            a = pi
        self.engine.add_bot(
            bottype=bottype,
            team=self.team,
            x=x,
            y=self.engine.world_height * 0.5,
            orientation=a,
            tower_orientation=0
        )

    def _get_bots(self):
        our = enemy = ctl = None
        for b in self.engine.iter_bots():
            if b.team == self.team:
                our = b
                ctl = self.engine.get_control(b)
            else:
                enemy = b
        if enemy is None:
            ctl.fire = False
            ctl.move = 0
            ctl.rotate = 0
            ctl.tower_rotate = 0
        return our, enemy, ctl


class RaiderVsSniper(SimpleDuelAI):

    def initialize(self):
        self._initialize(BotType.Raider, 0.05)
        self.was_attack = False
        self.orbit_k = 2

    def tick(self):
        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return

        orbit_k = 1 / 3
        orbit = orbit_k * bot.type.shot_range
        max_speed = sqrt(125 * orbit)

        dist = dist_points(bot.x, bot.y, enemy.x, enemy.y)
        enemy_angle = to_angle((enemy.x - bot.x), (enemy.y - bot.y), dist)
        ori_angle = norm_angle(bot.orientation)

        ctl.tower_rotate = navigate_gun(bot, enemy)

        ctl.shield = dist > 1.3 * orbit and should_fire(enemy, bot, dist)

        if dist > 1.3 * orbit or not is_at_back(enemy, enemy_angle + pi):
            # decide - move to left side from enemy or to right
            # determine target point - nearest point on orbit
            pt_angle = asin(orbit / dist) if orbit < dist else pi/2
            delta_angle = norm_angle(enemy_angle - ori_angle)
            pt_angle = enemy_angle - copysign(pt_angle, delta_angle)
            delta_angle = norm_angle(pt_angle - ori_angle)
            if delta_angle > 0:
                ctl.rotate = +1
            else:
                ctl.rotate = -1

            # always move ahead
            # limit speed if already at orbit to avoid drift
            if dist > 1.1 * orbit or vec_len(bot.vx,  bot.vy) < max_speed:
                ctl.move = +1
            else:
                ctl.move = 0
        else:
            ctl.move = 0
            ctl.rotate = 0

        # decide if we should fire
        ctl.fire = should_fire(bot, enemy, dist)


class SniperVsRaider(SimpleDuelAI):

    def initialize(self):
        self._initialize(BotType.Sniper, 0.3)
        self._fire_start = None

    def tick(self):
        max_ahead_v = 5

        bot, enemy, ctl = self._get_bots()
        if None in (bot, enemy):
            return

        # slowly move ahead is target is too far to shoot
        # move back if target is within fire range to keep distance
        dist = dist_points(bot.x, bot.y, enemy.x, enemy.y)
        if dist > bot.type.shot_range:
            v = vec_len(bot.vx, bot.vy)
            if v > max_ahead_v:
                ctl.move = 0
            else:
                ctl.move = +1
        else:
            ctl.move = -1

        # try to keep target in front
        enemy_angle = to_angle((enemy.x - bot.x), (enemy.y - bot.y), dist)
        ctl.rotate = ctl.tower_rotate = navigate_shortest(bot, enemy_angle)
        # ori_angle = norm_angle(bot.orientation)
        # if norm_angle(enemy_angle - ori_angle) > 0:
        #     ctl.rotate = +1
        # else:
        #     ctl.rotate = -1
        # ctl.tower_rotate = navigate_gun(bot, enemy)

        # decide if we should fire
        fire = should_fire(bot, enemy, dist)
        # if fire and self._fire_start is None:
        #     self._fire_start = self.engine.nticks + 1.5 * self.engine.ticks_per_sec
        # ctl.fire = fire and self.engine.nticks > self._fire_start
        ctl.fire = fire


def to_angle(dx, dy, dist):
    angle = acos(dx / dist)
    if dy < 0:
        angle = -angle
    return angle


def norm_angle(angle):
    angle %= (2 * pi)
    if angle > pi:
        angle -= 2 * pi
    return angle


def should_fire(bot, enemy, dist):
    gun_angle = bot.orientation + bot.tower_orientation
    kx = cos(gun_angle)
    ky = sin(gun_angle)
    fireline_dist = dist_line(
        point_x=enemy.x,
        point_y=enemy.y,
        line_cos=kx,
        line_sin=ky,
        line_x=bot.x,
        line_y=bot.y,
    )
    dot = vec_dot(kx, ky, (enemy.x - bot.x), (enemy.y - bot.y))
    return (dist < bot.type.shot_range) and (fireline_dist < BOT_RADIUS) and dot > 0


def navigate_shortest(bot, enemy_angle):
    gun_angle = bot.orientation + bot.tower_orientation
    need_to_rotate = norm_angle(enemy_angle - gun_angle)
    return +1 if need_to_rotate > 0 else -1


def navigate_gun(bot, enemy):
    dist = dist_points(bot.x, bot.y, enemy.x, enemy.y)
    enemy_angle = to_angle((enemy.x - bot.x), (enemy.y - bot.y), dist)
    gun_angle = norm_angle(bot.orientation + bot.tower_orientation)
    delta_angle = norm_angle(enemy_angle - gun_angle)
    if dist < bot.type.shot_range:
        limit_angle = asin((BOT_RADIUS / 2) / dist)
    else:
        limit_angle = pi / 6
    if abs(delta_angle) < limit_angle:
        return 0
    if delta_angle > 0:
        return +1
    else:
        return -1


def get_angle(from_bot, to_bot):
    dx = to_bot.x - from_bot.x
    dy = to_bot.y - from_bot.y
    dist = dist_points(to_bot.x, to_bot.y, from_bot.x, from_bot.y)
    return to_angle(dx, dy, dist)


def is_at_back(bot, angle):
    gun_angle = bot.orientation + bot.tower_orientation
    need_to_rotate = abs(norm_angle(angle - gun_angle))
    return need_to_rotate > 2*pi/3


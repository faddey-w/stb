from math import pi, acos, sqrt, asin, copysign, cos, sin, atan2
from strateobots.engine import dist_points, vec_len, dist_line, vec_dot
from strateobots.engine import Constants, BotType
from strateobots.util import objedict
from . import base
import logging


log = logging.getLogger(__name__)


class AIModule(base.AIModule):

    def __init__(self):
        self.config = {
            'short': (lambda: short_range_attack, 'Close distance attack'),
            'distance': (lambda: distance_attack, 'Long distance attack'),
            'ramming': (lambda: RammingAttack(), 'Ramming attack'),
            'hold': (lambda: hold_position, 'Hold position'),
        }

    def list_ai_function_descriptions(self):
        return [
            (name, key)
            for key, (func, name) in self.config.items()
        ]

    def list_bot_initializers(self):
        return []

    def construct_ai_function(self, team, parameters):
        return self.config[parameters][0]()


def short_range_attack(state):
    bot = objedict(state['friendly_bots'][0])
    bottype = BotType.by_code(bot.type)
    enemy = objedict(state['enemy_bots'][0])
    enemytype = BotType.by_code(enemy.type)
    ctl = objedict()
    ctl.id = bot.id

    orbit_k = 1 / 3
    orbit = orbit_k * bottype.shot_range
    max_speed = sqrt(500 * orbit)

    dist = dist_points(bot.x, bot.y, enemy.x, enemy.y)
    enemy_angle = atan2((enemy.y - bot.y), (enemy.x - bot.x))
    ori_angle = norm_angle(bot.orientation)

    ctl.tower_rotate = navigate_gun(bot, enemy)

    ctl.shield = dist > 1.3 * orbit and should_fire(enemy, bot, 1.5 * enemytype.shot_range, dist)

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
    ctl.fire = should_fire(bot, enemy, bottype.shot_range, dist)

    return [ctl]


def distance_attack(state):
    bot = objedict(state['friendly_bots'][0])
    bottype = BotType.by_code(bot.type)
    enemy = objedict(state['enemy_bots'][0])
    enemytype = BotType.by_code(enemy.type)
    ctl = objedict()
    ctl.id = bot.id
    max_ahead_v = 100

    # slowly move ahead is target is too far to shoot
    # move back if target is within fire range to keep distance
    dist = dist_points(bot.x, bot.y, enemy.x, enemy.y)
    if dist > 0.9 * bottype.shot_range:
        v = vec_len(bot.vx, bot.vy)
        if v > max_ahead_v:
            ctl.move = 0
        else:
            ctl.move = +1
    else:
        ctl.move = -1

    # try to keep target in front
    enemy_angle = atan2((enemy.y - bot.y), (enemy.x - bot.x))
    ctl.rotate = navigate_shortest(bot, enemy_angle, with_gun=False)
    ctl.tower_rotate = navigate_shortest(bot, enemy_angle)
    # ori_angle = norm_angle(bot.orientation)
    # if norm_angle(enemy_angle - ori_angle) > 0:
    #     ctl.rotate = +1
    # else:
    #     ctl.rotate = -1
    # ctl.tower_rotate = navigate_gun(bot, enemy)

    # decide if we should fire
    ctl.fire = fire = should_fire(bot, enemy, bottype.shot_range, dist)

    # decide if we should turn on the shield
    is_at_danger = should_fire(enemy, bot, 1.8 * enemytype.shot_range, dist)
    if is_at_danger:
        if bottype.shots_ray:
            if bot.is_firing:
                ctl.shield = not fire
            else:
                ctl.shield = not fire or bot.load < Constants.ray_min_load_required
        else:
            ctl.shield = not fire or not bot.shot_ready
        if ctl.shield and not bot.has_shield and bot.shield < 0.1:
            ctl.shield = False
    else:
        ctl.shield = False

    return [ctl]


class RammingAttack:
    def __init__(self):
        self.last_v = None
        self.is_ramming = True

    def __call__(self, state):
        bot = objedict(state['friendly_bots'][0])
        bottype = BotType.by_code(bot.type)
        enemy = objedict(state['enemy_bots'][0])
        if self.last_v is None:
            last_v = 0
        else:
            last_v = self.last_v
        # enemytype = BotType.by_code(enemy.type)
        ctl = objedict()
        ctl.id = bot.id

        limit_speed = bottype.max_ahead_speed / 2
        almost_zero_speed = bottype.acc * 0.2
        full_acc_dist = bottype.max_ahead_speed ** 2 / (2 * bottype.acc)
        dist_k = 1
        ramming_dist = dist_k * full_acc_dist

        dist = dist_points(bot.x, bot.y, enemy.x, enemy.y)
        v = vec_len(bot.vx, bot.vy)
        self.last_v = v
        ori_cos = cos(bot.orientation)
        ori_sin = sin(bot.orientation)
        enemy_angle = atan2((enemy.y - bot.y), (enemy.x - bot.x))

        # have_contact = dist < 2 * Constants.bot_radius + 3 * Constants.epsilon
        is_ahead = (bot.vx * ori_cos + bot.vy * ori_sin) > 0

        enemy_dir_cos = (enemy.x - bot.x) / dist
        enemy_dir_sin = (enemy.y - bot.y) / dist
        v_r_len = vec_len(bot.vx * enemy_dir_cos, bot.vy * enemy_dir_sin)
        v_r = v_r_len * (+1 if cos(bot.orientation - enemy_angle) > 0 else -1)
        time_to_max_speed = (bottype.max_ahead_speed - v_r) / bottype.acc
        dist_to_max_speed = time_to_max_speed * (bottype.max_ahead_speed + v_r) / 2
        contact_dist = dist - Constants.bot_radius
        if dist_to_max_speed > contact_dist:
            ts = solve_square_equation(bottype.acc / 2, v_r, -contact_dist)
            contact_time = max(ts)
            contact_speed = v_r + bottype.acc * contact_time
        else:
            contact_time = time_to_max_speed + (contact_dist - dist_to_max_speed) / bottype.max_ahead_speed
            contact_speed = bottype.max_ahead_speed

        if bot.has_shield:
            shield_start_time = 0
        else:
            shield_start_time = bottype.shield_warmup_period * (1 - bot.shield_warmup)
        shield_max_time = bot.shield * bottype.shield_energy / bottype.shield_regen

        if self.is_ramming:
            if v < last_v - 2:
                self.is_ramming = False
            do_ramming = self.is_ramming
        else:
            self.is_ramming = do_ramming = dist > ramming_dist

        can_ram_with_shield = shield_start_time + 0.1 < contact_time < shield_start_time + 0.9 * shield_max_time
        if can_ram_with_shield and do_ramming:
            ctl.shield = shield_start_time + 0.1 < contact_time < shield_start_time + min(1, 0.9 * shield_max_time)
        else:
            ctl.shield = False

        if do_ramming:
            ctl.move = +1
            ctl.rotate = navigate_shortest(bot, enemy_angle, with_gun=False)
        else:
            ctl.move = -1
            ctl.rotate = navigate_shortest(bot, enemy_angle, with_gun=False)
        ctl.tower_rotate = navigate_shortest(bot, enemy_angle, with_gun=True)
        ctl.fire = should_fire(bot, enemy, bottype.shot_range, dist)

        return [ctl]


def hold_position(state):
    [ctl] = distance_attack(state)
    ctl.move = 0
    return [ctl]


def norm_angle(angle):
    angle %= (2 * pi)
    if angle > pi:
        angle -= 2 * pi
    return angle


def should_fire(bot, enemy, shot_range, dist=None):
    if dist is None:
        dist = dist_points(bot.x, bot.y, enemy.x, enemy.y)
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
    return (dist < shot_range) and (fireline_dist < Constants.bot_radius) and dot > 0


def navigate_shortest(bot, enemy_angle, with_gun=True):
    if with_gun:
        angle = bot.orientation + bot.tower_orientation
    else:
        angle = bot.orientation
    need_to_rotate = norm_angle(enemy_angle - angle)
    return +1 if need_to_rotate > 0 else -1


def navigate_gun(bot, enemy):
    enemy_angle = atan2((enemy.y - bot.y), (enemy.x - bot.x))
    gun_angle = norm_angle(bot.orientation + bot.tower_orientation)
    delta_angle = norm_angle(enemy_angle - gun_angle)
    # if dist < bot.type.shot_range:
    #     limit_angle = asin((BOT_RADIUS / 2) / dist)
    # else:
    #     limit_angle = pi / 6
    # if abs(delta_angle) < limit_angle:
    #     return 0
    if delta_angle > 0:
        return +1
    else:
        return -1


def solve_square_equation(a, b, c):
    """"Solve equation a*x^2 + b*x + c = 0 in real numbers"""
    d = b*b - 4*a*c
    if d < 0:
        return []
    if d == 0:
        return [-b / (2*a)]
    d_sqrt = sqrt(d)
    return [(-b+d_sqrt) / (2*a), (-b-d_sqrt) / (2*a)]


def get_angle(from_bot, to_bot):
    dx = to_bot.x - from_bot.x
    dy = to_bot.y - from_bot.y
    return atan2(dy, dx)


def is_at_back(bot, angle):
    gun_angle = bot.orientation + bot.tower_orientation
    need_to_rotate = abs(norm_angle(angle - gun_angle))
    return need_to_rotate > 2*pi/3
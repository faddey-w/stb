from math import pi
from strateobots.engine import Constants, BotType
from strateobots.ai.evolution.lib import atan2, asin, cos, sin, binary_choice, Expression


def move_to_back(bot, enemy, orbit_radius, max_speed=None, apocenter_at_back_coeff=1.3):
    if max_speed is None:
        max_speed = (500 * orbit_radius) ** 0.5

    dist = dist_points(bot["x"], bot["y"], enemy["x"], enemy["y"])
    enemy_angle = atan2((enemy["y"] - bot["y"]), (enemy["x"] - bot["x"]))
    ori_angle = norm_angle(bot['orientation'])

    _is_at_back = (dist <= apocenter_at_back_coeff * orbit_radius) * is_at_back(
        enemy, enemy_angle + pi
    )
    _not_at_back = 1 - _is_at_back

    move, rotate = _movement_to_back(
        bot, orbit_radius, max_speed, dist, enemy_angle, ori_angle
    )
    return _not_at_back * move, _not_at_back * rotate


def _movement_to_back(bot, orbit_radius, max_speed, dist, enemy_angle, ori_angle):
    # decide - move to left side from enemy or to right
    # determine target point - nearest point on orbit
    pt_angle = asin(orbit_radius / dist) if orbit_radius < dist else pi / 2
    delta_angle = norm_angle(enemy_angle - ori_angle)
    pt_angle = enemy_angle - abs(pt_angle) * (2 * (delta_angle >= 0) - 1)
    delta_angle = norm_angle(pt_angle - ori_angle)
    rotate = delta_angle > 0

    # always move ahead
    # limit speed if already at orbit to avoid drift
    if dist > 1.1 * orbit_radius or vec_len(bot["vx"], bot["vy"]) < max_speed:
        move = +1
    else:
        move = 0

    return move, rotate


def get_bot_type_property(bot, prop_name):
    result = 0
    for name, bottype in BotType.__members__.items():
        result += (bot["type"][name] > 0.5) * getattr(bottype, prop_name)
    return result


def shot_ready(bot):
    return bot["load"] >= get_bot_type_property(bot, "shot_energy")


def keep_distance(bot, enemy, max_ahead_v=100):
    # slowly move ahead if target is too far to shoot
    # move back if target is within fire range to keep distance
    dist = dist_points(bot["x"], bot["y"], enemy["x"], enemy["y"])
    is_far = dist > 0.9 * get_bot_type_property(bot, "shot_range")
    v = vec_len(bot["vx"], bot["vy"])

    move = binary_choice(is_far, v < max_ahead_v, -1)

    # try to keep target in front
    enemy_angle = atan2((enemy["y"] - bot["y"]), (enemy["x"] - bot["x"]))
    rotate = navigate_shortest(bot, enemy_angle, with_gun=False)

    return dict(
        move=move,
        rotate=rotate,
        orientation=enemy_angle,
        # move_aim_x=enemy["x"],
        # move_aim_y=enemy["y"],
    )


def norm_angle(angle):
    graph = angle.graph
    with graph.register_mode(False):
        angle %= 2 * pi
        angle -= (angle > pi) * 2 * pi
    return Expression(graph, expr=angle.value)


def should_fire(bot, enemy, shot_range, dist=None):
    if dist is None:
        dist = dist_points(bot["x"], bot["y"], enemy["x"], enemy["y"])
    gun_angle = bot['orientation'] + bot['tower_orientation']
    kx = cos(gun_angle)
    ky = sin(gun_angle)
    fireline_dist = dist_line(
        point_x=enemy["x"],
        point_y=enemy["y"],
        line_cos=kx,
        line_sin=ky,
        line_x=bot["x"],
        line_y=bot["y"],
    )
    dot = vec_dot(kx, ky, (enemy["x"] - bot["x"]), (enemy["y"] - bot["y"]))
    return (dist < shot_range) * (fireline_dist < Constants.bot_radius) * (dot > 0)


def navigate_shortest(bot, enemy_angle, with_gun=True):
    if with_gun:
        angle = bot["orientation"] + bot["tower_orientation"]
    else:
        angle = bot["orientation"]
    need_to_rotate = norm_angle(enemy_angle - angle)
    return 2 * (need_to_rotate > 0) - 1


def navigate_gun(bot, enemy):
    enemy_angle = atan2((enemy["y"] - bot["y"]), (enemy["x"] - bot["x"]))
    gun_angle = norm_angle(bot["orientation"] + bot["tower_orientation"])
    delta_angle = norm_angle(enemy_angle - gun_angle)
    tower_rotate = 2 * (delta_angle > 0) - 1
    gun_orientation = enemy_angle
    # ctl.gun_aim_x, ctl.gun_aim_y = enemy["x"], enemy["y"]
    return tower_rotate, gun_orientation


def get_angle(from_bot, to_bot):
    dx = to_bot["x"] - from_bot["x"]
    dy = to_bot["y"] - from_bot["y"]
    return atan2(dy, dx)


def is_at_back(bot, angle):
    gun_angle = bot["orientation"] + bot["tower_orientation"]
    angle_to_rotate = abs(norm_angle(angle - gun_angle))
    return angle_to_rotate > (2 * pi / 3)


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
    with x1.graph.register_mode(False):
        z = x1 * x2 + y1 * y2
    return Expression(x1.graph, expr=z.value)


def half_chord_len(radius, distance):
    return (radius ** 2 - distance ** 2) ** 0.5


def dist_line(point_x, point_y, line_cos, line_sin, line_x, line_y):
    with point_x.graph.register_mode(False):
        q = line_sin * (point_x - line_x) - line_cos * (point_y - line_y)
        d = abs(q)
    return Expression(point_x.graph, expr=d.value)


def dist_points(x1, y1, x2, y2):
    with x1.graph.register_mode(False):
        z = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return Expression(x1.graph, expr=z.value)


def dist_bot(bot, x, y):
    return dist_points(bot["x"], bot["y"], x, y)


def vec_len(x, y):
    with x.graph.register_mode(False):
        z = (x ** 2 + y ** 2) ** 0.5
    return Expression(x.graph, expr=z.value)


def vec_len2(x, y):
    with x.graph.register_mode(False):
        z = x ** 2 + y ** 2
    return Expression(x.graph, expr=z.value)

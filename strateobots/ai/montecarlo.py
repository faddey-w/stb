from math import pi, acos, sqrt, asin, copysign, cos, sin, atan2
import numpy as np
from strateobots.engine import dist_points, vec_len, dist_line, vec_dot
from strateobots.engine import Constants, BotType
from strateobots.ai.lib import data
from strateobots.util import objedict
import itertools
from . import base
import logging


log = logging.getLogger(__name__)


class AIModule(base.AIModule):

    def __init__(self):
        self.config = {
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


class MonteCarloFunction:

    def __init__(self, keymapper, stats, how):
        self.keymapper = keymapper
        self.stats = stats
        self.how = how
        assert how in ('max', 'min')

    def __call__(self, state):
        key = self.keymapper(state)
        var_probs = self.stats[key]
        if self.how == 'max':
            idx = np.argmax(var_probs)
        elif self.how == 'min':
            idx = np.argmin(var_probs)
        else:
            raise ValueError(self.how)
        mov, rot, gun, action = CONTROL_VARIANTS[idx]
        return [{
            'id': state['friendly_bots'][0]['id'],
            'move': mov,
            'rotate': rot,
            'tower_rotate': gun,
            'fire': action == 'fire',
            'shield': action == 'shield',
        }]


CONTROL_VARIANTS = tuple(itertools.product(
    data.ctl_move.categories,
    data.ctl_rotate.categories,
    data.ctl_tower_rotate.categories,
    ['fire', 'shield', 'none'],
))


def map_key(state):
    bot = objedict(state['friendly_bots'][0])
    enemy = objedict(state['enemy_bots'][0])
    bottype = BotType.by_code(bot.type)

    vv = vec_len(bot.vx, bot.vy) / bottype.max_ahead_speed
    if vv > 0:
        va = atan2(bot.vy, bot.vx)
    else:
        va = 0

    bx = _section(bot.x, -1, 1001, 10)
    by = _section(bot.y, -1, 1001, 10)
    bbo = _section(bot.orientation % _2pi, 0, _2pi, 6)
    bgo = _section(bot.tower_orientation % _2pi, 0, _2pi, 6)
    bse = _section(bot.shield, 0, 1, 4)
    bsw = _section(bot.shield_warmup, -0.4, 1.4, 4)
    bvv = _section(vv, -0.01, 1, 4)
    bva = _section(va, 0, _2pi, 6)
    bl = _section(bot.load, 0, 1, 4)
    bhp = (
        0 if bot.hp > 0.9
        else 1 if bot.hp > 0.7
        else 2 if bot.hp > 0.4
        else 3 if bot.hp > 0.1
        else 4
    )

    ex = _section(enemy.x, -1, 1001, 10)
    ey = _section(enemy.y, -1, 1001, 10)
    ebo = _section(enemy.orientation % _2pi, 0, _2pi, 6)
    ego = _section(enemy.tower_orientation % _2pi, 0, _2pi, 6)
    ese = _section(enemy.shield, 0, 1, 4)
    ehp = (
        0 if enemy.hp > 0.9
        else 1 if enemy.hp > 0.7
        else 2 if enemy.hp > 0.4
        else 3 if enemy.hp > 0.1
        else 4
    )

    enemy_angle = atan2((enemy.y - bot.y), (enemy.x - bot.x))
    cos_nav = cos(bot.orientation + bot.tower_orientation - enemy_angle)
    nav = _section(cos_nav, -0.15, +0.15, 1, allow_out=True)

    return (
        bx,
        by,
        bbo,
        bgo,
        bse,
        bsw,
        bvv,
        # bva,
        bl,
        bhp,
        ex,
        ey,
        ebo,
        ego,
        ese,
        ehp,
        nav
    )


def make_empty_stats():
    return np.zeros([
        10,  # bx
        10,  # by
        6,  # bbo
        6,  # bgo
        4,  # bse
        4,  # bsw
        4,  # bvv
        # 6,  # bva
        4,  # bl
        5,  # bhp

        10,  # ex
        10,  # ey
        6,  # ebo
        6,  # ego
        4,  # ese
        5,  # ehp

        3,  # nav
    ], dtype=np.float)


def _section(value, low, high, n, allow_out=False):
    if allow_out and value < low:
        return 0
    if allow_out and value >= high:
        return n+1
    section = (high - low) / n
    r = int((value - low) / section)
    if allow_out:
        return r + 1
    else:
        return min(n-1, max(0, r))


_2pi = np.pi * 2

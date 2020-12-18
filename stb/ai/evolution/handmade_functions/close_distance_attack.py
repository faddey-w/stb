from stb.engine import Action
from ._base import BaseAlgorithm
from .lib import (
    navigate_gun,
    should_fire,
    get_bot_type_property,
    dist_points,
    move_to_back,
)


class CloseDistanceAttack(BaseAlgorithm):
    def _duel_control(self, bot, enemy):

        bot_shot_range = get_bot_type_property(bot, 'shot_range')
        enemy_shot_range = get_bot_type_property(enemy, 'shot_range')
        orbit = bot_shot_range / 3

        dist = dist_points(bot['x'], bot['y'], enemy['x'], enemy['y'])
        tower_rotate, enemy_angle = navigate_gun(bot, enemy)

        bot_should_fire = should_fire(bot, enemy, bot_shot_range)
        enemy_can_fire = should_fire(bot, enemy, 1.5 * enemy_shot_range)
        quite_far_away = dist > 1.5 * orbit

        act_fire = 1 * bot_should_fire
        act_shield_warmup = (1 - bot_should_fire) * enemy_can_fire
        act_acceleration = (1 - bot_should_fire) * (1 - enemy_can_fire) * quite_far_away
        act_idle = (1 - bot_should_fire) * (1 - enemy_can_fire) * (1 - quite_far_away)

        move, rotate, tgt_orientation = move_to_back(bot, enemy, orbit)

        ctl = {
            "move": move,
            "rotate": rotate,
            "tower_rotate": tower_rotate,
            "gun_orientation": enemy_angle,
            "orientation": tgt_orientation,
            Action.FIRE: act_fire,
            Action.SHIELD_WARMUP: act_shield_warmup,
            Action.ACCELERATION: act_acceleration,
            Action.IDLE: act_idle,
        }

        return ctl

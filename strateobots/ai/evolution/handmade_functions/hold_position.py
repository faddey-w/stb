from strateobots.engine import Action
from strateobots.ai.evolution.lib import logical_or
from ._base import BaseAlgorithm
from .lib import (
    keep_distance,
    navigate_gun,
    should_fire,
    get_bot_type_property,
)


class HoldPosition(BaseAlgorithm):
    def _duel_control(self, bot, enemy):

        ctl = keep_distance(bot, enemy)
        ctl["move"] = self.graph.const(0)
        tower_rotate, _ = navigate_gun(bot, enemy)
        ctl["tower_rotate"] = tower_rotate

        bot_shot_range = get_bot_type_property(bot, "shot_range")
        is_at_danger = should_fire(
            enemy, bot, 1.8 * get_bot_type_property(enemy, "shot_range")
        )
        potentially_can_fire = should_fire(bot, enemy, 1.2 * bot_shot_range)

        should_retreat = logical_or(
            is_at_danger * (1 - potentially_can_fire), bot["shot_ready"]
        )

        has_some_shield = bot["shield"] > 0.1
        bot_should_fire = should_fire(bot, enemy, bot_shot_range)
        act_shield_warmup = (1 - bot_should_fire) * has_some_shield * should_retreat
        act_fire = bot_should_fire
        act_idle = (1 - act_fire) * (1 - act_shield_warmup)

        ctl.update(
            {
                Action.FIRE: act_fire,
                Action.SHIELD_WARMUP: act_shield_warmup,
                Action.IDLE: act_idle,
            }
        )

        return ctl

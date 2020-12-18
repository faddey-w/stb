from math import pi, acos, sqrt, asin, copysign, cos, sin, atan2
from stb.models import Constants, BotType, Action
from stb.util import objedict, dist_points, vec_len, dist_line, vec_dot
from . import base
from stb.ai.simple_duel import LongDistanceAttack, CloseDistanceAttack, RammingAttack
import logging


log = logging.getLogger(__name__)


class AIModule(base.AIModule):
    def __init__(self):
        self.config = {
            "v1": (TeamCoordinatorV1, "Team V1"),
        }

    def list_ai_function_descriptions(self):
        return [(name, key) for key, (func, name) in self.config.items()]

    def list_bot_initializers(self):
        return []

    def construct_ai_function(self, team, parameters):
        return self.config[parameters][0]()

    def get_close_distance_attack(self):
        return self.construct_ai_function(None, "short")


class _BaseFunction:
    def __init__(self):
        self._bot_type_cache = {}

    def __call__(self, state):
        targets = self.assign_targets(state["friendly_bots"], state["enemy_bots"])
        ctls = []
        for bot, (opp_bot, strategy) in zip(state["friendly_bots"], targets):
            ctl = objedict()
            ctl.id = bot["id"]
            strategy(
                objedict(bot), self.get_type(bot), objedict(opp_bot), self.get_type(opp_bot), ctl
            )
            ctls.append(ctl)
        for bot, ctl, (opp_bot, _) in zip(state["friendly_bots"], ctls, targets):
            if ctl.action == Action.FIRE:
                if self.need_block_friendly_fire(bot, opp_bot, state["friendly_bots"]):
                    ctl.action = Action.IDLE
        return ctls

    def assign_targets(self, bots, opp_bots):
        raise NotImplementedError

    def get_type(self, bot):
        bot_id = bot["id"]
        typ = self._bot_type_cache.get(bot_id)
        if typ is None:
            typ = self._bot_type_cache[bot_id] = BotType.by_code(bot["type"])
        return typ

    def need_block_friendly_fire(self, bot, aim_bot, friendly_bots):
        fire_angle = bot["orientation"] + bot["tower_orientation"]
        fire_angle_cos = cos(fire_angle)
        fire_angle_sin = sin(fire_angle)
        bot_radius = Constants.bot_radius

        aim_dist_to_fire_line = dist_line(
            aim_bot["x"], aim_bot["y"], fire_angle_cos, fire_angle_sin, bot["x"], bot["y"]
        )
        aim_dist_to_bot = dist_points(aim_bot["x"], aim_bot["y"], bot["x"], bot["y"])
        aim_dist_along_fire_line = sqrt(aim_dist_to_bot ** 2 - aim_dist_to_fire_line ** 2)

        is_ray = self.get_type(bot).shots_ray

        affected_positions = []
        for f_bot in friendly_bots:
            if bot["id"] == f_bot["id"]:
                continue
            is_in_front = (
                vec_dot(
                    fire_angle_cos, fire_angle_sin, f_bot["x"] - bot["x"], f_bot["y"] - bot["y"]
                )
                > 0
            )
            if not is_in_front:
                continue
            f_dist_to_fire_line = dist_line(
                f_bot["x"], f_bot["y"], fire_angle_cos, fire_angle_sin, bot["x"], bot["y"]
            )
            if f_dist_to_fire_line > bot_radius:
                continue
            f_dist_to_bot = dist_points(f_bot["x"], f_bot["y"], bot["x"], bot["y"])
            f_dist_along_fire_line = sqrt(f_dist_to_bot ** 2 - f_dist_to_fire_line ** 2)

            if is_ray:
                affected_positions.append(
                    (f_dist_along_fire_line, f_dist_to_fire_line, True, f_bot)
                )
            elif f_dist_along_fire_line < aim_dist_along_fire_line:
                return True
        if is_ray:
            if not affected_positions:
                return False
            affected_positions.append(
                (aim_dist_along_fire_line, aim_dist_to_fire_line, False, aim_bot)
            )
            affected_positions.sort()
            damage_value = self.get_type(bot).damage * bot["load"]
            positive_damage = 0
            positive_kills = 0
            for _, dist_to_fire_line, is_friend, _bot in affected_positions:
                est_damage = damage_value * dist_to_fire_line / bot_radius
                damage_value /= 2
                positive_damage += est_damage * (-1 if is_friend else +1)
                if est_damage > _bot["hp"]:
                    positive_kills += -1 if is_friend else +1
            if positive_kills > 0:
                return False
            if positive_kills == 0:
                return positive_damage > 0
            else:
                return True

        else:
            return False


class TeamCoordinatorV1(_BaseFunction):

    _strategy_map = {
        (BotType.Raider.code, BotType.Raider.code): CloseDistanceAttack,
        (BotType.Raider.code, BotType.Heavy.code): CloseDistanceAttack,
        (BotType.Raider.code, BotType.Sniper.code): CloseDistanceAttack,
        (BotType.Heavy.code, BotType.Raider.code): LongDistanceAttack,
        (BotType.Heavy.code, BotType.Heavy.code): RammingAttack,
        (BotType.Heavy.code, BotType.Sniper.code): RammingAttack,
        (BotType.Sniper.code, BotType.Raider.code): LongDistanceAttack,
        (BotType.Sniper.code, BotType.Heavy.code): CloseDistanceAttack,
        (BotType.Sniper.code, BotType.Sniper.code): CloseDistanceAttack,
    }

    def __init__(self):
        super(TeamCoordinatorV1, self).__init__()
        self._former_targets = {}
        self._duel_strategies = {}

    def assign_targets(self, bots, opp_bots):
        targets = []
        for bot in bots:
            best_opp_bot = None
            best_dist = float("inf")
            former_opp_id = self._former_targets.get(bot["id"])

            for opp_bot in opp_bots:
                dist = dist_points(opp_bot["x"], opp_bot["y"], bot["x"], bot["y"])
                if opp_bot["id"] is former_opp_id:
                    dist -= 100
                if dist < best_dist:
                    best_dist = dist
                    best_opp_bot = opp_bot

            self._former_targets[bot["id"]] = best_opp_bot["id"]

            key = bot["id"], best_opp_bot["id"]
            strategy = self._duel_strategies.get(key)
            if strategy is None:
                strategy = self._strategy_map[bot["type"], best_opp_bot["type"]]()
                self._duel_strategies[key] = strategy

            targets.append((best_opp_bot, strategy.make_decision))

        return targets

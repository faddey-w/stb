import math
from .lib import ComputeGraph
from strateobots.engine import BotType, Action
from strateobots.ai.lib.util import map_struct


class AiFunction:
    def __init__(self, graph_builder_func, n_bots, memory_size):
        self.encoder = Encoder(n_bots, memory_size)
        self.graph = ComputeGraph(self.encoder.get_arg_names())
        expressions = graph_builder_func(self.graph, n_bots, memory_size)
        self.key_mappings = map_struct(self.graph.get_out_index, expressions)

        self._bot_id_lists = None
        self._memory = None

    def __call__(self, state):
        if self._bot_id_lists is None:
            friends_list = [bot["id"] for bot in state["friendly_bots"]]
            enemies_list = [bot["id"] for bot in state["enemy_bots"]]
            self._bot_id_lists = friends_list, enemies_list
        inputs = self.encoder.encode(
            state, self._bot_id_lists[0], self._bot_id_lists[1], self._memory
        )
        outputs = self.graph.graph.evaluate(inputs)

        ctl_dicts, self._memory = decode(
            outputs, self.key_mappings, self._bot_id_lists[0]
        )

        # add additional debugging info if possible:
        bots = {bot["id"]: bot for bot in state["friendly_bots"]}
        for ctl in ctl_dicts:
            bot = bots.get(ctl["id"])
            if bot is None:
                continue
            if "move_aim_x" not in ctl and "orientation" in ctl:
                ori = ctl["orientation"]
                ctl["move_aim_x"] = bot['x'] + 100 * math.cos(ori)
                ctl["move_aim_y"] = bot['y'] + 100 * math.sin(ori)

        return ctl_dicts


class Encoder:
    def __init__(self, n_bots, memory_size):
        self.n_bots = n_bots
        self.memory_size = memory_size

    def encode(self, state, friend_ids, enemy_ids, memory):
        friends = {bot["id"]: bot for bot in state["friendly_bots"]}
        enemies = {bot["id"]: bot for bot in state["enemy_bots"]}
        result = [state["tick"]]
        if memory is None:
            result += [0] * self.memory_size
        else:
            result.extend(memory)
        for b_id in friend_ids:
            result += _encode(friends.get(b_id), True)
        for _ in range(len(friend_ids), self.n_bots):
            result += _encode(None, True)
        for b_id in enemy_ids:
            result += _encode(enemies.get(b_id), False)
        for _ in range(len(enemy_ids), self.n_bots):
            result += _encode(None, False)
        return result

    def get_arg_names(self):
        names = ["tick"]
        names += [f"memory/{i}" for i in range(self.memory_size)]
        for i in range(self.n_bots):
            prefix = f"bot/{i}/"
            for bot_type_name in sorted(BotType.__members__):
                names.append(prefix + "type/" + bot_type_name)
            for field in _VISIBLE_BOT_FIELDS:
                names.append(prefix + field)
            for field in _PRIVATE_BOT_FIELDS:
                names.append(prefix + field)
        for i in range(self.n_bots):
            prefix = f"enemy/{i}/"
            for bot_type_name in sorted(BotType.__members__):
                names.append(prefix + "type/" + bot_type_name)
            for field in _VISIBLE_BOT_FIELDS:
                names.append(prefix + field)
        return names


def decode(output_vector, key_mapping, friend_ids):
    memory = [output_vector[i] for i in key_mapping["memory"]]
    ctls = []
    for bot_id, ctl_map in zip(friend_ids, key_mapping["controls"]):
        ctl_map = ctl_map.copy()
        values = [output_vector[ctl_map.pop(act)] for act in Action.ALL]
        act = max(Action.ALL, key=values.__getitem__)
        ctl = {"action": Action.ALL[act], "id": bot_id}
        for field in ["move", "rotate", "tower_rotate"]:
            value = output_vector[ctl_map.pop(field)]
            if value > 0.5:
                result = +1
            elif value < -0.5:
                result = -1
            else:
                result = 0
            ctl[field] = result

        # decode extra fields that provide debugging information:
        for field in ctl_map:
            ctl[field] = output_vector[ctl_map[field]]

        ctls.append(ctl)
    return ctls, memory


def _encode(bot, is_friend):
    if bot is None:
        return [0] * (_N_FRIEND_FIELDS if is_friend else _N_ENEMY_FIELDS)
    result = [0] * _N_BOT_TYPES
    result[bot["type"] - 1] = 1
    result.extend(bot[f] for f in _VISIBLE_BOT_FIELDS)
    if is_friend:
        result.extend(bot[f] for f in _PRIVATE_BOT_FIELDS)
    return result


_PRIVATE_BOT_FIELDS = ["load", "shot_ready", "shield_warmup"]
_VISIBLE_BOT_FIELDS = [
    "hp",
    "x",
    "y",
    "vx",
    "vy",
    "orientation",
    "tower_orientation",
    "shield",
    "has_shield",
    "is_firing",
]

_N_BOT_TYPES = len(BotType.__members__)
_N_FRIEND_FIELDS = len(_VISIBLE_BOT_FIELDS) + len(_PRIVATE_BOT_FIELDS) + _N_BOT_TYPES
_N_ENEMY_FIELDS = len(_VISIBLE_BOT_FIELDS) + _N_BOT_TYPES

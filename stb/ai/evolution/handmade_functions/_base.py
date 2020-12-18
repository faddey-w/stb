from stb.engine import Action
from stb.ai.evolution.lib import ComputeGraph, binary_choice
from .lib import dist_points


class BaseAlgorithm:
    @classmethod
    def build(cls, graph: ComputeGraph, n_bots, memory_size):
        return cls(graph, n_bots, memory_size).output

    def __init__(self, graph: ComputeGraph, n_bots, memory_size):
        self.arg_names = graph.arg_names
        self.graph = graph
        assert memory_size == n_bots * n_bots

        self.args_flat = {name: self.graph[name] for name in self.arg_names}
        self.args = _unflat_dict(self.args_flat)
        tgt_ema = self.graph.const(1)

        tgt_upd = tgt_ema
        tgt_keep = 1 - tgt_ema

        ctls = []
        memory = [None] * memory_size
        for b_i in range(n_bots):
            bot = self.args["bot"][b_i]
            mem_0 = b_i * n_bots
            enemy_0 = self.args["enemy"][0]
            best_ctl = self._duel_control(bot, enemy_0)
            best_tgt = self._target_fn(bot, enemy_0)
            # best_tgt = tgt_keep * self.args["memory"][mem_0] + tgt_upd * best_tgt
            best_tgt = enemy_0["is_alive"] * best_tgt
            memory[mem_0] = best_tgt
            for e_i in range(1, n_bots):
                mem_i = b_i * n_bots + e_i
                enemy = self.args["enemy"][e_i]
                ctl = self._duel_control(bot, enemy)
                tgt = self._target_fn(bot, enemy)
                # tgt = tgt_keep * self.args["memory"][mem_i] + tgt_upd * tgt
                tgt = enemy["is_alive"] * tgt
                memory[mem_i] = tgt
                is_better = tgt > best_tgt
                for key in ctl:
                    best_ctl[key] = binary_choice(is_better, ctl[key], best_ctl[key])
            ctls.append(best_ctl)

        for ctl in ctls:
            for act in Action.ALL:
                if act not in ctl:
                    ctl[act] = self.graph.const(0)
        assert all(x is not None for x in memory)
        self.output = {"memory": memory, "controls": ctls}

    def _duel_control(self, bot, enemy):
        raise NotImplementedError

    def _target_fn(self, bot, enemy):
        d = dist_points(bot["x"], bot["y"], enemy["x"], enemy["y"])
        return (100 / d) ** 2


def _unflat_dict(dictionary):
    result = {}
    for key, value in dictionary.items():
        path = []
        for item in key.split("/"):
            try:
                item = int(item)
            except ValueError:
                pass
            path.append(item)
        destination = result
        for item in path[:-1]:
            destination = destination.setdefault(item, {})
        destination[path[-1]] = value
    return result

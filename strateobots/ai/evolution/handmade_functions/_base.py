from strateobots.ai.evolution.ai_function import Encoder
from strateobots.engine import Action
from strateobots.ai.evolution.lib import ComputeGraph


class BaseAlgorithm:

    @classmethod
    def build(cls, graph: ComputeGraph, n_bots, memory_size):
        return cls(graph, n_bots, memory_size).output

    def __init__(self, graph: ComputeGraph, n_bots, memory_size):
        self.arg_names = graph.arg_names
        self.graph = graph
        assert n_bots == 1
        assert memory_size == 0

        self.args_flat = {name: self.graph[name] for name in self.arg_names}
        self.args = _unflat_dict(self.args_flat)

        ctl = self._duel_control(self.args["bot"][0], self.args["enemy"][0])
        for act in Action.ALL:
            if act not in ctl:
                ctl[act] = self.graph.const(0)
        self.output = {"memory": [], "controls": [ctl]}

    def _duel_control(self, bot, enemy):
        raise NotImplementedError


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




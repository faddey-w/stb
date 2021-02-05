import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from typing import Tuple, Sequence
from stb.ai import utils
from stb.ai.datacoding.definitions import (
    bot_full_coder,
    bot_visible_coder,
    bullet_coder,
    ray_coder,
    control_coder,
)


@dataclass
class WorldStateCodes:
    bots: "[ N BotDim ]" = None
    bullets: "[ N BulletDim ]" = None
    rays: "[ N RayDim ]" = None
    controls: "[ N ControlDim ]" = None

    @classmethod
    def from_replay_item(
        cls, replay_item, with_controls=False, only_teams=None, bot_full_data=True
    ) -> "WorldStateCodes":
        self = cls()
        if only_teams is None:
            only_teams = sorted(replay_item["bots"].keys())
        bot_coder = bot_full_coder if bot_full_data else bot_visible_coder
        self.bots = bot_coder.batch_encode(
            bot for team in only_teams for bot in replay_item["bots"][team]
        )

        self.bullets = bullet_coder.batch_encode(replay_item["bullets"])
        self.rays = ray_coder.batch_encode(replay_item["rays"])
        if with_controls:
            self.controls = control_coder.batch_encode(
                ctl for team in only_teams for ctl in replay_item["controls"][team]
            )

        return self

    @classmethod
    def from_engine(
        cls, engine, with_controls=False, only_teams=None, bot_full_data=True
    ) -> "WorldStateCodes":
        replay_item = engine.serialize_state()
        replay_item["controls"] = {
            team: [engine.get_control(engine.get_bot(bot["id"])).serialize() for bot in bots]
            for team, bots in replay_item["bots"].items()
            if only_teams is None or team in only_teams
        }
        return cls.from_replay_item(replay_item, with_controls, only_teams, bot_full_data)

    def to_replay_item(self):
        result = {}

        coders = {
            "bots": {bot_full_coder.dim: bot_full_coder, bot_visible_coder.dim: bot_visible_coder},
            "bullets": {bullet_coder.dim: bullet_coder},
            "rays": {ray_coder.dim: ray_coder},
            "controls": {control_coder.dim: control_coder},
        }

        def _decode_and_set(_, attr):
            data = getattr(self, attr)
            coder = coders[attr][data.shape[1]]
            result[attr] = coder.batch_decode(data)

        self._map(_decode_and_set)

        bots_per_team = defaultdict(list)
        ctls_per_team = defaultdict(list)
        if "controls" in result:
            for bot, ctl in zip(result["bots"], result["controls"]):
                team = bot["team"]
                bots_per_team[team].append(bot)
                ctls_per_team[team].append(ctl)
            result["controls"] = dict(ctls_per_team)
        else:
            for bot in result["bots"]:
                team = bot["team"]
                bots_per_team[team].append(bot)
        result["bots"] = dict(bots_per_team)

        return result

    @classmethod
    def to_batch(
        cls, states: Sequence["WorldStateCodes"], insert_batch_dim=0
    ) -> Tuple["WorldStateCodes", "WorldStateCodes"]:
        result = cls()
        pad_masks = cls()

        def _collate_and_set(_, attr):
            data, pad_mask = utils.collate_sequences_with_padding(
                [getattr(s, attr) for s in states], insert_batch_dim=insert_batch_dim
            )
            setattr(result, attr, data)
            setattr(pad_masks, attr, pad_mask)

        states[0]._map(_collate_and_set)

        return result, pad_masks

    def get_batch_size(self, batch_axis=0) -> int:
        result = None

        def _get_batch_size(_, attr):
            nonlocal result
            thing = getattr(self, attr)
            if result is None and thing.ndim > 2:
                result = thing.shape[batch_axis]

        self._map(_get_batch_size)

        # noinspection PyTypeChecker
        return result

    def unbatch(self, pad_masks=None, batch_axis=0):
        batch_size = self.get_batch_size(batch_axis)
        if batch_size is None:
            raise ValueError(f"{self} is not batched")
        result = [self.__class__() for _ in range(batch_size)]

        def mapfunc(_, attr):
            data = getattr(self, attr)
            if pad_masks is not None:
                mask = getattr(pad_masks, attr)
            else:
                mask = None
            for i in range(batch_size):
                matrix = np.take(data, i, batch_axis)
                if mask is not None:
                    matrix = matrix[np.take(mask == 0, i, batch_axis)]
                setattr(result[i], attr, matrix)

        self._map(mapfunc)

    def numpy_to_torch(self):
        import torch

        copy = self.__class__()

        def to_torch(_, attr):
            setattr(copy, attr, torch.tensor(getattr(self, attr)))

        self._map(to_torch)

        return copy

    def torch_to_numpy(self):

        copy = self.__class__()

        def to_numpy(_, attr):
            setattr(copy, attr, getattr(self, attr).numpy())

        self._map(to_numpy)

        return copy

    def _map(self, function):
        if self.bots is not None:
            function(self, "bots")
        if self.bullets is not None:
            function(self, "bullets")
        if self.rays is not None:
            function(self, "rays")
        if self.controls is not None:
            function(self, "controls")

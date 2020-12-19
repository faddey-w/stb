import numpy as np
from dataclasses import dataclass
from typing import Union, Dict, Tuple, Sequence
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
    bots: "Union[Dict[object, '[ N BotDim ]'], '[ N BotDim ]']" = None
    bullets: "[ N BulletDim ]" = None
    rays: "[ N RayDim ]" = None
    controls: "[ N ControlDim ]" = None

    @classmethod
    def from_engine(
        cls, engine, with_controls=False, split_teams=False, bot_full_data=True
    ) -> "WorldStateCodes":
        self = cls()
        if split_teams:
            if bot_full_data is True:
                bot_full_data = engine.teams
            self.bots = {
                team: (bot_full_coder if team in bot_full_data else bot_visible_coder).batch_encode(
                    bot for bot in engine.iter_bots() if bot.team == team
                )
                for team in engine.teams
            }
        else:
            coder = bot_full_coder if bot_full_data else bot_visible_coder
            self.bots = coder.batch_encode(engine.iter_bots())

        self.bullets = bullet_coder.batch_encode(engine.iter_bullets())
        self.rays = ray_coder.batch_encode(engine.iter_rays())
        if with_controls:
            self.controls = control_coder.batch_encode(map(engine.get_control, engine.iter_bots()))
        return self

    def decode(self):
        result = {}

        coders = {
            "bots": {bot_full_coder.dim: bot_full_coder, bot_visible_coder.dim: bot_visible_coder},
            "bullets": {bullet_coder.dim: bullet_coder},
            "rays": {ray_coder.dim: ray_coder},
            "controls": {control_coder.dim: control_coder},
        }

        def _decode_and_set(_, attr, key=None):
            data = self[attr, key]
            coder = coders[attr][data.shape[1]]
            objects = coder.batch_decode(data)
            if key is None:
                result[attr] = objects
            else:
                result.setdefault(attr, {})[key] = objects

        self._map(_decode_and_set)

        return result

    @classmethod
    def to_batch(
        cls, states: Sequence["WorldStateCodes"], insert_batch_dim=0
    ) -> Tuple["WorldStateCodes", "WorldStateCodes"]:
        result = cls()
        pad_masks = cls()

        def _collate_and_set(_, attr, key=None):
            result[attr, key], pad_masks[attr, key] = utils.collate_sequences_with_padding(
                [s[attr, key] for s in states], insert_batch_dim=insert_batch_dim
            )

        states[0]._map(_collate_and_set)

        return result, pad_masks

    def get_batch_size(self, batch_axis=0) -> int:
        result = None

        def _get_batch_size(_, attr, key=None):
            nonlocal result
            thing = self[attr, key]
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

        def mapfunc(_, attr, key=None):
            data = self[attr, key]
            if pad_masks is not None:
                mask = pad_masks[attr, key]
            else:
                mask = None
            for i in range(batch_size):
                matrix = np.take(data, i, batch_axis)
                if mask is not None:
                    matrix = matrix[np.take(mask == 0, i, batch_axis)]
                result[i][attr, key] = matrix

        self._map(mapfunc)

    def numpy_to_torch(self):
        import torch

        copy = self.__class__()

        def to_torch(_, attr, key=None):
            copy[attr, key] = torch.tensor(self[attr, key])

        self._map(to_torch)

        return copy

    def torch_to_numpy(self):

        copy = self.__class__()

        def to_numpy(_, attr, key=None):
            copy[attr, key] = self[attr, key].numpy()

        self._map(to_numpy)

        return copy

    def _map(self, function):
        if isinstance(self.bots, dict):
            for team in self.bots.keys():
                function(self, "bots", team)
        elif self.bots is not None:
            function(self, "bots")
        if self.bullets is not None:
            function(self, "bullets")
        if self.rays is not None:
            function(self, "rays")
        if self.controls is not None:
            function(self, "controls")

    def __getitem__(self, item):
        if isinstance(item, tuple):
            attr, key = item
        else:
            attr = item
            key = None
        value = getattr(self, attr)
        if key is not None:
            value = value[key]
        return value

    def __setitem__(self, item, value):
        if isinstance(item, tuple):
            attr, key = item
        else:
            attr = item
            key = None
        if key is not None:
            attr_value = getattr(self, attr)
            if attr_value is None:
                attr_value = {}
                setattr(self, attr, attr_value)
            attr_value[key] = value
        else:
            setattr(self, attr, value)

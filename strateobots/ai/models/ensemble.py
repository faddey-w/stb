import tensorflow as tf
import numpy as np
from strateobots.ai.lib import data


class Model:
    def __init__(self, name, models):
        self.name = name
        self.models = list(models)

        self._dim_start = []
        self._dim_end = []
        cumsum = 0
        for model in self.models:
            self._dim_start.append(cumsum)
            cumsum += model.state_dimension
            self._dim_end.append(cumsum)

        self.state_dimension = self._dim_end[-1]
        self.var_list = sum([model.var_list for model in self.models], [])
        self.init_op = tuple(model.init_op for model in self.models)

    def encode_prev_state(self, bot, enemy, bot_bullet, enemy_bullet):
        return np.concatenate(
            [
                model.encode_prev_state(bot, enemy, bot_bullet, enemy_bullet)
                for model in self.models
            ]
        )

    def encode_state(self, bot, enemy, bot_bullet, enemy_bullet):
        return np.concatenate(
            [
                model.encode_state(bot, enemy, bot_bullet, enemy_bullet)
                for model in self.models
            ]
        )

    def apply(self, state_vector_array):
        applies = [
            model.apply(state_vector_array[:, dim_start:dim_end])
            for model, dim_start, dim_end in zip(
                self.models, self._dim_start, self._dim_end
            )
        ]
        controls = {
            ctl: tf.reduce_mean(
                tf.stack([apply.controls[ctl] for apply in applies], 0), 0
            )
            for ctl in data.ALL_CONTROLS
        }
        # import pdb; pdb.set_trace()
        return self._Apply(state_vector_array, applies, controls)

    class _Apply:
        def __init__(self, state_vector_array, applies, controls):
            self.state_vector_array = state_vector_array
            self.applies = applies
            self.controls = controls

import tensorflow as tf
import os
from strateobots.ai import base
from strateobots.ai.lib import model_function
from strateobots.ai.models import ff_aim_angle


class AIModule(base.AIModule):
    def __init__(self, saved_models_dir=None):
        self.sess = tf.Session()
        self._saved_models_dir = saved_models_dir

    def list_ai_function_descriptions(self):
        if self._saved_models_dir is None:
            saved = []
        else:
            saved = [
                (
                    "saved: " + name,
                    (self._get_saved, (os.path.join(self._saved_models_dir, name),)),
                )
                for name in os.listdir(self._saved_models_dir)
                if _is_exported_model_dir(os.path.join(self._saved_models_dir, name))
            ]
        return saved

    def construct_ai_function(self, team, parameters):
        ctor, args = parameters
        return ctor(*args)

    def _get_saved(self, save_path):
        return model_function.ModelAiFunction.from_exported_model(save_path)

    def list_bot_initializers(self):
        return []

    def load(self, name):
        return self._get_saved(os.path.join(self._saved_models_dir, name))


def _is_exported_model_dir(path):
    return os.path.isfile(os.path.join(path, "inference.pb")) and os.path.isfile(
        os.path.join(path, "model-config.json")
    )

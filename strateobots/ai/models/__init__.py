import tensorflow as tf
import os
from strateobots.ai import base
from strateobots.ai.lib import model_function, model_saving
from strateobots.ai.models import ff_aim_angle


class AIModule(base.AIModule):
    def __init__(self, saved_models_dir=None):
        self.sess = tf.Session()
        self._untrained = None
        self._saved_model_managers = {}
        self._saved_models_dir = saved_models_dir
        self._namescope_counter = 0

    def list_ai_function_descriptions(self):
        if self._saved_models_dir is None:
            saved = []
        else:
            saved = [
                ('saved: '+name, (self._get_saved, (os.path.join(self._saved_models_dir, name), )))
                for name in os.listdir(self._saved_models_dir)
            ]
        return [
            ('Untrained', (self._get_untrained, ())),
            *saved
        ]

    def construct_ai_function(self, team, parameters):
        ctor, args = parameters
        return ctor(*args)

    def _get_untrained(self):
        if self._untrained is None:
            self._untrained = ff_aim_angle.Model('Untrained')
            self.sess.run(self._untrained.init_op)

        return model_function.ModelAiFunction(self._untrained, self.sess)

    def _get_saved(self, save_path):
        if save_path in self._saved_model_managers:
            model = self._saved_model_managers[save_path].model
        else:
            namescope = 'model-{}'.format(self._namescope_counter)
            self._namescope_counter += 1
            mgr = model_saving.ModelManager.load_existing_model(save_path, name_scope=namescope)
            mgr.load_vars(self.sess)
            self._saved_model_managers[save_path] = mgr
            model = mgr.model

        return model_function.ModelAiFunction(model, self.sess)

    def list_bot_initializers(self):
        return []

    def load(self, name):
        return self._get_saved(os.path.join(self._saved_models_dir, name))

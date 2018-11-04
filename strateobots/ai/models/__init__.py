import tensorflow as tf
from strateobots.ai import base
from strateobots.ai.lib import integration
from strateobots.ai.models import simple_ff


class AIModule(base.AIModule):
    def __init__(self):
        self.sess = tf.Session()
        self._untrained = None

    def list_ai_function_descriptions(self):
        return [
            ('Untrained', self._get_untrained),
        ]

    def construct_ai_function(self, team, parameters):
        return parameters()

    def _get_untrained(self):
        if self._untrained is None:
            self._untrained = simple_ff.Model('Untrained')
            self.sess.run(self._untrained.init_op)

        return integration.ModelAiFunction(self._untrained, self.sess)

    def list_bot_initializers(self):
        return []

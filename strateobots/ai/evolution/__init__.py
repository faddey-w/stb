from strateobots.ai import base


class AIModule(base.AIModule):
    def list_ai_function_descriptions(self):
        return [
            ('evocore: long distance attack', None)
        ]

    def list_bot_initializers(self):
        return []

    def construct_ai_function(self, team, parameters):
        from .handmade_functions.long_distance_attack import LongDistanceAttack
        from .ai_function import AiFunction
        return AiFunction(LongDistanceAttack.build, 1, 0)

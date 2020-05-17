from strateobots.ai import base


class AIModule(base.AIModule):
    def list_ai_function_descriptions(self):
        from .handmade_functions.long_distance_attack import LongDistanceAttack
        from .handmade_functions.hold_position import HoldPosition
        return [
            ('evocore: long distance attack', LongDistanceAttack.build),
            ('evocore: hold position', HoldPosition.build),
        ]

    def list_bot_initializers(self):
        return []

    def construct_ai_function(self, team, builder_fn):
        from .ai_function import AiFunction
        return AiFunction(builder_fn, 1, 0)

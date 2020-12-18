class AIModule:
    @property
    def name(self):
        # cls = self.__class__
        # return '{}.{}'.format(cls.__module__, cls.__name__)
        return self.__class__.__module__

    def list_bot_initializers(self):
        """
        Returns list of pairs (name, function)
        where name is a string, function is "initialize_bots" for StbEngine
        """
        raise NotImplementedError

    def list_ai_function_descriptions(self):
        """
        Returns list of pairs (name, any-object)
        where name is a string
        and any-object will be passed to `construct_ai_function`
        """
        raise NotImplementedError

    def construct_ai_function(self, team, parameters):
        """
        Returns a function that will be passed as ai1/ai2 to StbEngine
        """
        raise NotImplementedError


class DefaultAIModule(AIModule):
    def __init__(self, bot_initializers):
        self.bot_initializers = bot_initializers

    def list_bot_initializers(self):
        return self.bot_initializers

    def list_ai_function_descriptions(self):
        return []

    def construct_ai_function(self, team, parameters):
        raise RuntimeError("should never be called")

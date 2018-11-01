from tornado import web


class SimulationNotFound(web.HTTPError):

    def __init__(self, sim_id):
        super(SimulationNotFound, self).__init__(
            404, 'Simulation "%s" not found', sim_id)


class BotInitializerNotFound(web.HTTPError):

    def __init__(self, bot_initializer_id):
        super(BotInitializerNotFound, self).__init__(
            404, 'Bot initializer with id=%s not found', bot_initializer_id)


class AiModuleNotFound(web.HTTPError):

    def __init__(self, ai_id):
        super(AiModuleNotFound, self).__init__(
            404, 'AI module with id=%s not found', ai_id)


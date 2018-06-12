
class BaseAI:

    def __init__(self, team, engine):
        """
        :type team: int
        :type engine: strateobots.engine.StbEngine
        """
        self.team = team
        self.engine = engine

    def initialize(self):
        raise NotImplementedError

    def tick(self):
        raise NotImplementedError


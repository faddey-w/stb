
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

    def x_to_team_field(self, x0to1):
        idx = self.engine.teams.index(self.team)
        return self.engine.world_width * (x0to1 + idx) / 2

    def train(self):
        pass


class DuelAI(BaseAI):

    def _get_bots(self):
        our = enemy = ctl = None
        for b in self.engine.iter_bots():
            if b.team == self.team:
                our = b
                ctl = self.engine.get_control(b)
            else:
                enemy = b
        if enemy is None and ctl is not None:
            ctl.fire = False
            ctl.move = 0
            ctl.rotate = 0
            ctl.tower_rotate = 0
        return our, enemy, ctl

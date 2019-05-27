import tensorflow as tf
from strateobots.engine import BotType
from .train import AINDuelAI, model_based_function, adopt_handcrafted_function
from ..lib import model_saving, handcrafted


def AI(team, engine):
    if team == engine.teams[0]:
        state = _GlobalState.get()
        state.mgr.load_vars(state.session)
        function = state.function
    else:
        function = adopt_handcrafted_function(handcrafted.distance_attack)
    return AINDuelAI.parametrize(function=function, bot_type=BotType.Raider)(
        team, engine
    )


class _GlobalState:

    _instance = None

    def __init__(self):
        self.session = tf.Session()
        mgr = model_saving.ModelManager.load_existing_model("_data/AIN")
        mgr.load_vars(self.session)
        self.mgr = mgr
        self.model = mgr.model
        self.function = model_based_function(self.model, self.session)

    @classmethod
    def get(cls):
        """:rtype: _GlobalState"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

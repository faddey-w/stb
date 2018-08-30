from strateobots.ai.lib.data import action2vec


class ExplorationFunction:

    def __init__(self, replay_memory):
        self._states_tree = None
        self._actions = None
        self._mem = replay_memory
        self._state_vector = None
        self.update_index()

    def update_index(self):
        states, actions, _ = self._mem.get_last_entries(self._mem.used_size)
        self._actions = actions
        raise NotImplementedError

    def find_action_to_explore(self):
        raise NotImplementedError

    def set_state_vector(self, state_vector):
        self._state_vector = state_vector

    def __call__(self, bot, enemy, control, engine):
        assert self._state_vector is not None
        _, action = self.find_action_to_explore()
        action2vec.restore(action, control)

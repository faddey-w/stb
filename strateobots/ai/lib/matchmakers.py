import os
from ai.lib.model_saving import ModelManager, get_model_definition_hash
from ai.lib.integration import ModelFunction


class MatchmakerStorage:
    def __init__(self, directory):
        self.directory = directory

    def list_names(self):
        if os.path.exists(self.directory):
            return sorted(os.listdir(self.directory), key=int)
        else:
            return []

    def save(self, model_mgr, session):
        next_name = str(len(self.list_names()) + 1)
        save_path = os.path.join(self.directory, next_name)
        model_mgr.save_definition(save_path)
        model_mgr.save_vars(session, save_path=save_path)

    def load(self, name, session, manager=None, name_scope=None):
        save_path = self.make_path(name)
        if manager is None:
            manager = ModelManager.load_existing_model(save_path, name_scope)
        manager.load_vars(session, save_path, skip_not_changed=False)
        return manager

    def make_path(self, name):
        return os.path.join(self.directory, name)


class MatchmakerFunction:
    def __init__(self, matchmaker_store, session):
        self.storage = matchmaker_store  # type: MatchmakerStorage
        self.session = session
        self._managers = {}
        self._functions = {}
        self._current_name = None
        self._current_function = None

    def set_model(self, name):
        if name == self._current_name:
            return
        save_path = self.storage.make_path(name)
        def_hash = get_model_definition_hash(save_path)
        if def_hash not in self._managers:
            mgr = self.storage.load(name, self.session, name_scope=def_hash)
            func = ModelFunction(mgr.model, self.session)
            self._managers[def_hash] = mgr
            self._functions[def_hash] = func
        else:
            self.storage.load(name, self.session, self._managers[def_hash])
            func = self._functions[def_hash]
        self._current_name = name
        self._current_function = func

    def __call__(self, params):
        return self._current_function(params)

    def new_match(self, params):
        return self._current_function.new_match(params)

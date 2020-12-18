import logging
import time
import dataclasses
from tornado import gen
from stb.game import StbGame
from stb.engine import StbEngine
from stb.util import replay_descriptor_from_simulation
from stb.util import replay_descriptor_from_storage
from stb.util import make_metadata_before_game
from stb.util import fill_metadata_after_game
from stb.visualizer_app.exceptions import BotInitializerNotFound
from stb.visualizer_app.exceptions import AiModuleNotFound
from stb.visualizer_app.exceptions import SimulationNotFound
from stb.visualizer_app import config
from stb.replay import SimulationNotFound


log = logging.getLogger(__name__)


class ServerState:
    def __init__(self, ai_modules, storage):
        """
        :type storage: stb.replay.ReplayDataStorage
        """
        self.bot_initializers = []
        self.ai_function_descriptors = []
        self.storage = storage
        self._request_queue = []
        self.ai_modules = ai_modules

        self.refresh_launch_params()

    def add_game_simulation(self, bot_initializer_id, ai1_id, ai2_id) -> "SimulationState":
        try:
            bot_init_name, bot_init = self.bot_initializers[bot_initializer_id]
        except IndexError:
            raise BotInitializerNotFound(bot_initializer_id)
        try:
            func_name1, params1, ai1_mod = self.ai_function_descriptors[ai1_id]
        except IndexError:
            raise AiModuleNotFound(ai1_id)
        try:
            func_name2, params2, ai2_mod = self.ai_function_descriptors[ai2_id]
        except IndexError:
            raise AiModuleNotFound(ai2_id)

        team1, team2 = StbEngine.TEAMS
        ai1 = ai1_mod.construct_ai_function(team1, params1)
        ai2 = ai2_mod.construct_ai_function(team2, params2)

        simul = self._make_simulation(
            ai1,
            ai2,
            bot_init,
            metadata=make_metadata_before_game(
                init_name=bot_init_name,
                ai1_module=ai1_mod.name,
                ai1_name=func_name1,
                ai2_module=ai2_mod.name,
                ai2_name=func_name2,
            ),
        )
        log.info(
            "ENQUEUE simulation bot_init=%s ai1=%s ai2=%s -> %r",
            bot_init_name,
            "{}:{}".format(ai1_mod.name, func_name1),
            "{}:{}".format(ai2_mod.name, func_name2),
            simul.sim_id,
        )
        self._request_queue.append(simul)
        self._run_simulation(simul)
        return simul

    def list_replays(self) -> "list":
        running_replays = [replay_descriptor_from_simulation(sim) for sim in self._request_queue]

        keys = self.storage.list_keys()
        finished_replays = [replay_descriptor_from_storage(self.storage, key) for key in keys]

        replays = running_replays + finished_replays
        replays.sort(key=lambda rep: rep["id"], reverse=True)
        return replays

    def get_replay_data(self, sim_id: "str") -> "list":
        try:
            return self.storage.load_replay_data(sim_id)
        except KeyError:
            raise SimulationNotFound(sim_id)

    def remove_replay(self, sim_id):
        log.info("REMOVE simulation %r", sim_id)
        for simul in self._request_queue:
            if sim_id == simul.sim_id:
                simul.cancelled = True
                return
        else:
            self.storage.remove_replay(sim_id)

    def refresh_launch_params(self):
        self.bot_initializers = []
        self.ai_function_descriptors = []

        for ai_m in self.ai_modules:
            self.bot_initializers.extend(ai_m.list_bot_initializers())
            for func_name, params in ai_m.list_ai_function_descriptions():
                self.ai_function_descriptors.append((func_name, params, ai_m))

    @property
    def queue_size(self):
        return len(self._request_queue)

    @property
    def currently_runs(self):
        return len(self._request_queue) > 0

    def _make_simulation(self, ai1, ai2, bot_init, metadata):
        game = StbGame(
            StbEngine(max_ticks=config.GAME_MAX_TICKS, wait_after_win=1), ai1, ai2, bot_init
        )
        sim_id = time.strftime("%Y%m%d_%H%M%S")
        ex_keys = self._list_used_keys()
        suffix = 0
        unique_sim_id = sim_id
        while unique_sim_id in ex_keys:
            suffix += 1
            unique_sim_id = "{}_{}".format(sim_id, suffix)
        simul = SimulationState(game, metadata, unique_sim_id)
        return simul

    @gen.coroutine
    def _run_simulation(self, simul):
        # this is CPU-intensive, so yield between ticks
        log.info("RUN simulation %r", simul.sim_id)
        while not simul.game.is_finished and not simul.cancelled:
            yield
            simul.game.play_one_tick()
            if simul.game.engine.nticks % 100 == 0:
                log.debug("TICK %r: %s", simul.sim_id, simul.game.engine.nticks)
        yield
        fill_metadata_after_game(simul.metadata, simul.game)
        log.info(
            "FINISH simulation %r %s",
            simul.sim_id,
            "(cancelled)" if simul.cancelled else "",
        )
        if not simul.cancelled:
            self.storage.save_replay(simul.sim_id, simul.metadata, simul.game.replay)
        self._request_queue.remove(simul)

    def _list_used_keys(self):
        return self.storage.list_keys() + [sim.sim_id for sim in self._request_queue]


@dataclasses.dataclass
class SimulationState:
    game: StbGame
    metadata: dict
    sim_id: str
    cancelled: bool = False

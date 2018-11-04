import logging
import time
from tornado import gen
from strateobots.engine import StbEngine
from strateobots.util import replay_descriptor_from_simulation
from strateobots.util import replay_descriptor_from_storage
from strateobots.visualizer_app.exceptions import BotInitializerNotFound
from strateobots.visualizer_app.exceptions import AiModuleNotFound
from strateobots.visualizer_app.exceptions import SimulationNotFound
from strateobots.visualizer_app import config


log = logging.getLogger(__name__)


class ServerState:
    def __init__(self, ai_modules, storage):
        """
        :type storage: strateobots.replay.ReplayDataStorage
        """
        self.bot_initializers = []
        self.ai_function_descriptors = []
        self.storage = storage
        self._request_queue = []

        for ai_m in ai_modules:
            self.bot_initializers.extend(ai_m.list_bot_initializers())
            for func_name, params in ai_m.list_ai_function_descriptions():
                self.ai_function_descriptors.append((func_name, params, ai_m))

    def add_game_simulation(self, bot_initializer_id, ai1_id, ai2_id) -> 'SimulationState':
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
            metadata=dict(
                init_name=bot_init_name,
                ai1_module=ai1_mod.name,
                ai1_name=func_name1,
                ai2_module=ai2_mod.name,
                ai2_name=func_name2,
            ),
            params=dict(
                world_width=config.WORLD_WIDTH,
                world_height=config.WORLD_HEIGHT,
                ai1=ai1,
                ai2=ai2,
                initialize_bots=bot_init,
                max_ticks=config.GAME_MAX_TICKS,
                wait_after_win=1,
                stop_all_after_finish=True,
            ),
        )
        log.info('ENQUEUE simulation bot_init=%s ai1=%s ai2=%s -> %r',
                 bot_init_name,
                 '{}:{}'.format(ai1_mod.name, func_name1),
                 '{}:{}'.format(ai2_mod.name, func_name2),
                 simul.sim_id)
        self._request_queue.append(simul)
        self._run_simulation(simul)
        return simul

    def list_replays(self) -> 'list':
        running_replays = [
            replay_descriptor_from_simulation(sim)
            for sim in self._request_queue
        ]

        keys = self.storage.list_keys()
        finished_replays = [
            replay_descriptor_from_storage(self.storage, key)
            for key in keys
        ]

        replays = running_replays + finished_replays
        replays.sort(key=lambda rep: rep['id'], reverse=True)
        return replays

    def get_replay_data(self, sim_id: 'str') -> 'list':
        try:
            return self.storage.load_replay_data(sim_id)
        except KeyError:
            raise SimulationNotFound(sim_id)

    def remove_replay(self, sim_id):
        log.info('REMOVE simulation %r', sim_id)
        self.storage.remove_replay(sim_id)

    @property
    def queue_size(self):
        return len(self._request_queue)

    @property
    def currently_runs(self):
        return len(self._request_queue) > 0

    def _make_simulation(self, metadata, params):
        engine = StbEngine(**params)
        sim_id = time.strftime('%Y%m%d_%H%M%S')
        ex_keys = self._list_used_keys()
        suffix = 0
        unique_sim_id = sim_id
        while unique_sim_id in ex_keys:
            suffix += 1
            unique_sim_id = '{}_{}'.format(sim_id, suffix)
        simul = SimulationState(engine, metadata, unique_sim_id)
        return simul

    @gen.coroutine
    def _run_simulation(self, simul):
        # this is CPU-intensive, so yield between ticks
        log.info('RUN simulation %r', simul.sim_id)
        while not simul.engine.is_finished and not simul.cancelled:
            yield
            simul.engine.tick()
            if simul.engine.nticks % 100 == 0:
                log.debug('TICK %r: %s', simul.sim_id, simul.engine.nticks)
        yield
        simul.metadata['nticks'] = simul.engine.nticks
        log.info('FINISH simulation %r', simul.sim_id)
        self.storage.save_replay(simul.sim_id, simul.metadata, simul.engine.replay)
        self._request_queue.remove(simul)

    def _list_used_keys(self):
        return self.storage.list_keys() + [sim.sim_id for sim in self._request_queue]


class SimulationState:
    def __init__(self, engine, metadata, sim_id):
        """
        :type engine: strateobots.engine.StbEngine
        """
        self.sim_id = sim_id
        self.engine = engine
        self.metadata = metadata
        self.cancelled = False


import argparse
import importlib
import json
import logging.config
import random
import textwrap
import sys
from tornado import web, gen, ioloop
from strateobots.engine import StbEngine


log = logging.getLogger('strateobots.engine_server')


class schema:

    @staticmethod
    def bot(bot):
        return {
            'type': bot.type.code,
            'x': bot.x,
            'y': bot.y,
            'team': bot.team,
            'orientation': bot.orientation,
            'tower_orientation': bot.tower_orientation,
            'hp': bot.hp_ratio,
            'load': bot.load,
            'has_shield': bot.has_shield,
        }

    @staticmethod
    def bullet(bullet):
        return {
            'type': bullet.type.code,
            'x': bullet.x,
            'y': bullet.y,
            'orientation': bullet.orientation,
            'range': bullet.range,
        }

    @staticmethod
    def explosion(explosion):
        return {
            'x': explosion.x,
            'y': explosion.y,
            't': explosion.t_ratio,
            's': explosion.size,
        }


def get_ai_module():
    from strateobots import ai
    ai = importlib.reload(ai)
    log.info('LOADING AI: %s', ai.default_ai_name)
    ai_fullname = 'strateobots.ai.' + ai.default_ai_name
    for modname, mod in sys.modules.items():
        if modname.startswith('strateobots.ai.lib.'):
            importlib.reload(mod)
    if ai_fullname in sys.modules:
        ai_module = importlib.reload(sys.modules[ai_fullname])
    else:
        ai_module = importlib.import_module(ai_fullname)
    return ai_module


class SimulationState:
    def __init__(self, engine):
        """
        :type engine: strateobots.engine.StbEngine
        """
        self.engine = engine
        self.ticks_data = [self._render_state()]
        self.cancelled = False

    def process_tick(self):
        self.engine.tick()
        self.ticks_data.append(self._render_state())

    def _render_state(self):
        bots_data = [
            schema.bot(bot)
            for bot in self.engine.iter_bots()
        ]
        bullets_data = [
            schema.bullet(bullet)
            for bullet in self.engine.iter_bullets()
        ]
        rays_data = [
            schema.bullet(ray)
            for ray in self.engine.iter_rays()
        ]
        explosions_data = [
            schema.explosion(expl)
            for expl in self.engine.iter_explosions()
        ]
        return dict(
            bots=bots_data,
            bullets=bullets_data,
            rays=rays_data,
            explosions=explosions_data,
        )


class ServerState:
    def __init__(self, max_queue, expire_time, delay, extra_params):
        self.max_queue = max_queue
        self.delay = delay
        self.expire_time = expire_time
        self.extra_params = extra_params
        self._request_queue = []
        self._next_id = 1
        self._simulations = {}
        self._currently_runs = False

    def add_request(self, **params) -> 'str':
        params.update(self.extra_params)
        sim_id, simul = self._make_simulation(**params)
        log.info('ENQUEUE simulation params=%s -> %r', params, sim_id)
        if self._currently_runs:
            if len(self._request_queue) >= self.max_queue:
                raise RequestQueueIsFull()
            self._request_queue.append(sim_id)
        else:
            self._run_simulation(sim_id)
        return sim_id

    def get_simulation(self, sim_id: 'str') -> 'SimulationState':
        try:
            return self._simulations[sim_id]
        except KeyError:
            raise SimulationNotFound(sim_id)

    @property
    def queue_size(self):
        return len(self._request_queue)

    @property
    def currently_runs(self):
        return self._currently_runs

    def _make_simulation(self, **params):
        ai_cls = get_ai_module().AI
        engine = StbEngine(ai1_cls=ai_cls, ai2_cls=ai_cls, **params)
        simul = SimulationState(engine)
        sim_id = hex(abs(hash('{}_{}'.format(random.random(), self._next_id))))[2:]
        self._next_id += 1
        self._simulations[sim_id] = simul
        return sim_id, simul

    @gen.coroutine
    def _run_simulation(self, sim_id):
        # this is CPU-intensive, so yield between each tick
        log.info('RUN simulation %r', sim_id)
        self._currently_runs = True
        simul = self._simulations[sim_id]
        while not simul.engine.is_finished and not simul.cancelled:
            yield
            simul.process_tick()
            log.debug('TICK %r', sim_id)
        self._run_expiration(sim_id)
        yield gen.sleep(self.delay)
        if self._request_queue:
            self._run_simulation(self._request_queue.pop(0))
        else:
            self._currently_runs = False

    @gen.coroutine
    def _run_expiration(self, sim_id):
        yield gen.sleep(self.expire_time)
        log.info('EXPIRE simulation %r', sim_id)
        self._simulations.pop(sim_id)


class _BaseHandler(web.RequestHandler):

    def initialize(self, serverstate=None):
        self.state = serverstate  # type: ServerState

    def api_respond(self, data, http_code=200):
        self.set_status(http_code)
        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps(data))

    def options(self, *args, **kwargs):
        result = []
        for meth in self.SUPPORTED_METHODS:
            methfunc = getattr(self, meth.lower())
            if methfunc.__doc__:
                result.append({
                    'method': meth,
                    'description': textwrap.dedent(methfunc.__doc__),
                })
        self.api_respond(result)


class EnqueueSimulationHandler(_BaseHandler):

    def get(self):
        """
        Show current load status - whether simulation runs and how many
        simulation requests are currently enqueued.
        """
        self.api_respond({
            'queue_size': self.state.queue_size,
            'max_queue': self.state.max_queue,
            'currently_runs': self.state.currently_runs,
        })

    def post(self):
        """
        Creates a new simulation.
        Expects the following parameters:
         - width: int - width of simulation field
         - height: int - height of simulation field
        """
        w = int(self.get_body_argument('width'))
        h = int(self.get_body_argument('height'))
        sim_id = self.state.add_request(world_width=w, world_height=h)
        self.api_respond({'id': sim_id}, http_code=201)


class SimulationStatusHandler(_BaseHandler):

    def get(self, sim_id):
        """
        Shows current status of specified simulation
        """
        log.debug('looking for simulation %r', sim_id)
        simul = self.state.get_simulation(sim_id)
        nticks = len(simul.ticks_data)
        self.api_respond({
            'id': sim_id,
            'started': nticks > 1,
            'ticks_generated': nticks,
            'finished': simul.engine.is_finished,
            'cancelled': simul.cancelled,
        })

    def delete(self, sim_id):
        """
        Cancels simulation. Generated data will be preserved until expiration,
        but generation of next ticks will be stopped.
        """
        self.state.get_simulation(sim_id).cancelled = True
        self.set_status(200)


class SimulationDataHandler(_BaseHandler):

    def get(self, sim_id):
        """
        Returns segment of generated data. Query parameters:
         - start: int - index of first tick in segment
         - count: int - number of ticks to send, length of segment
        """
        start = int(self.get_query_argument('start'))
        count = int(self.get_query_argument('count'))
        simul = self.state.get_simulation(sim_id)
        data = simul.ticks_data[start:start+count]
        self.api_respond({
            'count': len(data),
            'data': data
        })


class RequestQueueIsFull(web.HTTPError):

    def __init__(self):
        super(RequestQueueIsFull, self).__init__(
            503, "Simulation queue full")


class SimulationNotFound(web.HTTPError):

    def __init__(self, sim_id):
        super(SimulationNotFound, self).__init__(
            404, 'Simulation "%s" not found', sim_id)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--static-dir', required=True)
    parser.add_argument('--port', '-P', default=9999, type=int)
    parser.add_argument('--max-queue', '-Q', default=10, type=int)
    parser.add_argument('--max-ticks', '-T', default=2000, type=int)
    parser.add_argument('--expire-time', '-E', default=7200, type=int)
    parser.add_argument('--delay', '-D', default=5, type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv)
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {
            'stderr': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'default',
            }
        },
        'filters': {},
        'formatters': {
            'default': {
                'format': '%(levelname)s:%(name)s:%(message)s',
            }
        },
        'loggers': {
            '': {
                'handlers': ['stderr'],
                'level': 'INFO' if args.debug else 'WARNING',
            },
            'strateobots': {
                'handlers': ['stderr'],
                'level': 'DEBUG' if args.debug else 'INFO',
                'propagate': False,
            }
        }
    })

    state = ServerState(max_queue=args.max_queue,
                        expire_time=args.expire_time,
                        delay=args.delay,
                        extra_params={
                            'max_ticks': args.max_ticks
                        })
    initargs = dict(serverstate=state)
    fileserver_args = dict(path=args.static_dir, default_filename='index.html')
    app = web.Application([
        ('/api/v1/simulation', EnqueueSimulationHandler, initargs),
        ('/api/v1/simulation/([0-9a-f]+)', SimulationStatusHandler, initargs),
        ('/api/v1/simulation/([0-9a-f]+)/data', SimulationDataHandler, initargs),
        ('/(.*)', web.StaticFileHandler, fileserver_args)
    ], debug=args.debug)
    app.listen(args.port)
    log.info('listening at 0.0.0.0:%s', args.port)
    ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    main()

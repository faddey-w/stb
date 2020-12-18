import logging
import textwrap
import json
from tornado import web, escape
from stb import util, replay
from stb.visualizer_app import exceptions


log = logging.getLogger(__name__)


class CertificateSignatureAuthHandler:
    def __init__(self, certificate_path):
        from stb import cryptoutil
        with open(certificate_path) as f:
            self.key = cryptoutil.using_key(f.read())

    def __call__(self, request):
        # TODO parse signature from request
        pass


def noop_auth_handler(request):
    pass


class _BaseHandler(web.RequestHandler):
    def initialize(self, serverstate=None, auth_handler=None, program_storage=None):
        """
        :type serverstate: stb.engine_server.controller.ServerState
        :type auth_handler: callable
        """
        self.state = serverstate
        self.authenticate = auth_handler
        self.program_storage = program_storage

    def api_respond(self, data, http_code=200):
        self.set_status(http_code)
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(data))

    def options(self, *args, **kwargs):
        result = []
        for meth in self.SUPPORTED_METHODS:
            methfunc = getattr(self, meth.lower())
            if methfunc.__doc__:
                result.append(
                    {"method": meth, "description": textwrap.dedent(methfunc.__doc__)}
                )
        self.api_respond(result)

    def get_int(self, value):
        try:
            return int(value)
        except:
            raise web.HTTPError(400)

    def get_json_argument(self, key):
        if not hasattr(self.request, "json_data"):
            self.request.json_data = escape.json_decode(self.request.body)
        return self.request.json_data[key]

    def check_etag_header(self):
        return False


class GameLaunchParametersHandler(_BaseHandler):
    def get(self):
        self.state.refresh_launch_params()
        self.api_respond(
            {
                "bot_initializers": [
                    {"name": name} for name, func in self.state.bot_initializers
                ],
                "ai_functions": [
                    {"module": ai_m.name, "name": name}
                    for name, params, ai_m in self.state.ai_function_descriptors
                ],
            }
        )


class GameListHandler(_BaseHandler):
    def get(self):
        """
        Lists game replays available for view
        """
        replays = self.state.list_replays()
        self.api_respond({"result": replays})

    def post(self):
        """
        Creates a new game.
        Expects the following parameters:
         - initializer_id: id of bot initializer to use
         - ai1_id: id of AI for team 1
         - ai2_id: id of AI for team 2
        """
        self.authenticate(self.request)
        init_id = self.get_int(self.get_json_argument("initializer_id"))
        ai1_id = self.get_int(self.get_json_argument("ai1_id"))
        ai2_id = self.get_int(self.get_json_argument("ai2_id"))
        simul = self.state.add_game_simulation(init_id, ai1_id, ai2_id)
        self.api_respond(util.replay_descriptor_from_simulation(simul), http_code=201)


class GameViewHandler(_BaseHandler):
    def get(self, sim_id):
        """
        Returns segment of generated data. Query parameters:
         - start: int - index of first tick in segment
         - count: int - number of ticks to send, length of segment
        """
        start = self.get_int(self.get_query_argument("start"))
        count = self.get_int(self.get_query_argument("count"))
        try:
            data = self.state.get_replay_data(sim_id)
        except replay.SimulationNotFound as snf:
            raise exceptions.SimulationNotFound(snf.key)
        total_len = len(data)
        data = data[start : start + count]
        self.api_respond({"count": len(data), "total": total_len, "data": data})

    def delete(self, sim_id):
        """
        Removes simulation from the storage
        """
        self.authenticate(self.request)
        try:
            self.state.remove_replay(sim_id)
        except replay.SimulationNotFound as snf:
            raise exceptions.SimulationNotFound(snf.key)
        self.set_status(204)


class UserProgramsViewHandler(_BaseHandler):
    def post(self, program_name):
        if self.program_storage is None:
            raise web.HTTPError(404)
        self.authenticate(self.request)
        self.program_storage.save_program(
            program_name, self.request.body.decode("utf-8")
        )
        self.state.refresh_launch_params()

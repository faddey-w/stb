from stb.ai import base


class AIModule(base.AIModule):
    def __init__(self, guiding_ai_module, guided_ai_module):
        self.guiding_ai_module = guiding_ai_module
        self.guided_ai_module = guided_ai_module

    def list_ai_function_descriptions(self):
        return [
            ('"{}" guided by "{}"'.format(n2, n1), (p1, p2))
            for n1, p1 in self.guiding_ai_module.list_ai_function_descriptions()
            for n2, p2 in self.guided_ai_module.list_ai_function_descriptions()
        ]

    def list_bot_initializers(self):
        return []

    def construct_ai_function(self, team, parameters):
        p1, p2 = parameters
        guide = self.guiding_ai_module.construct_ai_function(team, p1)
        guided = self.guided_ai_module.construct_ai_function(team, p2)
        return guided_ai_function(guide, guided)


def guided_ai_function(guide_func, demo_func):
    def function(state):
        resp = guide_func(state)
        demo_resp = demo_func(state)

        resp_ctls = resp["controls"] if isinstance(resp, dict) else resp
        demo_ctls = demo_resp["controls"] if isinstance(demo_resp, dict) else demo_resp

        resp_ctls = {ctl["id"]: ctl for ctl in resp_ctls}
        demo_ctls = {ctl["id"]: ctl for ctl in demo_ctls}

        for bot_id, demo_ctl in demo_ctls.items():
            if bot_id in resp_ctls:
                ctl = resp_ctls[bot_id]
                if "orientation" in demo_ctl:
                    ctl["orientation"] = demo_ctl["orientation"]
                if "gun_orientation" in demo_ctl:
                    ctl["gun_orientation"] = demo_ctl["gun_orientation"]
                if "move_aim_x" in demo_ctl:
                    ctl["move_aim_x"] = demo_ctl["move_aim_x"]
                    ctl["move_aim_y"] = demo_ctl["move_aim_y"]
                if "gun_aim_x" in demo_ctl:
                    ctl["gun_aim_x"] = demo_ctl["gun_aim_x"]
                    ctl["gun_aim_y"] = demo_ctl["gun_aim_y"]
        return resp

    return function

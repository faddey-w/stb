from math import pi, acos, sqrt, asin, copysign, cos, sin, atan2
import numpy as np

# from scipy import optimize
from strateobots.engine import dist_points, vec_len, dist_line, vec_dot
from strateobots.engine import Constants, BotType, StbEngine
from strateobots.ai.simple_duel import (
    norm_angle,
    navigate_gun,
    navigate_shortest,
    should_fire,
)
from strateobots.util import objedict
import itertools
import random
from functools import partial
from . import base
import logging


log = logging.getLogger(__name__)


class AIModule(base.AIModule):
    def __init__(self):
        self.config = {
            # should be: depth * resolution ~= 150
            # for all branching^depth should be approximately same
            "MCTS-0": lambda: MCTSAiFunction(20, [(2, 4)]),
            "MCTS-1": lambda: MCTSAiFunction(37, [(4, 3), (2, 1)]),
            "MCTS-2": lambda: MCTSAiFunction(30, [(3, 3), [2, 2]]),
            "MCTS-3": lambda: MCTSAiFunction(22, [(2, 7)]),
        }

    def list_ai_function_descriptions(self):
        return [(name, name) for name, func in self.config.items()]

    def list_bot_initializers(self):
        return []

    def construct_ai_function(self, team, name):
        return self.config[name]()


class MCTSAiFunction:
    def __init__(self, resolution, branching_config):
        self.resolution = resolution
        self.branching = []
        for n_br, dep in branching_config:
            self.branching.extend([n_br] * dep)
        self.depth = len(self.branching)
        self.enemy_load = 1
        self.since_enemy_shield = 9999999999
        self.since_enemy_fire = 9999999999
        self.dt = 1.0 / Constants.ticks_per_sec

    def __call__(self, state):
        bot = state["friendly_bots"][0]
        enemy = state["enemy_bots"][0]
        bottype = BotType.by_code(bot["type"])
        enemytype = BotType.by_code(enemy["type"])

        if enemy["is_firing"]:
            if enemytype.shots_ray:
                self.enemy_load -= self.dt * Constants.ray_charge_per_sec
            else:
                self.enemy_load = 0
            self.since_enemy_fire = 0
        else:
            self.enemy_load += self.dt / enemytype.cd_period
            self.since_enemy_fire += 1
        if enemy["has_shield"]:
            self.since_enemy_shield = 0
        else:
            self.since_enemy_shield += 1

        possible_time_warming = self.dt * min(
            self.since_enemy_shield, self.since_enemy_fire
        )
        warmup = max(1.0, possible_time_warming / enemytype.shield_warmup_period)

        value, ctl = mcts_ai(
            bot,
            enemy,
            bottype,
            enemytype,
            resolution=self.resolution,
            depth=self.depth,
            heuristic=_heuristic,
            branching=self.branching,
            enemy_modelled_fields={
                "load": min(1, self.enemy_load),
                "shield_warmup": warmup,
            },
        )
        # ctl = {'id': bot['id']}
        # bot = objedict(bot)
        # enemy = objedict(enemy)
        # ctl.update(_target_enemy(bot, enemy))
        # ctl.update(_fire_shield_policy(bot, enemy, bottype, False))
        # ctl.update(_flank_move(bot, enemy, -1))
        return [ctl]


def mcts_ai(
    bot,
    enemy,
    bottype,
    enemytype,
    resolution,
    depth,
    branching,
    heuristic,
    enemy_modelled_fields,
):

    all_actions = tuple(
        itertools.product(
            [_target_enemy],
            [
                partial(_fire_shield_policy, bottype=bottype, shield=True),
                partial(_fire_shield_policy, bottype=bottype, shield=False),
            ],
            [
                partial(_frontal_move, direction=+1),
                partial(_frontal_move, direction=-1),
                partial(_flank_move, side=+1),
                partial(_flank_move, side=-1),
            ],
        )
    )
    n_actions = len(all_actions)

    def sim_ai():
        def function(_state):
            b = _state["friendly_bots"][0]
            e = _state["enemy_bots"][0]
            ctl = {"id": b.id}
            for action in function.actions_list:
                ctl.update(action(b, e))
            return [ctl]

        function.actions_list = []
        return function

    initial_game = StbEngine(
        ai1=sim_ai(),
        ai2=sim_ai(),
        initialize_bots=lambda g: None,
        wait_after_win_ticks=0,
    )
    sim_bot = initial_game.add_bot(
        bottype,
        initial_game.team1,
        bot["x"],
        bot["y"],
        bot["orientation"],
        bot["tower_orientation"],
        bot["hp"] * bottype.max_hp,
    )
    sim_bot.is_firing = bot["is_firing"]
    sim_bot.load = bot["load"]
    sim_bot.shield = bot["shield"]
    sim_bot.shield_warmup = bot["shield_warmup"]
    sim_bot.vx = bot["vx"]
    sim_bot.vy = bot["vy"]

    sim_enemy = initial_game.add_bot(
        enemytype,
        initial_game.team2,
        enemy["x"],
        enemy["y"],
        enemy["orientation"],
        enemy["tower_orientation"],
        enemy["hp"] * enemytype.max_hp,
    )
    sim_enemy.is_firing = enemy["is_firing"]
    sim_enemy.shield = enemy["shield"]
    sim_enemy.vx = enemy["vx"]
    sim_enemy.vy = enemy["vy"]
    for key, val in enemy_modelled_fields.items():
        setattr(sim_enemy, key, val)

    initial_branches = [(i, random.randrange(n_actions)) for i in range(n_actions)]

    value_stats = np.zeros([n_actions, 2], dtype=np.float)
    for st_act_idx in initial_branches:
        v, (i, _) = _treesearch_subtree(
            initial_game,
            st_act_idx,
            all_actions,
            branching,
            resolution,
            heuristic,
            depth,
        )
        value_stats[i] += v, 1

    values = value_stats[:, 0] / np.maximum(1, value_stats[:, 1])
    best_action_i = np.argmax(values)
    value, best_action_list = values[best_action_i], all_actions[best_action_i]

    ctl = {"id": bot["id"]}
    bot_obj = objedict(bot)
    enemy_obj = objedict(enemy)
    for action in best_action_list:
        ctl.update(action(bot_obj, enemy_obj))

    return value, ctl


def _treesearch_subtree(
    initial_game,
    starting_action_idx,
    all_actions,
    branching,
    resolution,
    heuristic,
    depth,
):
    n_actions = len(all_actions)
    stack = [
        _Matrix(n_actions, initial_game, 1, explicit_branches=[starting_action_idx])
    ]
    while stack:
        top = stack[-1]
        n_branches = branching[len(stack) - 1]
        if top.tests_left > 0:
            game = top.game.clone(
                with_explosions=False, with_replay=False, using_class=_SimulationEngine
            )

            bot_act_idx, enemy_act_idx = top.next_idx
            game.ai1.actions_list[:] = all_actions[bot_act_idx]
            game.ai1.actions_list[:] = all_actions[enemy_act_idx]
            for _ in range(resolution):
                game.tick()
            if game.has_error:
                _, e, tb = game.exc_info
                raise e.with_traceback(tb)

            if game.is_finished or len(stack) >= depth:
                value = heuristic(game)
                top.add_test(value)
            else:
                stack.append(_Matrix(n_actions, game, n_branches))
        else:
            # bot_probs = _solve_zero_sum_game(top.mat)
            # value = np.sum(top.mat[:, 0] * bot_probs)
            stack.pop(-1)
            if stack:
                value = np.sum(top.values) / n_branches
                prev = stack[-1]
                # prev.mat[prev.next_idx] = value
                prev.add_test(value)
            else:
                i = np.argmax(top.values)
                value = top.values[i]
                return value, top.get_action_idx(i)


class _Matrix:
    def __init__(self, n_actions, game, n_branches, explicit_branches=None):
        self.game = game
        self.n_actions = n_actions
        # self.mat = np.zeros([n_actions, n_actions], dtype=np.float)
        self._selected_actions = (
            [
                (b_act, random.choice(range(n_actions)))
                for b_act in random.sample(range(n_actions), n_branches)
            ]
            if explicit_branches is None
            else explicit_branches
        )
        self.values = np.zeros([n_branches], dtype=np.float)
        self.tests_left = n_branches

    @property
    def next_idx(self):
        return self.get_action_idx(-self.tests_left)

    def get_action_idx(self, i):
        return self._selected_actions[i]

    def add_test(self, value):
        self.values[-self.tests_left] = value
        self.tests_left -= 1


class _SimulationEngine(StbEngine):
    def _serialize_game_state(self):
        team_bots_visible_data = {t: [] for t in self.teams}
        team_bots_full_data = {t: [] for t in self.teams}
        bullets = list(self.iter_bullets())
        rays = list(self.iter_rays())
        explosions = []
        for bot in self.iter_bots():
            team_bots_visible_data[bot.team].append(bot)
            team_bots_full_data[bot.team].append(bot)

        return team_bots_full_data, team_bots_visible_data, bullets, rays, explosions


def _heuristic(game):
    t1, t2 = game.teams
    n_t1 = n_t2 = 0
    hp_t1 = hp_t2 = 0
    for bot in game.iter_bots():
        if bot.team == t1:
            n_t1 += 1
            hp_t1 += bot.hp_ratio
        else:
            n_t2 += 1
            hp_t2 += bot.hp_ratio

    if n_t1 == 0:
        return -1
    if n_t2 == 0:
        return +1
    else:
        return hp_t1 / n_t1 - hp_t2 / n_t2


# def _solve_zero_sum_game(paymatrix):
#     n_a, n_b = paymatrix.shape
#
#     min_payment = np.min(paymatrix)
#     if min_payment > 0:
#         restr_mat = -paymatrix.T
#     else:
#         restr_mat = -(paymatrix.T - (min_payment - 1))
#     restr_b = -np.ones([n_b])
#     objective = np.ones([n_a])
#     result = optimize.linprog(objective, restr_mat, restr_b)  # type: optimize.OptimizeResult
#     assert result.status == 0
#
#     x_res = result.x
#     a_probs = x_res / np.sum(x_res)
#
#     return a_probs


def _target_enemy(bot, enemy):
    return {"tower_rotate": navigate_gun(bot, enemy)}


def _fire_shield_policy(bot, enemy, bottype, shield):
    if shield:
        fire = False
    else:
        fire = should_fire(bot, enemy, bottype.shot_range)
    return {"shield": shield, "fire": fire}


def _frontal_move(bot, enemy, direction):
    enemy_angle = atan2((enemy.y - bot.y), (enemy.x - bot.x))
    rotate = navigate_shortest(bot, enemy_angle, with_gun=False)
    return {"move": direction, "rotate": rotate}


def _flank_move(bot, enemy, side):
    orbit_radius = Constants.bot_radius * 3

    dist = dist_points(bot.x, bot.y, enemy.x, enemy.y)
    enemy_angle = atan2((enemy.y - bot.y), (enemy.x - bot.x))
    ori_angle = norm_angle(bot.orientation)

    pt_angle = asin(min(dist, orbit_radius) / dist)
    pt_angle = enemy_angle - side * pt_angle
    delta_angle = norm_angle(pt_angle - ori_angle)
    if delta_angle > 0:
        rotate = +1
    else:
        rotate = -1

    return {"move": +1, "rotate": rotate}


def _const_move(bot, enemy, move, rotate):
    return {"move": move, "rotate": rotate}

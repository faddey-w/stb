import random
import logging
import copy
from typing import Iterable
from math import sin, cos, sqrt
from stb.util import (
    vec_rotate,
    vec_len,
    vec_len2,
    vec_dot,
    vec_sum,
    dist_points,
    dist_line,
    half_chord_len,
)
from stb.models import (
    Constants,
    BotTypeProperties,
    BotModel,
    BulletModel,
    ExplosionModel,
    BotControl,
    Action,
)


log = logging.getLogger(__name__)


class StbEngine:

    TEAMS = "1", "2"

    def __init__(
        self, max_ticks=1000, wait_after_win=1, wait_after_win_ticks=None, teams=None,
    ):
        self.teams = self.team1, self.team2 = teams or self.TEAMS
        self._bots = {}
        self._rays = {}
        self._bullets = []
        self._explosions = []

        self._controls = {}

        self._n_bots = {self.team1: 0, self.team2: 0}

        self.nticks = 0
        self.max_ticks = max_ticks
        self._win_reached_at = None
        if wait_after_win_ticks is None:
            wait_after_win_ticks = Constants.ticks_per_sec * wait_after_win
        self._wait_after_win = max(1, wait_after_win_ticks)

    def clone(self, with_explosions, using_class=None):
        if using_class is None:
            using_class = self.__class__
        clone = using_class()
        clone.teams = self.teams
        clone.team1 = self.team1
        clone.team2 = self.team2
        clone._bots = {bot_id: copy.copy(bot) for bot_id, bot in self._bots.items()}
        clone._rays = {bot_id: copy.copy(ray) for bot_id, ray in self._rays.items()}
        clone._bullets = [copy.copy(bullet) for bullet in self._bullets]
        if with_explosions:
            clone._explosions = [copy.copy(expl) for expl in self._explosions]
        else:
            clone._explosions = []

        clone._controls = {bot_id: copy.copy(ctl) for bot_id, ctl in self._controls.items()}
        clone._n_bots = self._n_bots.copy()

        clone.nticks = self.nticks
        clone.max_ticks = self.max_ticks
        clone._win_reached_at = self._win_reached_at
        clone._wait_after_win = self._wait_after_win

        return clone

    def iter_bots(self) -> Iterable[BotModel]:
        return self._bots.values()

    def iter_rays(self) -> Iterable[BulletModel]:
        return self._rays.values()

    def iter_bullets(self) -> Iterable[BulletModel]:
        return self._bullets

    def iter_explosions(self) -> Iterable[ExplosionModel]:
        return self._explosions

    def get_control(self, bot) -> BotControl:
        return self._controls[bot.id]

    def get_bot(self, bot_id):
        return self._bots[bot_id]

    def add_bot(self, bottype, team, x, y, orientation, tower_orientation, hp=None):
        bot = BotModel(
            type=bottype,
            id=len(self._bots),
            orientation=orientation,
            tower_orientation=tower_orientation,
            hp=hp,
            team=team,
            x=x,
            y=y,
        )
        self._bots[bot.id] = bot
        self._controls[bot.id] = BotControl()
        self._n_bots[team] += 1
        return bot

    def tick(self):
        next_bullets = []
        next_rays = {}
        tps = float(Constants.ticks_per_sec)
        bullet_speed = Constants.bullet_speed / tps
        bot_radius = Constants.bot_radius
        friction_factor = Constants.friction_factor
        collision_factor = Constants.collision_factor
        rotation_smoothness = Constants.rotation_smoothness
        eps = Constants.epsilon
        min_collision_speed = Constants.min_collision_speed
        world_width = Constants.world_width
        world_height = Constants.world_height
        load_with_action = Constants.load_with_action
        shield_leak_factor = 1 - Constants.shield_half_leak_period ** (1 / tps)

        # move bullets
        for bullet in self._bullets:
            bullet.x += bullet_speed * bullet.cos
            bullet.y += bullet_speed * bullet.sin
            bullet.remaining_range -= bullet_speed
            if (
                0 <= bullet.x <= world_width
                and 0 <= bullet.y <= world_height
                and bullet.remaining_range > 0
            ):
                next_bullets.append(bullet)

        # move bots
        for b_id, bot in self._bots.items():
            ctl = self._controls[bot.id]  # type: BotControl
            typ = bot.type  # type: BotTypeProperties

            rot_speed = typ.rot_speed
            if ctl.action == Action.ACCELERATION:
                rot_speed += typ.bonus_rot_speed
            bot.rot_speed = (rotation_smoothness * bot.rot_speed + ctl.rotate * rot_speed) / (
                rotation_smoothness + 1
            )
            ori_change = little_noise(bot.rot_speed) / tps

            a_angle = bot.orientation + ori_change / 2
            a_sin = sin(a_angle)
            a_cos = cos(a_angle)

            # acceleration
            f_cos = f_sin = None
            if ctl.move != 0:
                # if we move in positive direction to acceleration
                # then engine accelerates in needed direction
                # and friction reduces other part of velocity vector
                a_cos *= ctl.move
                a_sin *= ctl.move
                # if vec_dot(a_cos, a_sin, v_cos, v_sin) > 0:
                av = bot.vx * a_cos + bot.vy * a_sin
                fvx = bot.vx - av * a_cos
                fvy = bot.vy - av * a_sin
                acc = typ.acc / tps
                if ctl.action == Action.ACCELERATION:
                    acc += typ.bonus_acc / tps
                bot.vx -= fvx
                bot.vy -= fvy
            else:
                acc = 0
                fvx = bot.vx
                fvy = bot.vy
                bot.vx = bot.vy = 0

            # friction
            fv = vec_len(fvx, fvy) or 1
            f_cos = f_cos if f_cos is not None else fvx / fv
            f_sin = f_sin if f_sin is not None else fvy / fv
            fax = friction_factor * f_cos / tps
            fay = friction_factor * f_sin / tps
            if abs(fvx) <= abs(fax):
                fvx = 0
            else:
                fvx -= fax
            if abs(fvy) <= abs(fay):
                fvy = 0
            else:
                fvy -= fay

            # apply acceleration and friction
            bot.vx += a_cos * acc + fvx
            bot.vy += a_sin * acc + fvy

            # maximum speed
            v = sqrt(bot.vx * bot.vx + bot.vy * bot.vy)
            if ctl.action == Action.ACCELERATION:
                extra_speed = typ.bonus_max_speed
            else:
                extra_speed = 0
            if ctl.move == 1:
                v_coeff = max(1.0, v / (typ.max_ahead_speed + extra_speed))
            else:
                v_coeff = max(1.0, v / (typ.max_back_speed + extra_speed))
            bot.vx /= v_coeff
            bot.vy /= v_coeff

            # change position and orientation
            bot.x += bot.vx / tps
            bot.y += bot.vy / tps
            bot.orientation += ori_change
            if bot.x < bot_radius:
                bot.x = bot_radius
            elif bot.x > world_width - bot_radius:
                bot.x = world_width - bot_radius
            if bot.y < bot_radius:
                bot.y = bot_radius
            elif bot.y > world_height - bot_radius:
                bot.y = world_height - bot_radius

            bot.tower_rot_speed = (
                rotation_smoothness * bot.tower_rot_speed + ctl.tower_rotate * typ.gun_rot_speed
            ) / (1 + rotation_smoothness)
            bot.tower_orientation += little_noise(bot.tower_rot_speed) / tps

        # shield
        for b_id, bot in self._bots.items():
            ctl = self._controls[bot.id]  # type: BotControl
            typ = bot.type  # type: BotTypeProperties
            if ctl.action == Action.SHIELD_WARMUP:
                bot.shield_warmup += 1 / (tps * typ.shield_warmup_period)
                bot.shield_warmup = min(1.0, bot.shield_warmup)
            else:
                regen = typ.shield_regen
                if ctl.action == Action.SHIELD_REGEN:
                    regen += typ.bonus_shield_regen
                bot.shield_warmup *= shield_leak_factor
                if bot.shield < typ.shield_energy:
                    bot.shield = min(typ.shield_energy, bot.shield + regen / tps)

        # firing
        for b_id, bot in self._bots.items():
            ctl = self._controls[bot.id]  # type: BotControl
            typ = bot.type  # type: BotTypeProperties
            wants_fire = ctl.action == Action.FIRE
            if wants_fire and not typ.shots_ray and bot.shot_ready and not bot.is_firing:
                angle = random.gauss(
                    mu=bot.orientation + bot.tower_orientation, sigma=typ.fire_scatter
                )
                bullet = BulletModel(typ, b_id, angle, bot.x, bot.y, typ.shot_range)
                next_bullets.append(bullet)
                bot.load -= typ.shot_energy
                bot.is_firing = True
            elif (
                wants_fire and typ.shots_ray and bot.load > typ.shot_energy / tps and bot.is_firing
            ):
                # ray should already be in rays dict
                pass
            elif wants_fire and typ.shots_ray and bot.load > Constants.ray_min_load_required:
                if bot.id not in self._rays:
                    bullet = BulletModel(
                        typ,
                        b_id,
                        bot.orientation + bot.tower_orientation,
                        bot.x,
                        bot.y,
                        typ.shot_range,
                    )
                    self._rays[bot.id] = bullet
                bot.is_firing = True
            else:
                if bot.load < 1:
                    if ctl.action != Action.IDLE and ctl.action != Action.FIRE:
                        coeff = load_with_action
                    else:
                        coeff = 1.0
                    bot.load += coeff / (typ.reload_period * tps)
                bot.is_firing = False

        # update rays
        for ray in self._rays.values():
            bot = self._bots.get(ray.origin_id)
            if bot is None or bot.load < 0 or self._controls[bot.id].action != Action.FIRE:
                continue
            bot.load -= bot.type.shot_energy / tps
            next_rays[bot.id] = ray
            position_ray(bot, ray)

        # make bullet damage
        next_bullets_after_damage = []
        for bullet in next_bullets:
            for bot in self._bots.values():
                if bullet.origin_id == bot.id:
                    continue
                d = dist_points(bullet.x, bullet.y, bot.x, bot.y)
                if d > bot_radius:
                    continue
                h = dist_line(bot.x, bot.y, bullet.cos, bullet.sin, bullet.x, bullet.y)
                dx = bullet.x - bot.x
                dy = bullet.y - bot.y
                bsin = sin(bot.orientation)
                bcos = cos(bot.orientation)

                hit_factor = half_chord_len(bot_radius, h) / bot_radius
                armor_factor = vec_dot(dx, dy, bcos, bsin) / (vec_len(dx, dy) + eps)

                damage = bullet.type.damage * hit_factor / (2 + armor_factor)
                if bot.has_shield:
                    damage = absorb_damage_by_shield(bot, damage)
                bot.hp -= damage
                self._explosions.append(
                    ExplosionModel(bullet.x, bullet.y, 0.75 * tps, 0.5 * bot_radius)
                )
                break
            else:
                next_bullets_after_damage.append(bullet)
        next_bullets = next_bullets_after_damage

        # make ray damage
        for ray in next_rays.values():
            base_dmg = ray.type.damage / tps
            damaged = []
            for bot in self._bots.values():
                if ray.origin_id == bot.id:
                    continue
                d = dist_line(bot.x, bot.y, ray.cos, ray.sin, ray.x, ray.y)
                if d > bot_radius:
                    continue
                dx = bot.x - ray.x
                dy = bot.y - ray.y
                t = sqrt(dx * dx + dy * dy - d * d)
                if vec_dot(ray.cos, ray.sin, dx, dy) < 0:
                    t = -t
                if not (0 <= t <= ray.range):
                    continue

                hit_factor = half_chord_len(bot_radius, d) / bot_radius
                damaged.append((t, base_dmg * hit_factor, bot))

                dt = sqrt(bot_radius * bot_radius - d * d)
                for t_i in range(int(t - dt), int(t + dt + 1), 2):
                    self._explosions.append(
                        ExplosionModel(
                            x=ray.x + t_i * ray.cos, y=ray.y + t_i * ray.sin, size=8, duration=2,
                        )
                    )
            damaged.sort(key=lambda item: item[0])
            decay_factor = 1.0
            for _, dmg, bot in damaged:
                dmg *= decay_factor
                if bot.has_shield:
                    bot.hp -= absorb_damage_by_shield(bot, dmg)
                else:
                    bot.hp -= dmg
                decay_factor /= 2

        # make collisions damage, fix coordinates
        all_bots = list(self._bots.values())
        for i1, b1 in enumerate(all_bots):
            m1 = b1.type.mass
            for i2 in range(i1 + 1, len(all_bots)):
                b2 = all_bots[i2]
                d = dist_points(b1.x, b1.y, b2.x, b2.y)
                if d >= 2 * bot_radius:
                    continue
                d = max(d, eps)

                m2 = b2.type.mass

                v1x, v1y = b1.vx, b1.vy
                v2x, v2y = b2.vx, b2.vy
                e1b = vec_len2(v1x, v1y)
                e2b = vec_len2(v2x, v2y)

                c_cos = (b2.x - b1.x) / d
                c_sin = (b2.y - b1.y) / d

                v1r = v1x * c_cos + v1y * c_sin
                v1t = -v1x * c_sin + v1y * c_cos

                v2r = v2x * c_cos + v2y * c_sin
                v2t = -v2x * c_sin + v2y * c_cos

                vr = (v1r * m1 + v2r * m2) / (m1 + m2)

                b1.vx = vr * c_cos - v1t * c_sin
                b1.vy = vr * c_sin + v1t * c_cos
                b2.vx = vr * c_cos - v2t * c_sin
                b2.vy = vr * c_sin + v2t * c_cos

                mx = (b1.x + b2.x) / 2
                my = (b1.y + b2.y) / 2
                b1.x = mx - (bot_radius + eps) * c_cos
                b1.y = my - (bot_radius + eps) * c_sin
                b2.x = mx + (bot_radius + eps) * c_cos
                b2.y = my + (bot_radius + eps) * c_sin

                # make damage

                if abs(v1r - v2r) < min_collision_speed:
                    continue

                e1a = vec_len2(b1.vx, b1.vy)
                e2a = vec_len2(b2.vx, b2.vy)

                h = max(0, m1 * (e1b - e1a) + m2 * (e2b - e2a))
                cf1 = 2 - vec_dot(cos(b1.orientation), sin(b1.orientation), c_cos, c_sin)
                cf2 = 2 + vec_dot(cos(b2.orientation), sin(b2.orientation), c_cos, c_sin)

                dmg1 = collision_factor * cf1 * h * m2 / (m1 + m2)
                dmg2 = collision_factor * cf2 * h * m1 / (m1 + m2)

                if b1.has_shield:
                    dmg1 = absorb_damage_by_shield(b1, dmg1)
                if b2.has_shield:
                    dmg2 = absorb_damage_by_shield(b2, dmg2)

                b1.hp -= dmg1
                b2.hp -= dmg2

        # shield discharge
        for bot in self._bots.values():
            if bot.has_shield:
                bot.shield -= bot.type.shield_regen / tps
                bot.shield = max(0.0, bot.shield)

        # remove killed bots
        next_bots = {}
        for bot in self._bots.values():
            if bot.hp > 0:
                next_bots[bot.id] = bot
            else:
                self._explosions.append(ExplosionModel(bot.x, bot.y, tps, 2 * bot_radius))
                self._n_bots[bot.team] -= 1
                bot.hp = 0
        self._bots = next_bots

        # update explosions
        for expl in self._explosions:
            expl.t += 1
        self._explosions = [e for e in self._explosions if not e.is_ended]

        self._bullets = next_bullets
        self._rays = next_rays
        self.nticks += 1

    @property
    def is_finished(self):
        if self._win_reached_at is None:
            if self.win_condition_reached:
                self._win_reached_at = self.nticks
        return (
            self._win_reached_at is not None
            and self.nticks >= self._win_reached_at + self._wait_after_win
            or self.nticks >= self.max_ticks
        )

    @property
    def win_condition_reached(self):
        return sum(1 for n in self._n_bots.values() if n > 0) <= 1

    def serialize_state(self):
        team_bots = {t: [] for t in self.teams}
        bullets = [bullet.serialize() for bullet in self.iter_bullets()]
        rays = [ray.serialize() for ray in self.iter_rays()]
        explosions = [explosion.serialize() for explosion in self.iter_explosions()]
        for bot in self.iter_bots():
            team_bots[bot.team].append(bot.serialize())
        return dict(bots=team_bots, bullets=bullets, rays=rays, explosions=explosions)

    def serialize_bots_visible_state(self):
        team_bots_visible_data = {t: [] for t in self.teams}
        for bot in self.iter_bots():
            team_bots_visible_data[bot.team].append(bot.serialize(with_hidden=False))
        return team_bots_visible_data

    @property
    def time(self):
        return self.nticks / Constants.ticks_per_sec

    def get_constants(self):
        return Constants

    def get_any_nonloser_team(self):
        for team in self.teams:
            n = self._n_bots.get(team, 0)
            if n > 0:
                return team


def absorb_damage_by_shield(bot, damage):
    absorbed = min(bot.shield, Constants.shield_damage_absorption * damage)
    damage -= absorbed
    bot.shield -= absorbed
    bot.shield = max(0.0, bot.shield)
    return damage


def little_noise(x):
    if abs(x) < Constants.epsilon:
        return x
    return random.gauss(x, x / 10)


def position_ray(bot, ray):
    angle = bot.orientation + bot.tower_orientation
    tower_shift = vec_rotate(-12, 0, bot.orientation)
    ray_start_shift = vec_rotate(53, 0, angle)
    x, y = vec_sum((bot.x, bot.y), tower_shift, ray_start_shift)
    ray.orientation = angle
    ray.x = x
    ray.y = y
    ray.cos = cos(angle)
    ray.sin = sin(angle)

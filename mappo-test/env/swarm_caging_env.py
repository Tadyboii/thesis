"""
swarm_caging_env.py
======================
A multi-agent differential-drive object-pushing environment built on
PettingZoo (ParallelEnv), Box2D physics, and Pygame rendering.

Agents must cooperatively surround a circular object and push it to a
target position.  Training is split into curriculum *phases*:

    Phase 0   — random spawn, free navigation, learn to reach the object.
    Phase 1   — all entities spawn randomly (object/target in 3.5x3.5 centre
                area, agents anywhere in 5x5); with 5 dummy agents when
                n_agents == 1.  Learn to navigate to the object.
    Phase 2   — on the standoff ring, learn equal angular spacing.
    Phase 3   — equal-spaced ring spawn, learn coordinated pushing.

Dummy agents (static, non-learning) are injected during phases 1 / 2
when only a single learning agent is present.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pygame
from Box2D import b2RayCastCallback, b2Vec2, b2World
from gymnasium.spaces import Box
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv


def raw_env(**kwargs) -> "SwarmCagingEnv":
    return SwarmCagingEnv(**kwargs)


class SwarmCagingEnv(ParallelEnv):
    """
    Parallel multi-agent environment: differential-drive robots push a
    circular object to a goal position.

    Observation space (per agent, 8 floats in [-1, 1]):
        [0:2]  nearest-neighbor   angle (÷pi) & distance (÷ WORLD_SIZE)
        [2:4]  second-nearest-neighbor angle & distance  (same encoding)
        [4:6]  target angle & distance                   (same encoding)
        [6:8]  object angle & surface-distance           (same encoding)

    Action space (per agent, 2 floats in [-1, 1]):
        [0]  forward / backward throttle
        [1]  angular (yaw) rate
    """

    metadata = {"render_modes": ["human"]}

    # World / agent geometry
    PIXELS_PER_METER   = 200
    WORLD_SIZE         = 5.0
    AGENT_RADIUS       = 0.125
    MAX_SPEED          = 1.0
    TURN_RATE          = 3.0
    OBJECT_RADIUS      = 0.5

    # Rendering
    AGENT_COLORS = [
        (255,   0,   0), (  0, 255,   0), (  0,   0, 255),
        (255, 255,   0), (255,   0, 255), (  0, 255, 255),
    ]
    DUMMY_COLOR = (128, 128, 128)

    # Cross-phase
    STEP_PENALTY      = 0.02
    STANDOFF_DISTANCE = 0.125
    SPAWN_AREA        = 3.5

    # Phase 0
    FACING_PENALTY_P0  = 0.3
    REPULSION_SCALE_P0 = 10.0
    REPULSION_THRESH_P0 = 0.10

    # Phase 1
    N_DUMMIES_P1        = 5
    STANDOFF_DELTA_P1   = 2.0
    STANDOFF_BONUS_P1   = 0.05
    CLOSE_PENALTY_P1    = 400.0
    FACING_PENALTY_P1   = 0.3
    REPULSION_SCALE_P1  = 100.0
    REPULSION_THRESH_P1 = 0.1

    # Phase 2
    NEIGHBOR_RANGE_P2 = math.pi / 2.0
    ARC_DELTA_P2      = 50.0
    ARC_BONUS_P2      = 0.08
    CLOSE_PENALTY_P2  = 400.0
    FAR_PENALTY_P2    = 200.0

    # Phase 3
    PROGRESS_SCALE_P3  = 10.0
    FORMATION_SCALE_P3 = 1.0
    TARGET_DIST_P3     = 2.0

    def __init__(self, 
                 n_agents: int = 1,
                 phase: int = 1, 
                 render_mode: Optional[str] = None, 
                 max_steps: int = 300) -> None:
        self.n_agents = n_agents
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_name_mapping = {f"agent_{i}": i for i in range(self.n_agents)}

        self.max_steps   = max_steps
        self.phase       = phase
        self.render_mode = render_mode

        self.phase3_target_distance = self.TARGET_DIST_P3
        self.phase1_standoff_dist      = self.STANDOFF_DISTANCE
        self.phase3_isHeadingRandom    = False
        self.phase2_isEqualSpacing     = False
        self.phase1_is_dummies_caged   = False
        self.phase2_special            = False

        self.screen_width  = int(self.WORLD_SIZE * self.PIXELS_PER_METER)
        self.screen_height = int(self.WORLD_SIZE * self.PIXELS_PER_METER)

        self._action_space      = Box(-1.0, 1.0, (2,), np.float32)
        self._observation_space = Box(-1.0, 1.0, (8,), np.float32)

        self.np_random, _ = seeding.np_random(None)
        self.world: Optional[b2World] = None
        self.agent_bodies  = []
        self.dummy_bodies  = []
        self.object_body   = None
        self.target_pos    = None

        self.step_count             = 0
        self.agent_hit_object_flags = [False]
        self.prev_dist_to_target    = 0.0
        self.agents_spawn_offset_angle = 0.0
        self.spawn_slots: list[int] = []

        self._prev_standoff_devs  = [0.0]
        self._prev_arc_imbalances = [0.0]

        self.paused = False
        self.screen = None
        self.clock  = None
        self.font   = None

        self.last_obs     = [np.zeros(8, np.float32) for _ in range(self.n_agents)]
        self.last_actions = [np.zeros(2, np.float32) for _ in range(self.n_agents)]
        self.last_rewards = [0.0 for _ in range(self.n_agents)]

        self.cumulative_reward          = 0.0
        self.agent_cumulative_rewards   = [0.0 for _ in range(self.n_agents)]
        self.last_progress_reward       = 0.0
        self.cumulative_progress_reward = 0.0

        self._p2s_dummy0_angle     = 0.0
        self._p2s_dummy1_side      = 1
        self._p2s_dummy1_arc_start = 0.0
        self._p2s_dummy1_arc_end   = 0.0
        self._p2s_agent_arc_start  = 0.0
        self._p2s_agent_arc_end    = 0.0

    def action_space(self, agent: str) -> Box:
        return self._action_space

    def observation_space(self, agent: str) -> Box:
        return self._observation_space

    def _dummies_active(self) -> bool:
        return self.phase in (1, 2) and self.n_agents == 1

    def _phase2_special_active(self) -> bool:
        return self.phase2_special and self.phase == 2 and self.n_agents == 1

    def reset(self, seed=None, options=None):
        self.np_random, _ = seeding.np_random(seed)
        self.agents    = self.possible_agents[:]
        self.step_count = 0
        self.agent_hit_object_flags = [False] * self.n_agents
        self.agents_spawn_offset_angle = self.np_random.uniform(-math.pi, math.pi)
        self.spawn_slots = list(range(self.n_agents))
        self.np_random.shuffle(self.spawn_slots)

        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.agent_bodies = []
        self.dummy_bodies = []
        self._spawn_object()
        self._spawn_target()

        if self._phase2_special_active():
            self._spawn_phase2_special_dummies()
            for i in range(self.n_agents):
                self._spawn_agent(i)
        else:
            for i in range(self.n_agents):
                self._spawn_agent(i)
            if self._dummies_active():
                if self.phase == 1:
                    for _ in range(self.N_DUMMIES_P1):
                        self._spawn_dummy_agent_phase1_random()
                else:
                    total = self.N_DUMMIES_P1 + self.n_agents
                    for d in range(self.N_DUMMIES_P1):
                        self._spawn_dummy_agent(slot_index=d + 1, total_slots=total)

        self._initialise_reward_baselines()

        if self.render_mode == "human" and self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("DiffDrivePush")
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.SysFont("Consolas", 14)

        self.cumulative_reward          = 0.0
        self.agent_cumulative_rewards   = [0.0 for _ in range(self.n_agents)]
        self.last_obs     = [np.zeros(8, np.float32) for _ in range(self.n_agents)]
        self.last_actions = [np.zeros(2, np.float32) for _ in range(self.n_agents)]
        self.last_rewards = [0.0 for _ in range(self.n_agents)]
        self.last_progress_reward       = 0.0
        self.cumulative_progress_reward = 0.0
        self.prev_dist_to_target        = self._object_to_target_dist()

        obs   = {a: self._get_obs(i) for i, a in enumerate(self.possible_agents)}
        infos = {a: {}               for a in self.possible_agents}
        return obs, infos

    def step(self, actions: dict):
        obs, rewards, terms, truncs, infos = {}, {}, {}, {}, {}

        for i, name in enumerate(self.possible_agents):
            self._apply_agent_action(i, actions[name])
        for db in self.dummy_bodies:
            db.linearVelocity  = b2Vec2(0, 0)
            db.angularVelocity = 0.0

        self.world.Step(1 / 30.0, 6, 2)
        self.step_count += 1

        dist_to_target = self._object_to_target_dist()
        reached_target = dist_to_target <= (self.object_body.fixtures[0].shape.radius + 0.2)
        dist_delta     = self.prev_dist_to_target - dist_to_target

        progress_reward = 0.0 if self.phase in (1, 2) else dist_delta * self.PROGRESS_SCALE_P3
        self.last_progress_reward        = progress_reward
        self.cumulative_progress_reward += progress_reward
        self.prev_dist_to_target         = dist_to_target

        both_neighbors_lost = False
        if self._phase2_special_active():
            obs_0 = self._get_obs(0)
            both_neighbors_lost = np.allclose(obs_0[0:2], 0.0) and np.allclose(obs_0[2:4], 0.0)

        for i, name in enumerate(self.possible_agents):
            obs_i = self._get_obs(i)
            self.last_obs[i] = obs_i
            obs[name] = obs_i

            total_reward = (
                self._compute_positioning_reward(i)
                + self._compute_facing_penalty(i)
                - self._compute_spacing_penalty(i)
                + progress_reward
                - self.STEP_PENALTY
            )

            terms[name]  = reached_target or both_neighbors_lost
            truncs[name] = self.step_count >= self.max_steps
            rewards[name]                  = total_reward
            self.last_rewards[i]           = total_reward
            self.agent_cumulative_rewards[i] += total_reward
            infos[name]                    = {}

        self.cumulative_reward += float(sum(rewards.values()))
        if self.render_mode == "human":
            self.render()
        return obs, rewards, terms, truncs, infos

    # -----------------------------------------------------------------------
    # Reward sub-computations
    # -----------------------------------------------------------------------

    def _compute_positioning_reward(self, agent_idx: int) -> float:
        agent_pos    = self.agent_bodies[agent_idx].position
        obj_pos      = self.object_body.position
        dist_center  = math.hypot(agent_pos.x - obj_pos.x, agent_pos.y - obj_pos.y)
        dist_surface = max(0.0, dist_center - self.object_body.fixtures[0].shape.radius - self.AGENT_RADIUS)
        deviation    = dist_surface - self.phase1_standoff_dist

        if self.phase == 1:
            return self._standoff_ring_reward(agent_idx, deviation)
        if self.phase == 2:
            return self._arc_spacing_reward(agent_idx, deviation, agent_pos, obj_pos)
        return -(dist_surface ** 2 / self.WORLD_SIZE ** 2) * 10.0

    def _standoff_ring_reward(self, agent_idx: int, deviation: float) -> float:
        prev_dev = self._prev_standoff_devs[agent_idx]
        reward   = (abs(prev_dev) - abs(deviation)) * self.STANDOFF_DELTA_P1
        reward  += self.STANDOFF_BONUS_P1 if abs(deviation) <= 0.05 else 0.0
        if deviation < -0.05:
            reward -= (deviation ** 2 / self.WORLD_SIZE ** 2) * self.CLOSE_PENALTY_P1
        self._prev_standoff_devs[agent_idx] = deviation
        return reward

    def _arc_spacing_reward(self, agent_idx: int, deviation: float, agent_pos, obj_pos) -> float:
        my_angle = math.atan2(agent_pos.y - obj_pos.y, agent_pos.x - obj_pos.x)

        neighbor_angles = [
            math.atan2(b.position.y - obj_pos.y, b.position.x - obj_pos.x)
            for j, b in enumerate(self.agent_bodies) if j != agent_idx
        ] + [
            math.atan2(db.position.y - obj_pos.y, db.position.x - obj_pos.x)
            for db in self.dummy_bodies
        ]

        def signed_arc(a, b):
            return (a - b + math.pi) % (2.0 * math.pi) - math.pi

        offsets      = [signed_arc(a, my_angle) for a in neighbor_angles]
        gap_left     = min((o for o in offsets if o > 0), default=math.pi)
        gap_right    = min((abs(o) for o in offsets if o < 0), default=math.pi)
        arc_imbalance = (gap_right - gap_left) / math.pi

        delta = abs(self._prev_arc_imbalances[agent_idx]) - abs(arc_imbalance)
        reward = delta * self.ARC_DELTA_P2
        reward += self.ARC_BONUS_P2 if abs(arc_imbalance) <= 0.1 and abs(deviation) <= 0.3 else 0.0

        max_dist2 = self.WORLD_SIZE ** 2
        if deviation < -0.15:
            reward -= (deviation ** 2 / max_dist2) * self.CLOSE_PENALTY_P2
        if deviation > 0.15:
            reward -= (deviation ** 2 / max_dist2) * self.FAR_PENALTY_P2

        self._prev_standoff_devs[agent_idx]  = deviation
        self._prev_arc_imbalances[agent_idx] = arc_imbalance
        return reward

    def _compute_facing_penalty(self, agent_idx: int) -> float:
        if self.phase == 0:
            scale = self.FACING_PENALTY_P0
        elif self.phase == 1:
            scale = self.FACING_PENALTY_P1
        else:
            return 0.0

        body         = self.agent_bodies[agent_idx]
        dx           = self.object_body.position.x - body.position.x
        dy           = self.object_body.position.y - body.position.y
        rel_angle    = (math.atan2(dy, dx) - body.angle + math.pi) % (2 * math.pi) - math.pi

        if abs(rel_angle) <= math.pi / 2:
            return 0.0
        excess = abs(rel_angle) - math.pi / 2
        return -((excess / (math.pi / 2)) ** 2) * scale

    def _compute_spacing_penalty(self, agent_idx: int) -> float:
        d1, d2 = self._two_nearest_neighbor_dists(agent_idx)

        if self.phase == 0:
            nearest_norm = min(d1, d2) / self.WORLD_SIZE
            if nearest_norm < self.REPULSION_THRESH_P0:
                diff = self.REPULSION_THRESH_P0 - nearest_norm
                return diff * diff * self.REPULSION_SCALE_P0
        elif self.phase == 1:
            nearest_norm = min(d1, d2) / self.WORLD_SIZE
            if nearest_norm < self.REPULSION_THRESH_P1:
                diff = self.REPULSION_THRESH_P1 - nearest_norm
                return diff * diff * self.REPULSION_SCALE_P1
        elif self.phase == 3:
            return (abs(d1 - d2) ** 2) * self.FORMATION_SCALE_P3
        return 0.0

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------

    def _get_obs(self, agent_idx: int) -> np.ndarray:
        body     = self.agent_bodies[agent_idx]
        max_dist = self.WORLD_SIZE

        class ClosestCallback(b2RayCastCallback):
            def __init__(self):
                super().__init__()
                self.hit_body     = None
                self.hit_fraction = 1.0

            def ReportFixture(self, fixture, point, normal, fraction):
                self.hit_body     = fixture.body
                self.hit_fraction = fraction
                return fraction

        def check_visibility(target_pos, target_body=None, radius_adjustment=0.0):
            if target_body is not None:
                try:
                    shape_radius = float(getattr(target_body.fixtures[0].shape, "radius", 0.0))
                except Exception:
                    shape_radius = 0.0
                sample_radius = max(shape_radius, float(radius_adjustment))

                if sample_radius <= 0.0:
                    cb = ClosestCallback()
                    self.world.RayCast(cb, body.position, b2Vec2(float(target_pos[0]), float(target_pos[1])))
                    if cb.hit_body is not None and cb.hit_body != target_body:
                        return np.zeros(2, np.float32)
                    dx = float(target_pos[0]) - float(body.position.x)
                    dy = float(target_pos[1]) - float(body.position.y)
                    angle        = (math.atan2(dy, dx) - body.angle + math.pi) % (2 * math.pi) - math.pi
                    surface_dist = max(0.0, math.hypot(dx, dy) - sample_radius - self.AGENT_RADIUS)
                    return np.array([angle / math.pi, surface_dist / max_dist], np.float32)

                best_norm_dist = float('inf')
                best_angle     = 0.0
                for k in range(12):
                    theta = (2.0 * math.pi * k) / 12
                    tgt_x = float(target_body.position.x + math.cos(theta) * sample_radius * 0.95)
                    tgt_y = float(target_body.position.y + math.sin(theta) * sample_radius * 0.95)

                    cb = ClosestCallback()
                    self.world.RayCast(cb, body.position, b2Vec2(tgt_x, tgt_y))

                    if cb.hit_body is not None and cb.hit_body == target_body:
                        dx       = tgt_x - body.position.x
                        dy       = tgt_y - body.position.y
                        raw_dist = math.hypot(dx, dy)
                        norm_dist = max(0.0, raw_dist * float(cb.hit_fraction) - self.AGENT_RADIUS) / max_dist
                        angle     = (math.atan2(dy, dx) - body.angle + math.pi) % (2 * math.pi) - math.pi
                        if norm_dist < best_norm_dist:
                            best_norm_dist = norm_dist
                            best_angle     = angle / math.pi

                if best_norm_dist < float('inf'):
                    return np.array([best_angle, best_norm_dist], np.float32)
                return np.zeros(2, np.float32)

            cb = ClosestCallback()
            self.world.RayCast(cb, body.position, b2Vec2(float(target_pos[0]), float(target_pos[1])))
            dx           = float(target_pos[0]) - float(body.position.x)
            dy           = float(target_pos[1]) - float(body.position.y)
            angle        = (math.atan2(dy, dx) - body.angle + math.pi) % (2 * math.pi) - math.pi
            surface_dist = max(0.0, math.hypot(dx, dy) - radius_adjustment - self.AGENT_RADIUS)

            if cb.hit_body is not None and cb.hit_body != body:
                ud = cb.hit_body.userData
                if ud and ud.get("type") in ("agent", "object", "dummy"):
                    return np.zeros(2, np.float32)

            return np.array([angle / math.pi, surface_dist / max_dist], np.float32)

        all_neighbors = sorted(
            [(math.hypot(b.position.x - body.position.x, b.position.y - body.position.y), b)
             for j, b in enumerate(self.agent_bodies) if j != agent_idx] +
            [(math.hypot(db.position.x - body.position.x, db.position.y - body.position.y), db)
             for db in self.dummy_bodies],
            key=lambda x: x[0]
        )

        agent_obs = []
        for _, other in all_neighbors:
            candidate = check_visibility(other.position, target_body=other)
            if not np.allclose(candidate, 0.0):
                agent_obs.append(candidate)
                if len(agent_obs) == 2:
                    break
        while len(agent_obs) < 2:
            agent_obs.append(np.zeros(2, np.float32))

        target_obs = check_visibility(self.target_pos)
        object_obs = check_visibility(self.object_body.position,
                                      target_body=self.object_body,
                                      radius_adjustment=self.object_body.fixtures[0].shape.radius)

        return np.concatenate([agent_obs[0], agent_obs[1], target_obs, object_obs]).astype(np.float32)

    # -----------------------------------------------------------------------
    # Physics helpers
    # -----------------------------------------------------------------------

    def _apply_agent_action(self, agent_idx: int, action: np.ndarray) -> None:
        body = self.agent_bodies[agent_idx]
        forward, angular = float(action[0]), float(action[1])
        self.last_actions[agent_idx] = np.array([forward, angular], np.float32)
        body.linearVelocity  = b2Vec2(math.cos(body.angle) * forward * self.MAX_SPEED,
                                      math.sin(body.angle) * forward * self.MAX_SPEED)
        body.angularVelocity = angular * self.TURN_RATE

    def _object_to_target_dist(self) -> float:
        return math.hypot(float(self.target_pos[0]) - float(self.object_body.position.x),
                          float(self.target_pos[1]) - float(self.object_body.position.y))

    def _two_nearest_neighbor_dists(self, agent_idx: int):
        body  = self.agent_bodies[agent_idx]
        dists = sorted(
            math.hypot(o.position.x - body.position.x, o.position.y - body.position.y)
            for o in ([b for j, b in enumerate(self.agent_bodies) if j != agent_idx] + self.dummy_bodies)
        )
        return (dists[0] if dists else float("inf")), (dists[1] if len(dists) > 1 else float("inf"))

    # -----------------------------------------------------------------------
    # Reward baseline initialisation
    # -----------------------------------------------------------------------

    def _initialise_reward_baselines(self) -> None:
        obj_pos   = self.object_body.position
        obj_r     = self.object_body.fixtures[0].shape.radius

        def signed_arc(a, b):
            return (a - b + math.pi) % (2.0 * math.pi) - math.pi

        prev_devs, prev_imbs = [], []
        for i, ab in enumerate(self.agent_bodies):
            ap = ab.position
            dist_surface = max(0.0, math.hypot(ap.x - obj_pos.x, ap.y - obj_pos.y) - obj_r - self.AGENT_RADIUS)
            prev_devs.append(dist_surface - self.phase1_standoff_dist)

            my_angle = math.atan2(ap.y - obj_pos.y, ap.x - obj_pos.x)
            neighbor_angles = [
                math.atan2(ob.position.y - obj_pos.y, ob.position.x - obj_pos.x)
                for j, ob in enumerate(self.agent_bodies) if j != i
            ] + [
                math.atan2(db.position.y - obj_pos.y, db.position.x - obj_pos.x)
                for db in self.dummy_bodies
            ]
            offsets  = [signed_arc(a, my_angle) for a in neighbor_angles]
            gap_l    = min((o for o in offsets if o > 0), default=math.pi)
            gap_r    = min((abs(o) for o in offsets if o < 0), default=math.pi)
            prev_imbs.append((gap_r - gap_l) / math.pi)

        self._prev_standoff_devs  = prev_devs
        self._prev_arc_imbalances = prev_imbs

    # -----------------------------------------------------------------------
    # Spawn helpers
    # -----------------------------------------------------------------------

    def _spawn_object(self) -> None:
        W, H   = self.WORLD_SIZE, self.WORLD_SIZE
        obj_r  = self.OBJECT_RADIUS
        margin = obj_r + 0.1

        half = self.SPAWN_AREA / 2.0
        cx, cy = W / 2.0, H / 2.0
        lo_x = max(margin, cx - half); hi_x = min(W - margin, cx + half)
        lo_y = max(margin, cy - half); hi_y = min(H - margin, cy + half)
        x = self.np_random.uniform(lo_x, hi_x)
        y = self.np_random.uniform(lo_y, hi_y)
        density = 1.0

        self.object_body = self.world.CreateDynamicBody(
            position=(float(x), float(y)), linearDamping=8.0, angularDamping=6.0
        )
        self.object_body.CreateCircleFixture(radius=obj_r, density=density, friction=5.0)
        self.object_body.userData = {"type": "object"}

    def _spawn_target(self) -> None:
        t_radius = 0.2
        margin   = t_radius + 0.2
        W, H     = self.WORLD_SIZE, self.WORLD_SIZE

        if self.phase in (0, 1, 2):
            ox, oy   = self.object_body.position
            min_dist = self.object_body.fixtures[0].shape.radius + t_radius + 0.3
            half     = self.SPAWN_AREA / 2.0
            cx, cy   = W / 2.0, H / 2.0
            lo_x = max(margin, cx - half); hi_x = min(W - margin, cx + half)
            lo_y = max(margin, cy - half); hi_y = min(H - margin, cy + half)
            for _ in range(200):
                tx = self.np_random.uniform(lo_x, hi_x)
                ty = self.np_random.uniform(lo_y, hi_y)
                if math.hypot(tx - ox, ty - oy) >= min_dist:
                    self.target_pos = np.array([tx, ty], np.float32)
                    return
            self.target_pos = np.array([
                float(max(lo_x, min(hi_x, W - ox))),
                float(max(lo_y, min(hi_y, H - oy))),
            ], np.float32)
            return

        if self.phase == 3:
            ox, oy = self.object_body.position
            for _ in range(100):
                angle = self.np_random.uniform(0.0, 2 * math.pi)
                tx    = ox + math.cos(angle) * self.phase3_target_distance
                ty    = oy + math.sin(angle) * self.phase3_target_distance
                if margin < tx < W - margin and margin < ty < H - margin:
                    self.target_pos = np.array([float(tx), float(ty)], np.float32)
                    return

            self.target_pos = np.array([
                float(self.np_random.uniform(margin, W - margin)),
                float(self.np_random.uniform(margin, H - margin)),
            ], np.float32)

    def _spawn_agent(self, agent_idx: int) -> None:
        radius = self.AGENT_RADIUS

        if self.phase == 3 and self.object_body:
            slot  = self.spawn_slots[agent_idx]
            theta = 2 * math.pi * slot / self.n_agents + self.agents_spawn_offset_angle
            dist  = self.object_body.fixtures[0].shape.radius + radius + 0.2
            pos_x = self.object_body.position.x + math.cos(theta) * dist
            pos_y = self.object_body.position.y + math.sin(theta) * dist
            angle = (float(self.np_random.uniform(-math.pi, math.pi))
                     if self.phase3_isHeadingRandom else theta + math.pi)

        elif self._phase2_special_active() and self.object_body:
            pos_x, pos_y, angle = self._find_phase2_special_agent_spawn(radius)

        elif self.phase == 2 and self.object_body:
            dist = self.object_body.fixtures[0].shape.radius + radius + self.phase1_standoff_dist
            if self.phase2_isEqualSpacing:
                theta = 2 * math.pi * self.spawn_slots[agent_idx] / self.n_agents + self.agents_spawn_offset_angle
            else:
                theta = self.np_random.uniform(-math.pi, math.pi)
            pos_x = self.object_body.position.x + math.cos(theta) * dist
            pos_y = self.object_body.position.y + math.sin(theta) * dist
            angle = theta + math.pi

        else:
            pos_x, pos_y, angle = self._find_open_map_spawn(radius)

        body = self.world.CreateDynamicBody(position=(float(pos_x), float(pos_y)), angle=float(angle))
        body.CreateCircleFixture(radius=radius, density=1.0, friction=0.8, restitution=0.2)
        body.userData = {"type": "agent", "index": agent_idx}
        self.agent_bodies.append(body)

    def _find_open_map_spawn(self, radius: float):
        margin = radius + 0.1
        W, H   = self.WORLD_SIZE, self.WORLD_SIZE
        angle  = float(self.np_random.uniform(-math.pi, math.pi))
        pos_x  = self.np_random.uniform(margin, W - margin)
        pos_y  = self.np_random.uniform(margin, H - margin)

        obj_r = self.object_body.fixtures[0].shape.radius if self.object_body else 0.0
        for _ in range(200):
            cx = self.np_random.uniform(margin, W - margin)
            cy = self.np_random.uniform(margin, H - margin)
            if self.object_body and math.hypot(cx - self.object_body.position.x,
                                               cy - self.object_body.position.y) < radius + obj_r + 0.2:
                continue
            if (self.target_pos is not None
                    and math.hypot(cx - self.target_pos[0], cy - self.target_pos[1]) < radius + 0.4):
                continue
            if self._overlaps_agents(cx, cy, radius) or self._overlaps_dummies(cx, cy, radius):
                continue
            pos_x, pos_y = cx, cy
            break

        return pos_x, pos_y, angle

    def _spawn_dummy_agent_phase1_random(self) -> None:
        radius = self.AGENT_RADIUS
        pos_x, pos_y, angle = self._find_open_map_spawn(radius)
        body = self.world.CreateStaticBody(position=(float(pos_x), float(pos_y)), angle=float(angle))
        body.CreateCircleFixture(radius=radius, friction=0.8, restitution=0.2)
        body.userData = {"type": "dummy", "index": len(self.dummy_bodies)}
        self.dummy_bodies.append(body)

    def _spawn_dummy_agent(self, slot_index: int, total_slots: int) -> None:
        radius     = self.AGENT_RADIUS
        spawn_dist = self.object_body.fixtures[0].shape.radius + radius + self.phase1_standoff_dist

        if self.phase1_is_dummies_caged:
            theta = 2.0 * math.pi * slot_index / total_slots + self.agents_spawn_offset_angle
            pos_x = max(radius, min(self.WORLD_SIZE - radius, self.object_body.position.x + math.cos(theta) * spawn_dist))
            pos_y = max(radius, min(self.WORLD_SIZE - radius, self.object_body.position.y + math.sin(theta) * spawn_dist))
            angle = theta + math.pi
        else:
            pos_x, pos_y, angle = self._find_ring_random_spawn(radius, spawn_dist)

        body = self.world.CreateStaticBody(position=(float(pos_x), float(pos_y)), angle=float(angle))
        body.CreateCircleFixture(radius=radius, friction=0.8, restitution=0.2)
        body.userData = {"type": "dummy", "index": len(self.dummy_bodies)}
        self.dummy_bodies.append(body)

    def _find_ring_random_spawn(self, radius: float, spawn_dist: float):
        obj   = self.object_body.position
        pos_x, pos_y, angle = obj.x, obj.y, 0.0

        for _ in range(200):
            theta = self.np_random.uniform(-math.pi, math.pi)
            cx    = obj.x + math.cos(theta) * spawn_dist
            cy    = obj.y + math.sin(theta) * spawn_dist
            if not self._in_bounds(cx, cy, radius):
                continue
            if self._overlaps_agents(cx, cy, radius) or self._overlaps_dummies(cx, cy, radius):
                continue
            pos_x, pos_y, angle = cx, cy, theta + math.pi
            return pos_x, pos_y, angle

        n     = len(self.dummy_bodies) + 1
        theta = 2.0 * math.pi / n
        pos_x = max(radius, min(self.WORLD_SIZE - radius, obj.x + math.cos(theta) * spawn_dist))
        pos_y = max(radius, min(self.WORLD_SIZE - radius, obj.y + math.sin(theta) * spawn_dist))
        return pos_x, pos_y, theta + math.pi

    def _find_phase2_special_agent_spawn(self, radius: float):
        dist = self.object_body.fixtures[0].shape.radius + radius + self.phase1_standoff_dist
        mid  = (self._p2s_agent_arc_start + self._p2s_agent_arc_end) / 2.0
        pos_x = self.object_body.position.x + math.cos(mid) * dist
        pos_y = self.object_body.position.y + math.sin(mid) * dist
        angle = mid + math.pi

        for _ in range(200):
            theta = self.np_random.uniform(self._p2s_agent_arc_start, self._p2s_agent_arc_end)
            cx = self.object_body.position.x + math.cos(theta) * dist
            cy = self.object_body.position.y + math.sin(theta) * dist
            if not self._in_bounds(cx, cy, radius) or self._overlaps_dummies(cx, cy, radius):
                continue
            pos_x, pos_y, angle = cx, cy, theta + math.pi
            break

        return pos_x, pos_y, angle

    def _spawn_phase2_special_dummies(self) -> None:
        radius     = self.AGENT_RADIUS
        spawn_dist = self.object_body.fixtures[0].shape.radius + radius + self.phase1_standoff_dist
        obj        = self.object_body.position
        nr         = self.NEIGHBOR_RANGE_P2

        d0_angle = self.np_random.uniform(-math.pi, math.pi)
        self._p2s_dummy0_angle = d0_angle
        d0_x = max(radius, min(self.WORLD_SIZE - radius, obj.x + math.cos(d0_angle) * spawn_dist))
        d0_y = max(radius, min(self.WORLD_SIZE - radius, obj.y + math.sin(d0_angle) * spawn_dist))

        body0 = self.world.CreateStaticBody(position=(float(d0_x), float(d0_y)), angle=float(d0_angle + math.pi))
        body0.CreateCircleFixture(radius=radius, friction=0.8, restitution=0.2)
        body0.userData = {"type": "dummy", "index": 0}
        self.dummy_bodies.append(body0)

        d1_side = 1 if self.np_random.random() < 0.5 else -1
        self._p2s_dummy1_side = d1_side

        if d1_side == 1:
            self._p2s_dummy1_arc_start = d0_angle
            self._p2s_dummy1_arc_end   = d0_angle + nr
            self._p2s_agent_arc_start  = d0_angle - nr
            self._p2s_agent_arc_end    = d0_angle
        else:
            self._p2s_dummy1_arc_start = d0_angle - nr
            self._p2s_dummy1_arc_end   = d0_angle
            self._p2s_agent_arc_start  = d0_angle
            self._p2s_agent_arc_end    = d0_angle + nr

        for _ in range(200):
            theta = self.np_random.uniform(self._p2s_dummy1_arc_start, self._p2s_dummy1_arc_end)
            cx = obj.x + math.cos(theta) * spawn_dist
            cy = obj.y + math.sin(theta) * spawn_dist
            if not self._in_bounds(cx, cy, radius):
                continue
            if math.hypot(cx - d0_x, cy - d0_y) < radius * 2 + 0.05:
                continue
            body1 = self.world.CreateStaticBody(position=(float(cx), float(cy)), angle=float(theta + math.pi))
            body1.CreateCircleFixture(radius=radius, friction=0.8, restitution=0.2)
            body1.userData = {"type": "dummy", "index": 1}
            self.dummy_bodies.append(body1)
            return

        theta = self._p2s_dummy1_arc_end if d1_side == 1 else self._p2s_dummy1_arc_start
        cx = max(radius, min(self.WORLD_SIZE - radius, obj.x + math.cos(theta) * spawn_dist))
        cy = max(radius, min(self.WORLD_SIZE - radius, obj.y + math.sin(theta) * spawn_dist))
        body1 = self.world.CreateStaticBody(position=(float(cx), float(cy)), angle=float(theta + math.pi))
        body1.CreateCircleFixture(radius=radius, friction=0.8, restitution=0.2)
        body1.userData = {"type": "dummy", "index": 1}
        self.dummy_bodies.append(body1)

    # -----------------------------------------------------------------------
    # Collision / bounds utilities
    # -----------------------------------------------------------------------

    def _in_bounds(self, x: float, y: float, radius: float) -> bool:
        return radius <= x <= self.WORLD_SIZE - radius and radius <= y <= self.WORLD_SIZE - radius

    def _overlaps_agents(self, x: float, y: float, radius: float, min_gap: float = 0.1) -> bool:
        return any(math.hypot(x - b.position.x, y - b.position.y) < radius * 2 + min_gap
                   for b in self.agent_bodies)

    def _overlaps_dummies(self, x: float, y: float, radius: float, min_gap: float = 0.05) -> bool:
        return any(math.hypot(x - b.position.x, y - b.position.y) < radius * 2 + min_gap
                   for b in self.dummy_bodies)

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def render(self) -> None:
        if self.screen is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_q:
                    self.close(); raise KeyboardInterrupt

        while self.paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close(); raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = False
                    if event.key == pygame.K_q:
                        self.close(); raise KeyboardInterrupt
            pygame.time.wait(100)

        self.screen.fill((30, 30, 30))
        ppm = self.PIXELS_PER_METER

        grid_color = (55, 55, 55)
        for m in range(1, int(self.WORLD_SIZE)):
            px = int(m * ppm)
            pygame.draw.line(self.screen, grid_color, (px, 0), (px, self.screen_height), 1)
            pygame.draw.line(self.screen, grid_color, (0, px), (self.screen_width, px), 1)

        self._render_spawn_area(ppm)

        tx, ty = self.target_pos
        cs = 0.1
        pygame.draw.line(self.screen, (255, 255, 255),
                         (int((tx - cs) * ppm), int((ty - cs) * ppm)),
                         (int((tx + cs) * ppm), int((ty + cs) * ppm)), 2)
        pygame.draw.line(self.screen, (255, 255, 255),
                         (int((tx - cs) * ppm), int((ty + cs) * ppm)),
                         (int((tx + cs) * ppm), int((ty - cs) * ppm)), 2)

        ox, oy = self.object_body.position
        pygame.draw.circle(self.screen, (200, 200, 200),
                           (int(ox * ppm), int(oy * ppm)),
                           int(self.object_body.fixtures[0].shape.radius * ppm))

        if self.phase in (0, 1, 2, 3):
            self._render_ring_overlays(ox, oy, ppm)

        for d_idx, db in enumerate(self.dummy_bodies):
            dx, dy = db.position
            px, py = int(dx * ppm), int(dy * ppm)
            pygame.draw.circle(self.screen, self.DUMMY_COLOR, (px, py), int(self.AGENT_RADIUS * ppm))
            pygame.draw.circle(self.screen, (200, 200, 200), (px, py), int(self.AGENT_RADIUS * ppm), 1)
            if self.font:
                self.screen.blit(self.font.render(f"D{d_idx}", True, (200, 200, 200)), (px - 8, py - 7))

        for i, body in enumerate(self.agent_bodies):
            self._render_agent(i, body, ox, oy, ppm)

        if self.font:
            self._render_global_hud(ppm)

        pygame.display.flip()
        self.clock.tick(60)

    def _render_spawn_area(self, ppm: int) -> None:
        half   = self.SPAWN_AREA / 2.0
        cx, cy = self.WORLD_SIZE / 2.0, self.WORLD_SIZE / 2.0
        margin = self.object_body.fixtures[0].shape.radius
        lo_x   = int((cx - half - margin) * ppm)
        lo_y   = int((cy - half - margin) * ppm)
        w_px   = int((self.SPAWN_AREA + 2 * margin) * ppm)

        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        pygame.draw.rect(overlay, (128, 128, 128, 40), (lo_x, lo_y, w_px, w_px))
        self.screen.blit(overlay, (0, 0))
        pygame.draw.rect(self.screen, (128, 128, 128), (lo_x, lo_y, w_px, w_px), 1)

    def _render_ring_overlays(self, ox: float, oy: float, ppm: int) -> None:
        cx    = int(ox * ppm)
        cy    = int(oy * ppm)
        std_r = self.object_body.fixtures[0].shape.radius + self.AGENT_RADIUS + self.phase1_standoff_dist
        pygame.draw.circle(self.screen, (128, 128, 128), (cx, cy), int(std_r * ppm), 1)

        if self._phase2_special_active() and len(self.dummy_bodies) == 2:
            self._render_phase2_special_arcs(cx, cy, std_r, ppm)

    def _render_phase2_special_arcs(self, cx: int, cy: int, ring_r: float, ppm: int) -> None:
        r_px     = int(ring_r * ppm)
        rect     = pygame.Rect(cx - r_px, cy - r_px, r_px * 2, r_px * 2)
        thick    = max(3, r_px // 8)
        arc_surf = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)

        def draw_ring_arc(color, start, end, t):
            span = (end - start) % (2.0 * math.pi)
            if span:
                pygame.draw.arc(arc_surf, color, rect, -start - span, -start, t)

        draw_ring_arc((200, 200, 200, 120), self._p2s_dummy1_arc_start, self._p2s_dummy1_arc_end, thick)
        draw_ring_arc((255, 80, 80, 120),   self._p2s_agent_arc_start,  self._p2s_agent_arc_end,  thick)
        draw_ring_arc((255, 255, 255, 200), self._p2s_dummy0_angle - 0.04, self._p2s_dummy0_angle + 0.04, max(2, thick // 2))
        self.screen.blit(arc_surf, (0, 0))

    def _render_agent(self, agent_idx: int, body, ox: float, oy: float, ppm: int) -> None:
        x, y  = body.position
        angle = body.angle
        color = self.AGENT_COLORS[agent_idx % len(self.AGENT_COLORS)]

        pygame.draw.circle(self.screen, color, (int(x * ppm), int(y * ppm)), int(self.AGENT_RADIUS * ppm))
        hx = x + math.cos(angle) * self.AGENT_RADIUS
        hy = y + math.sin(angle) * self.AGENT_RADIUS
        pygame.draw.line(self.screen, (255, 255, 255), (int(x * ppm), int(y * ppm)), (int(hx * ppm), int(hy * ppm)), 2)

        current_obs = self._get_obs(agent_idx)
        max_dist    = self.WORLD_SIZE


        # Use agent color for all observation lines
        obs_line_color = color
        def draw_obs_ray(obs_slice):
            if not np.allclose(obs_slice, 0.0):
                rel = obs_slice[0] * math.pi
                dist = obs_slice[1] * max_dist
                gdir = angle + rel
                sx = x + math.cos(gdir) * self.AGENT_RADIUS
                sy = y + math.sin(gdir) * self.AGENT_RADIUS
                pygame.draw.line(self.screen, obs_line_color,
                                 (int(sx * ppm), int(sy * ppm)),
                                 (int((sx + math.cos(gdir) * dist) * ppm),
                                  int((sy + math.sin(gdir) * dist) * ppm)), 1)

        draw_obs_ray(current_obs[0:2])
        draw_obs_ray(current_obs[2:4])
        draw_obs_ray(current_obs[4:6])
        draw_obs_ray(current_obs[6:8])

        if not self.font:
            return

        obs    = self.last_obs[agent_idx]
        act    = self.last_actions[agent_idx]
        text_y = 10 + 90 * agent_idx
        for j, line in enumerate([
            f"agent_{agent_idx}",
            f"Action: ({act[0]:.2f},{act[1]:.2f})",
            f"Near_1: ({obs[0]:.2f},{obs[1]:.2f})",
            f"Near_2: ({obs[2]:.2f},{obs[3]:.2f})",
            f"Target: ({obs[4]:.2f},{obs[5]:.2f})",
            f"Object: ({obs[6]:.2f},{obs[7]:.2f})",
        ]):
            self.screen.blit(self.font.render(line, True, color), (10, text_y + j * 14))

        r_val  = self.last_rewards[agent_idx]
        bar_x, bar_y, bar_w, bar_h = 10, text_y + 6 * 14 + 5, 100, 6
        zero_x = bar_x + bar_w - 20

        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.line(self.screen, (150, 150, 150), (zero_x, bar_y), (zero_x, bar_y + bar_h), 1)

        bar_len = int(abs(r_val) * 100.0)
        if r_val < 0:
            pygame.draw.rect(self.screen, (255, 50, 50), (zero_x - bar_len, bar_y, bar_len, bar_h))
        else:
            pygame.draw.rect(self.screen, (50, 255, 50), (zero_x, bar_y, bar_len, bar_h))

        self.screen.blit(self.font.render(f"{r_val:.2f}", True, color), (bar_x + bar_w + 5, bar_y - 4))
        self.screen.blit(self.font.render(f"Tot: {self.agent_cumulative_rewards[agent_idx]:.1f}", True, color),
                         (bar_x + bar_w + 60, bar_y - 4))

    def _render_global_hud(self, ppm: int) -> None:
        W, H = self.screen_width, self.screen_height
        step_txt = self.font.render(f"Step: {self.step_count}/{self.max_steps}", True, (255, 255, 255))
        self.screen.blit(step_txt, (W - step_txt.get_width() - 10, 10))

        y_br = H - 10
        for text in [
            f"Total Reward: {self.cumulative_reward:.3f}",
            f"Progress (Step): {self.last_progress_reward:.4f}",
            f"Progress (Total): {self.cumulative_progress_reward:.4f}",
        ]:
            lbl = self.font.render(text, True, (255, 255, 255))
            y_br -= lbl.get_height()
            self.screen.blit(lbl, (W - lbl.get_width() - 10, y_br))

        legend = []
        if self.phase in (1, 2, 3):
            legend.append(((100, 160, 255),
                f"Object/Target area: {self.SPAWN_AREA}x{self.SPAWN_AREA}m centred"))
        if self.phase == 1:
            if self._dummies_active():
                legend.append((self.DUMMY_COLOR, f"Dummy agents ({self.N_DUMMIES_P1}) — random 5x5 spawn"))
        if self.phase == 2:
            ring_r = self.object_body.fixtures[0].shape.radius + self.AGENT_RADIUS + self.phase1_standoff_dist
            legend.append(((255, 165, 0), f"Standoff ring ({ring_r:.2f}m)"))
            if self._phase2_special_active():
                deg = math.degrees(self.NEIGHBOR_RANGE_P2)
                legend += [
                    (self.DUMMY_COLOR, f"Dummy agents (2) — special mode, neighbor_range={deg:.0f}deg"),
                    ((200, 200, 200), f"D1 spawn arc (silver) +-{deg:.0f}deg from D0"),
                    ((255, 80, 80),   f"Agent spawn arc (red) +-{deg:.0f}deg opposite"),
                ]
            elif self._dummies_active():
                legend.append((self.DUMMY_COLOR, f"Dummy agents ({self.N_DUMMIES_P1}) — caged, random spacing"))

        leg_y = H - len(legend) * 16 - 8
        for col, label in legend:
            self.screen.blit(self.font.render(label, True, col), (10, leg_y))
            leg_y += 16

    def close(self) -> None:
        if self.screen:
            pygame.quit()
            self.screen = None
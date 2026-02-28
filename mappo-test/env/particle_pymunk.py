import math
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import functools

from pettingzoo import ParallelEnv
from gymnasium.spaces import Box
from gymnasium.utils import seeding


def raw_env(render_mode=None):
    return DiffDrivePushEnv(render_mode=render_mode)


class DiffDrivePushEnv(ParallelEnv):
    metadata = {
        "name": "diffdrive_push_v0",
        "render_modes": ["human"],
    }

    AGENT_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
    (255, 255, 0),(255, 0, 255), (0, 255, 255), 
    (127, 0, 0), (0, 127, 0), (0, 0, 127),
    (127, 127,0), (127, 0, 127), (0, 127, 127),
    (63, 0, 0), (0, 63, 0), (0, 0, 63),
    (63, 63, 0), (63, 0, 63), (0, 63, 63),
    ]

    def __init__(self, n_agents=6, width=400, height=400, max_steps=500, object_radius=70, render_mode=None):
        self.n_agents = n_agents
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_name_mapping = {f"agent_{i}": i for i in range(self.n_agents)}

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.object_radius = object_radius
        self.render_mode = render_mode
        self.max_speed = 100.0
        self.wheel_base = 40.0
        self.turn_sensitivity = 8.0
        self.target_pos = None
        self.prev_object_target_dist = None

        # Action & observation spaces
        self._action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Expanded observation space: object (angle, dist) + 2 nearest agents (angle, dist) each
        self._observation_space = Box(
            low=np.array(
                [-1.0, 0.0,   # object: angle, dist
                -1.0, 0.0,   # nearest agent 1
                -1.0, 0.0,   # nearest agent 2
                -1.0, 0.0],  # target: angle, dist
                dtype=np.float32
            ),
            high=np.array(
                [1.0, 1.0,
                1.0, 1.0,
                1.0, 1.0,
                1.0, 1.0],
                dtype=np.float32
            ),
)

        self.np_random, _ = seeding.np_random(None)

        # Physics
        self.space = None
        self.agent_bodies = [None for _ in range(self.n_agents)]
        self.object_body = None

        # Collision tracking
        self.agent_hit_object_flags = [False] * self.n_agents

        # Rendering
        self.screen = None
        self.clock = None
        self.draw_options = None
        self.font = None

        self.step_count = 0
        self.last_actions = [np.array([0, 0], dtype=np.float32) for _ in range(self.n_agents)]
        self.last_observations = [np.zeros(8, dtype=np.float32) for _ in range(self.n_agents)]
        self.last_rewards = [0.0 for _ in range(self.n_agents)]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_space

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_space

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        self.np_random, _ = seeding.np_random(seed)
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.agent_hit_object_flags = [False] * self.n_agents

        # Physics world
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)

        self._add_object()
        self._add_target()
        self.prev_object_target_dist = self._distance_object_to_target()
        self.agent_bodies = []
        for i in range(self.n_agents):
            self._add_agent(i)

        # Collision handler
        self._setup_collision_handler()

        # Rendering
        if self.render_mode == "human" and self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("DiffDrive Push Environment")
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        obs = {f"agent_{i}": self._get_obs(i) for i in range(self.n_agents)}
        infos = {f"agent_{i}": {} for i in range(self.n_agents)}
        return obs, infos

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, actions):
        rewards = {}
        obs = {}
        terminations = {}
        truncations = {}
        infos = {}

        # reset collision flags
        self.agent_hit_object_flags = [False] * self.n_agents

        # Apply actions
        for i, agent_name in enumerate(self.possible_agents):
            action = actions[agent_name]
            self.last_actions[i] = action.copy()

            # action = [forward_speed, angular_speed] in [-1, 1]
            forward_cmd, angular_cmd = action

            # Scale commands
            linear_vel = forward_cmd * self.max_speed
            angular_vel = angular_cmd * self.turn_sensitivity

            body = self.agent_bodies[i]
            angle = body.angle
            vx = math.cos(angle) * linear_vel
            vy = math.sin(angle) * linear_vel
            body.velocity = (vx, vy)
            body.angular_velocity = angular_vel

        # Step physics
        self.space.step(1 / 60.0)
        self.step_count += 1

        # ----------------------------------
        # TERMINATION CHECK
        # ----------------------------------
        obj_pos = np.array(self.object_body.position)
        tgt_pos = np.array(self.target_pos)

        dist = np.linalg.norm(obj_pos - tgt_pos)

        # success threshold
        if dist <= self.object_radius:
            self.terminated = True
        else:
            self.terminated = False

        # Precompute
        max_dist2 = (math.hypot(self.width, self.height)) ** 2

        # # Object to target penalty
        # obj_to_target_dist = self._distance_object_to_target() - self.object_radius
        # object_target_penalty = (obj_to_target_dist ** 2 / max_dist2) * 8.0
        
        # --- Object progress reward ---
        current_dist = self._distance_object_to_target()
        prev_dist = self.prev_object_target_dist

        progress = prev_dist - current_dist  # positive = moved toward target

        # Normalize by arena size
        max_dist = math.hypot(self.width, self.height)
        progress_norm = progress / max_dist

        # Scale and clip
        object_progress_reward = np.clip(progress_norm * 10.0, -1.0, 1.0)

        # Update for next step
        self.prev_object_target_dist = current_dist

        # Compute rewards & observations
        for i, agent_name in enumerate(self.possible_agents):

            # DISTANCE PENALITY
            dist = self._distance_to_object(i) - self.object_radius
            reward = (-dist * dist /  max_dist2) * 10

            # FACING PENALTY
            body = self.agent_bodies[i]
            dx = self.object_body.position.x - body.position.x
            dy = self.object_body.position.y - body.position.y
            angle_to_obj = math.atan2(dy, dx)
            rel_angle = angle_to_obj - body.angle
            # Normalize to [-pi, pi]
            rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
            # Squared facing penalty
            # reward -= (rel_angle / math.pi) ** 2 * 0.1
            reward -= (rel_angle / math.pi) ** 2 * 0.02

            # NEARBY AGENT PENALTY
            dist1, dist2 = self._distance_to_nearest_agents(i)
            max_dist = math.hypot(self.width, self.height)
            # Normalize distances
            dist1_norm = dist1 / max_dist
            dist2_norm = dist2 / max_dist
            # Safe threshold (normalized)
            safe_threshold = 0.100   # adjust this value depending on arena size
            penalty = 0.0
            for d in (dist1_norm, dist2_norm):
                if d < safe_threshold:
                    # closer than safe distance → penalty
                    diff = safe_threshold - d
                    penalty += diff * diff * 4  # squared normalized difference
            reward -= penalty

            # OBJECT TO TARGET PENALTY
            # reward -= object_target_penalty
            reward += object_progress_reward

            self.last_rewards[i] = reward
            self.last_observations[i] = self._get_obs(i)
            rewards[agent_name] = reward
            obs[agent_name] = self.last_observations[i]
            truncations[agent_name] = self.step_count >= self.max_steps
            # terminations[agent_name] = False
            terminations[agent_name] = self.terminated
            infos[agent_name] = {"distance": dist, "collided": self.agent_hit_object_flags[i]}

        ## end episode if all agent collided with object
        # if all(self.agent_hit_object_flags):
        #     for agent_name in self.possible_agents:
        #         rewards[agent_name] += 1.0  # bonus for all agents
        #         terminations[agent_name] = True

        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self, index):
        body = self.agent_bodies[index]

        # --- Object observation ---
        dx_obj = self.object_body.position.x - body.position.x
        dy_obj = self.object_body.position.y - body.position.y
        dist_obj = math.hypot(dx_obj, dy_obj) - self.object_radius
        max_dist = math.hypot(self.width, self.height)
        dist_obj_norm = dist_obj / max_dist
        angle_to_obj = math.atan2(dy_obj, dx_obj)
        rel_angle_obj = angle_to_obj - body.angle
        rel_angle_obj = (rel_angle_obj + math.pi) % (2 * math.pi) - math.pi

        # --- Target observation ---
        # dx_t = self.target_pos[0] - body.position.x
        # dy_t = self.target_pos[1] - body.position.y
        # dist_t = math.hypot(dx_t, dy_t)
        # dist_t_norm = dist_t / max_dist

        # angle_t = math.atan2(dy_t, dx_t)
        # rel_angle_t = angle_t - body.angle
        # rel_angle_t = (rel_angle_t + math.pi) % (2 * math.pi) - math.pi

        # --- Target observation with LOS ---
        if self._target_in_line_of_sight(index):
            dx_t = self.target_pos[0] - body.position.x
            dy_t = self.target_pos[1] - body.position.y
            dist_t = math.hypot(dx_t, dy_t)
            dist_t_norm = dist_t / max_dist

            angle_t = math.atan2(dy_t, dx_t)
            rel_angle_t = angle_t - body.angle
            rel_angle_t = (rel_angle_t + math.pi) % (2 * math.pi) - math.pi

            target_angle_norm = rel_angle_t / math.pi
            target_dist_norm = dist_t_norm
        else:
            # Target not visible
            target_angle_norm = 0.0
            target_dist_norm = 0.0

        # --- Other agents ---
        other_bodies = [b for i_, b in enumerate(self.agent_bodies) if i_ != index]
        distances = []
        for other in other_bodies:
            dx = other.position.x - body.position.x
            dy = other.position.y - body.position.y
            dist = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)
            rel_angle = angle - body.angle
            rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
            distances.append((dist, rel_angle, other))

        distances.sort(key=lambda x: x[0])

        # Nearest agent
        if len(distances) > 0:
            dist1, angle1, _ = distances[0]
            dist1_norm = dist1 / max_dist
            angle1_norm = angle1 / math.pi
        else:
            dist1_norm = 0.0
            angle1_norm = 0.0

        # Second nearest agent
        if len(distances) > 1:
            dist2, angle2, _ = distances[1]
            dist2_norm = dist2 / max_dist
            angle2_norm = angle2 / math.pi
        else:
            dist2_norm = 0.0
            angle2_norm = 0.0

        obs = np.array(
            [rel_angle_obj / math.pi, dist_obj_norm,
            angle1_norm, dist1_norm, 
            angle2_norm, dist2_norm,
            target_angle_norm, target_dist_norm],
            # angle1_norm, dist1_norm, 
            # angle2_norm, dist2_norm,
            # rel_angle_t / math.pi, dist_t_norm],
            dtype=np.float32
        )
        return obs

    def _distance_to_object(self, index):
        body = self.agent_bodies[index]
        dx = self.object_body.position.x - body.position.x
        dy = self.object_body.position.y - body.position.y
        return math.hypot(dx, dy)

    def _distance_object_to_target(self):
        dx = self.target_pos[0] - self.object_body.position.x
        dy = self.target_pos[1] - self.object_body.position.y
        return math.hypot(dx, dy)

    def _distance_to_nearest_agents(self, index):
        body = self.agent_bodies[index]
        other_bodies = [b for i_, b in enumerate(self.agent_bodies) if i_ != index]

        distances = []
        for other in other_bodies:
            dx = other.position.x - body.position.x
            dy = other.position.y - body.position.y
            dist = math.hypot(dx, dy)
            distances.append(dist)

        distances.sort()
        if len(distances) > 0:
            dist1 = distances[0]
        else:
            dist1 = 0.0
        if len(distances) > 1:
            dist2 = distances[1]
        else:
            dist2 = 0.0

        return dist1, dist2

    # ------------------------------------------------------------------
    # Physics setup
    # ------------------------------------------------------------------
    def _add_agent(self, index):
        mass = 2.0
        radius = 15
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, inertia)

        # allow spawning closer to edges by using a margin computed from radii
        max_attempts = 100
        margin = max(radius, self.object_radius) + 10
        for _ in range(max_attempts):
            x = self.np_random.uniform(margin, self.width - margin)
            y = self.np_random.uniform(margin, self.height - margin)
            pos = (x, y)
            safe = True

            obj_pos = (self.object_body.position.x, self.object_body.position.y)
            if math.hypot(pos[0] - obj_pos[0], pos[1] - obj_pos[1]) < (self.object_radius + radius + 5):
                safe = False

            for other_body in self.agent_bodies:
                other_pos = (other_body.position.x, other_body.position.y)
                if math.hypot(pos[0] - other_pos[0], pos[1] - other_pos[1]) < (2 * radius + 5):
                    safe = False

            if safe:
                body.position = pos
                break
        else:
            body.position = pos

        body.angle = self.np_random.uniform(-math.pi, math.pi)
        shape = pymunk.Circle(body, radius)
        shape.friction = 0.9
        shape.elasticity = 0.2
        shape.collision_type = 1
        shape.color = self.AGENT_COLORS[index] + (255,)
        self.space.add(body, shape)
        self.agent_bodies.append(body)

    def _add_object(self):
        mass = 2000.0
        vertices = [(0,0), (10,0), (10,10), (0,10)]
        moment = pymunk.moment_for_poly(mass, vertices)

        radius = self.object_radius
        self.object_body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)
        margin = self.object_radius + 10
        self.object_body.position = (
            self.np_random.uniform(margin, self.width - margin),
            self.np_random.uniform(margin, self.height - margin),
        )
        shape = pymunk.Circle(self.object_body, radius)
        shape.friction = 1.0
        shape.elasticity = 0.0
        shape.color = (200, 200, 200, 255)
        shape.collision_type = 2
        self.space.add(self.object_body, shape)

    def _add_target(self):
        max_attempts = 100

        obj_x = self.object_body.position.x
        obj_y = self.object_body.position.y

        min_dist = self.object_radius + 20  # small margin outside object
        margin = self.object_radius + 20

        for _ in range(max_attempts):
            x = self.np_random.uniform(margin, self.width - margin)
            y = self.np_random.uniform(margin, self.height - margin)

            if math.hypot(x - obj_x, y - obj_y) >= min_dist:
                self.target_pos = np.array([x, y], dtype=np.float32)
                return

        # Fallback (should almost never happen)
        self.target_pos = np.array(
            [obj_x + min_dist, obj_y],
            dtype=np.float32
        )

    def _target_in_line_of_sight(self, agent_index):
        body = self.agent_bodies[agent_index]

        start = body.position
        end = pymunk.Vec2d(self.target_pos[0], self.target_pos[1])

        hits = self.space.segment_query(
            start,
            end,
            0,
            pymunk.ShapeFilter()
        )

        # Sort by distance from agent
        hits.sort(key=lambda h: h.alpha)

        for hit in hits:
            shape = hit.shape

            # Ignore self and other agents
            if shape.collision_type == 1:
                continue

            # Object blocks LOS
            if shape.collision_type == 2:
                return False

        # Nothing blocks the ray
        return True


    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------
    def _setup_collision_handler(self):
        self.space.on_collision(
            collision_type_a=1,
            collision_type_b=2,
            post_solve=self._on_agent_hits_object
        )

    def _on_agent_hits_object(self, arbiter, space, data):
        shape_a, shape_b = arbiter.shapes
        for i, body in enumerate(self.agent_bodies):
            for shape in body.shapes:
                if shape == shape_a or shape == shape_b:
                    self.agent_hit_object_flags[i] = True
                    break
        return True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        if self.screen is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.screen.fill((30, 30, 30))
        self.space.debug_draw(self.draw_options)

        if not self.font:
            pygame.font.init()
            self.font = pygame.font.SysFont("Consolas", 16)

        step_label = self.font.render(f"Step: {self.step_count}/{self.max_steps}", True, (255, 255, 0))
        self.screen.blit(step_label, (10, 10))

        max_dist = math.hypot(self.width, self.height)

        # --- Draw target as an "X" ---
        tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
        size = 15
        color = (255, 255, 255)

        pygame.draw.line(self.screen, color, (tx - size, ty - size), (tx + size, ty + size), 3)
        pygame.draw.line(self.screen, color, (tx - size, ty + size), (tx + size, ty - size), 3)

        # Draw each agent info and lines to nearest agents
        for i, agent_name in enumerate(self.possible_agents):
            body = self.agent_bodies[i]
            color = self.AGENT_COLORS[i]
            obs_vals = self.last_observations[i]

            # --- Lines to nearest agents ---
            other_bodies = [b for j, b in enumerate(self.agent_bodies) if j != i]
            distances = [(math.hypot(b.position.x - body.position.x, b.position.y - body.position.y), b) for b in other_bodies]
            distances.sort(key=lambda x: x[0])
            for k in range(min(2, len(distances))):
                other_body = distances[k][1]
                pygame.draw.line(
                    self.screen, color,
                    (int(body.position.x), int(body.position.y)),
                    (int(other_body.position.x), int(other_body.position.y)), 2
                )
            
            # --- Line of sight to target ---
            if self._target_in_line_of_sight(i):
                pygame.draw.line(
                    self.screen,
                    (255, 255, 255),
                    (int(body.position.x), int(body.position.y)),
                    (int(self.target_pos[0]), int(self.target_pos[1])),
                    2
                )

            # --- Debug text ---
            action_vals = self.last_actions[i]
            dist_obj = self._distance_to_object(i)
            collided = self.agent_hit_object_flags[i]

            info_text = (
                f"{agent_name}\n"
                f"Action: ({action_vals[0]:.2f}, {action_vals[1]:.2f})\n"
                f"Obs_obj: ({obs_vals[0]:.2f}, {obs_vals[1]:.2f})\n"
                f"Near_1: ({obs_vals[2]:.2f}, {obs_vals[3]:.2f})\n"
                f"Near_2: ({obs_vals[4]:.2f}, {obs_vals[5]:.2f})\n"
                f"Obs_tgt: ({obs_vals[6]:.2f}, {obs_vals[7]:.2f})\n"
                f"Dist: {dist_obj:.2f}\n"
                f"Collided: {collided}"
            )

            lines = info_text.split("\n")
            for j, line in enumerate(lines):
                label = self.font.render(line, True, color)
                self.screen.blit(label, (10 + i * 200, 30 + j * 18))

            # Reward bar
            reward = self.last_rewards[i]
            bar_max_length = self.width // 4
            bar_height = 20
            bar_bottom = self.height - 10
            length = int(min(-reward, 1.0) * bar_max_length)
            pygame.draw.rect(
                self.screen,
                color,
                (10 + i * 200, bar_bottom - bar_height, length, bar_height)
            )
            reward_label = self.font.render(f"Penalty: {reward:.3f}", True, color)
            self.screen.blit(reward_label, (10 + i * 200, bar_bottom - bar_height - 20))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

import math
import numpy as np
import pygame
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box
from gymnasium.utils import seeding
from Box2D import b2World, b2Vec2, b2RayCastCallback

def raw_env(**kwargs):
    return DiffDrivePushEnv(**kwargs)

class DiffDrivePushEnv(ParallelEnv):
    metadata = {"render_modes": ["human"]}
    AGENT_COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    DUMMY_COLOR = (128, 128, 128)  # Gray for dummy agents

    # Number of dummy agents to spawn when phase==1/2 and n_agents==1
    N_PHASE1_DUMMIES = 5

    # Fixed angular gap between the two dummies in phase2_special mode (radians).
    # The learning agent spawns randomly within this arc.
    PHASE2_SPECIAL_GAP = math.pi * 2 / 2.1   # 120 degrees

    # Angular threshold (radians) defining how far Dummy 1 can be from Dummy 0,
    # and how far the learning agent can spawn from Dummy 0 on the opposite side.
    # Both ranges use the same value so the spacing is symmetric.
    PHASE2_SPECIAL_NEIGHBOR_RANGE = math.pi / 2.0   # 30 degrees

    def __init__(self, n_agents=1, width=15.0, height=15.0,
                 max_steps=300, object_radius=1.5,
                 render_mode=None, pixels_per_meter=60, phase=1,
                 target_distance=5.0, phase1_spawn_bound=5.0,
                 phase1_standoff_dist=0.4,
                 phase3_isHeadingRandom=False,
                 phase2_isEqualSpacing=False,
                 phase1_is_dummies_caged=False,
                 phase2_special=False):
        self.n_agents = n_agents
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agent_name_mapping = {a:i for i,a in enumerate(self.possible_agents)}

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.object_radius = object_radius
        self.target_distance = target_distance
        self.phase1_spawn_bound = phase1_spawn_bound
        self.phase1_standoff_dist = phase1_standoff_dist
        self.phase3_isHeadingRandom = phase3_isHeadingRandom
        self.phase2_isEqualSpacing = phase2_isEqualSpacing
        self.phase1_is_dummies_caged = phase1_is_dummies_caged
        self.phase2_special = phase2_special
        self.render_mode = render_mode
        self.pixels_per_meter = pixels_per_meter
        self.phase = phase
        self.screen_width = int(width*pixels_per_meter)
        self.screen_height = int(height*pixels_per_meter)

        self.max_speed = 2.0
        self.turn_sensitivity = 3.0
        # Fixed normalization base so observations don't scale with map size.
        # 10.0 approximates the diagonal of a 7x7 map.
        self.max_obs_dist = 10.0

        # action/obs spaces
        self._action_space = Box(-1.0,1.0,(2,),np.float32)
        self._observation_space = Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # environment variables
        self.np_random, _ = seeding.np_random(None)
        self.world = None
        self.agent_bodies = []
        self.dummy_bodies = []   # static dummy agents (phase 1/2, n_agents==1)
        self.object_body = None
        self.target_pos = None
        self.agent_hit_object_flags = [False]*self.n_agents
        self.step_count = 0

        # rendering
        self.paused = False
        self.screen = None
        self.clock = None
        self.font = None
        self.last_obs = [np.zeros(8,np.float32) for _ in range(self.n_agents)]
        self.last_actions = [np.zeros(2,np.float32) for _ in range(self.n_agents)]
        self.last_rewards = [0.0] * self.n_agents
        # live cumulative total reward across the episode
        self.cumulative_reward = 0.0
        self.agent_cumulative_rewards = [0.0] * self.n_agents
        self.last_progress_reward = 0.0
        self.cumulative_progress_reward = 0.0
        self.agents_spawn_offset_angle = 0.0

        # object-target distance
        self.prev_dist_to_target = 0.0

        # prev deviation tracking (always initialized to avoid hasattr checks)
        self._prev_devs = [0.0] * self.n_agents
        self._prev_arc_imbalances = [0.0] * self.n_agents

        # phase2_special bookkeeping — set during _add_phase2_special_dummies so
        # render() can draw the per-entity spawn range arcs each frame.
        # _p2s_dummy0_angle : world-space ring angle of Dummy 0 (anchor)
        # _p2s_dummy1_side  : +1 if Dummy 1 is CCW (left) of D0, -1 if CW (right)
        # _p2s_agent_arc_start / _end : angular range the agent may spawn in
        # _p2s_dummy1_arc_start / _end: angular range Dummy 1 may spawn in
        self._p2s_dummy0_angle = 0.0
        self._p2s_dummy1_side = 1
        self._p2s_agent_arc_start = 0.0
        self._p2s_agent_arc_end = 0.0
        self._p2s_dummy1_arc_start = 0.0
        self._p2s_dummy1_arc_end = 0.0

    def action_space(self, agent): return self._action_space
    def observation_space(self, agent): return self._observation_space

    def _use_dummies(self):
        """Returns True when phase==1/2 (or manual -2) and only 1 learning agent — dummy ring is active."""
        return self.phase in [1, 2, -2] and self.n_agents == 1

    def _use_phase2_special(self):
        """Returns True when phase2_special is active and conditions are met."""
        return self.phase2_special and self.phase in [2, -2] and self.n_agents == 1

    def reset(self, seed=None, options=None):
        self.np_random, _ = seeding.np_random(seed)
        # PHASE -1 / -2: Manual control, force one agent
        if self.phase in [-1, -2]:
            self.n_agents = 1
            self.possible_agents = ["agent_0"]
            self.agent_name_mapping = {"agent_0": 0}
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.agent_hit_object_flags = [False]*self.n_agents
        # Reset prev deviations for delta reward calculation — zero first so
        # the first step of a new episode never sees a stale cross-episode delta.
        self._prev_devs = [0.0 for _ in range(self.n_agents)]
        self._prev_arc_imbalances = [0.0 for _ in range(self.n_agents)]
        # New episode random offset for ring formation
        self.agents_spawn_offset_angle = self.np_random.uniform(-math.pi, math.pi)
        # Shuffle spawn slots if we are spawning around object
        self.spawn_slots = list(range(self.n_agents))
        self.np_random.shuffle(self.spawn_slots)
        # create world
        self.world = b2World(gravity=(0,0), doSleep=True)
        self.agent_bodies = []
        self.dummy_bodies = []
        self._add_object()
        self._add_target()

        # ---------------------------------------------------------------
        # phase2_special: spawn dummies FIRST so _add_agent can use their
        # positions to find a valid slot between them.
        # ---------------------------------------------------------------
        if self._use_phase2_special():
            self._add_phase2_special_dummies()
            for i in range(self.n_agents):
                self._add_agent(i)
        else:
            for i in range(self.n_agents):
                self._add_agent(i)

            # Spawn dummy agents when phase==1 or phase==2 (or their manual counterparts) and n_agents==1
            if self._use_dummies():
                total = self.N_PHASE1_DUMMIES + self.n_agents  # treat as 6-agent ring
                for d in range(self.N_PHASE1_DUMMIES):
                    self._add_dummy_agent(slot_index=d + 1, total_agents=total)

        # -----------------------------
        # Initialize _prev_devs based on ACTUAL spawn positions so the very
        # first step produces a delta of ~0 (not a large spike from 0.0 vs
        # the true spawn deviation).
        # -----------------------------
        self._prev_devs = []
        self._prev_arc_imbalances = []
        for i in range(self.n_agents):
            agent_pos = self.agent_bodies[i].position
            obj_pos = self.object_body.position
            # distance from agent surface to object surface
            dist_center = math.hypot(agent_pos.x - obj_pos.x, agent_pos.y - obj_pos.y)
            dist_surface = max(0.0, dist_center - self.object_radius - 0.4)
            deviation = dist_surface - self.phase1_standoff_dist
            self._prev_devs.append(deviation)

            # Arc imbalance: compute from actual positions so step-1 delta is ~0
            my_angle = math.atan2(agent_pos.y - obj_pos.y, agent_pos.x - obj_pos.x)
            neighbor_arcs = []
            for j, other in enumerate(self.agent_bodies):
                if j == i:
                    continue
                a = math.atan2(other.position.y - obj_pos.y, other.position.x - obj_pos.x)
                neighbor_arcs.append(a)
            for db in self.dummy_bodies:
                a = math.atan2(db.position.y - obj_pos.y, db.position.x - obj_pos.x)
                neighbor_arcs.append(a)

            def arc_diff(a, b):
                return (a - b + math.pi) % (2.0 * math.pi) - math.pi

            offsets = [arc_diff(a, my_angle) for a in neighbor_arcs]
            left_offsets  = sorted([o      for o in offsets if o > 0])
            right_offsets = sorted([abs(o) for o in offsets if o < 0])
            gap_left  = left_offsets[0]  if left_offsets  else math.pi
            gap_right = right_offsets[0] if right_offsets else math.pi
            arc_imbalance = (gap_right - gap_left) / math.pi
            self._prev_arc_imbalances.append(arc_imbalance)
        # -----------------------------

        if self.render_mode=="human" and self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width,self.screen_height))
            pygame.display.set_caption("DiffDrivePush")
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.SysFont("Consolas",14)

        # reset cumulative reward each episode
        self.cumulative_reward = 0.0
        self.agent_cumulative_rewards = [0.0] * self.n_agents
        self.last_progress_reward = 0.0
        self.cumulative_progress_reward = 0.0
        self.agents_spawn_offset_angle = 0.0

        # object-target distance
        self.prev_dist_to_target = self._distance_object_to_target()

        obs = {agent:self._get_obs(i) for i,agent in enumerate(self.possible_agents)}
        infos = {agent:{} for agent in self.possible_agents}
        return obs, infos

    def step(self, actions):
        rewards, obs, terms, truncs, infos = {}, {}, {}, {}, {}

        # move agents
        if self.phase in [-1, -2]:
            # Manual control: one agent, controlled by arrow keys
            body = self.agent_bodies[0]
            forward = 0.0
            angular = 0.0
            # Poll keyboard state
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                forward = 1.0
            if keys[pygame.K_DOWN]:
                forward = -1.0
            if keys[pygame.K_LEFT]:
                angular = -1.0
            if keys[pygame.K_RIGHT]:
                angular = 1.0
            self.last_actions[0] = np.array([forward, angular], np.float32)
            vx = float(math.cos(body.angle) * forward * self.max_speed)
            vy = float(math.sin(body.angle) * forward * self.max_speed)
            body.linearVelocity = b2Vec2(vx, vy)
            body.angularVelocity = float(angular * self.turn_sensitivity)
        else:
            for i, agent_name in enumerate(self.possible_agents):
                body = self.agent_bodies[i]
                forward, angular = actions[agent_name]
                self.last_actions[i] = np.array([forward, angular], np.float32)
                vx = float(math.cos(body.angle) * forward * self.max_speed)
                vy = float(math.sin(body.angle) * forward * self.max_speed)
                body.linearVelocity = b2Vec2(vx, vy)
                body.angularVelocity = float(angular * self.turn_sensitivity)

        # Dummy agents are static — explicitly zero their velocities every step
        for db in self.dummy_bodies:
            db.linearVelocity = b2Vec2(0, 0)
            db.angularVelocity = 0.0

        # step physics
        self.world.Step(1/30.0, 6, 2)
        self.step_count += 1

        # compute object-target distance
        dist_to_target = self._distance_object_to_target()
        
        # Terminate if object touches target (defined as distance < sum of radii)
        # Target is treated as having radius 0.2 (visual size), object has self.object_radius
        terminated = dist_to_target <= (self.object_radius + 0.2)

        dist_delta = self.prev_dist_to_target - dist_to_target
        
        # Scale the progress reward so that good movement outweighs the step penalty
        if self.phase in [1, 2, -2]:
            progress_reward = 0.0
        else:
            progress_scale = 10.0
            progress_reward = dist_delta * progress_scale

        self.last_progress_reward = progress_reward
        self.cumulative_progress_reward += progress_reward
        self.prev_dist_to_target = dist_to_target

        max_dist = self.max_obs_dist
        max_dist2 = max_dist ** 2

        # ------------------------------------------------------------------
        # phase2_special neighbor-loss termination: terminate the episode if
        # the agent has lost observation of BOTH nearest neighbors (both
        # agent_obs slots are all-zero vectors in the current observation).
        # Only applies when phase2_special is active (phase 2 or -2, n_agents==1).
        # ------------------------------------------------------------------
        p2s_neighbor_lost = False
        if self._use_phase2_special():
            obs_i = self._get_obs(0)
            neighbor1_lost = np.allclose(obs_i[0:2], 0.0)
            neighbor2_lost = np.allclose(obs_i[2:4], 0.0)
            if neighbor1_lost and neighbor2_lost:
                p2s_neighbor_lost = True

        # fill dictionaries
        for i,agent_name in enumerate(self.possible_agents):
            obs_i = self._get_obs(i)
            self.last_obs[i] = obs_i
            obs[agent_name] = obs_i
            
            # ------------------------------------------------------------------
            # 1. DISTANCE / POSITIONING REWARD
            # ------------------------------------------------------------------
            agent_pos = self.agent_bodies[i].position
            obj_pos = self.object_body.position
            dist_center = math.hypot(agent_pos.x - obj_pos.x, agent_pos.y - obj_pos.y)
            dist_surface = max(0.0, dist_center - self.object_radius - 0.4)

            if self.phase in [1, -1]:
                # --- Phase 1 / manual phase -1: delta reward toward standoff ring ---
                deviation = dist_surface - self.phase1_standoff_dist
                prev_dev = self._prev_devs[i]

                delta_dev    = abs(prev_dev) - abs(deviation)
                delta_reward = delta_dev * 2.0
                stay_bonus   = 0.05 if abs(deviation) <= 0.3 else 0.0

                close_penalty = 0.0
                if deviation < -0.15:
                    close_penalty = -(deviation**2 / max_dist2) * 200.0

                self._prev_devs[i] = deviation
                dist_penalty = delta_reward + stay_bonus + close_penalty

            elif self.phase in [2, -2]:
                # ----------------------------------------------------------
                # Phase 2: two sub-rewards combined
                #
                # (A) STANDOFF reward — same delta system as phase 1, keeps
                #     the agent glued to the ring radius.
                # (B) ARC-SPACING reward — agent computes its arc angle on
                #     the ring and the arc angles of its 2 nearest neighbors.
                #     It should sit exactly halfway between them.
                #     Error = signed gap imbalance; reward = delta(|error|).
                # ----------------------------------------------------------

                # --- (A) Standoff ring reward (identical to phase 1) ---
                deviation = dist_surface - self.phase1_standoff_dist
                prev_dev = self._prev_devs[i]

                self._prev_devs[i] = deviation

                # --- (B) Arc-spacing reward ---
                # Compute arc angle of this agent relative to the object center
                my_angle = math.atan2(agent_pos.y - obj_pos.y,
                                      agent_pos.x - obj_pos.x)

                # Gather arc angles of all neighbors (real agents + dummies)
                neighbor_arcs = []
                for j, other in enumerate(self.agent_bodies):
                    if j == i:
                        continue
                    a = math.atan2(other.position.y - obj_pos.y,
                                   other.position.x - obj_pos.x)
                    neighbor_arcs.append(a)
                for db in self.dummy_bodies:
                    a = math.atan2(db.position.y - obj_pos.y,
                                   db.position.x - obj_pos.x)
                    neighbor_arcs.append(a)

                def arc_diff(a, b):
                    """Signed shortest arc from b to a (a - b), range (-pi, pi]."""
                    return (a - b + math.pi) % (2.0 * math.pi) - math.pi

                offsets = [arc_diff(a, my_angle) for a in neighbor_arcs]

                # Left neighbors: CCW from agent (positive offset), nearest = smallest
                # Right neighbors: CW from agent (negative offset), nearest = smallest abs
                left_offsets  = sorted([o        for o in offsets if o > 0])
                right_offsets = sorted([abs(o)   for o in offsets if o < 0])

                # Gap to nearest neighbor on each side.
                # If a side is empty the agent is alone on that half → gap = pi
                gap_left  = left_offsets[0]  if left_offsets  else math.pi
                gap_right = right_offsets[0] if right_offsets else math.pi

                # Signed imbalance: > 0 means right gap > left gap → too close to left → move right
                # Normalized by pi so range is (-1, 1)
                arc_imbalance = (gap_right - gap_left) / math.pi

                # Delta reward: positive when |imbalance| is shrinking
                prev_imb  = self._prev_arc_imbalances[i]
                delta_imb = abs(prev_imb) - abs(arc_imbalance)
                arc_delta_reward = delta_imb * 50.0

                # Stay bonus: flat reward when nearly centered (|imbalance| < 0.1)
                arc_stay_bonus = (
                    0.08
                    if abs(arc_imbalance) <= 0.1 and abs(deviation) <= 0.3
                    else 0.0
                )

                self._prev_arc_imbalances[i] = arc_imbalance

                dist_penalty = arc_stay_bonus + arc_delta_reward

                if deviation < -0.15:
                    dist_penalty -= (deviation**2 / max_dist2) * 400.0

                # Bleeding penalty for being too far from standoff ring
                if deviation > 0.15:
                    dist_penalty += -(deviation**2 / max_dist2) * 200.0

            else:
                # All other phases: strong penalty for being far from object
                dist_penalty = (-dist_surface * dist_surface / max_dist2) * 10.0
            
            # ------------------------------------------------------------------
            # 2. FACING PENALTY (Encourage facing the object)
            # ------------------------------------------------------------------
            facing_penalty = 0.0
            # Phase 0, 1, and -1 have facing penalty, but NOT phase 2 / -2
            if self.phase in [0, 1, -1]:
                body = self.agent_bodies[i]
                dx = self.object_body.position.x - body.position.x
                dy = self.object_body.position.y - body.position.y
                angle_to_obj = math.atan2(dy, dx)
                rel_angle = angle_to_obj - body.angle
                # Normalize to [-pi, pi]
                rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi

                # No penalty if within ±90 degrees (±π/2)
                if abs(rel_angle) <= (math.pi / 2):
                    facing_penalty = 0.0
                else:
                    # Penalty increases outside ±90 degrees, quadratic scaling
                    excess = abs(rel_angle) - (math.pi / 2)
                    facing_penalty = -((excess / (math.pi / 2)) ** 2) * 0.3

            # ------------------------------------------------------------------
            # 3. COLLISION/PROXIMITY OR FORMATION PENALTY
            # ------------------------------------------------------------------
            dist1, dist2 = self._distance_to_nearest_agents(i)

            near_agent_penalty = 0.0
            if self.phase in [0, 1]:
                # REPULSION PENALTY (Phase 0, 1, -1, and 2)
                max_d = self.max_obs_dist
                # Find which is nearer: dist1 or dist2
                nearest_dist = min(dist1, dist2)
                nearest_dist_norm = nearest_dist / max_d
                safe_threshold = 0.20 # roughly 2 meters

                if nearest_dist_norm < safe_threshold:
                    diff = safe_threshold - nearest_dist_norm
                    near_agent_penalty = diff * diff * 10.0
            elif self.phase == 3:
                # FORMATION MAINTENANCE PENALTY (Phase 3 only)
                diff = abs(dist1 - dist2)
                near_agent_penalty = (diff ** 2) * 1.0
            
            # ------------------------------------------------------------------
            # 4. TOTAL REWARD AGGREGATION
            # ------------------------------------------------------------------
            # Start with individual penalties
            total_reward = dist_penalty + facing_penalty - near_agent_penalty

            # Add shared progress reward (calculated outside loop)
            # This incentivizes moving the object towards the target
            total_reward += progress_reward 

            # Time step penalty (not applied in phase 2 / -2)
            # if self.phase not in [2, -2]:
            total_reward -= 0.02

            # For phase -1 / -2, max steps is infinite (never truncates)
            if self.phase in [-1, -2]:
                truncs[agent_name] = False
                # Terminate if neighbor observation is lost in phase2_special mode
                terms[agent_name] = p2s_neighbor_lost
            else:
                truncs[agent_name] = self.step_count >= self.max_steps
                # Terminate on object reaching target OR neighbor observation lost
                terms[agent_name] = terminated or p2s_neighbor_lost
            
            rewards[agent_name] = total_reward
            self.last_rewards[i] = total_reward
            self.agent_cumulative_rewards[i] += total_reward

            infos[agent_name] = {}

        # accumulate total reward this step
        try:
            step_total = float(sum(rewards.values()))
        except Exception:
            step_total = 0.0
        self.cumulative_reward += step_total

        # render
        if self.render_mode=="human":
            self.render()

        return obs, rewards, terms, truncs, infos

    def _get_obs(self, index):
        body = self.agent_bodies[index]
        max_dist = self.max_obs_dist

        class ClosestCallback(b2RayCastCallback):
            def __init__(self):
                super().__init__()
                self.hit_body = None
                self.hit_fraction = 1.0
            def ReportFixture(self, fixture, point, normal, fraction):
                self.hit_body = fixture.body
                self.hit_fraction = fraction
                return fraction

        def check_visibility(target_pos, target_body=None, radius_adjustment=0.0):
            # Multi-sample visibility for bodies (so seeing any part counts)
            if target_body is not None:
                # try to infer fixture radius
                shape_radius = 0.0
                try:
                    shape = target_body.fixtures[0].shape
                    shape_radius = float(getattr(shape, "radius", 0.0))
                except Exception:
                    shape_radius = 0.0

                sample_radius = max(shape_radius, float(radius_adjustment))

                # fallback to single ray if no meaningful radius
                if sample_radius <= 0.0:
                    cb = ClosestCallback()
                    self.world.RayCast(cb, body.position, b2Vec2(float(target_pos[0]), float(target_pos[1])))
                    if cb.hit_body is not None and cb.hit_body != target_body:
                        return np.array([0.0, 0.0], np.float32)
                    dx = float(target_pos[0]) - float(body.position.x)
                    dy = float(target_pos[1]) - float(body.position.y)
                    angle = (math.atan2(dy, dx) - body.angle + math.pi) % (2*math.pi) - math.pi
                    raw_dist = math.hypot(dx, dy)
                    surface_dist = max(0.0, raw_dist - sample_radius)
                    return np.array([angle/math.pi, surface_dist / max_dist], np.float32)

                # sample rays around the perimeter
                num_samples = 12
                best_norm_dist = float('inf')
                best_angle = 0.0
                for k in range(num_samples):
                    theta = (2.0 * math.pi * k) / num_samples
                    tgt_x = float(target_body.position.x + math.cos(theta) * sample_radius * 0.95)
                    tgt_y = float(target_body.position.y + math.sin(theta) * sample_radius * 0.95)

                    cb = ClosestCallback()
                    self.world.RayCast(cb, body.position, b2Vec2(tgt_x, tgt_y))

                    # If this ray first hits the target body, consider it visible
                    if cb.hit_body is not None and cb.hit_body == target_body:
                        dx = tgt_x - body.position.x
                        dy = tgt_y - body.position.y
                        raw_dist = math.hypot(dx, dy)
                        hit_dist = raw_dist * float(cb.hit_fraction)
                        norm_dist = max(0.0, hit_dist) / max_dist
                        angle = (math.atan2(dy, dx) - body.angle + math.pi) % (2*math.pi) - math.pi
                        if norm_dist < best_norm_dist:
                            best_norm_dist = norm_dist
                            best_angle = angle / math.pi

                if best_norm_dist < float('inf'):
                    return np.array([best_angle, best_norm_dist], np.float32)
                # all perimeter samples blocked
                return np.array([0.0, 0.0], np.float32)

            # Point-target logic (unchanged)
            cb = ClosestCallback()
            self.world.RayCast(cb, body.position, b2Vec2(float(target_pos[0]), float(target_pos[1])))

            dx = float(target_pos[0]) - float(body.position.x)
            dy = float(target_pos[1]) - float(body.position.y)
            angle = (math.atan2(dy, dx) - body.angle + math.pi) % (2*math.pi) - math.pi

            raw_dist = math.hypot(dx, dy)
            surface_dist = max(0.0, raw_dist - radius_adjustment)
            dist = surface_dist / max_dist

            # Visibility Logic for point targets
            if cb.hit_body is not None and cb.hit_body != body:
                ud = cb.hit_body.userData
                if ud and ud.get("type") in ["agent", "object", "dummy"]:
                    return np.array([0.0, 0.0], np.float32)

            return np.array([angle/math.pi, dist], np.float32)

        # -------------------------------------------------------------------
        # Build neighbor list: real agents + dummy agents (if active)
        # -------------------------------------------------------------------
        all_neighbor_bodies = []
        for j, other in enumerate(self.agent_bodies):
            if j == index:
                continue
            dist = math.hypot(other.position.x - body.position.x,
                              other.position.y - body.position.y)
            all_neighbor_bodies.append((dist, other))

        # Include dummy bodies as observable neighbors
        for db in self.dummy_bodies:
            dist = math.hypot(db.position.x - body.position.x,
                              db.position.y - body.position.y)
            all_neighbor_bodies.append((dist, db))

        # Sort by distance and take 2 nearest
        all_neighbor_bodies.sort(key=lambda x: x[0])
        nearest_candidates = all_neighbor_bodies[:2]

        agent_obs = []
        for _, other in nearest_candidates:
            obs_val = check_visibility(other.position, target_body=other)
            agent_obs.append(obs_val)

        # Pad if needed
        while len(agent_obs) < 2:
            agent_obs.append(np.zeros(2, np.float32))

        # Target (zeroed out for phase -1, -2 only — manual modes have no target goal)
        # Phase 1, 2, 3, and 0 all observe the target position.
        if self.phase in [-1, -2]:
            target_obs = np.zeros(2, np.float32)
        else:
            target_obs = check_visibility(self.target_pos, target_body=None)

        # Object (adjust distance by radius to measure to surface)
        object_obs = check_visibility(self.object_body.position, target_body=self.object_body, radius_adjustment=self.object_radius)

        return np.concatenate([agent_obs[0], agent_obs[1], target_obs, object_obs]).astype(np.float32)

    def render(self):
        if self.screen is None: return

        # Event handling with Pause support
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT: self.close(); raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.paused = not self.paused
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                self.close(); raise KeyboardInterrupt

        while self.paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.close(); raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    self.close(); raise KeyboardInterrupt
            pygame.time.wait(100)

        self.screen.fill((30,30,30))
        ppm = self.pixels_per_meter

        # target
        tx, ty = self.target_pos
        pygame.draw.line(self.screen,(255,255,255),(int((tx-0.2)*ppm),int((ty-0.2)*ppm)),
                         (int((tx+0.2)*ppm),int((ty+0.2)*ppm)),3)
        pygame.draw.line(self.screen,(255,255,255),(int((tx-0.2)*ppm),int((ty+0.2)*ppm)),
                         (int((tx+0.2)*ppm),int((ty-0.2)*ppm)),3)

        # object
        ox, oy = self.object_body.position
        pygame.draw.circle(self.screen,(200,200,200),(int(ox*ppm),int(oy*ppm)),int(self.object_radius*ppm))

        # Phase 1, 2, -1, -2: draw standoff-distance overlays
        if self.phase in [1, 2, -1, -2]:
            agent_radius = 0.4
            cx, cy = int(ox * ppm), int(oy * ppm)
            if self.phase in [1, -1]:
                # Set minimum spawn inner radius to standoff distance
                standoff_r = self.object_radius + agent_radius + self.phase1_standoff_dist  # standoff ring (center)
                inner_r = standoff_r  # min spawn dist (center) is standoff
                outer_r = self.object_radius + self.phase1_spawn_bound  # max spawn dist (center)

                # Semi-transparent gray annulus showing the spawn region
                overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
                pygame.draw.circle(overlay, (180, 180, 180, 40), (cx, cy), int(outer_r * ppm))
                pygame.draw.circle(overlay, (0, 0, 0, 0), (cx, cy), int(inner_r * ppm))
                self.screen.blit(overlay, (0, 0))

                # Outer spawn-bound border (gray)
                pygame.draw.circle(self.screen, (160, 160, 160), (cx, cy), int(outer_r * ppm), 1)
                # Inner spawn-bound border (dark gray)
                pygame.draw.circle(self.screen, (100, 100, 100), (cx, cy), int(inner_r * ppm), 1)

            # Standoff ring (orange dashed approximation via solid thin ring)
            standoff_r = self.object_radius + agent_radius + self.phase1_standoff_dist
            pygame.draw.circle(self.screen, (255, 165, 0), (cx, cy), int(standoff_r * ppm), 1)

            # ------------------------------------------------------------------
            # phase2_special: draw per-entity spawn range arcs on the standoff ring.
            #
            # Layout (set in _add_phase2_special_dummies):
            #   Dummy 0  — anchor, shown as a thin gray arc spanning NEIGHBOR_RANGE
            #              on both sides just to mark its exact position clearly.
            #   Dummy 1  — its valid spawn arc (NEIGHBOR_RANGE wide) rendered in
            #              light-gray/silver.
            #   Agent    — its valid spawn arc (NEIGHBOR_RANGE wide) rendered in
            #              the agent's color (red for agent_0).
            #
            # Arc drawing convention: pygame.draw.arc works in screen space where
            # Y increases downward.  A world-space angle θ maps to pygame angle
            # atan2(-sin θ, cos θ) = -θ.  Equivalently, for a world point at angle
            # θ relative to the object centre:
            #   pygame_angle = atan2(-(world_y - obj_y), world_x - obj_x)
            # We draw arcs from the smaller pygame angle to the larger one.
            # ------------------------------------------------------------------
            if self._use_phase2_special() and len(self.dummy_bodies) == 2:
                arc_surf = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
                r_px = int(standoff_r * ppm)
                rect = pygame.Rect(cx - r_px, cy - r_px, r_px * 2, r_px * 2)
                arc_thickness = max(3, r_px // 8)

                def world_to_pygame_angle(world_angle):
                    """Convert a world-space ring angle to pygame arc angle (flipped Y)."""
                    return -world_angle

                def draw_ring_arc(surf, color, a_world_start, a_world_end, thickness):
                    """Draw an arc on the standoff ring between two world-space angles.

                    Handles the CCW / CW ambiguity by always drawing the arc that
                    spans from a_world_start to a_world_end in the CCW (positive)
                    direction in world space, which corresponds to the CW direction
                    in pygame screen space (Y-flipped).
                    """
                    # Ensure we draw the intended arc direction by normalising the
                    # angular span to [0, 2π).
                    span = (a_world_end - a_world_start) % (2.0 * math.pi)
                    if span == 0:
                        return
                    # In pygame (Y-flipped) a CCW world arc becomes CW, so we pass
                    # the start angle as the *end* of the pygame arc and subtract
                    # the span to get the start.
                    pg_end   = world_to_pygame_angle(a_world_start)
                    pg_start = pg_end - span          # subtract because Y is flipped
                    pygame.draw.arc(surf, color, rect, pg_start, pg_end, thickness)

                # --- Dummy 1 spawn range arc (silver / light gray) ---
                draw_ring_arc(arc_surf, (200, 200, 200, 120),
                              self._p2s_dummy1_arc_start,
                              self._p2s_dummy1_arc_end,
                              arc_thickness)

                # --- Agent spawn range arc (red, matching agent_0 color) ---
                draw_ring_arc(arc_surf, (255, 80, 80, 120),
                              self._p2s_agent_arc_start,
                              self._p2s_agent_arc_end,
                              arc_thickness)

                # --- Dummy 0 position marker (thin bright white tick arc) ---
                tick_half = 0.04   # ±0.04 rad tick around D0's exact angle
                draw_ring_arc(arc_surf, (255, 255, 255, 200),
                              self._p2s_dummy0_angle - tick_half,
                              self._p2s_dummy0_angle + tick_half,
                              max(2, arc_thickness // 2))

                self.screen.blit(arc_surf, (0, 0))

        # Draw dummy agents (gray, with a small "D" label, no heading line)
        for d_idx, db in enumerate(self.dummy_bodies):
            dx, dy = db.position
            pygame.draw.circle(self.screen, self.DUMMY_COLOR,
                               (int(dx * ppm), int(dy * ppm)), int(0.4 * ppm))
            # Draw a thin border so they're clearly distinct from the object
            pygame.draw.circle(self.screen, (200, 200, 200),
                               (int(dx * ppm), int(dy * ppm)), int(0.4 * ppm), 1)
            if self.font:
                lbl = self.font.render(f"D{d_idx}", True, (200, 200, 200))
                self.screen.blit(lbl, (int(dx * ppm) - 8, int(dy * ppm) - 7))

        # agents
        for i,body in enumerate(self.agent_bodies):
            x, y = body.position
            angle = body.angle
            color = self.AGENT_COLORS[i%len(self.AGENT_COLORS)]
            pygame.draw.circle(self.screen,color,(int(x*ppm),int(y*ppm)),int(0.4*ppm))
            hx = x + math.cos(angle)*0.6
            hy = y + math.sin(angle)*0.6
            pygame.draw.line(self.screen,(255,255,255),(int(x*ppm),int(y*ppm)),(int(hx*ppm),int(hy*ppm)),2)

            obs = self.last_obs[i]
            act = self.last_actions[i]
            dist_obj = math.hypot(ox-x, oy-y)

            # Get fresh observation for rendering lines
            current_obs = self._get_obs(i)
            max_dist = self.max_obs_dist

            # Helper to draw observation line
            def draw_obs_line(obs_slice, line_color, is_object=False):
                if not np.allclose(obs_slice, 0.0):
                    rel_angle = obs_slice[0] * math.pi
                    dist = obs_slice[1] * max_dist
                    
                    global_angle = angle + rel_angle
                    tx = x + math.cos(global_angle) * dist
                    ty = y + math.sin(global_angle) * dist
                    pygame.draw.line(
                        self.screen,
                        line_color,
                        (int(x*ppm), int(y*ppm)),
                        (int(tx*ppm), int(ty*ppm)),
                        1
                    )

            # Draw lines to nearest agents (Cyan) — these can now point to dummies
            draw_obs_line(current_obs[0:2], (0, 255, 255))
            draw_obs_line(current_obs[2:4], (0, 255, 255))

            # Draw line to target if visible (White)
            draw_obs_line(current_obs[4:6], (255, 255, 255))

            # Draw line to object if visible (Yellow)
            draw_obs_line(current_obs[6:8], (255, 255, 0), is_object=True)

            info_text = (
                f"agent_{i}\n"
                f"Action: ({act[0]:.2f},{act[1]:.2f})\n"
                f"Near_1: ({obs[0]:.2f},{obs[1]:.2f})\n"
                f"Near_2: ({obs[2]:.2f},{obs[3]:.2f})\n"
                f"Target: ({obs[4]:.2f},{obs[5]:.2f})\n"
                f"Object: ({obs[6]:.2f},{obs[7]:.2f})"
            )
            
            text_y_start = 10 + 90*i
            for j,line in enumerate(info_text.split("\n")):
                lbl = self.font.render(line,True,color)
                self.screen.blit(lbl,(10, text_y_start + j*14))
            
            # Render Reward Bar
            r_val = self.last_rewards[i]
            bar_x = 10
            bar_y = text_y_start + 6 * 14 + 5
            bar_width = 100
            bar_height = 6
            zero_x = bar_x + bar_width - 20

            # Background
            pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height))
            # Zero line
            pygame.draw.line(self.screen, (150,150,150), (zero_x, bar_y), (zero_x, bar_y+bar_height), 1)
            
            # Bar
            scale = 100.0
            bar_len = int(abs(r_val) * scale)
            if r_val < 0:
                rect = (zero_x - bar_len, bar_y, bar_len, bar_height)
                pygame.draw.rect(self.screen, (255, 50, 50), rect)
            else:
                rect = (zero_x, bar_y, bar_len, bar_height)
                pygame.draw.rect(self.screen, (50, 255, 50), rect)

            r_lbl = self.font.render(f"{r_val:.2f}", True, color)
            self.screen.blit(r_lbl, (bar_x + bar_width + 5, bar_y - 4))

            cum_val = self.agent_cumulative_rewards[i]
            c_lbl = self.font.render(f"Tot: {cum_val:.1f}", True, color)
            self.screen.blit(c_lbl, (bar_x + bar_width + 60, bar_y - 4))


        # Draw cumulative total reward in bottom-right (pixels)
        if self.font:
            txt = self.font.render(f"Total Reward: {self.cumulative_reward:.3f}", True, (255,255,255))
            tx = max(10, self.screen_width - txt.get_width() - 10)
            ty = max(10, self.screen_height - txt.get_height() - 10)
            self.screen.blit(txt, (tx, ty))

            ptxt = self.font.render(f"Progress (Step): {self.last_progress_reward:.4f}", True, (255,255,255))
            px = max(10, self.screen_width - ptxt.get_width() - 10)
            py = ty - 20 
            self.screen.blit(ptxt, (px, py))

            cptxt = self.font.render(f"Progress (Total): {self.cumulative_progress_reward:.4f}", True, (255,255,255))
            cpx = max(10, self.screen_width - cptxt.get_width() - 10)
            cpy = py - 20
            self.screen.blit(cptxt, (cpx, cpy))

            # Render steps at upper right
            step_txt = self.font.render(f"Step: {self.step_count}/{self.max_steps}", True, (255,255,255))
            sx = max(10, self.screen_width - step_txt.get_width() - 10)
            sy = 10
            self.screen.blit(step_txt, (sx, sy))

            # Phase 1/2/-2 ring legend (bottom-left)
            if self.phase in [1, 2, -2]:
                legend_items = []
                if self.phase == 1:
                    legend_items += [
                        ((160, 160, 160), f"Spawn outer  ({self.object_radius + self.phase1_spawn_bound:.1f}m)"),
                        ((100, 100, 100), f"Spawn inner  ({self.object_radius + 0.4 + 0.1:.1f}m)"),
                    ]
                legend_items.append(
                    ((255, 165, 0), f"Standoff     ({self.object_radius + 0.4 + self.phase1_standoff_dist:.1f}m)")
                )
                if self._use_phase2_special():
                    legend_items.append(
                        (self.DUMMY_COLOR, f"Dummy agents (2) — special mode, neighbor_range={math.degrees(self.PHASE2_SPECIAL_NEIGHBOR_RANGE):.0f}°")
                    )
                    legend_items.append(
                        ((200, 200, 200), f"D1 spawn arc  (silver) ±{math.degrees(self.PHASE2_SPECIAL_NEIGHBOR_RANGE):.0f}° from D0")
                    )
                    legend_items.append(
                        ((255, 80, 80), f"Agent spawn arc (red)  ±{math.degrees(self.PHASE2_SPECIAL_NEIGHBOR_RANGE):.0f}° opposite side")
                    )
                elif self._use_dummies():
                    cage_label = "caged, random spacing" if self.phase in [2, -2] else ("caged" if self.phase1_is_dummies_caged else "free-roaming")
                    legend_items.append(
                        (self.DUMMY_COLOR, f"Dummy agents ({self.N_PHASE1_DUMMIES}) — {cage_label}")
                    )
                legend_y = self.screen_height - len(legend_items) * 16 - 8
                for color, label in legend_items:
                    lbl = self.font.render(label, True, color)
                    self.screen.blit(lbl, (10, legend_y))
                    legend_y += 16

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen: pygame.quit(); self.screen=None

    def _distance_object_to_target(self):
        dx = float(self.target_pos[0]) - float(self.object_body.position.x)
        dy = float(self.target_pos[1]) - float(self.object_body.position.y)
        return math.hypot(dx, dy)

    def _add_phase2_special_dummies(self):
        """Spawn exactly 2 static dummy agents beside each other on the standoff ring.

        Spawn order and geometry
        ------------------------
        Step 1 — Dummy 0 (anchor):
            Placed at a uniformly random angle on the standoff ring.

        Step 2 — Dummy 1 (neighbor):
            Randomly placed within PHASE2_SPECIAL_NEIGHBOR_RANGE of Dummy 0 on
            either the LEFT (CCW, positive angular offset) or the RIGHT (CW,
            negative angular offset).  The side is chosen with equal probability
            each episode.

        Step 3 — Learning agent (handled in _add_agent):
            Placed within PHASE2_SPECIAL_NEIGHBOR_RANGE on the OPPOSITE side of
            Dummy 0 from Dummy 1.

        The following bookkeeping attributes are set so that _add_agent and
        render() can use them:
            _p2s_dummy0_angle      — world ring angle of Dummy 0
            _p2s_dummy1_side       — +1 if D1 is CCW (left), -1 if CW (right)
            _p2s_dummy1_arc_start  — CCW start of D1's valid arc
            _p2s_dummy1_arc_end    — CCW end   of D1's valid arc
            _p2s_agent_arc_start   — CCW start of agent's valid arc
            _p2s_agent_arc_end     — CCW end   of agent's valid arc
        """
        radius = 0.4
        spawn_dist = self.object_radius + radius + self.phase1_standoff_dist
        obj_pos = self.object_body.position
        nr = self.PHASE2_SPECIAL_NEIGHBOR_RANGE   # shorthand

        # --- Step 1: Dummy 0 at a random angle ---
        d0_angle = self.np_random.uniform(-math.pi, math.pi)
        self._p2s_dummy0_angle = d0_angle

        d0_x = obj_pos.x + math.cos(d0_angle) * spawn_dist
        d0_y = obj_pos.y + math.sin(d0_angle) * spawn_dist
        d0_x = max(radius, min(self.width - radius, d0_x))
        d0_y = max(radius, min(self.height - radius, d0_y))

        body0 = self.world.CreateStaticBody(
            position=(float(d0_x), float(d0_y)),
            angle=float(d0_angle + math.pi)   # face inward
        )
        body0.CreateCircleFixture(radius=radius, friction=0.8, restitution=0.2)
        body0.userData = {"type": "dummy", "index": 0}
        self.dummy_bodies.append(body0)

        # --- Step 2: Dummy 1 to the left OR right of Dummy 0 ---
        # side: +1 → CCW (left in math convention), -1 → CW (right)
        d1_side = 1 if self.np_random.random() < 0.5 else -1
        self._p2s_dummy1_side = d1_side

        # Valid arc for Dummy 1: [d0_angle, d0_angle + side*nr]  (CCW positive)
        if d1_side == 1:
            # D1 is to the LEFT (CCW) of D0 → arc goes from d0 CCW by nr
            self._p2s_dummy1_arc_start = d0_angle
            self._p2s_dummy1_arc_end   = d0_angle + nr
        else:
            # D1 is to the RIGHT (CW) of D0 → arc goes from d0 - nr to d0
            self._p2s_dummy1_arc_start = d0_angle - nr
            self._p2s_dummy1_arc_end   = d0_angle

        # Agent spawns on the OPPOSITE side of D0 from D1
        if d1_side == 1:
            # D1 left → agent to the RIGHT (CW) → arc from d0 - nr to d0
            self._p2s_agent_arc_start = d0_angle - nr
            self._p2s_agent_arc_end   = d0_angle
        else:
            # D1 right → agent to the LEFT (CCW) → arc from d0 to d0 + nr
            self._p2s_agent_arc_start = d0_angle
            self._p2s_agent_arc_end   = d0_angle + nr

        # Try up to 200 times to place Dummy 1 within its valid arc
        d1_placed = False
        for _ in range(200):
            theta = self.np_random.uniform(self._p2s_dummy1_arc_start,
                                           self._p2s_dummy1_arc_end)
            cx = obj_pos.x + math.cos(theta) * spawn_dist
            cy = obj_pos.y + math.sin(theta) * spawn_dist

            if (cx < radius or cx > self.width - radius or
                    cy < radius or cy > self.height - radius):
                continue

            # Must not overlap with Dummy 0
            if math.hypot(cx - d0_x, cy - d0_y) < (radius * 2 + 0.05):
                continue

            body1 = self.world.CreateStaticBody(
                position=(float(cx), float(cy)),
                angle=float(theta + math.pi)   # face inward
            )
            body1.CreateCircleFixture(radius=radius, friction=0.8, restitution=0.2)
            body1.userData = {"type": "dummy", "index": 1}
            self.dummy_bodies.append(body1)
            d1_placed = True
            break

        if not d1_placed:
            # Fallback: place at the far edge of the valid arc
            theta = self._p2s_dummy1_arc_end if d1_side == 1 else self._p2s_dummy1_arc_start
            cx = obj_pos.x + math.cos(theta) * spawn_dist
            cy = obj_pos.y + math.sin(theta) * spawn_dist
            cx = max(radius, min(self.width - radius, cx))
            cy = max(radius, min(self.height - radius, cy))
            body1 = self.world.CreateStaticBody(
                position=(float(cx), float(cy)),
                angle=float(theta + math.pi)
            )
            body1.CreateCircleFixture(radius=radius, friction=0.8, restitution=0.2)
            body1.userData = {"type": "dummy", "index": 1}
            self.dummy_bodies.append(body1)

    def _add_agent(self,index):
        radius = 0.4
        
        pos_x, pos_y = 0.0, 0.0

        if self.phase == 3 and self.object_body:
            # PHASE 3: Spawning around object (Ring Formation - Equally Spaced)
            slot_idx = self.spawn_slots[index]
            
            theta = (2 * math.pi * slot_idx) / self.n_agents + self.agents_spawn_offset_angle
            spawn_dist = self.object_radius + radius + 0.2
            
            pos_x = self.object_body.position.x + math.cos(theta) * spawn_dist
            pos_y = self.object_body.position.y + math.sin(theta) * spawn_dist
            
            if self.phase3_isHeadingRandom:
                angle = float(self.np_random.uniform(-math.pi, math.pi))
            else:
                angle = theta + math.pi

        elif self._use_phase2_special() and self.object_body:
            # -----------------------------------------------------------------
            # phase2_special: agent spawns randomly within its designated arc on
            # the standoff ring — on the OPPOSITE side of Dummy 0 from Dummy 1.
            # The arc bounds were stored in _add_phase2_special_dummies.
            # -----------------------------------------------------------------
            spawn_dist = self.object_radius + radius + self.phase1_standoff_dist

            # Fallback position: midpoint of the agent arc
            mid_theta = (self._p2s_agent_arc_start + self._p2s_agent_arc_end) / 2.0
            pos_x = self.object_body.position.x + math.cos(mid_theta) * spawn_dist
            pos_y = self.object_body.position.y + math.sin(mid_theta) * spawn_dist
            angle = mid_theta + math.pi   # face inward

            for _ in range(200):
                theta = self.np_random.uniform(self._p2s_agent_arc_start,
                                               self._p2s_agent_arc_end)
                cx = self.object_body.position.x + math.cos(theta) * spawn_dist
                cy = self.object_body.position.y + math.sin(theta) * spawn_dist

                if (cx < radius or cx > self.width - radius or
                        cy < radius or cy > self.height - radius):
                    continue

                # Must not overlap with already-placed dummies
                collision = False
                for db in self.dummy_bodies:
                    if math.hypot(cx - db.position.x, cy - db.position.y) < (radius * 2 + 0.05):
                        collision = True
                        break
                if collision:
                    continue

                pos_x, pos_y = cx, cy
                angle = theta + math.pi  # face inward
                break

        elif self.phase in [2, -2] and self.object_body:
            spawn_dist = self.object_radius + radius + self.phase1_standoff_dist
            if self.phase2_isEqualSpacing:
                slot_idx = self.spawn_slots[index]
                theta = (2 * math.pi * slot_idx) / self.n_agents + self.agents_spawn_offset_angle
            else:
                theta = self.np_random.uniform(-math.pi, math.pi)
            pos_x = self.object_body.position.x + math.cos(theta) * spawn_dist
            pos_y = self.object_body.position.y + math.sin(theta) * spawn_dist
            angle = theta + math.pi

        elif self.phase == 1 and self.object_body:
            # PHASE 1: Random spawn NEAR object (within bounds)
            angle = float(self.np_random.uniform(-math.pi, math.pi))

            min_spawn_dist = self.object_radius + radius + self.phase1_standoff_dist
            max_spawn_dist = self.object_radius + self.phase1_spawn_bound

            for _ in range(100):
                r = math.sqrt(self.np_random.uniform(min_spawn_dist**2, max_spawn_dist**2))
                theta = self.np_random.uniform(-math.pi, math.pi)

                pos_x = self.object_body.position.x + r * math.cos(theta)
                pos_y = self.object_body.position.y + r * math.sin(theta)

                if (pos_x < radius or pos_x > self.width - radius or 
                    pos_y < radius or pos_y > self.height - radius):
                    continue

                if self.target_pos is not None:
                    dist_tgt = math.hypot(pos_x - self.target_pos[0], pos_y - self.target_pos[1])
                    if dist_tgt < (radius + 0.4): 
                        continue

                collision = False
                for other in self.agent_bodies:
                    dist_ag = math.hypot(pos_x - other.position.x, pos_y - other.position.y)
                    if dist_ag < (radius * 2 + 0.1):
                        collision = True
                        break

                if not collision:
                    break

        else:
            # PHASE 0: Random spawn anywhere in the box
            angle = float(self.np_random.uniform(-math.pi, math.pi))
            
            margin = radius + 0.1
            
            for _ in range(100):
                pos_x = self.np_random.uniform(margin, self.width - margin)
                pos_y = self.np_random.uniform(margin, self.height - margin)
                
                if self.object_body:
                    dist_obj = math.hypot(pos_x - self.object_body.position.x, pos_y - self.object_body.position.y)
                    if dist_obj < (radius + self.object_radius + 0.2):
                        continue

                if self.target_pos is not None:
                    dist_tgt = math.hypot(pos_x - self.target_pos[0], pos_y - self.target_pos[1])
                    if dist_tgt < (radius + 0.4): 
                        continue

                collision = False
                for other in self.agent_bodies:
                    dist_ag = math.hypot(pos_x - other.position.x, pos_y - other.position.y)
                    if dist_ag < (radius * 2 + 0.1):
                        collision = True
                        break
                
                if not collision:
                    break

        body = self.world.CreateDynamicBody(position=(float(pos_x), float(pos_y)), angle=float(angle))
        body.CreateCircleFixture(radius=radius, density=1.0, friction=0.8, restitution=0.2)
        body.userData = {"type":"agent","index":index}
        self.agent_bodies.append(body)

    def _add_dummy_agent(self, slot_index: int, total_agents: int):
        """Spawn a static dummy agent on the standoff ring.

        Phase 2 (always):                        caged on the standoff ring at a
                                                  randomly drawn angle each episode
                                                  (non-overlapping, not equally spaced).
        Phase 1 (phase1_is_dummies_caged=True):  fixed equally-spaced ring position.
        Phase 1 (phase1_is_dummies_caged=False):  random spawn within phase 1 bounds.
        """
        radius = 0.4
        spawn_dist = self.object_radius + radius + self.phase1_standoff_dist

        if self.phase in [2, -2]:
            # -----------------------------------------------------------------
            # Phase 2: caged on standoff ring, RANDOM angular positions
            # -----------------------------------------------------------------
            # Try up to 200 random angles; reject any that overlap existing
            # agents or dummies on the ring.
            pos_x = self.object_body.position.x  # fallback (object center)
            pos_y = self.object_body.position.y
            angle = 0.0
            placed = False

            for _ in range(200):
                theta = self.np_random.uniform(-math.pi, math.pi)
                cx = self.object_body.position.x + math.cos(theta) * spawn_dist
                cy = self.object_body.position.y + math.sin(theta) * spawn_dist

                # Map boundary check
                if (cx < radius or cx > self.width - radius or
                        cy < radius or cy > self.height - radius):
                    continue

                # Avoid overlap with learning agents
                collision = False
                for other in self.agent_bodies:
                    if math.hypot(cx - other.position.x, cy - other.position.y) < (radius * 2 + 0.05):
                        collision = True
                        break
                if collision:
                    continue

                # Avoid overlap with already-placed dummies
                for other in self.dummy_bodies:
                    if math.hypot(cx - other.position.x, cy - other.position.y) < (radius * 2 + 0.05):
                        collision = True
                        break
                if collision:
                    continue

                pos_x, pos_y = cx, cy
                angle = theta + math.pi   # face inward
                placed = True
                break

            if not placed:
                # Fallback: use equally-spaced slot
                theta = (2.0 * math.pi * slot_index) / total_agents
                pos_x = self.object_body.position.x + math.cos(theta) * spawn_dist
                pos_y = self.object_body.position.y + math.sin(theta) * spawn_dist
                pos_x = max(radius, min(self.width - radius, pos_x))
                pos_y = max(radius, min(self.height - radius, pos_y))
                angle = theta + math.pi

        elif self.phase1_is_dummies_caged:
            # -----------------------------------------------------------------
            # Phase 1 caged: fixed equally-spaced ring position
            # -----------------------------------------------------------------
            theta = (2.0 * math.pi * slot_index) / total_agents + self.agents_spawn_offset_angle
            pos_x = self.object_body.position.x + math.cos(theta) * spawn_dist
            pos_y = self.object_body.position.y + math.sin(theta) * spawn_dist
            pos_x = max(radius, min(self.width - radius, pos_x))
            pos_y = max(radius, min(self.height - radius, pos_y))
            angle = theta + math.pi

        else:
            # -----------------------------------------------------------------
            # Phase 1 not caged: random spawn within phase 1 bounds
            # -----------------------------------------------------------------
            min_spawn_dist = self.object_radius + radius + self.phase1_standoff_dist
            max_spawn_dist = self.object_radius + self.phase1_spawn_bound
            angle = float(self.np_random.uniform(-math.pi, math.pi))
            pos_x, pos_y = self.object_body.position.x, self.object_body.position.y  # fallback

            for _ in range(200):
                r = math.sqrt(self.np_random.uniform(min_spawn_dist**2, max_spawn_dist**2))
                theta = self.np_random.uniform(-math.pi, math.pi)
                cx = self.object_body.position.x + r * math.cos(theta)
                cy = self.object_body.position.y + r * math.sin(theta)

                if (cx < radius or cx > self.width - radius or
                        cy < radius or cy > self.height - radius):
                    continue

                if self.target_pos is not None:
                    if math.hypot(cx - self.target_pos[0], cy - self.target_pos[1]) < (radius + 0.4):
                        continue

                collision = False
                for other in self.agent_bodies:
                    if math.hypot(cx - other.position.x, cy - other.position.y) < (radius * 2 + 0.1):
                        collision = True
                        break
                if collision:
                    continue

                for other in self.dummy_bodies:
                    if math.hypot(cx - other.position.x, cy - other.position.y) < (radius * 2 + 0.1):
                        collision = True
                        break
                if collision:
                    continue

                pos_x, pos_y = cx, cy
                angle = float(self.np_random.uniform(-math.pi, math.pi))
                break

        # Static body — never moves
        body = self.world.CreateStaticBody(position=(float(pos_x), float(pos_y)),
                                           angle=float(angle))
        body.CreateCircleFixture(radius=radius, friction=0.8, restitution=0.2)
        body.userData = {"type": "dummy", "index": len(self.dummy_bodies)}
        self.dummy_bodies.append(body)

    def _add_object(self):
        # Center spawn for phase 1, 2, -1, -2
        if self.phase in [1, 2, -1, -2]:
            x = self.width / 2.0
            y = self.height / 2.0
        else:
            if self.phase in [3]:
                extra_margin = 0.7
            else:
                extra_margin = 0.1
            margin = self.object_radius + extra_margin
            x = self.np_random.uniform(margin, self.width - margin)
            y = self.np_random.uniform(margin, self.height - margin)
        self.object_body = self.world.CreateDynamicBody(position=(float(x), float(y)), linearDamping=8.0, angularDamping=6.0)
        if self.phase in [0, 3]:
            obj_density = 1.0
        else:
            obj_density = 10000.0
        self.object_body.CreateCircleFixture(radius=self.object_radius, density=obj_density, friction=5.0)
        self.object_body.userData = {"type":"object"}

    def _add_target(self):
        target_radius = 0.2
        margin = target_radius + 0.2

        # Phase 1 and 2 (including manual -1/-2): spawn target randomly anywhere
        # in the map, only avoiding overlap with the object.
        # target_distance is NOT used for these phases.
        if self.phase in [1, 2, -1, -2]:
            ox = self.object_body.position.x
            oy = self.object_body.position.y
            min_dist_from_object = self.object_radius + target_radius + 0.3

            for _ in range(200):
                tx = self.np_random.uniform(margin, self.width - margin)
                ty = self.np_random.uniform(margin, self.height - margin)

                if math.hypot(tx - ox, ty - oy) < min_dist_from_object:
                    continue

                self.target_pos = np.array([float(tx), float(ty)], np.float32)
                return

            # Fallback: place target on the opposite side of the map from the object
            tx = self.width - ox
            ty = self.height - oy
            tx = max(margin, min(self.width - margin, tx))
            ty = max(margin, min(self.height - margin, ty))
            self.target_pos = np.array([float(tx), float(ty)], np.float32)
            return

        # Phase 3 and all others: use fixed target_distance from object (original behaviour)
        fixed_distance = self.target_distance
        ox = self.object_body.position.x
        oy = self.object_body.position.y

        valid_position_found = False

        for _ in range(100):
            angle = self.np_random.uniform(0, 2 * math.pi)
            tx = ox + math.cos(angle) * fixed_distance
            ty = oy + math.sin(angle) * fixed_distance
            
            if (tx > margin and tx < self.width - margin and 
                ty > margin and ty < self.height - margin):
                self.target_pos = np.array([float(tx), float(ty)], np.float32)
                valid_position_found = True
                break
        
        if not valid_position_found:
             super_margin = margin
             self.target_pos = np.array([
                float(self.np_random.uniform(super_margin, self.width - super_margin)),
                float(self.np_random.uniform(super_margin, self.height - super_margin))
            ], np.float32)

    def _distance_to_nearest_agents(self, index):
        body = self.agent_bodies[index]
        other_bodies = [b for j, b in enumerate(self.agent_bodies) if j != index]
        # Include dummy agents so that phase-1/2 single-agent scenarios
        # don't default to dist=0 (which always triggers the repulsion penalty)
        other_bodies += self.dummy_bodies

        distances = []
        for other in other_bodies:
            dx = other.position.x - body.position.x
            dy = other.position.y - body.position.y
            distances.append(math.hypot(dx, dy))

        distances.sort()
        if len(distances) > 0:
            dist1 = distances[0]
        else:
            dist1 = float('inf')
        if len(distances) > 1:
            dist2 = distances[1]
        else:
            dist2 = float('inf')

        return dist1, dist2
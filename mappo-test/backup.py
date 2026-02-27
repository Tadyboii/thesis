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

    def __init__(self, n_agents=6, width=10.0, height=10.0,
                 max_steps=250, object_radius=1.5,
                 render_mode="human", pixels_per_meter=60):
        self.n_agents = n_agents
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agent_name_mapping = {a:i for i,a in enumerate(self.possible_agents)}

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.object_radius = object_radius
        self.render_mode = render_mode
        self.pixels_per_meter = pixels_per_meter
        self.screen_width = int(width*pixels_per_meter)
        self.screen_height = int(height*pixels_per_meter)

        self.max_speed = 6.0
        self.turn_sensitivity = 4.0

        # action/obs spaces
        self._action_space = Box(-1.0,1.0,(2,),np.float32)
        self._observation_space = Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # environment variables
        self.np_random, _ = seeding.np_random(None)
        self.world = None
        self.agent_bodies = []
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

    def action_space(self, agent): return self._action_space
    def observation_space(self, agent): return self._observation_space

    def reset(self, seed=None, options=None):
        self.np_random, _ = seeding.np_random(seed)
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.agent_hit_object_flags = [False]*self.n_agents

        # create world
        self.world = b2World(gravity=(0,0), doSleep=True)
        self.agent_bodies = []

        self._add_object()
        self._add_target()
        for i in range(self.n_agents):
            self._add_agent(i)

        if self.render_mode=="human" and self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width,self.screen_height))
            pygame.display.set_caption("DiffDrivePush")
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.SysFont("Consolas",14)

        # store previous object-target distance for rewards
        self.prev_dist_to_target = self._distance_object_to_target()

        obs = {agent:self._get_obs(i) for i,agent in enumerate(self.possible_agents)}
        infos = {agent:{} for agent in self.possible_agents}
        return obs, infos

    def step(self, actions):
        rewards, obs, terms, truncs, infos = {}, {}, {}, {}, {}

        # move agents
        for i, agent_name in enumerate(self.possible_agents):
            body = self.agent_bodies[i]
            forward, angular = actions[agent_name]
            self.last_actions[i] = np.array([forward, angular], np.float32)

            vx = float(math.cos(body.angle) * forward * self.max_speed)
            vy = float(math.sin(body.angle) * forward * self.max_speed)
            body.linearVelocity = b2Vec2(vx, vy)
            body.angularVelocity = float(angular * self.turn_sensitivity)

        # step physics
        self.world.Step(1/60.0, 6, 2)
        self.step_count += 1

        # compute object-target distance
        dist_to_target = self._distance_object_to_target()
        terminated = dist_to_target <= self.object_radius

        # compute reward as progress
        progress_reward = self.prev_dist_to_target - dist_to_target
        self.prev_dist_to_target = dist_to_target

        max_dist = math.hypot(self.width, self.height)
        max_dist2 = max_dist ** 2

        # fill dictionaries
        for i,agent_name in enumerate(self.possible_agents):
            obs_i = self._get_obs(i)
            self.last_obs[i] = obs_i
            obs[agent_name] = obs_i
            
            # DISTANCE PENALTY
            # Calculate distance to object surface
            agent_pos = self.agent_bodies[i].position
            obj_pos = self.object_body.position
            dist_center = math.hypot(agent_pos.x - obj_pos.x, agent_pos.y - obj_pos.y)
            dist_surface = max(0.0, dist_center - self.object_radius)
            
            penalty = (-dist_surface * dist_surface / max_dist2) * 10.0
            
            # FACING PENALTY
            body = self.agent_bodies[i]
            dx = self.object_body.position.x - body.position.x
            dy = self.object_body.position.y - body.position.y
            angle_to_obj = math.atan2(dy, dx)
            rel_angle = angle_to_obj - body.angle
            # Normalize to [-pi, pi]
            rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
            # Squared facing penalty
            facing_penalty = -(rel_angle / math.pi) ** 2 * 0.3
            
            total_reward = penalty + facing_penalty
            
            rewards[agent_name] = total_reward
            self.last_rewards[i] = total_reward

            terms[agent_name] = terminated
            truncs[agent_name] = self.step_count >= self.max_steps
            infos[agent_name] = {}

        # render
        if self.render_mode=="human":
            self.render()

        return obs, rewards, terms, truncs, infos

    def _get_obs(self, index):
        body = self.agent_bodies[index]
        max_dist = math.hypot(self.width,self.height)

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
            cb = ClosestCallback()
            self.world.RayCast(cb, body.position, b2Vec2(float(target_pos[0]), float(target_pos[1])))
            
            dx = float(target_pos[0]) - float(body.position.x)
            dy = float(target_pos[1]) - float(body.position.y)
            angle = (math.atan2(dy, dx) - body.angle + math.pi) % (2*math.pi) - math.pi
            
            raw_dist = math.hypot(dx, dy)
            # Adjust distance for surface observation
            surface_dist = max(0.0, raw_dist - radius_adjustment)
            dist = surface_dist / max_dist

            # Visibility Logic
            if target_body: 
                 # We are looking at a specific agent or object
                 if cb.hit_body is not None and cb.hit_body != target_body:
                     return np.array([0.0, 0.0], np.float32) # Blocked by something else
            else: 
                 # We are looking at a point (target position)
                 if cb.hit_body is not None and cb.hit_body != body:
                     ud = cb.hit_body.userData
                     # If we hit an agent or object, view to target is blocked
                     if ud and ud.get("type") in ["agent", "object"]:
                         return np.array([0.0, 0.0], np.float32)
            
            return np.array([angle/math.pi, dist], np.float32)

        # 1. Find all neighbors and their true distances
        neighbors = []
        for j, other in enumerate(self.agent_bodies):
            if j == index: continue
            dist = math.hypot(other.position.x - body.position.x, other.position.y - body.position.y)
            neighbors.append((dist, other))

        # 2. Sort by distance
        neighbors.sort(key=lambda x: x[0])

        # 3. Take top 2
        nearest_candidates = neighbors[:2]

        # 4. Get observations for these candidates
        agent_obs = []
        for _, other in nearest_candidates:
             obs_val = check_visibility(other.position, target_body=other)
             agent_obs.append(obs_val)

        # Pad if needed
        while len(agent_obs) < 2:
             agent_obs.append(np.zeros(2, np.float32))

        # Target
        target_obs = check_visibility(self.target_pos, target_body=None)

        # Object (adjust distance by radius to measure to surface)
        object_obs = check_visibility(self.object_body.position, target_body=self.object_body, radius_adjustment=self.object_radius)

        return np.concatenate([agent_obs[0], agent_obs[1], target_obs, object_obs]).astype(np.float32)

    def render(self):
        if self.screen is None: return

        # Event handling with Pause support
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT: self.close(); return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.paused = not self.paused

        while self.paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.close(); return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.paused = not self.paused
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
            max_dist = math.hypot(self.width, self.height)

            # Helper to draw observation line
            def draw_obs_line(obs_slice, line_color, is_object=False):
                if not np.allclose(obs_slice, 0.0):
                    rel_angle = obs_slice[0] * math.pi
                    # obs_slice[1] is normalized distance to surface (if object) or center (if target/agent)
                    # We just draw the line of that length.
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

            # Draw lines to nearest agents (Cyan)
            draw_obs_line(current_obs[0:2], (0, 255, 255))
            draw_obs_line(current_obs[2:4], (0, 255, 255))

            # Draw line to target if visible (White)
            draw_obs_line(current_obs[4:6], (255, 255, 255))

            # Draw line to object if visible (Yellow) - logic handles surface dist automatically
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
            bar_y = text_y_start + 6 * 14 + 5 # Below the 6 lines of text
            bar_width = 100
            bar_height = 6
            zero_x = bar_x + bar_width - 20 # Offset zero point to the right to accommodate large negative penalty
            
            # Background
            pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height))
            # Zero line
            pygame.draw.line(self.screen, (150,150,150), (zero_x, bar_y), (zero_x, bar_y+bar_height), 1)
            
            # Bar
            scale = 100.0 # pixels per unit
            bar_len = int(abs(r_val) * scale)
            if r_val < 0:
                # Red bar to the left
                rect = (zero_x - bar_len, bar_y, bar_len, bar_height)
                pygame.draw.rect(self.screen, (255, 50, 50), rect)
            else:
                # Green bar to the right
                rect = (zero_x, bar_y, bar_len, bar_height)
                pygame.draw.rect(self.screen, (50, 255, 50), rect)

            # Render numeric reward value
            r_lbl = self.font.render(f"{r_val:.2f}", True, color)
            self.screen.blit(r_lbl, (bar_x + bar_width + 5, bar_y - 4))


        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen: pygame.quit(); self.screen=None

    def _distance_object_to_target(self):
        dx = float(self.target_pos[0]) - float(self.object_body.position.x)
        dy = float(self.target_pos[1]) - float(self.object_body.position.y)
        return math.hypot(dx, dy)

    def _add_agent(self,index):
        radius=0.4
        x=self.np_random.uniform(2,self.width-2)
        y=self.np_random.uniform(2,self.height-2)
        body=self.world.CreateDynamicBody(position=(float(x), float(y)), angle=float(self.np_random.uniform(-math.pi, math.pi)))
        body.CreateCircleFixture(radius=radius,density=1.0,friction=0.8,restitution=0.2)
        body.userData={"type":"agent","index":index}
        self.agent_bodies.append(body)

    def _add_object(self):
        x=self.np_random.uniform(4,self.width-4)
        y=self.np_random.uniform(4,self.height-4)
        self.object_body=self.world.CreateDynamicBody(position=(float(x), float(y)),linearDamping=8.0,angularDamping=6.0)
        self.object_body.CreateCircleFixture(radius=self.object_radius,density=10.0,friction=5.0)
        self.object_body.userData={"type":"object"}

    def _add_target(self):
        self.target_pos=np.array([float(self.np_random.uniform(2,self.width-2)),
                                  float(self.np_random.uniform(2,self.height-2))],np.float32)

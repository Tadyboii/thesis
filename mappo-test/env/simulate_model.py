import os
import time
import math
import argparse
import torch
import numpy as np
import pygame
import torch.nn as nn
from torch.distributions.normal import Normal

# ======================
# Argument Parsing
# ======================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    type=str,
    default="ros",
    choices=["ros", "box2d", "pymunk", "swarm"],
    help="Environment to use: 'ros' (ros_caging_env), 'box2d' (particle_box2d), 'pymunk' (particle_pymunk)",
)
args = parser.parse_args()

if args.env == "ros":
    from ros_caging_env import DiffDrivePushEnv
elif args.env == "box2d":
    from particle_box2d import DiffDrivePushEnv
elif args.env == "pymunk":
    from particle_pymunk import DiffDrivePushEnv
elif args.env == "swarm":
    from swarm_caging_env import SwarmCagingEnv

print(f"[INFO] Using environment: {args.env}")

from env.pettingzoo_wrapper import PettingZooWrapper

# ======================
# Device
# ======================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# ======================
# Checkpoint    
# ======================
checkpoint_approach  = "checkpoints/pz__DiffDrivePushEnv__2026-02-28_14-19-44_.pt"
checkpoint_cage      = "checkpoints/phase2.pt"
checkpoint_transport = "checkpoints/phase1_2.pt"
current_checkpoint_path = checkpoint_approach

CHECK_INTERVAL_SEC = 2.0   # how often to check file timestamp


# ===========================================================================
# Menu screen (only used when args.env == "swarm")
# ===========================================================================

_MENU_W, _MENU_H = 960, 540
_BG_DARK   = (10,  12,  18)
_BG_PANEL  = (18,  22,  32)
_GRID_COL  = (28,  34,  50)
_ACCENT    = (0,  210, 180)
_ACCENT2   = (255, 160,  40)
_WHITE     = (220, 230, 240)
_MUTED     = (100, 115, 140)
_HOVER_BG  = (30,  40,  60)
_PRESS_BG  = (0,  130, 115)
_SEP       = (40,  50,  70)


class _Button:
    def __init__(self, rect, label, accent_color, config, sub_label=None):
        self.rect      = pygame.Rect(rect)
        self.label     = label
        self.sub_label = sub_label
        self.accent    = accent_color
        self.config    = config
        self.hovered   = False
        self.pressed   = False
        self._press_t  = 0.0

    def handle_event(self, event):
        mx, my = pygame.mouse.get_pos()
        self.hovered = self.rect.collidepoint(mx, my)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.hovered:
            self.pressed  = True
            self._press_t = time.time()
            return self.config
        return None

    def update(self):
        if self.pressed and (time.time() - self._press_t) > 0.15:
            self.pressed = False

    def draw(self, surf, phase_font, sub_font):
        bg = _PRESS_BG if self.pressed else (_HOVER_BG if self.hovered else _BG_PANEL)
        pygame.draw.rect(surf, bg, self.rect, border_radius=8)
        border_col = self.accent if (self.hovered or self.pressed) else _SEP
        border_w   = 2 if (self.hovered or self.pressed) else 1
        pygame.draw.rect(surf, border_col, self.rect, border_w, border_radius=8)
        if self.hovered or self.pressed:
            top_bar = pygame.Rect(self.rect.x + border_w, self.rect.y + border_w,
                                  self.rect.width - border_w * 2, 3)
            pygame.draw.rect(surf, self.accent, top_bar, border_radius=2)
        text_color = _WHITE if not self.pressed else _BG_DARK
        lbl = phase_font.render(self.label, True, text_color)
        surf.blit(lbl, lbl.get_rect(centerx=self.rect.centerx,
                                    centery=self.rect.centery - (10 if self.sub_label else 0)))
        if self.sub_label:
            sub = sub_font.render(self.sub_label, True,
                                  self.accent if not self.pressed else _BG_DARK)
            surf.blit(sub, sub.get_rect(centerx=self.rect.centerx,
                                        centery=self.rect.centery + 14))


def _draw_grid(surf):
    for x in range(0, _MENU_W, 40):
        pygame.draw.line(surf, _GRID_COL, (x, 0), (x, _MENU_H))
    for y in range(0, _MENU_H, 40):
        pygame.draw.line(surf, _GRID_COL, (0, y), (_MENU_W, y))


def _draw_corners(surf, rect, color, size=18, thick=2):
    x, y, w, h = rect
    segs = [
        ((x, y + size), (x, y), (x + size, y)),
        ((x + w - size, y), (x + w, y), (x + w, y + size)),
        ((x + w, y + h - size), (x + w, y + h), (x + w - size, y + h)),
        ((x + size, y + h), (x, y + h), (x, y + h - size)),
    ]
    for pts in segs:
        pygame.draw.lines(surf, color, False, pts, thick)


def run_swarm_menu() -> dict:
    """Show a phase-selection menu and return {phase, n_agents}."""
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((_MENU_W, _MENU_H))
    pygame.display.set_caption("SWARM CAGING — Mission Select")
    clock = pygame.time.Clock()

    def load_font(names, size):
        for name in names:
            try:
                return pygame.font.SysFont(name, size)
            except Exception:
                pass
        return pygame.font.Font(None, size)

    title_font = load_font(["Courier New", "Lucida Console", "monospace"], 38)
    label_font = load_font(["Courier New", "Lucida Console", "monospace"], 15)
    phase_font = load_font(["Courier New", "Lucida Console", "monospace"], 22)
    sub_font   = load_font(["Courier New", "Lucida Console", "monospace"], 14)
    hint_font  = load_font(["Courier New", "Lucida Console", "monospace"], 13)

    row_y    = _MENU_H // 2 - 55
    btn_h    = 110
    x_start  = 110

    p0_rect   = (x_start, row_y, 130, btn_h)
    p1_x      = x_start + 130 + 20
    p1_1_rect = (p1_x, row_y,                       210, btn_h // 2 - 4)
    p1_6_rect = (p1_x, row_y + btn_h // 2 + 4,      210, btn_h // 2 - 4)
    p2_x      = p1_x + 210 + 20
    p2_1_rect = (p2_x, row_y,                       210, btn_h // 2 - 4)
    p2_6_rect = (p2_x, row_y + btn_h // 2 + 4,      210, btn_h // 2 - 4)
    p3_x      = p2_x + 210 + 20
    p3_rect   = (p3_x, row_y, 130, btn_h)

    buttons = [
        _Button(p0_rect,   "PHASE 0", _ACCENT,  {"phase": 0, "n_agents": 6}),
        _Button(p1_1_rect, "PHASE 1", _ACCENT,  {"phase": 1, "n_agents": 1}, sub_label="1-agent"),
        _Button(p1_6_rect, "PHASE 1", _ACCENT,  {"phase": 1, "n_agents": 6}, sub_label="6-agent"),
        _Button(p2_1_rect, "PHASE 2", _ACCENT2, {"phase": 2, "n_agents": 1}, sub_label="1-agent"),
        _Button(p2_6_rect, "PHASE 2", _ACCENT2, {"phase": 2, "n_agents": 6}, sub_label="6-agent"),
        _Button(p3_rect,   "PHASE 3", _ACCENT,  {"phase": 3, "n_agents": 6}),
    ]

    group_labels = [
        (x_start + 65,  row_y - 32, "FREE NAV",    _MUTED),
        (p1_x + 105,    row_y - 32, "STANDOFF",    _MUTED),
        (p2_x + 105,    row_y - 32, "ARC SPACING", _MUTED),
        (p3_x + 65,     row_y - 32, "PUSH",        _MUTED),
    ]
    sep_xs = [p1_x - 10, p2_x - 10, p3_x - 10]
    legend_items = [
        (x_start + 65,  "n=6"),
        (p1_x + 55,     "n=1"),
        (p1_x + 160,    "n=6"),
        (p2_x + 55,     "n=1"),
        (p2_x + 160,    "n=6"),
        (p3_x + 65,     "n=6"),
    ]

    scan_y = 0.0
    t0 = time.time()

    while True:
        dt  = clock.tick(60) / 1000.0
        now = time.time() - t0
        pulse = 0.5 + 0.5 * math.sin(now * 2.5)
        chosen = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys; sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                import sys; sys.exit()
            for btn in buttons:
                result = btn.handle_event(event)
                if result is not None:
                    chosen = result

        for btn in buttons:
            btn.update()

        if chosen is not None:
            # Close just the display surface; pygame stays initialised so
            # the env can open its own window without re-calling pygame.init()
            pygame.display.quit()
            return chosen

        screen.fill(_BG_DARK)
        _draw_grid(screen)

        scan_y = (scan_y + 80.0 * dt) % _MENU_H
        scan_surf = pygame.Surface((_MENU_W, 2), pygame.SRCALPHA)
        scan_surf.fill((0, 210, 180, 18))
        screen.blit(scan_surf, (0, int(scan_y)))

        title_lbl = title_font.render("// SWARM CAGING — MISSION SELECT", True, _WHITE)
        screen.blit(title_lbl, title_lbl.get_rect(centerx=_MENU_W // 2, y=48))

        uw = title_lbl.get_width() + 40
        ux = _MENU_W // 2 - uw // 2
        uy = 48 + title_lbl.get_height() + 6
        line_surf = pygame.Surface((uw, 2), pygame.SRCALPHA)
        line_surf.fill((*_ACCENT, int(160 + 80 * pulse)))
        screen.blit(line_surf, (ux, uy))

        sub_lbl = label_font.render("Select a training phase to initialise the environment", True, _MUTED)
        screen.blit(sub_lbl, sub_lbl.get_rect(centerx=_MENU_W // 2, y=uy + 14))

        for sx in sep_xs:
            pygame.draw.line(screen, _SEP, (sx, row_y - 40), (sx, row_y + btn_h + 10), 1)

        for gx, gy, gtxt, gcol in group_labels:
            gl = label_font.render(gtxt, True, gcol)
            screen.blit(gl, gl.get_rect(centerx=gx, y=gy))

        for btn in buttons:
            btn.draw(screen, phase_font, sub_font)

        legend_y = row_y + btn_h + 22
        for lx, ltxt in legend_items:
            ll = hint_font.render(ltxt, True, _MUTED)
            screen.blit(ll, ll.get_rect(centerx=lx, y=legend_y))

        sel_rect = (x_start - 16, row_y - 46,
                    p3_x + 130 + 16 - x_start + 16, btn_h + 72)
        _draw_corners(screen, sel_rect,
                      (*_ACCENT, int(120 + 80 * pulse)), size=22, thick=2)

        esc_lbl = hint_font.render("ESC to quit", True, _MUTED)
        screen.blit(esc_lbl, (_MENU_W - esc_lbl.get_width() - 16, _MENU_H - 22))

        pygame.display.flip()


# ===========================================================================
# Helper: build a SwarmCagingEnv from a menu config dict
# ===========================================================================
def _build_swarm_env(menu_config: dict):
    phase    = menu_config["phase"]
    n_agents = menu_config["n_agents"]
    print(f"[INFO] Selected phase={phase}, n_agents={n_agents}")

    raw = SwarmCagingEnv(phase=phase, render_mode="human", max_steps=300)
    raw.n_agents                 = n_agents
    raw.possible_agents          = [f"agent_{i}" for i in range(n_agents)]
    raw.agent_name_mapping       = {f"agent_{i}": i for i in range(n_agents)}
    raw.last_obs                 = [np.zeros(8, np.float32) for _ in range(n_agents)]
    raw.last_actions             = [np.zeros(2, np.float32) for _ in range(n_agents)]
    raw.last_rewards             = [0.0 for _ in range(n_agents)]
    raw.agent_cumulative_rewards = [0.0 for _ in range(n_agents)]
    raw._prev_standoff_devs      = [0.0] * n_agents
    raw._prev_arc_imbalances     = [0.0] * n_agents
    raw.agent_hit_object_flags   = [False] * n_agents
    raw.spawn_slots              = list(range(n_agents))
    return raw


# ======================
# Actor (MUST match training EXACTLY)
# ======================
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim):
        super().__init__()

        mean_layers = [nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())]
        for _ in range(num_layer):
            mean_layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        mean_layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))

        self.mean_layers = nn.ModuleList(mean_layers)

        # Learnable log std (same as training)
        self.logstd_layer = nn.Parameter(torch.zeros(output_dim) - 2)

    def mean(self, x):
        for layer in self.mean_layers:
            x = layer(x)
        return x

    def act(self, x):
        mean = self.mean(x)
        std = self.logstd_layer.exp()
        dist = Normal(mean, std)

        u = dist.sample()
        action = torch.tanh(u)

        log_prob_u = dist.log_prob(u).sum(dim=-1)
        log_prob_tanh = torch.sum(torch.log(1.0 - action.pow(2) + 1e-6), dim=-1)
        log_prob = log_prob_u - log_prob_tanh

        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


# ======================
# Checkpoint watcher
# ======================
last_mtime = None
actor = None  # built after first env is created (need obs/act dims)


def load_checkpoint(path):
    global last_mtime
    if not os.path.exists(path):
        print(f"[WARN] Checkpoint not found: {path}")
        return
    checkpoint = torch.load(path, map_location=device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    last_mtime = os.path.getmtime(path)
    print(
        f"\n[CHECKPOINT LOADED] {path}\n"
        f"step={checkpoint.get('step', 'N/A')} "
        f"time={time.strftime('%H:%M:%S')}"
    )


def maybe_reload_checkpoint():
    global last_mtime, current_checkpoint_path
    if not os.path.exists(current_checkpoint_path):
        return
    mtime = os.path.getmtime(current_checkpoint_path)
    if last_mtime is None or mtime > last_mtime:
        load_checkpoint(current_checkpoint_path)


# ======================
# Keyboard action helper
# ======================
def get_keyboard_action() -> np.ndarray:
    """Read arrow keys and return a (2,) action array [forward, angular]."""
    keys = pygame.key.get_pressed()
    forward = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])
    angular = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
    return np.array([forward, angular], dtype=np.float32)


# ======================
# Main
# ======================
print("[INFO] Starting infinite evaluation (Ctrl+C to stop)")
print("[INFO] BACKSPACE → menu | 1/2/3 → switch model | P → manual | R → reset | Q → quit\n")

if args.env != "swarm":
    # ------------------------------------------------------------------
    # Non-swarm: original single-env behaviour, no menu
    # ------------------------------------------------------------------
    raw_env = DiffDrivePushEnv(render_mode="human")
    env = PettingZooWrapper(env=raw_env, agent_ids=False, vector_rewards=True)

    obs_dim = env.get_obs_size()
    act_dim = env.get_action_size()
    print(f"[INFO] obs_dim={obs_dim}, act_dim={act_dim}")

    actor = Actor(input_dim=obs_dim, hidden_dim=64, num_layer=1, output_dim=act_dim).to(device)
    actor.eval()
    load_checkpoint(current_checkpoint_path)

    manual_override    = False
    p_key_pressed_last = False
    enter_pressed      = False
    ep = 0
    last_check_time = 0.0

    try:
        while True:
            ep += 1
            now = time.time()
            if now - last_check_time > CHECK_INTERVAL_SEC:
                maybe_reload_checkpoint()
                last_check_time = now

            obs, _ = env.reset()
            done = truncated = False
            ep_reward = ep_length = 0

            while not (done or truncated):
                pygame.event.pump()
                keys = pygame.key.get_pressed()

                if keys[pygame.K_q]:
                    raise KeyboardInterrupt
                if keys[pygame.K_r]:
                    time.sleep(0.2)
                    done = True
                    break

                p_key_down_now = bool(keys[pygame.K_p])
                if p_key_down_now and not p_key_pressed_last:
                    manual_override = not manual_override
                    print(f"[INFO] Control: {'MANUAL' if manual_override else 'MODEL'}")
                p_key_pressed_last = p_key_down_now

                if keys[pygame.K_1] and not enter_pressed:
                    current_checkpoint_path = checkpoint_approach
                    print(">>> APPROACH MODEL (1) <<<")
                    load_checkpoint(current_checkpoint_path)
                    enter_pressed = True
                elif keys[pygame.K_2] and not enter_pressed:
                    current_checkpoint_path = checkpoint_cage
                    print(">>> CAGE MODEL (2) <<<")
                    load_checkpoint(current_checkpoint_path)
                    enter_pressed = True
                elif keys[pygame.K_3] and not enter_pressed:
                    current_checkpoint_path = checkpoint_transport
                    print(">>> TRANSPORT MODEL (3) <<<")
                    load_checkpoint(current_checkpoint_path)
                    enter_pressed = True
                elif not keys[pygame.K_1] and not keys[pygame.K_2] and not keys[pygame.K_3]:
                    enter_pressed = False

                obs_tensor = torch.from_numpy(obs).float().to(device)
                with torch.no_grad():
                    actions, _, _ = actor.act(obs_tensor)
                actions_np = actions.cpu().numpy()

                next_obs, reward, done, truncated, infos = env.step(actions_np)
                ep_reward += float(np.sum(reward))
                ep_length += 1
                obs = next_obs

            print(f"Episode {ep:04d} | Reward: {ep_reward:+.3f} | Steps: {ep_length}")

    except KeyboardInterrupt:
        print("\n[INFO] Evaluation stopped by user")
    finally:
        env.close()

else:
    # ------------------------------------------------------------------
    # SWARM: outer loop — menu → sim → Backspace → menu
    # ------------------------------------------------------------------
    actor_built = False

    try:
        while True:
            # Show menu, get config
            menu_config = run_swarm_menu()
            raw_env = _build_swarm_env(menu_config)
            env = PettingZooWrapper(env=raw_env, agent_ids=False, vector_rewards=True)

            obs_dim = env.get_obs_size()
            act_dim = env.get_action_size()
            print(f"[INFO] obs_dim={obs_dim}, act_dim={act_dim}")

            # Build actor only once (dims are fixed across phases)
            if not actor_built:
                actor = Actor(
                    input_dim=obs_dim, hidden_dim=64, num_layer=1, output_dim=act_dim
                ).to(device)
                actor.eval()
                load_checkpoint(current_checkpoint_path)
                actor_built = True

            manual_override    = False
            p_key_pressed_last = False
            enter_pressed      = False
            ep = 0
            last_check_time = 0.0
            go_to_menu = False

            # Episode loop
            while not go_to_menu:
                ep += 1
                now = time.time()
                if now - last_check_time > CHECK_INTERVAL_SEC:
                    maybe_reload_checkpoint()
                    last_check_time = now

                obs, _ = env.reset()
                done = truncated = False
                ep_reward = ep_length = 0

                while not (done or truncated):
                    pygame.event.pump()
                    keys = pygame.key.get_pressed()

                    # QUIT
                    if keys[pygame.K_q]:
                        raise KeyboardInterrupt

                    # BACK TO MENU
                    if keys[pygame.K_BACKSPACE]:
                        print("[INFO] Returning to menu...")
                        time.sleep(0.2)  # debounce
                        go_to_menu = True
                        done = True
                        break

                    # RESET
                    if keys[pygame.K_r]:
                        print("[INFO] Resetting environment...")
                        time.sleep(0.2)
                        done = True
                        break

                    # Manual override toggle (P key, edge-detected)
                    p_key_down_now = bool(keys[pygame.K_p])
                    if p_key_down_now and not p_key_pressed_last:
                        if raw_env.n_agents == 1 and raw_env.phase in (1, 2):
                            manual_override = not manual_override
                            mode_str = "MANUAL (arrow keys)" if manual_override else "MODEL"
                            print(f"[INFO] Control mode switched to: {mode_str}")
                        else:
                            print("[INFO] Manual override only available for n_agents=1, phase 1 or 2")
                    p_key_pressed_last = p_key_down_now

                    # 1 / 2 / 3 — switch between models (edge-detected)
                    if keys[pygame.K_1] and not enter_pressed:
                        current_checkpoint_path = checkpoint_approach
                        print(">>> SWITCHING TO APPROACH MODEL (1) <<<")
                        load_checkpoint(current_checkpoint_path)
                        enter_pressed = True
                    elif keys[pygame.K_2] and not enter_pressed:
                        current_checkpoint_path = checkpoint_cage
                        print(">>> SWITCHING TO CAGE MODEL (2) <<<")
                        load_checkpoint(current_checkpoint_path)
                        enter_pressed = True
                    elif keys[pygame.K_3] and not enter_pressed:
                        current_checkpoint_path = checkpoint_transport
                        print(">>> SWITCHING TO TRANSPORT MODEL (3) <<<")
                        load_checkpoint(current_checkpoint_path)
                        enter_pressed = True
                    elif not keys[pygame.K_1] and not keys[pygame.K_2] and not keys[pygame.K_3]:
                        enter_pressed = False

                    # Determine actions
                    if manual_override and raw_env.n_agents == 1 and raw_env.phase in (1, 2):
                        obs_tensor = torch.from_numpy(obs).float().to(device)
                        with torch.no_grad():
                            dummy_actions, _, _ = actor.act(obs_tensor)
                        dummy_np   = dummy_actions.cpu().numpy()
                        kb_action  = get_keyboard_action()
                        actions_np = np.broadcast_to(kb_action, dummy_np.shape).copy()
                    else:
                        obs_tensor = torch.from_numpy(obs).float().to(device)
                        with torch.no_grad():
                            actions, _, _ = actor.act(obs_tensor)
                        actions_np = actions.cpu().numpy()

                    next_obs, reward, done, truncated, infos = env.step(actions_np)
                    ep_reward += float(np.sum(reward))
                    ep_length += 1
                    obs = next_obs

                if not go_to_menu:
                    mode_str = "MANUAL" if manual_override else "MODEL"
                    print(
                        f"Episode {ep:04d} | "
                        f"Reward: {ep_reward:+.3f} | "
                        f"Steps: {ep_length} | "
                        f"Control: {mode_str}"
                    )

            # Close env window before re-showing menu
            env.close()

    except KeyboardInterrupt:
        print("\n[INFO] Evaluation stopped by user")
        try:
            env.close()
        except Exception:
            pass
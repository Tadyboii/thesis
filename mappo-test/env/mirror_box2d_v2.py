import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
import pygame
import numpy as np
import threading
import time
import math

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    """
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class GazeboMirror(Node):
    def __init__(self, n_agents=6):
        super().__init__('gazebo_mirror')
        self.n_agents = n_agents

        # --- Visualization Settings ---
        self.width_m = 20.0 # Make sure world is big enough
        self.height_m = 20.0
        self.ppm = 30 # pixels per meter
        self.screen_width = int(self.width_m * self.ppm)
        self.screen_height = int(self.height_m * self.ppm)
        self.offset_x = self.screen_width // 2
        self.offset_y = self.screen_height // 2

        self.agent_colors = [
            (255,0,0), (0,255,0), (0,0,255), 
            (255,255,0), (255,0,255), (0,255,255)
        ]

        # --- State Storage ---
        self.robot_states = {}  # {agent_id: (x, y, theta)}
        self.object_state = None # (x, y, theta)
        self.target_state = None # (x, y)

        # --- Subscribers ---
        for i in range(self.n_agents):
            agent_id = f"diff_drive_robot{i+1}"
            topic = f"/{agent_id}/odom" 
            self.create_subscription(
                Odometry,
                topic,
                lambda msg, aid=i: self.robot_odom_callback(msg, aid),
                10
            )

        # Object (Assuming bridged topic /model/object/pose)
        self.create_subscription(
            Pose,
            '/model/object/pose', 
            self.object_pose_callback,
            10
        )

        # Target (Assuming bridged topic /model/target/pose)
        self.create_subscription(
            Pose,
            '/model/target/pose',
            self.target_pose_callback,
            10
        )
        
        self.get_logger().info("Mirror Node Started. Waiting for ROS messages...")

    def robot_odom_callback(self, msg: Odometry, agent_idx):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        
        if np.isnan([pos.x, pos.y, ori.x, ori.y, ori.z, ori.w]).any():
            return

        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.robot_states[agent_idx] = (pos.x, pos.y, yaw)

    def object_pose_callback(self, msg: Pose):
        pos = msg.position
        ori = msg.orientation

        if np.isnan([pos.x, pos.y, ori.x, ori.y, ori.z, ori.w]).any():
            return
            
        _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.object_state = (pos.x, pos.y, yaw)

    def target_pose_callback(self, msg: Pose):
        pos = msg.position
        if np.isnan([pos.x, pos.y]).any():
            return
        self.target_state = (pos.x, pos.y)

    def to_screen(self, x, y):
        # Center is (0,0) in Sim -> Offset in Screen
        # Y is inverted (Screen Y down)
        sx = int(x * self.ppm) + self.offset_x
        sy = self.offset_y - int(y * self.ppm)
        return sx, sy

    def render_loop(self):
        pygame.init()
        pygame.font.init() # Ensure font module is initialized
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Gazebo Mirror")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 24)
        
        running = True
        while running and rclpy.ok():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill((30, 30, 30))
            
            # --- Draw Grid/Axes ---
            cx, cy = self.to_screen(0, 0)
            pygame.draw.line(self.screen, (100, 100, 100), (cx, 0), (cx, self.screen_height), 1)
            pygame.draw.line(self.screen, (100, 100, 100), (0, cy), (self.screen_width, cy), 1)

            # --- Draw Target ---
            if self.target_state:
                tx, ty = self.target_state
                sx, sy = self.to_screen(tx, ty)
                radius = 5
                # Draw X
                pygame.draw.line(self.screen, (255, 255, 255), (sx-radius, sy-radius), (sx+radius, sy+radius), 2)
                pygame.draw.line(self.screen, (255, 255, 255), (sx-radius, sy+radius), (sx+radius, sy-radius), 2)

            # --- Draw Object ---
            if self.object_state:
                ox, oy, o_yaw = self.object_state
                sx, sy = self.to_screen(ox, oy)
                # Radius 1.5m from particle_box2d
                obj_radius_px = int(1.5 * self.ppm)
                pygame.draw.circle(self.screen, (200, 200, 200), (sx, sy), obj_radius_px)
                # Orientation line
                end_x = ox + math.cos(o_yaw) * 1.5
                end_y = oy + math.sin(o_yaw) * 1.5
                ex, ey = self.to_screen(end_x, end_y)
                pygame.draw.line(self.screen, (0,0,0), (sx, sy), (ex, ey), 2)

            # --- Draw Agents ---
            for i in range(self.n_agents):
                if i in self.robot_states:
                    rx, ry, rtheta = self.robot_states[i]
                    sx, sy = self.to_screen(rx, ry)
                    
                    # Radius 0.4m
                    agent_radius_px = int(0.4 * self.ppm)
                    color = self.agent_colors[i % len(self.agent_colors)]
                    
                    pygame.draw.circle(self.screen, color, (sx, sy), agent_radius_px)
                    
                    # Heading
                    end_x = rx + math.cos(rtheta) * 0.5
                    end_y = ry + math.sin(rtheta) * 0.5
                    ex, ey = self.to_screen(end_x, end_y)
                    pygame.draw.line(self.screen, (255, 255, 255), (sx, sy), (ex, ey), 2)
                    
                    # ID
                    label = font.render(str(i+1), True, (255, 255, 255))
                    self.screen.blit(label, (sx, sy))

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        # Clean shutdown
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    # Create valid node first
    mirror_node = GazeboMirror(n_agents=6)

    # Run ROS spin in a separate thread so render loop isn't blocked
    spin_thread = threading.Thread(target=rclpy.spin, args=(mirror_node,), daemon=True)
    spin_thread.start()

    try:
        mirror_node.render_loop()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()
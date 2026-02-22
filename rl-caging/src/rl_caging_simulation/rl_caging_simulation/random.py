#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import random

class RandomActionNode(Node):
    def __init__(self):
        super().__init__('random_action_node')

        # ======= Robot namespaces =======
        self.robot_names = [f'diff_drive_robot{i}' for i in range(1, 7)]

        # ======= Publishers for all robots =======
        self.robot_publishers = {
            name: self.create_publisher(Twist, f'/{name}/cmd_vel', 10)
            for name in self.robot_names
        }

        # ======= Action limits =======
        self.v_min, self.v_max = 0.0, 0.5      # linear velocity range (m/s)
        self.w_min, self.w_max = -2.0, 2.0     # angular velocity range (rad/s)

        # ======= Smoothing settings =======
        self.alpha = 0.90        # smoothing factor (0 → no change, 1 → instant)

        # Each robot has its own current [v, w]
        self.velocities = {name: {'v': 0.0, 'w': 0.0} for name in self.robot_names}

        # ======= Timer =======
        self.timer_period = 1   # 10 Hz
        self.timer = self.create_timer(self.timer_period, self.publish_random_actions)

        self.get_logger().info(f'Independent random [v, ω] publisher started for {len(self.robot_names)} robots at 10 Hz.')
        self.get_logger().info(f'Linear range: {self.v_min}–{self.v_max} m/s, Angular range: {self.w_min}–{self.w_max} rad/s')

    def publish_random_actions(self):
        for name, pub in self.robot_publishers.items():
            # Generate random target velocities for each robot
            v_target = random.uniform(self.v_min, self.v_max)
            w_target = random.uniform(self.w_min, self.w_max)

            # Retrieve current smoothed values
            v_current = self.velocities[name]['v']
            w_current = self.velocities[name]['w']

            # Apply exponential smoothing
            v_new = self.alpha * v_target + (1 - self.alpha) * v_current
            w_new = self.alpha * w_target + (1 - self.alpha) * w_current

            # Update stored values
            self.velocities[name]['v'] = v_new
            self.velocities[name]['w'] = w_new

            # Build TwistStamped message
            msg = Twist()

            msg.linear.x = v_new
            msg.angular.z = w_new

            # Publish
            pub.publish(msg)

        # Log one line summarizing all
        summary = ', '.join([f'{n}: v={self.velocities[n]["v"]:.2f}, ω={self.velocities[n]["w"]:.2f}' for n in self.robot_names])
        self.get_logger().info(f'{summary}', throttle_duration_sec=1.0)

    def stop_all_robots(self):
        """Send zero velocity to stop all robots."""
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0

        for pub in self.robot_publishers.values():
            pub.publish(stop_msg)

        self.get_logger().info('All robots stopped.')

def main():
    rclpy.init()
    node = RandomActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_all_robots()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

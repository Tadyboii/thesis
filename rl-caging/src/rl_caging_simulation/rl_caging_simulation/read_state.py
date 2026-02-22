from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
import math


class ReadState(Node):

    def __init__(self):
        super().__init__('read_state')
        self.get_logger().info("Read State Node has been started.")
        self.scan_ranges = []
        self.scan_intensities = []
        self.has_scan_received = False

        self.stop_distance = 1.5
        self.tele_twist = TwistStamped()
        self.tele_twist.twist.linear.x = 2.0
        self.tele_twist.twist.angular.z = 0.0

        self.angle_min = 0.00  # in radians
        self.angle_increment = 0.0174533  # 1 degree in radians

        # --- Object intensity range ---
        self.object_intensity_threshold_max = 500.0
        self.object_intensity_threshold_min = 100.0

        # --- Target intensity parameters ---
        self.target_intensity_value = 500.0
        self.target_intensity_tolerance = 10.0  # ±10 around 500

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/diff_drive_robot1/scan',
            self.scan_callback,
            qos_profile=qos_profile_sensor_data
        )

    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges
        self.scan_intensities = msg.intensities
        self.scan_obstacles()

    def scan_obstacles(self):
        robot_obstacles = []
        current_robot_obstacle_distance = []
        current_robot_obstacle_angle = []

        first_robot_obstacle_distance = []
        first_robot_obstacle_angle = []

        current_object_obstacle_distance = []
        current_object_obstacle_angle = []

        current_target_obstacle_distance = []
        current_target_obstacle_angle = []

        # --- Scan through each LiDAR ray ---
        for i, distance in enumerate(self.scan_ranges):
            intensity = self.scan_intensities[i]
            angle = math.degrees(self.angle_min + i * self.angle_increment)

            # --- Detect cylindrical object (medium intensity range) ---
            if self.object_intensity_threshold_min < intensity < self.object_intensity_threshold_max:
                current_object_obstacle_distance.append(distance)
                current_object_obstacle_angle.append(angle)

            # --- Detect target (very high intensity ≈ 500 ± tolerance) ---
            elif abs(intensity - self.target_intensity_value) <= self.target_intensity_tolerance:
                current_target_obstacle_distance.append(distance)
                current_target_obstacle_angle.append(angle)

            # --- Detect robot clusters (low-intensity and valid distances) ---
            elif not math.isinf(distance) and not math.isnan(distance) and intensity <= self.object_intensity_threshold_min:
                current_robot_obstacle_distance.append(distance)
                current_robot_obstacle_angle.append(angle)
            else:
                # End of cluster: save it if exists
                if current_robot_obstacle_distance:
                    avg_distance = sum(current_robot_obstacle_distance) / len(current_robot_obstacle_distance)
                    avg_angle = sum(current_robot_obstacle_angle) / len(current_robot_obstacle_angle)
                    robot_obstacles.append({
                        'angle': round(avg_angle, 2),
                        'distance': round(avg_distance, 3),
                        'num_points': len(current_robot_obstacle_distance),
                    })

                    if not first_robot_obstacle_distance:
                        first_robot_obstacle_distance = current_robot_obstacle_distance
                        first_robot_obstacle_angle = current_robot_obstacle_angle

                    current_robot_obstacle_distance = []
                    current_robot_obstacle_angle = []

        # --- Handle last robot cluster if open ---
        if current_robot_obstacle_distance:
            if first_robot_obstacle_distance:
                current_robot_obstacle_distance = first_robot_obstacle_distance + current_robot_obstacle_distance
                current_robot_obstacle_angle = first_robot_obstacle_angle + current_robot_obstacle_angle
                avg_distance = sum(current_robot_obstacle_distance) / len(current_robot_obstacle_distance)
                avg_angle = sum(current_robot_obstacle_angle) / len(current_robot_obstacle_angle)
                robot_obstacles[0] = {
                    'angle': round(avg_angle, 2),
                    'distance': round(avg_distance, 3),
                    'num_points': len(current_robot_obstacle_distance),
                }
            else:
                avg_distance = sum(current_robot_obstacle_distance) / len(current_robot_obstacle_distance)
                avg_angle = sum(current_robot_obstacle_angle) / len(current_robot_obstacle_angle)
                robot_obstacles.append({
                    'angle': round(avg_angle, 2),
                    'distance': round(avg_distance, 3),
                    'num_points': len(current_robot_obstacle_distance),
                })

        # --- Compute averages for object and target ---
        def avg_cluster(distances, angles):
            if not distances:
                return 0.0, 0.0

            avg_distance = sum(distances) / len(distances)

            # Convert to radians for trigonometric mean
            angles_rad = [math.radians(a) for a in angles]
            sin_sum = sum(math.sin(a) for a in angles_rad)
            cos_sum = sum(math.cos(a) for a in angles_rad)
            mean_angle_rad = math.atan2(sin_sum / len(angles_rad), cos_sum / len(angles_rad))

            # Convert back to degrees in [0, 360)
            mean_angle_deg = math.degrees(mean_angle_rad) % 360

            return round(avg_distance, 3), round(mean_angle_deg, 2)


        object_distance, object_angle = avg_cluster(current_object_obstacle_distance, current_object_obstacle_angle)
        target_distance, target_angle = avg_cluster(current_target_obstacle_distance, current_target_obstacle_angle)

        # --- Sort and pick 2 nearest robots ---
        robot_obstacles.sort(key=lambda obs: obs['distance'])
        nearest_two = robot_obstacles[:2]

        robot1_distance = nearest_two[0]['distance'] if len(nearest_two) > 0 else 0.0
        robot1_angle = nearest_two[0]['angle'] if len(nearest_two) > 0 else 0.0
        robot2_distance = nearest_two[1]['distance'] if len(nearest_two) > 1 else 0.0
        robot2_angle = nearest_two[1]['angle'] if len(nearest_two) > 1 else 0.0

        # --- Log detected data ---
        self.get_logger().info(f"""
        OBJECT -> dist: {object_distance:.3f}, angle: {object_angle:.2f}
        TARGET -> dist: {target_distance:.3f}, angle: {target_angle:.2f}
        AGENT1 -> dist: {robot1_distance:.3f}, angle: {robot1_angle:.2f}
        AGENT2 -> dist: {robot2_distance:.3f}, angle: {robot2_angle:.2f}
        """)


def main(args=None):
    rclpy.init(args=args)
    read_state_node = ReadState()
    rclpy.spin(read_state_node)
    read_state_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

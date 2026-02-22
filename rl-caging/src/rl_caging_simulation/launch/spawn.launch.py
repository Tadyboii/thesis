from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import GroupAction
from ament_index_python.packages import get_package_share_directory
import os
import tempfile
import yaml
import random
import xml.etree.ElementTree as ET
import math
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnShutdown

def generate_random_position(existing_positions, min_distance, arena_bounds):
    """Generate a random (x, y) position within arena_bounds 
    that is at least min_distance away from all existing positions."""
    while True:
        x = random.uniform(-arena_bounds, arena_bounds)
        y = random.uniform(-arena_bounds, arena_bounds)
        if all(math.dist((x, y), pos) >= min_distance for pos in existing_positions):
            return x, y

def generate_bridge_yaml(robot_namespace):
    """Generates a temporary YAML file for a robot with all topics namespaced."""
    bridge_config = [
        {"ros_topic_name": "cmd_vel",
         "gz_topic_name": f"{robot_namespace}/cmd_vel",
         "ros_type_name": "geometry_msgs/msg/Twist",
         "gz_type_name": "gz.msgs.Twist",
         "direction": "ROS_TO_GZ"},

        {"ros_topic_name": "scan",
         "gz_topic_name": f"{robot_namespace}/scan",
         "ros_type_name": "sensor_msgs/msg/LaserScan",
         "gz_type_name": "gz.msgs.LaserScan",
         "direction": "GZ_TO_ROS"},
    ]

    tmp_file = os.path.join(tempfile.gettempdir(), f"{robot_namespace}_bridge.yaml")
    with open(tmp_file, 'w') as f:
        yaml.dump(bridge_config, f)
    return tmp_file

def generate_launch_description():
    diff_drive_bot_sdf_template = os.path.join(
        get_package_share_directory('rl_caging_simulation'),
        'models',
        'diff_drive_bot',
        'diff_drive_robot.sdf'
    )

    cylindrical_object_sdf = os.path.join(
        get_package_share_directory('rl_caging_simulation'),
        'models',
        'cylindrical_object',
        'model.sdf'
    )

    target_flag_sdf = os.path.join(
        get_package_share_directory('rl_caging_simulation'),
        'models',
        'target_flag',
        'model.sdf'
    )

    save_path = os.path.join(
        get_package_share_directory('rl_caging_simulation'),
        'models',
        'diff_drive_bot',
        'tmp'
    )
    os.makedirs(save_path, exist_ok=True)

    number_of_robots = 6
    arena_bounds = 2
    min_distance = 0.6
    positions = []

    # Unique front_marker colors per robot (R G B A)
    robot_colors = [
        "1 0 0 1",       # robot1 – red
        "0 1 0 1",       # robot2 – green
        "0 0 1 1",       # robot3 – blue
        "1 1 0 1",       # robot4 – yellow
        "1 0 1 1",       # robot5 – magenta
        "0 1 1 1",       # robot6 – cyan
    ]

    ld_actions = []

    # Spawn Cylindrical Object
    cyl_x, cyl_y = generate_random_position(positions, min_distance, arena_bounds)
    positions.append((cyl_x, cyl_y))  # prevent collisions with robots

    cylinder_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'object',
            '-file', cylindrical_object_sdf,
            '-x', str(cyl_x),
            '-y', str(cyl_y),
            '-z', '0.1',
        ],
        output='screen',
    )
    ld_actions.append(cylinder_node)

    # Spawn Target Flag
    flag_x, flag_y = generate_random_position(positions, min_distance, arena_bounds)
    positions.append((flag_x, flag_y))  # prevent collisions with robots
    flag_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'target',
            '-file', target_flag_sdf,
            '-x', str(flag_x),
            '-y', str(flag_y),
            '-z', '0.0',
        ],
        output='screen',
    )
    ld_actions.append(flag_node)

    # Rviz
    # rviz_config_file = os.path.join(
    #     get_package_share_directory('rl_caging_simulation'),
    #     'rviz',
    #     'caging_config.rviz'
    # )
    
    # # Check if rviz dir exists (it might not be installed yet)
    # # If not found, just launch default rviz
    # if not os.path.exists(rviz_config_file):
    #     # Allow fallback to source dir for development convenience
    #     src_rviz = os.path.join(os.getcwd(), 'src', 'rl_caging_simulation', 'rviz', 'caging_config.rviz')
    #     if os.path.exists(src_rviz):
    #         rviz_config_file = src_rviz

    # rviz_node = Node(
    #    package='rviz2',
    #    executable='rviz2',
    #    name='rviz2',
    #    arguments=['-d', rviz_config_file] if os.path.exists(rviz_config_file) else [],
    #    output='screen'
    # )
    # ld_actions.append(rviz_node)

    # Spawn robots
    for i in range(number_of_robots):
        # Random position
        x, y = generate_random_position(positions, min_distance, arena_bounds)
        positions.append((x, y))

        namespace = f'diff_drive_robot{i+1}'
        robot_name = f'diff_drive_robot{i+1}'

        # Load SDF template and modify topics
        tree = ET.parse(diff_drive_bot_sdf_template)
        root = tree.getroot()

        for plugin_tag in root.iter('plugin'):
            for topic_tag in plugin_tag.iter('topic'):
                if topic_tag.text:
                    topic_tag.text = f'{namespace}/{topic_tag.text}'

        for sensor_tag in root.iter('sensor'):
            topic_tag = sensor_tag.find('topic')
            if topic_tag is not None and topic_tag.text:
                topic_tag.text = f'{namespace}/{topic_tag.text}'

        # Set unique front_marker color for this robot
        color = robot_colors[i % len(robot_colors)]
        for visual_tag in root.iter('visual'):
            if visual_tag.get('name') == 'front_marker':
                material_tag = visual_tag.find('material')
                if material_tag is not None:
                    for channel in ('ambient', 'diffuse'):
                        ch_tag = material_tag.find(channel)
                        if ch_tag is not None:
                            ch_tag.text = color

        urdf_modified = ET.tostring(root, encoding='unicode')
        urdf_modified = '<?xml version="1.0" ?>\n'+urdf_modified
        sdf_file = os.path.join(save_path, f'{i+1}.sdf')
        with open(sdf_file, 'w') as file:
            file.write(urdf_modified)

        # Bridge YAML
        bridge_yaml_file = generate_bridge_yaml(namespace)

        spawn_node = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', robot_name,
                '-file', sdf_file,
                '-x', str(x),
                '-y', str(y),
                '-z', '0.5',
            ],
            output='screen',
        )

        bridge_node = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '--ros-args',
                '-p',
                f'config_file:={bridge_yaml_file}',
            ],
            output='screen',
        )

        ld_actions.append(
            GroupAction([
                PushRosNamespace(namespace),
                spawn_node,
                bridge_node
            ])
        )

        ld_actions.append(
            RegisterEventHandler(
                OnShutdown(
                    on_shutdown=lambda event, context: [
                        os.remove(os.path.join(save_path, f'{count+1}.sdf')) 
                        for count in range(number_of_robots)
                        if os.path.exists(os.path.join(save_path, f'{count+1}.sdf'))
                    ]
                )
            )
        )

    return LaunchDescription(ld_actions)

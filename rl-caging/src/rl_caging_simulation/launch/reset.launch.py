#!/usr/bin/env python3
import random
import math
from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_random_position(existing_positions, min_distance, arena_bounds):
    while True:
        x = random.uniform(-arena_bounds, arena_bounds)
        y = random.uniform(-arena_bounds, arena_bounds)
        if all(math.dist((x, y), pos) >= min_distance for pos in existing_positions):
            return x, y

def generate_launch_description():
    number_of_robots = 5
    robot_names = [f"diff_drive_robot{i+1}" for i in range(number_of_robots)]
    object_name = "object"

    arena_bounds = 4.5
    min_distance = 0.6
    positions = []
    processes = []

    # Object (cylinder)
    cyl_x, cyl_y = generate_random_position(positions, min_distance, arena_bounds)
    positions.append((cyl_x, cyl_y))
    cyl_pose_req = (
        f'name: "{object_name}" '
        f'position: {{x: {cyl_x}, y: {cyl_y}, z: 0.1}} '
        f'orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}'
    )

    processes.append(ExecuteProcess(
        cmd=[
            'gz', 'service',
            '-s', '/world/car_world/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--req', cyl_pose_req
        ],
        output='screen'
    ))

    # Robots
    for name in robot_names:
        x, y = generate_random_position(positions, min_distance, arena_bounds)
        positions.append((x, y))
        pose_req = (
            f'name: "{name}" '
            f'position: {{x: {x}, y: {y}, z: 0.5}} '
            f'orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}'
        )

        processes.append(ExecuteProcess(
            cmd=[
                'gz', 'service',
                '-s', '/world/car_world/set_pose',
                '--reqtype', 'gz.msgs.Pose',
                '--reptype', 'gz.msgs.Boolean',
                '--req', pose_req
            ],
            output='screen'
        ))

    return LaunchDescription(processes)

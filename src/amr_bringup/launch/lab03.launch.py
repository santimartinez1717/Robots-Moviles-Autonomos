from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node

import math


def generate_launch_description():
    world = "lab03"
    start = (0.0, -0.8, math.radians(90))
    particles = 2000
    sigma_v = 0.05
    sigma_w = 0.1
    sigma_z = 0.2

    particle_filter_node = LifecycleNode(
        package="amr_localization",
        executable="particle_filter",
        name="particle_filter",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "INFO"],
        parameters=[
            {
                "enable_plot": True,
                "global_localization": True,
                "particles": particles,
                "sigma_v": sigma_v,
                "sigma_w": sigma_w,
                "sigma_z": sigma_z,
                "world": world,
            }
        ],
    )

    wall_follower_node = LifecycleNode(
        package="amr_control",
        executable="wall_follower",
        name="wall_follower",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
    )

    coppeliasim_node = LifecycleNode(
        package="amr_simulation",
        executable="coppeliasim",
        name="coppeliasim",
        namespace="",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[{"enable_localization": True, "start": start}],
    )

    lifecycle_manager_node = Node(
        package="amr_bringup",
        executable="lifecycle_manager",
        output="screen",
        arguments=["--ros-args", "--log-level", "WARN"],
        parameters=[
            {
                "node_startup_order": (
                    "particle_filter",
                    "wall_follower",
                    "coppeliasim",  # Must be started last
                )
            }
        ],
    )

    return LaunchDescription(
        [
            particle_filter_node,
            wall_follower_node,
            coppeliasim_node,
            lifecycle_manager_node,  # Must be launched last
        ]
    )

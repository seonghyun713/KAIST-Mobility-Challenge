from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    problem = LaunchConfiguration("problem")

    return LaunchDescription([
        DeclareLaunchArgument(
            "problem",
            default_value="2",
            description="Problem ID (Task 1-2)"
        ),

        Node(
            package="pkg_task_1_2",
            executable="task1_2.py",
            name="task1_2_node",
            output="screen",
            parameters=[
                {"problem": problem}
            ],
        ),
    ])


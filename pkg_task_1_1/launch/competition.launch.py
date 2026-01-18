from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    problem = LaunchConfiguration("problem")

    return LaunchDescription([
        DeclareLaunchArgument(
            "problem",
            default_value="1",
            description="Problem ID (Task 1-1)"
        ),

        Node(
            package="pkg_task_1_1",
            executable="task1_1.py",
            name="task1_1_node",
            output="screen",
            parameters=[
                {"problem": problem}
            ],
        ),
    ])


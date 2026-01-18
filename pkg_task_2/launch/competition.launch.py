from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    problem = LaunchConfiguration("problem")

    return LaunchDescription([
        DeclareLaunchArgument(
            "problem",
            default_value="3",
            description="Problem ID (Task 2)"
        ),

        Node(
            package="pkg_task_2",
            executable="task2.py",
            name="task2_node",
            output="screen",
            parameters=[
                {"problem": problem}
            ],
        ),
    ])


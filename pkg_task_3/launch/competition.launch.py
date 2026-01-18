from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    problem = LaunchConfiguration("problem")

    return LaunchDescription([
        DeclareLaunchArgument(
            "problem",
            default_value="4",
            description="Problem ID (Task 3)"
        ),

        Node(
            package="pkg_task_3",
            executable="task3.py",
            name="task3_node",
            output="screen",
            parameters=[
                {"problem": problem}
            ],
        ),
    ])


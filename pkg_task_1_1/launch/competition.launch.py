from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=["python3", "/ws/src/pkg_task_1_1/src/task1_1.py"],
            output="screen"
        )
    ])
cmake_minimum_required(VERSION 3.8)
project(pkg_task_2)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

install(
  PROGRAMS
    src/task2.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY src/path/
  DESTINATION lib/${PROJECT_NAME}/path
)


ament_package()


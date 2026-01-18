cmake_minimum_required(VERSION 3.8)
project(pkg_task_1_2)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

# Python 노드
install(
  PROGRAMS
    src/task1_2.py
  DESTINATION lib/${PROJECT_NAME}
)

# path 폴더 전체 install
install(
  DIRECTORY
    src/path
  DESTINATION lib/${PROJECT_NAME}
)

# launch 파일
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()


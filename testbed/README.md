**아키텍처:**

```
[노트북]
  (Task3 python)  →  publish /cmd_vel  (Wi-Fi, ROS2 DDS)
                         ↓
[Jetson Orin (차량 위)]
  subscribe /cmd_vel
  SDK/ROS2 C++ node: drv.setCommand(linear, omega)
  → UART (0xA5) 송신 (/dev/ttyKMC)
                         ↓
[Raspberry Pi (차량 내부)]
  UART 수신 → 모터/조향 구동
```

 `/cmd_vel`

- CAV1 Jetson: `ROS_DOMAIN_ID=0`
- CAV2 Jetson: `ROS_DOMAIN_ID=1`
- CAV3 Jetson: `ROS_DOMAIN_ID=2`
- CAV4 Jetson: `ROS_DOMAIN_ID=3`

```bash
ROS_DOMAIN_ID=0 python3 task3.py# CAV1 제어
ROS_DOMAIN_ID=1 python3 task3.py# CAV2 제어
ROS_DOMAIN_ID=2 python3 task3.py# CAV3 제어
ROS_DOMAIN_ID=3 python3 task3.py# CAV4 제어
```

근데 너 코드 그대로는 한 번에 4대를 다루는 구조라서,

실차 도메인 방식이면 보통 **“1대용으로 쪼개서 실행”**하는 게 깔끔해.

### 1. SSH 연결

**초기 세팅:**

```bash
cd KAIST_Mobility_Challenge_SDK/
ls -l /dev/ttyUSB*
sudo picocom -b 115200 /dev/ttyACM0 
```

**WIFI 세팅:**

```bash
#jetson terminal 접속
ssh jetson
```

**Dashboard 실행:**

```bash
cd ~/Desktop/Dashboard
./KMC_Dashboard-x86_64.AppImage
```

—-

### 2. 차량 초기 세팅

```bash
ls /opt/ros

sudo apt update
sudo apt install -y curl gnupg lsb-release

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu$(lsb_release -cs) main" | \
sudotee /etc/apt/sources.list.d/ros2.list

sudo apt update
sudo apt install -y ros-foxy-ros-base

#필수 패키지 설치
sudo apt install -y \
  ros-foxy-rclcpp \
  ros-foxy-geometry-msgs \
  ros-foxy-std-msgs

source /opt/ros/foxy/setup.bash

```

---

### 3. 실행

**정지코드(로컬 혹은 ssh):**

```
#CAV 01 정지
source /opt/ros/foxy/setup.bash
ros2 topic pub --once /CAV_01/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0}, angular: {z: 0.0}}"
```

**Domain ID 확인:**

```
env | grep ROS_DOMAIN_ID
```

**Domain 변경: Domain ID 4대 동일하게 해야됨 무조건**

```
export ROS_DOMAIN_ID=0
```

**ssh 실행시켜야할 코드**: driver_read_allstate_node.cpp

- **차량별 초기 pose를 쏴줘야 출발함**
- **차량별 코드 CAV07:**

```
export ROS_DOMAIN_ID=0
source /opt/ros/foxy/setup.bash
source ~/KAIST_Mobility_Challenge_SDK/examples/Driver_ROS2/install/setup.bash

ros2 run kmc_hardware_driver_node kmc_hardware_driver_read_allstate_node \
--ros-args -p port:=/dev/ttyKMC -r cmd_vel:=/CAV_07/cmd_vel
```

- **차량별 코드 CAV09:**

```
export ROS_DOMAIN_ID=0
source /opt/ros/foxy/setup.bash
source ~/KAIST_Mobility_Challenge_SDK/examples/Driver_ROS2/install/setup.bash

ros2 run kmc_hardware_driver_node kmc_hardware_driver_read_allstate_node \
--ros-args -p port:=/dev/ttyKMC -r cmd_vel:=/CAV_09/cmd_vel
```

- **차량별 코드 CAV10:**

```
export ROS_DOMAIN_ID=0
source /opt/ros/foxy/setup.bash
source ~/KAIST_Mobility_Challenge_SDK/examples/Driver_ROS2/install/setup.bash

ros2 run kmc_hardware_driver_node kmc_hardware_driver_read_allstate_node \
--ros-args -p port:=/dev/ttyKMC -r cmd_vel:=/CAV_10/cmd_vel
```

- **차량별 코드 CAV28:**

```
export ROS_DOMAIN_ID=0
source /opt/ros/foxy/setup.bash
source ~/KAIST_Mobility_Challenge_SDK/examples/Driver_ROS2/install/setup.bash

ros2 run kmc_hardware_driver_node kmc_hardware_driver_read_allstate_node \
--ros-args -p port:=/dev/ttyKMC -r cmd_vel:=/CAV_28/cmd_vel
```

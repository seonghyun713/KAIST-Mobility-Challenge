## 1. Architecture

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

- CAV01 Jetson: `ROS_DOMAIN_ID=100`
- CAV02 Jetson: `ROS_DOMAIN_ID=100`
- CAV03 Jetson: `ROS_DOMAIN_ID=100`
- CAV04 Jetson: `ROS_DOMAIN_ID=100`


---

## 1. 차량 초기 세팅

- password: 1234

Domain ID

```
# Domain_ID 확인
env | grep ROS_DOMAIN_ID

# Domain_ID 변경
export ROS_DOMAIN_ID=100
```

Git clone

```
git clone https://github.com/Seo12044/KAIST_Mobility_Challenge_SDK.git
```

ROS2 Foxy:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository universe

sudo apt update
sudo apt install -y curl ca-certificates gnupg lsb-release

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
| sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-foxy-ros-base python3-colcon-common-extensions python3-rosdep python3-argcomplete

sudo rosdep init || true
rosdep update

echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
source /opt/ros/foxy/setup.bash

sudo apt update
sudo apt install -y ros-foxy-ros-base

```

SDK 연결:

```bash
cd ~/KAIST_Mobility_Challenge_SDK
mkdir -p src
```

Driver 패키지 링크: 

```bash
rm -rf src/kmc_hardware_driver_node
ln -s ../examples/Driver_ROS2 src/kmc_hardware_driver_node
```

Build 및 확인

```bash
source /opt/ros/foxy/setup.bash
cd ~/KAIST_Mobility_Challenge_SDK
rm -rf build installlog

colcon list --base-paths src
colcon build --symlink-install --base-paths src --packages-select kmc_hardware_driver_node

source install/setup.bash
ros2 pkg executables kmc_hardware_driver_node
```

도메인 고정

```bash
export ROS_DOMAIN_ID=100
source /opt/ros/foxy/setup.bash
source ~/KAIST_Mobility_Challenge_SDK/install/setup.bash
```

포트 확인

```bash
ls -l /dev/ttyKMC /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```
- ttySUB0 또는 ttyUSB1

**Driver 노드 실행**

```bash
ros2 run kmc_hardware_driver_node kmc_hardware_driver_read_allstate_node \
  --ros-args -p port:=/dev/ttyKMC -r cmd_vel:=/CAV_09/cmd_vel
```

- `/dev/ttyKMC` 없을경우:

```bash
ros2 run kmc_hardware_driver_node kmc_hardware_driver_read_allstate_node \
  --ros-args -p port:=/dev/ttyUSB0 -r cmd_vel:=/CAV_28/cmd_vel
```

---

## 2. 차량 제어

### 2-1. 정지 (ssh)
- 각 CAV별 번호 입력
- CAV_28 예시
```bash
export ROS_DOMAIN_ID=100
source /opt/ros/foxy/setup.bash
ros2 topic pub --once /CAV_28/cmd_vel geometry_msgs/msg/Twist \
"{linear: {x: 0.0}, angular: {z: 0.0}}"
```


### 2-2. 고속 제어: 단일 실행 (ssh)
- 각 CAV별 번호 입력
- CAV_28 예시
```bash
ros2 run kmc_hardware_driver_node kmc_hardware_high_rate_control_node \
  --ros-args \
  -p port:=/dev/ttyUSB0 \
  -p baud:=1000000 \
  -p control_rate_hz:=1000.0 \
  -p vehicle_speed_rate_hz:=100.0 \
  -p command_refresh_hz:=50.0 \
  -r cmd_vel:=/CAV_28/cmd_vel
```

### 2-4. [task3.py](http://task3.py) 실행 (local)

```bash
export ROS_DOMAIN_ID=100 
source /opt/ros/foxy/setup.bash 
python3 task3.py
```


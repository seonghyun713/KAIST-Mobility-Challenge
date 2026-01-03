import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import json
import csv
import os
import math
import sys
from geometry_msgs.msg import Accel, PoseStamped


ZONE_1_CSV = 'path1_1_rotary.csv' 
ZONE_2_CSV = 'path1_2_zone.csv' 

# 기본 속도 및 제어 상수
TARGET_VELOCITY = 0.48
LOOK_AHEAD_DISTANCE = 0.23  
ZONE_TOLERANCE = 0.2  #구역 인식 반경(m)

# PID 제어 게인
Kp = 5.0      
Ki = 0.05
Kd = 2.3      
K_cte = 6.0  

target_vehicle_id = 2
if len(sys.argv) > 1:
    try: target_vehicle_id = int(sys.argv[1])
    except: pass

if target_vehicle_id == 1:
    MY_PATH_FILE = 'path1_1.json'
    MY_TOPIC = '/CAV_01'
    OTHER_TOPIC = '/CAV_02'
    print(f"\n [차량 1] ready")
else:
    MY_PATH_FILE = 'path1_2.json'
    MY_TOPIC = '/CAV_02'
    OTHER_TOPIC = '/CAV_01'
    print(f"\n  [차량 2] ready")

class ZonePriorityDriver(Node):
    def __init__(self):
        super().__init__(f'zone_driver_{target_vehicle_id}')
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publisher & Subscriber
        self.accel_publisher = self.create_publisher(Accel, f'{MY_TOPIC}_accel', 10)
        self.create_subscription(PoseStamped, MY_TOPIC, self.pose_callback, qos_profile)
        self.create_subscription(PoseStamped, OTHER_TOPIC, self.other_pose_callback, qos_profile)
        
        # 경로 및 구역 데이터
        self.path_x, self.path_y = [], []
        self.zone1_points = [] 
        self.zone2_points = [] 
        
        # 파일 로드
        self.load_my_path()
        self.zone1_points = self.load_csv_zone(ZONE_1_CSV)
        self.zone2_points = self.load_csv_zone(ZONE_2_CSV)

        # 주행 상태 변수
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.is_pose_received = False
        
        # 상대 차량 위치
        self.other_car_x = None
        self.other_car_y = None
        
        # PID 변수
        self.prev_error = 0.0
        self.integral_error = 0.0
        
        # 0.05초마다 제어 루프 실행
        self.dt = 0.05 
        self.timer = self.create_timer(self.dt, self.drive_callback)
        self.log_counter = 0

    def load_my_path(self):
        if os.path.exists(MY_PATH_FILE):
            with open(MY_PATH_FILE, 'r') as f:
                data = json.load(f)
                self.path_x = data.get('X') or data.get('x') or []
                self.path_y = data.get('Y') or data.get('y') or []
            print(f"경로 로드 완료: {MY_PATH_FILE}")
        else: 
            self.get_logger().error(f"경로 파일 없음: {MY_PATH_FILE}")

    def load_csv_zone(self, filename):
        points = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        px, py = float(row[0]), float(row[1])
                        points.append((px, py))
                    except: continue
            print(f"Zone 파일 로드: {filename}")
        else:
            print(f"Zone 파일 없음: {filename} (경고)")
        return points

    def pose_callback(self, msg):
        self.is_pose_received = True
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        # 쿼터니언 -> 오일러(Yaw)
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def other_pose_callback(self, msg):
        self.other_car_x = msg.pose.position.x
        self.other_car_y = msg.pose.position.y

    def check_in_zone(self, x, y, zone_points):
        if not zone_points or x is None: return False
        
        for (zx, zy) in zone_points:
            dist = math.hypot(zx - x, zy - y)
            if dist < ZONE_TOLERANCE:
                return True
        return False

    def drive_callback(self):
        if not self.is_pose_received or not self.path_x: return

        # 1. Pure Pursuit 목표점 찾기
        min_dist = float('inf')
        current_idx = 0
        
        # 가장 가까운 경로 점 찾기
        for i in range(len(self.path_x)):
            dist = math.hypot(self.path_x[i] - self.current_x, self.path_y[i] - self.current_y)
            if dist < min_dist:
                min_dist = dist
                current_idx = i

        # Look Ahead
        target_idx = current_idx
        for i in range(current_idx, len(self.path_x)):
            if math.hypot(self.path_x[i] - self.current_x, self.path_y[i] - self.current_y) >= LOOK_AHEAD_DISTANCE:
                target_idx = i
                break
        
        tx = self.path_x[target_idx]
        ty = self.path_y[target_idx]

        # 2. 속도 제어
        final_velocity = TARGET_VELOCITY
        status_msg = "주행 중"

        if target_vehicle_id == 2:
            # (1) 내가 (Zone 2)에 있는가?
            am_i_in_zone = self.check_in_zone(self.current_x, self.current_y, self.zone2_points)
            
            # (2) 상대가 (Zone 1)에 있는가?
            is_opponent_in_zone = self.check_in_zone(self.other_car_x, self.other_car_y, self.zone1_points)

            # (3) 둘 다 해당되면 deacceleration
            if am_i_in_zone and is_opponent_in_zone:
                final_velocity = 0.05
                status_msg = "[양보] 상대방 통과 대기 중"
            elif am_i_in_zone:
                status_msg = "[진입] 상대방 없음 -> 통과"
            else:
                status_msg = "[일반]"

        # 3. 조향 제어 (PID + Stanley 보정)
        desired_yaw = math.atan2(ty - self.current_y, tx - self.current_x)
        yaw_err = desired_yaw - self.current_yaw
        
        # 각도 정규화
        while yaw_err > math.pi: yaw_err -= 2 * math.pi
        while yaw_err < -math.pi: yaw_err += 2 * math.pi

        self.integral_error = max(-1.0, min(1.0, self.integral_error + yaw_err * self.dt))
        
        p_term = Kp * yaw_err
        i_term = Ki * self.integral_error
        d_term = Kd * (yaw_err - self.prev_error) / self.dt 
        
        # CTE(횡방향 오차) 보정
        cte_correction = min_dist * K_cte  
        if yaw_err < 0: cte_correction = -cte_correction 
        
        final_steering = max(min(p_term + i_term + d_term + cte_correction, 1.0), -1.0)
        self.prev_error = yaw_err
        
        # 4. 명령 전송
        cmd = Accel()
        cmd.linear.x = final_velocity
        cmd.angular.z = final_steering
        self.accel_publisher.publish(cmd)

        self.log_counter += 1
        if self.log_counter % 20 == 0:
            print(f"[{status_msg}] Vel: {final_velocity:.1f}, Idx: {current_idx}")

def main(args=None):
    rclpy.init(args=args)
    node = ZonePriorityDriver()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

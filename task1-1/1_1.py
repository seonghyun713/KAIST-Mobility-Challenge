import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import json
import os
import math
from geometry_msgs.msg import Accel, PoseStamped

# ===============================================================
# [1] 1번 차량 전용 설정
# ===============================================================
VEHICLE_ID = 1
PATH_FILENAME = 'path1_1_xplus0p2.json'
VEHICLE_TOPIC_NAME = '/CAV_01'

print(f"\n🔵 [차량 {VEHICLE_ID}] 솔로 주행 모드 시작 (Steering Boost ON)")
print(f"   - 경로 파일: {PATH_FILENAME}")
print(f"   - 토픽 이름: {VEHICLE_TOPIC_NAME}")

# ===============================================================
# [2] 튜닝 파라미터 (회전교차로 최적화 값 유지)
# ===============================================================
TARGET_VELOCITY = 0.48      
LOOK_AHEAD_DISTANCE = 0.23  # 짧게 설정하여 코너 안쪽 공략

# 조향 강화 파라미터
Kp = 4.0      # 각도 오차에 민감하게 반응
Ki = 0.05
Kd = 1.7      # 진동 방지
K_cte = 6.0   # 경로 이탈 시 3배 강하게 복귀 (핸들 팍 꺾음)

class Vehicle1Driver(Node):
    def __init__(self):
        super().__init__(f'driver_vehicle_{VEHICLE_ID}')
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 내 정보 구독 및 제어
        self.create_subscription(PoseStamped, VEHICLE_TOPIC_NAME, self.pose_callback, qos_profile)
        self.accel_publisher = self.create_publisher(Accel, f'{VEHICLE_TOPIC_NAME}_accel', 10)
        
        self.path_x = []
        self.path_y = []
        self.load_path_file()

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.is_pose_received = False
        
        # PID 제어 변수
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.dt = 0.05 
        self.timer = self.create_timer(self.dt, self.drive_callback)
        self.log_counter = 0

    def load_path_file(self):
        if os.path.exists(PATH_FILENAME):
            with open(PATH_FILENAME, 'r') as f:
                data = json.load(f)
                self.path_x = data.get('X') or data.get('x') or []
                self.path_y = data.get('Y') or data.get('y') or []
                self.get_logger().info(f"✅ 경로 파일 로드 완료: {len(self.path_x)} points")
        else:
            self.get_logger().error(f"❌ 경로 파일 없음: {PATH_FILENAME}")

    def pose_callback(self, msg):
        self.is_pose_received = True
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def drive_callback(self):
        if not self.is_pose_received or len(self.path_x) == 0:
            return

        # 1. 내 위치에서 가장 가까운 경로점 찾기 (CTE 계산용)
        min_dist = float('inf')
        current_idx = 0
        for i in range(len(self.path_x)):
            dist = math.hypot(self.path_x[i] - self.current_x, self.path_y[i] - self.current_y)
            if dist < min_dist:
                min_dist = dist
                current_idx = i

        # 2. Look Ahead Point 찾기
        target_idx = current_idx
        for i in range(current_idx, len(self.path_x)):
            dist = math.hypot(self.path_x[i] - self.current_x, self.path_y[i] - self.current_y)
            if dist >= LOOK_AHEAD_DISTANCE:
                target_idx = i
                break
        
        tx = self.path_x[target_idx]
        ty = self.path_y[target_idx]

        # 3. 조향각 계산 (Pure Pursuit + PID + CTE Boost)
        desired_yaw = math.atan2(ty - self.current_y, tx - self.current_x)
        yaw_err = desired_yaw - self.current_yaw
        
        # 각도 정규화 (-pi ~ pi)
        while yaw_err > math.pi: yaw_err -= 2 * math.pi
        while yaw_err < -math.pi: yaw_err += 2 * math.pi

        # PID
        self.integral_error += yaw_err * self.dt
        self.integral_error = max(-1.0, min(1.0, self.integral_error))
        
        p = Kp * yaw_err
        i = Ki * self.integral_error
        d = Kd * (yaw_err - self.prev_error) / self.dt 
        
        # [CTE Boost] 경로 이탈 시 핸들 강하게 보정
        cte_correction = min_dist * K_cte  
        
        # 방향 결정 (yaw_err 부호와 맞춤 - 간단한 휴리스틱)
        if yaw_err < 0: 
            cte_correction = -cte_correction 
        
        final_steering = p + i + d + cte_correction
        self.prev_error = yaw_err
        
        # 하드웨어 제한 (-1.0 ~ 1.0)
        final_steering = max(min(final_steering, 1.0), -1.0)
        
        # 명령 발행
        cmd = Accel()
        cmd.linear.x = TARGET_VELOCITY
        cmd.angular.z = final_steering
        self.accel_publisher.publish(cmd)
        
	# [디버깅용 로그 출력 코드]  
        self.log_counter += 1
        if self.log_counter % 5 == 0:  # 5번에 한 번씩 출력 (자주 확인)
            print(f"[{current_idx}] "
                  f"Err(거리):{min_dist:.3f}m | "
                  f"YawErr(각도):{math.degrees(yaw_err):.1f}° | "
                  f"Steer(명령):{final_steering:.3f} | "
                  f"Boost(보정):{cte_correction:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = Vehicle1Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

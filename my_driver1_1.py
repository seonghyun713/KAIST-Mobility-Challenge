import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import json
import os
import math
from geometry_msgs.msg import Accel, PoseStamped

# === [최종 PID 제어 설정] ===
PATH_FILENAME = 'path_shifted_1_1.json'
TARGET_VELOCITY = 0.20      # (m/s)

# 1. 주시 거리 (Look Ahead)
# 회전교차로에서 안쪽을 파고들지 않으려면 짧게 봐야 합니다.
# 0.2m ~ 0.25m 추천
LOOK_AHEAD_DISTANCE = 0.4  

# 2. PID 게인 튜닝 (오차 5cm 목표)
Kp = 2.0   # P: 현재 오차만큼 핸들을 팍 꺾음 (기본 힘)
Ki = 0.05  # I: 오차가 안 줄어들면 힘을 '누적'시킴 (밀림 방지 핵심)
Kd = 1.0  # D: 핸들이 흔들리지 않게 잡아주는 댐퍼 (진동 방지)

class PIDIntegralDriver(Node):
    def __init__(self):
        super().__init__('pid_integral_driver')
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(PoseStamped, '/CAV_01', self.pose_callback, qos_profile)
        self.accel_publisher = self.create_publisher(Accel, '/CAV_01_accel', 10)
        
        self.path_x = []
        self.path_y = []
        self.load_path_file()

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.is_pose_received = False
        
        # PID 제어 상태 변수
        self.prev_error = 0.0
        self.integral_error = 0.0  # 오차 누적통 (I항)

        self.timer = self.create_timer(0.05, self.drive_callback)

    def load_path_file(self):
        if os.path.exists(PATH_FILENAME):
            with open(PATH_FILENAME, 'r') as f:
                data = json.load(f)
                self.path_x = data.get('X', [])
                self.path_y = data.get('Y', [])
                self.get_logger().info(f'📂 경로 로드 완료: {len(self.path_x)}점')

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

        # 1. 내 차에서 가장 가까운 경로 점 찾기 (현재 오차 측정용)
        min_dist = float('inf')
        current_idx = 0
        for i in range(len(self.path_x)):
            dx = self.path_x[i] - self.current_x
            dy = self.path_y[i] - self.current_y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                min_dist = dist
                current_idx = i

        # 2. 횡방향 오차(CTE) 계산 및 방향 판별
        # 로컬 좌표계 변환: 내 차 기준으로 점이 왼쪽(+)인지 오른쪽(-)인지 계산
        closest_x = self.path_x[current_idx]
        closest_y = self.path_y[current_idx]
        
        dx = closest_x - self.current_x
        dy = closest_y - self.current_y
        
        # Rotation Matrix를 이용한 Y축 오차 계산
        # local_y > 0 이면 왼쪽 오차, < 0 이면 오른쪽 오차
        local_y = -dx * math.sin(self.current_yaw) + dy * math.cos(self.current_yaw)
        current_error = local_y 

        # 3. 목표점(Look Ahead) 찾기 (Pure Pursuit 헤딩용)
        target_idx = current_idx
        for i in range(current_idx, len(self.path_x)):
            dx = self.path_x[i] - self.current_x
            dy = self.path_y[i] - self.current_y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist >= LOOK_AHEAD_DISTANCE:
                target_idx = i
                break
        
        target_x = self.path_x[target_idx]
        target_y = self.path_y[target_idx]
        
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        target_angle = math.atan2(dy, dx)
        
        # 헤딩 에러 (가야 할 방향 - 내 방향)
        heading_error = target_angle - self.current_yaw
        while heading_error > math.pi: heading_error -= 2 * math.pi
        while heading_error < -math.pi: heading_error += 2 * math.pi

        # ==========================================================
        # ★ [PID 제어 로직] - 적분(I) 포함
        # ==========================================================
        
        # (1) 적분항 계산 (Accumulate)
        self.integral_error += current_error
        
        # [중요] Anti-Windup (적분 누적 제한)
        # 오차가 너무 오랫동안 쌓이면, 직진할 때도 핸들이 안 돌아옵니다.
        # 그래서 누적값을 일정 범위(-1.0 ~ 1.0)로 강제로 자릅니다.
        if self.integral_error > 1.0: self.integral_error = 1.0
        if self.integral_error < -1.0: self.integral_error = -1.0
        
        # (2) PID 계산
        p_term = Kp * current_error
        i_term = Ki * self.integral_error
        d_term = Kd * (current_error - self.prev_error)
        
        # (3) 최종 조향각 = (방향 잡기) + (오차 수정 PID)
        final_steering = heading_error + p_term + i_term + d_term
        
        # 상태 업데이트
        self.prev_error = current_error

        # 물리적 한계 (-1.0 ~ 1.0)
        MAX_STEER = 1.0
        if final_steering > MAX_STEER: final_steering = MAX_STEER
        if final_steering < -MAX_STEER: final_steering = -MAX_STEER
        
        cmd = Accel()
        cmd.linear.x = TARGET_VELOCITY
        cmd.angular.z = final_steering
        
        self.accel_publisher.publish(cmd)
        
        # 로그 (I-term이 작동하는지 확인하세요)
        now_sec = self.get_clock().now().seconds_nanoseconds()[0]
        if now_sec % 1 == 0:
            self.get_logger().info(f'🏁 오차: {abs(current_error)*100:.2f} cm | I-힘: {i_term:.3f}')

def main(args=None):
    rclpy.init(args=args)
    node = PIDIntegralDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

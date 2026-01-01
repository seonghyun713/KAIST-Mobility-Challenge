import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import json
import os
import math
from geometry_msgs.msg import Accel, PoseStamped

VEHICLE_ID = 1
PATH_FILENAME = 'converted_path1_1_final.json'
VEHICLE_TOPIC_NAME = '/CAV_01'

print(f"\n [차량 {VEHICLE_ID}] 주행 시작")
print(f"   - 경로 파일: {PATH_FILENAME}")
print(f"   - 토픽 이름: {VEHICLE_TOPIC_NAME}")


TARGET_VELOCITY = 0.48     
LOOK_AHEAD_DISTANCE = 0.23 

Kp = 4.5      # 각도 오차에 민감하게 반응
Ki = 0.05
Kd = 2.3      # 진동 방지
K_cte = 5.0   # 경로 이탈 시 복귀

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
                self.get_logger().info(f"경로 파일 로드 완료: {len(self.path_x)} points")
        else:
            self.get_logger().error(f"경로 파일 없음: {PATH_FILENAME}")

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

        # 1. 내 위치에서 가장 가까운 경로점
        min_dist = float('inf')
        current_idx = 0
        for i in range(len(self.path_x)):
            dist = math.hypot(self.path_x[i] - self.current_x, self.path_y[i] - self.current_y)
            if dist < min_dist:
                min_dist = dist
                current_idx = i

        # 2. Look Ahead Point
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
        
        # [CTE]
        cte_correction = min_dist * K_cte  
        
        # 방향 결정
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
        
	# 디버깅용
        self.log_counter += 1
        if self.log_counter % 5 == 0:
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

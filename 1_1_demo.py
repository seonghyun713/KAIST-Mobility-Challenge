#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import json
import os
import math
from geometry_msgs.msg import Accel, PoseStamped

# ============================================================
# [설정]
# ============================================================
VEHICLE_ID = 1
PATH_FILENAME = 'path1_1.json'
VEHICLE_TOPIC_NAME = '/CAV_01'

print(f"\n [차량 {VEHICLE_ID}] 지도 기반 예지 주행 (Map Prediction)")
print(f"   - 방식: 핸들 흔들림 무시, 도로 형상 직접 분석")
print(f"   - 목표: 직진 구간 1.8m/s ")

# ============================================================
# [파라미터 세트]
# ============================================================
CURVE_PARAMS = {
    "vel": 0.58,
    "look_ahead": 0.38,
    "kp": 6.3,    
    "ki": 0.05,
    "kd": 1.0,
    "k_cte": 5.0
}

# 직진: 흔들림을 잡기 위해 Kp는 낮추고 Kd(댐핑)는 높임
STRAIGHT_PARAMS = {
    "vel": 1.8,         # 목표 속도
    "look_ahead": 0.47,  # 멀리 봄
    "kp": 4.0,          # 핸들을 부드럽게 (진동 방지)
    "ki": 0.005,         # I항 제거
    "kd": 1.5,          # 댐핑 극대화 (핸들 고정)
    "k_cte": 0.8        # 복귀 천천히
}

WHEELBASE = 0.211
DIST_CENTER_TO_REAR = WHEELBASE / 2.0
TICK_RATE = 0.05

ACCEL_LIMIT = 0.8
DECEL_LIMIT = 3.0

class Vehicle1Driver(Node):
    def __init__(self):
        super().__init__(f'driver_vehicle_{VEHICLE_ID}')
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        def load_path_points(json_file):
            if not os.path.exists(json_file): return []
            with open(json_file, "r") as f: data = json.load(f)
            xs = data.get("x") or data.get("X")
            ys = data.get("y") or data.get("Y")
            if not xs or not ys: return []
            return [(float(x), float(y)) for x, y in zip(xs, ys)]

        self.path_pts = load_path_points(PATH_FILENAME)
        self.path_x = [p[0] for p in self.path_pts]
        self.path_y = [p[1] for p in self.path_pts]
        
        if not self.path_pts:
            self.get_logger().error(f"❌ 경로 파일 없음: {PATH_FILENAME}")

        self.create_subscription(PoseStamped, VEHICLE_TOPIC_NAME, self.pose_callback, qos_profile)
        self.accel_publisher = self.create_publisher(Accel, f'{VEHICLE_TOPIC_NAME}_accel', 10)

        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0
        self.got_pose = False

        self.prev_err = 0.0
        self.int_err = 0.0
        self.last_time = self.get_clock().now()
        
        self.current_vel_cmd = 0.48
        self.mode = "CURVE"          
        
        self.log_counter = 0
        self.create_timer(TICK_RATE, self.drive_loop)

    def pose_callback(self, msg):
        self.got_pose = True
        self.curr_yaw = msg.pose.orientation.z
        self.curr_x = msg.pose.position.x - (DIST_CENTER_TO_REAR * math.cos(self.curr_yaw))
        self.curr_y = msg.pose.position.y - (DIST_CENTER_TO_REAR * math.sin(self.curr_yaw))

    def get_road_curvature(self, current_idx):
        # [핵심 로직] 도로 형상 분석
        # 현재 위치에서 약 10포인트(약 0.5~1m) 앞과 
        # 거기서 10포인트 더 앞의 각도를 비교
        
        idx_now = current_idx
        idx_near = min(len(self.path_x) - 1, current_idx + 50)
        idx_far  = min(len(self.path_x) - 1, current_idx + 100)

        if idx_near == idx_far: return 0.0 # 끝부분

        # 1. 가까운 도로 벡터
        dx1 = self.path_x[idx_near] - self.path_x[idx_now]
        dy1 = self.path_y[idx_near] - self.path_y[idx_now]
        angle1 = math.atan2(dy1, dx1)

        # 2. 먼 도로 벡터
        dx2 = self.path_x[idx_far] - self.path_x[idx_near]
        dy2 = self.path_y[idx_far] - self.path_y[idx_near]
        angle2 = math.atan2(dy2, dx2)

        # 3. 각도 차이 (도로가 얼마나 휘었는가?)
        diff = abs(angle1 - angle2)
        while diff > math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        
        return abs(diff)

    def drive_loop(self):
        if not self.got_pose or not self.path_pts: return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0.001 or dt > 0.1: return

        # --------------------------------------------------------
        # [Step 1] 가장 가까운 경로점 찾기
        # --------------------------------------------------------
        min_d = float('inf')
        curr_idx = 0
        for i, (px, py) in enumerate(zip(self.path_x, self.path_y)):
            d = math.hypot(px - self.curr_x, py - self.curr_y)
            if d < min_d:
                min_d = d
                curr_idx = i

        # --------------------------------------------------------
        # [Step 2] 모드 결정 (지도 기반) - 여기가 바뀜!
        # --------------------------------------------------------
        road_curve_amount = self.get_road_curvature(curr_idx)

        # 도로가 거의 일직선(0.15 rad 약 8도 이하)이면 직진 모드
        # 단, 경로 이탈(min_d)이 0.4m 이상이면 안전을 위해 커브 모드(감속)
        if road_curve_amount < 0.15 and min_d < 0.4:
            self.mode = "STRGT"
        else:
            self.mode = "CURVE"

        if self.mode == "STRGT":
            params = STRAIGHT_PARAMS
        else:
            params = CURVE_PARAMS

        # --------------------------------------------------------
        # [Step 3] PID 제어
        # --------------------------------------------------------
        target_idx = curr_idx
        for i in range(curr_idx, len(self.path_x)):
            d = math.hypot(self.path_x[i] - self.curr_x, self.path_y[i] - self.curr_y)
            if d >= params["look_ahead"]: 
                target_idx = i
                break
        
        tx, ty = self.path_x[target_idx], self.path_y[target_idx]
        desired_yaw = math.atan2(ty - self.curr_y, tx - self.curr_x)
        yaw_err = desired_yaw - self.curr_yaw
        
        while yaw_err > math.pi: yaw_err -= 2 * math.pi
        while yaw_err < -math.pi: yaw_err += 2 * math.pi

        self.int_err = max(-1.0, min(1.0, self.int_err + yaw_err * dt))
        
        p = params["kp"] * yaw_err
        i_term = params["ki"] * self.int_err
        d = params["kd"] * (yaw_err - self.prev_err) / dt
        cte = min_d * params["k_cte"] * (-1 if yaw_err < 0 else 1)

        raw_steer = p + i_term + d + cte
        final_steer = max(-1.0, min(1.0, raw_steer))
        self.prev_err = yaw_err

        # --------------------------------------------------------
        # [Step 4] 속도 결정 및 발행
        # --------------------------------------------------------
        target_v = params["vel"] # 모드에 따라 1.8 or 0.48

        if target_v > self.current_vel_cmd:
            self.current_vel_cmd = min(target_v, self.current_vel_cmd + ACCEL_LIMIT * dt)
        else:
            self.current_vel_cmd = max(target_v, self.current_vel_cmd - DECEL_LIMIT * dt)

        cmd = Accel()
        cmd.linear.x = float(self.current_vel_cmd)
        cmd.angular.z = float(final_steer)
        self.accel_publisher.publish(cmd)

        self.log_counter += 1
        if self.log_counter % 20 == 0:
            # RoadCurve가 0에 가까워야 직진입니다
            print(f"[{self.mode}] Vel:{self.current_vel_cmd:.2f} | RoadCurve:{road_curve_amount:.3f} | DistErr:{min_d:.3f}")

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

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import json
import os
import math
from geometry_msgs.msg import Accel, PoseStamped


# [설정]
VEHICLE_ID = 1
PATH_FILENAME = 'path1_1.json'
VEHICLE_TOPIC_NAME = '/CAV_01'

print(f"\n [차량 {VEHICLE_ID}] 3단 변속 모드 적용")
print(f"   - STRGT: 1.8 m/s")
print(f"   - EASY: 0.7 m/s")
print(f"   - HARD: 0.5 m/s")

# 1. 급커브 (Hard Curve)
HARD_PARAMS = {
    "vel": 0.5,      
    "look_ahead": 0.37,
    "kp": 6.5,         
    "ki": 0.045,
    "kd": 1.0,
    "k_cte": 5.0
}

# 2. 완만한 커브 (Easy Curve)
EASY_PARAMS = {
    "vel": 0.8,     
    "look_ahead": 0.39, 
    "kp": 6.0,         
    "ki": 0.05,
    "kd": 1.0,          
    "k_cte": 4.0
}

# 3. 직진 (Straight)
STRAIGHT_PARAMS = {
    "vel": 1.8,        
    "look_ahead": 0.5,  
    "kp": 4.0,          
    "ki": 0.005,
    "kd": 2.0,        
    "k_cte": 1.0
}

WHEELBASE = 0.211
DIST_CENTER_TO_REAR = WHEELBASE / 2.0
TICK_RATE = 0.05

ACCEL_LIMIT = 3.0
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
        
        self.current_vel_cmd = 0.5
        self.mode = "HARD" # 안전하게 시작
        self.avg_steer_signed = 0.0 

        self.log_counter = 0
        self.create_timer(TICK_RATE, self.drive_loop)

    def pose_callback(self, msg):
        self.got_pose = True
        self.curr_yaw = msg.pose.orientation.z
        self.curr_x = msg.pose.position.x - (DIST_CENTER_TO_REAR * math.cos(self.curr_yaw))
        self.curr_y = msg.pose.position.y - (DIST_CENTER_TO_REAR * math.sin(self.curr_yaw))

    def drive_loop(self):
        if not self.got_pose or not self.path_pts: return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0.001 or dt > 0.1: return

        # [Step 1] 현재 모드에 따른 파라미터 선택
        if self.mode == "HARD":
            params = HARD_PARAMS
        elif self.mode == "EASY":
            params = EASY_PARAMS
        else: # STRGT
            params = STRAIGHT_PARAMS

        # [Step 2] Pure Pursuit & PID
        min_d = float('inf')
        curr_idx = 0
        for i, (px, py) in enumerate(zip(self.path_x, self.path_y)):
            d = math.hypot(px - self.curr_x, py - self.curr_y)
            if d < min_d: min_d = d; curr_idx = i

        target_idx = curr_idx
        for i in range(curr_idx, len(self.path_x)):
            d = math.hypot(self.path_x[i] - self.curr_x, self.path_y[i] - self.curr_y)
            if d >= params["look_ahead"]: target_idx = i; break
        
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

        # [Step 3] 모드 전환 로직 (히스테리시스)
        self.avg_steer_signed = 0.85 * self.avg_steer_signed + 0.15 * final_steer
        filter_val = abs(self.avg_steer_signed)
        if abs(final_steer) > 0.90:
            self.mode = "HARD"
            # 필터값 조정 (모드 유지용)
            if final_steer > 0: self.avg_steer_signed = 0.7 
            else: self.avg_steer_signed = -0.7
        else:
            if self.mode == "STRGT":
                if filter_val > 0.30:
                    self.mode = "EASY"
            
            elif self.mode == "EASY":
                if filter_val < 0.15:       
                    self.mode = "STRGT"
                elif filter_val > 0.80:     
                    self.mode = "HARD"
            
            elif self.mode == "HARD":
                if filter_val < 0.70:       
                    self.mode = "EASY"

        # [Step 4] 결정된 모드에 맞는 속도 설정
        if self.mode == "HARD":
            target_v = HARD_PARAMS["vel"]
        elif self.mode == "EASY":
            target_v = EASY_PARAMS["vel"]
        else:
            target_v = STRAIGHT_PARAMS["vel"]

        # [Step 5] 속도 제어 및 발행
        if target_v > self.current_vel_cmd:
            self.current_vel_cmd = min(target_v, self.current_vel_cmd + ACCEL_LIMIT * dt)
        else:
            self.current_vel_cmd = max(target_v, self.current_vel_cmd - DECEL_LIMIT * dt)

        cmd = Accel()
        cmd.linear.x = float(self.current_vel_cmd)
        cmd.angular.z = float(final_steer)
        self.accel_publisher.publish(cmd)

        self.log_counter += 1

        # [Debugging] 로직 값 기반 모니터링
        self.log_counter += 1
        
        # Terminal 출력
        if self.log_counter % 5 == 0:
            bar_len = 10 
            fill = int(abs(final_steer) * bar_len) 
            fill = min(fill, bar_len)
            
            if final_steer < 0: # 왼쪽
                bar_str = " " * (bar_len - fill) + "<" * fill + "|" + " " * bar_len
                dir_str = "LFT" # Left
            else: # 오른쪽
                bar_str = " " * bar_len + "|" + ">" * fill + " " * (bar_len - fill)
                dir_str = "RGT"

            # 최종 출력
            # [모드] 속도 | 방향 | Filter(평균) | Raw(순간) | 막대그래프
            print(f"[{self.mode}] {self.current_vel_cmd:.1f}m/s | "
                  f"{dir_str} | "
                  f"Filter:{filter_val:.3f} | " 
                  f"Raw:{abs(final_steer):.3f} | " 
                  f"[{bar_str}]")

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

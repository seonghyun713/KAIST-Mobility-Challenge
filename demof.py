#!/usr/bin/env python3
import os
import csv
import math
import json

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Accel

# ============================================================
# [IO helpers]
# ============================================================
def load_dz_points(csv_file: str):
    pts = []
    if not os.path.exists(csv_file): return pts
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2: continue
            try: pts.append((float(row[0]), float(row[1])))
            except: pass
    return pts

def load_path_points(json_file: str):
    if not os.path.exists(json_file): return []
    with open(json_file, "r") as f: data = json.load(f)
    pts = []
    xs = data.get("x") or data.get("X")
    ys = data.get("y") or data.get("Y")
    if xs and ys:
        for x, y in zip(xs, ys): pts.append((float(x), float(y)))
    return pts

# ============================================================
# [Geometry helpers]
# ============================================================
def min_dist_to_points(x, y, pts):
    if not pts: return float("inf")
    return min(math.hypot(x - px, y - py) for px, py in pts)

def build_cumulative_s(pts):
    s = [0.0]
    for i in range(1, len(pts)):
        s.append(s[-1] + math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]))
    return s

def project_point_to_polyline_s(px, py, pts, s_cum):
    best_s, best_d2 = 0.0, float("inf")
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i+1]
        vx, vy = bx - ax, by - ay
        det = vx*vx + vy*vy
        if det < 1e-12: continue
        t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / det))
        qx, qy = ax + t * vx, ay + t * vy
        d2 = (px - qx)**2 + (py - qy)**2
        if d2 < best_d2:
            best_d2 = d2
            best_s = s_cum[i] + t * math.sqrt(det)
    return best_s

# ============================================================
# [Global Settings]
# ============================================================
TARGET_VELOCITY = 0.48
LOOK_AHEAD_DISTANCE = 0.37
ZONE_TOLERANCE = 0.3
WHEELBASE = 0.211            
DIST_CENTER_TO_REAR = WHEELBASE / 2.0 

Kp, Ki, Kd = 6.0, 0.055, 1.0
K_cte = 5.0  

# Guardian과 Driver 간 통신용 전역 변수 (간소화된 아키텍처)
GUARDIAN_LIMITS = {1: 99.0, 2: 99.0}

# ============================================================
# [CODE A] Driver (Yaw 계산 단순화 적용)
# ============================================================
class ZonePriorityDriver(Node):
    def __init__(self, vehicle_id: int):
        super().__init__(f'zone_driver_{vehicle_id}')
        self.vehicle_id = int(vehicle_id)

        if self.vehicle_id == 1:
            self.MY_PATH_FILE = 'path1_1.json'
            self.MY_TOPIC, self.OTHER_TOPIC = '/CAV_01', '/CAV_02'
        else:
            self.MY_PATH_FILE = 'path1_2.json'
            self.MY_TOPIC, self.OTHER_TOPIC = '/CAV_02', '/CAV_01'
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.accel_publisher = self.create_publisher(Accel, f'{self.MY_TOPIC}_accel', 10)
        self.create_subscription(PoseStamped, self.MY_TOPIC, self.pose_callback, qos)
        
        self.path_points = load_path_points(self.MY_PATH_FILE)
        self.path_x = [p[0] for p in self.path_points]
        self.path_y = [p[1] for p in self.path_points]

        self.current_x = 0.0; self.current_y = 0.0; self.current_yaw = 0.0
        self.is_pose_received = False
        self.prev_error = 0.0; self.integral_error = 0.0
        
        self.last_time = self.get_clock().now()
        self.create_timer(0.05, self.drive_callback) 
        self.log_counter = 0

    def pose_callback(self, msg):
        self.is_pose_received = True
        
        # [수정됨] 명세서에 따라 orientation.z가 Yaw임 (복잡한 계산 삭제)
        raw_yaw = msg.pose.orientation.z
        
        # 좌표 보정 (중심 -> 뒷바퀴)
        self.current_yaw = raw_yaw
        self.current_x = msg.pose.position.x - (DIST_CENTER_TO_REAR * math.cos(self.current_yaw))
        self.current_y = msg.pose.position.y - (DIST_CENTER_TO_REAR * math.sin(self.current_yaw))

    def drive_callback(self):
        if not self.is_pose_received or not self.path_x: return

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        if dt <= 0.001: return 
        if dt > 0.1: dt = 0.1

        # Pure Pursuit
        min_dist = float('inf'); current_idx = 0
        for i, (px, py) in enumerate(zip(self.path_x, self.path_y)):
            d = math.hypot(px - self.current_x, py - self.current_y)
            if d < min_dist: min_dist = d; current_idx = i
        
        target_idx = current_idx
        for i in range(current_idx, len(self.path_x)):
            if math.hypot(self.path_x[i]-self.current_x, self.path_y[i]-self.current_y) >= LOOK_AHEAD_DISTANCE:
                target_idx = i; break
        tx, ty = self.path_x[target_idx], self.path_y[target_idx]

        # 속도 결정 (Guardian 우선)
        driver_velocity = TARGET_VELOCITY
        guardian_limit = GUARDIAN_LIMITS.get(self.vehicle_id, 99.0)
        final_velocity = min(driver_velocity, guardian_limit)

        status_msg = "주행"
        if guardian_limit < driver_velocity: status_msg = f"🛡️ [Grd] {guardian_limit:.2f}"

        # 조향 (PID)
        desired_yaw = math.atan2(ty - self.current_y, tx - self.current_x)
        yaw_err = desired_yaw - self.current_yaw
        while yaw_err > math.pi: yaw_err -= 2*math.pi
        while yaw_err < -math.pi: yaw_err += 2*math.pi
        
        self.integral_error = max(-1.0, min(1.0, self.integral_error + yaw_err*dt))
        p = Kp * yaw_err
        i = Ki * self.integral_error
        d = Kd * (yaw_err - self.prev_error) / dt 
        cte = min_dist * K_cte * (-1 if yaw_err < 0 else 1)
        
        final_steering = max(-1.0, min(1.0, p + i + d + cte))
        self.prev_error = yaw_err

        cmd = Accel()
        cmd.linear.x = float(final_velocity)
        cmd.angular.z = float(final_steering)
        self.accel_publisher.publish(cmd)

        self.log_counter += 1
        if self.log_counter % 20 == 0:
            print(f"[Car{self.vehicle_id}] {status_msg} | V:{final_velocity:.2f}")

# ============================================================
# [CODE B] Guardian (Right + Upper + Rotary)
# ============================================================
class UpperRightCollisionGuardianSplitDZ(Node):
    def __init__(self):
        super().__init__("guardian_node")
        
        # 파일 경로
        self.DZ_RIGHT = "right_1_2.csv"
        self.DZ_UPPER_V1 = "upper_dz_v1.csv"
        self.DZ_UPPER_V2 = "upper_dz_v2.csv"
        self.DZ_ROTARY_V1 = "path1_1_rotary.csv" # 사용자 요청 파일
        self.DZ_ZONE_V2   = "path1_2_zone.csv"   # 사용자 요청 파일

        # 파라미터
        self.NEAR_TOL_RIGHT, self.SAFETY_DIST_RIGHT = 0.40, 1.50
        self.NEAR_TOL_UPPER, self.SAFETY_DIST_UPPER = 0.80, 1.50
        self.UPPER_RELEASE_DIST = 2.20
        self.NEAR_TOL_ROTARY = 0.40 

        self.ACCEL_YIELD, self.ACCEL_RESUME = 0.10, 0.05
        self.MIN_ACCEL, self.TICK, self.RAMP_PER_SEC = 0.0, 0.05, 0.35

        # 데이터 로드
        self.dz_right = load_dz_points(self.DZ_RIGHT)
        self.dz_upper_v1 = load_dz_points(self.DZ_UPPER_V1)
        self.dz_upper_v2 = load_dz_points(self.DZ_UPPER_V2)
        self.dz_rotary_v1 = load_dz_points(self.DZ_ROTARY_V1)
        self.dz_zone_v2 = load_dz_points(self.DZ_ZONE_V2)

        # Upper Logic용 경로
        self.path1_pts = load_path_points("path1_1.json")
        self.path2_pts = load_path_points("path1_2.json")
        self.path1_s = build_cumulative_s(self.path1_pts)
        self.path2_s = build_cumulative_s(self.path2_pts)

        # Upper Merge Point
        upper_all = self.dz_upper_v1 + self.dz_upper_v2
        self.upper_merge = None
        self.s_merge_1, self.s_merge_2 = None, None
        if upper_all:
            mx = sum(p[0] for p in upper_all)/len(upper_all)
            my = sum(p[1] for p in upper_all)/len(upper_all)
            self.upper_merge = (mx, my)
            self.s_merge_1 = project_point_to_polyline_s(mx, my, self.path1_pts, self.path1_s)
            self.s_merge_2 = project_point_to_polyline_s(mx, my, self.path2_pts, self.path2_s)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.p1, self.p2 = None, None
        self.create_subscription(PoseStamped, "/CAV_01", self.cb1, qos)
        self.create_subscription(PoseStamped, "/CAV_02", self.cb2, qos)

        self.cmd_accel = {1: 0.0, 2: 0.0}
        self.target_accel = {1: None, 2: None}
        self.resume_ticks_left = {1: 0, 2: 0}
        self.upper_lock_yield_id = None
        self.RESUME_PULSE_TICKS = int(1.5 / self.TICK)
        
        self.create_timer(self.TICK, self.tick)
        print("✅ Guardian Running (All Zones Active)")

    def cb1(self, msg): self.p1 = (msg.pose.position.x, msg.pose.position.y)
    def cb2(self, msg): self.p2 = (msg.pose.position.x, msg.pose.position.y)

    def _update_global_limit(self, vid, speed):
        GUARDIAN_LIMITS[vid] = float(max(self.MIN_ACCEL, speed))

    def _ramp_publish(self, vid):
        tgt = self.target_accel[vid]
        if tgt is None:
            self.cmd_accel[vid] = 99.0
            self._update_global_limit(vid, 99.0)
            return
        cur = self.cmd_accel[vid]
        if cur > 50: cur = 0.5 
        step = self.RAMP_PER_SEC * self.TICK
        if tgt > cur: cur = min(tgt, cur + step)
        else: cur = max(tgt, cur - step)
        self.cmd_accel[vid] = cur
        self._update_global_limit(vid, cur)

    def _start_resume_pulse(self, vid):
        self.resume_ticks_left[vid] = self.RESUME_PULSE_TICKS
        self.target_accel[vid] = self.ACCEL_RESUME

    def _cancel_resume(self, vid):
        self.resume_ticks_left[vid] = 0

    def tick(self):
        if self.p1 is None or self.p2 is None: return
        x1, y1 = self.p1
        x2, y2 = self.p2
        d12 = math.hypot(x1 - x2, y1 - y2)

        for vid in (1, 2):
            if self.resume_ticks_left[vid] > 0:
                self._ramp_publish(vid)
                self.resume_ticks_left[vid] -= 1
                if self.resume_ticks_left[vid] <= 0:
                    self.target_accel[vid] = None 

        # [Logic 1] RIGHT
        if self.dz_right:
            n1 = min_dist_to_points(x1, y1, self.dz_right) <= self.NEAR_TOL_RIGHT
            n2 = min_dist_to_points(x2, y2, self.dz_right) <= self.NEAR_TOL_RIGHT
            if n1 and n2 and d12 <= self.SAFETY_DIST_RIGHT:
                self._cancel_resume(1)
                self.target_accel[1] = self.ACCEL_YIELD
                self.target_accel[2] = None
                self._ramp_publish(1); self._ramp_publish(2)
                return
            elif self.target_accel[1] == self.ACCEL_YIELD and self.resume_ticks_left[1] == 0:
                self._start_resume_pulse(1)

        # [Logic 2] UPPER
        if self.upper_merge:
            n1 = min_dist_to_points(x1, y1, self.dz_upper_v1) <= self.NEAR_TOL_UPPER
            n2 = min_dist_to_points(x2, y2, self.dz_upper_v2) <= self.NEAR_TOL_UPPER
            
            if self.upper_lock_yield_id:
                if (not (n1 and n2)) or (d12 > self.UPPER_RELEASE_DIST):
                    yid = self.upper_lock_yield_id
                    self.upper_lock_yield_id = None
                    if self.target_accel[yid] == self.ACCEL_YIELD and self.resume_ticks_left[yid] == 0:
                        self._start_resume_pulse(yid)
                else:
                    yid = self.upper_lock_yield_id
                    self._cancel_resume(yid)
                    self.target_accel[yid] = self.ACCEL_YIELD
                    self.target_accel[3-yid] = None
                    self._ramp_publish(1); self._ramp_publish(2)
                    return
            elif n1 and n2 and d12 <= self.SAFETY_DIST_UPPER:
                s1 = project_point_to_polyline_s(x1, y1, self.path1_pts, self.path1_s)
                s2 = project_point_to_polyline_s(x2, y2, self.path2_pts, self.path2_s)
                rem1 = max(0.0, self.s_merge_1 - s1)
                rem2 = max(0.0, self.s_merge_2 - s2)
                yield_id = 2 if rem1 < rem2 - 0.1 else 1 

                self.upper_lock_yield_id = yield_id
                self._cancel_resume(yield_id)
                self.target_accel[yield_id] = self.ACCEL_YIELD
                self.target_accel[3-yield_id] = None
                self._ramp_publish(1); self._ramp_publish(2)
                return
            
            for vid in (1, 2):
                if self.target_accel[vid] == self.ACCEL_YIELD and self.resume_ticks_left[vid] == 0 and not self.upper_lock_yield_id:
                    self._start_resume_pulse(vid)

        # [Logic 3] ROTARY (님이 원하신 기능)
        if self.dz_rotary_v1 and self.dz_zone_v2:
            in_rotary = min_dist_to_points(x1, y1, self.dz_rotary_v1) <= self.NEAR_TOL_ROTARY
            in_zone   = min_dist_to_points(x2, y2, self.dz_zone_v2)   <= self.NEAR_TOL_ROTARY
            
            if in_rotary and in_zone:
                self._cancel_resume(2)
                self.target_accel[2] = self.ACCEL_YIELD
                self._ramp_publish(2)
                return
            elif self.target_accel[2] == self.ACCEL_YIELD and self.resume_ticks_left[2] == 0 and not self.upper_lock_yield_id:
                 self._start_resume_pulse(2)
            elif self.target_accel[2] is None:
                self._ramp_publish(2)

def main():
    rclpy.init(args=None)
    node_driver_1 = ZonePriorityDriver(1)
    node_driver_2 = ZonePriorityDriver(2)
    node_guardian = UpperRightCollisionGuardianSplitDZ()

    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node_driver_1)
    ex.add_node(node_driver_2)
    ex.add_node(node_guardian)

    try: ex.spin()
    except KeyboardInterrupt: pass
    finally:
        ex.shutdown()
        node_driver_1.destroy_node()
        node_driver_2.destroy_node()
        node_guardian.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

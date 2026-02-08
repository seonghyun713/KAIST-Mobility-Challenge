#!/usr/bin/env python3
"""
Unified CAV Control System for V2X Competition
- Integrates path following, roundabout control, and safety logic
- Conservative improvements for stability and lap time
"""
import os
import math
import json
import csv
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Accel, PoseStamped, Twist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DIR = os.path.join(BASE_DIR, "path")

# ============================================================
# [TOPIC CONFIG] Change only here at the venue
# ============================================================
CAV_LOGICAL_IDS = [1, 2, 3, 4]

CAV_TOPIC_NUM = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
}

HV_TOPICS = {
    "HV1": "/HV_19",
    "HV2": "/HV_20",
}

def cav_topic(logical_id: int) -> str:
    n = int(CAV_TOPIC_NUM[logical_id])
    return f"/CAV_{n:02d}"

def cav_cmd_topic(logical_id: int) -> str:
    return f"{cav_topic(logical_id)}/cmd_vel"

def cav_accel_raw_topic(logical_id: int) -> str:
    return f"{cav_topic(logical_id)}_accel_raw"

def hv_topic(role: str) -> str:
    return HV_TOPICS[role]

# ============================================================
# [CONFIGURATION] Vehicle & Track Parameters
# ============================================================
WHEELBASE = 0.211  # 21.1cm
DIST_CENTER_TO_REAR = WHEELBASE / 2.0
TRACK_LENGTH_PER_LAP = 9.4  # meters
TARGET_LAPS = 5

# ============================================================
# [SLOW ZONES] Intersection Areas
# ============================================================
RAW_SLOW_ZONES = [
    (-0.5, -4.6,  2.0, -2.0), # 사지교차로
    (-0.5,  2.7,  2.0, -1.4), # 회전교차로
]

SLOW_ZONES = [
    (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
    for (x1, x2, y1, y2) in RAW_SLOW_ZONES
]

# 회전교차로 cutout (진입 차선 제외)
SLOW2_CUTOUT = {
    "x_min": -0.50,
    "x_max": 0.37,
    "y_min":  0.00,
    "y_max":  1.80,
}

# ============================================================
# [CONTROL PARAMS] Speed Profiles & PID Gains
# ============================================================
SLOW_PARAMS = {
    "vel": 1.5,          
    "look_ahead": 0.50, 
    "kp": 6.0,          
    "ki": 0.045,
    "kd": 1.0,          
    "k_cte": 4.0  # Original: 4.0
}

HARD_PARAMS = {
    "vel": 1.2, 
    "look_ahead": 0.65, 
    "kp": 6.0, 
    "ki": 0.045, 
    "kd": 1.0, 
    "k_cte": 2.5  # Original: 3.0 → 감소 (진동 완화)
}

EASY_PARAMS = {
    "vel": 1.3, 
    "look_ahead": 0.65, 
    "kp": 6.0, 
    "ki": 0.05, 
    "kd": 1.0, 
    "k_cte": 2.5  # Original: 3.0 → 감소 (진동 완화)
}

STRAIGHT_PARAMS = {
    "vel": 1.8, 
    "look_ahead": 1.2, 
    "kp": 2.0, 
    "ki": 0.002, 
    "kd": 2.5, 
    "k_cte": 1.0
}

MAX_SPEED = 1.5
TICK_RATE = 0.02      # 50 Hz
ACCEL_LIMIT = 1.5     # m/s²
DECEL_LIMIT = 2.0     # m/s²

# ============================================================
# [ROUNDABOUT PARAMS] Gate & HV Safety
# ============================================================
GATE_X, GATE_Y = 1.7, 0.0
GATE_SLOW_DIST = 2.1      # 게이트 눈치보기 시작 거리
GATE_RESET_DIST = 2.5     # 리셋 거리 (다음 바퀴 준비)
GATE_SLOW_VEL = 0.05      # 게이트 대기 속도

HV_DETECT_RADIUS = 0.3    # Original: 0.3
ZONE_RADIUS = 0.25        # Original: 0.25
HV_HOLD_TICKS = 5         # HV 감지 유지 시간 (깜빡임 방지)

# Smart ACC (CAV 3, 4)
ACC_DIST_LIMIT = 0.6      # 추종 거리
ACC_P_GAIN = 2.5          # 거리 오차 게인
SLOW_VELOCITY = 0.2       # ACC 감속 시
MAX_ACC_VELOCITY = 2.0    # ACC 가속 시

# ============================================================
# [COMMON] File Loaders
# ============================================================
def load_path_points(json_file):
    if not os.path.exists(json_file): 
        return []
    with open(json_file, "r") as f: 
        data = json.load(f)
    xs = data.get("x") or data.get("X")
    ys = data.get("y") or data.get("Y")
    if not xs or not ys: 
        return []
    return [(float(x), float(y)) for x, y in zip(xs, ys)]

def load_zone_from_csv(filename):
    points = []
    if not filename or not os.path.exists(filename): 
        return []
    try:
        with open(filename, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and len(row) >= 2:
                    try: 
                        points.append((float(row[0]), float(row[1])))
                    except: 
                        continue
    except: 
        pass
    return points

# ============================================================
# [NODE] Unified Vehicle Driver
# ============================================================
class UnifiedVehicleDriver(Node):
    def __init__(self, vehicle_id: int, path_filename: str,
                 start_zone_file: str,        # ⚠️ 미사용 (Exit Conflict 삭제로 불필요)
                 start_trigger_file: str,     # ✅ 사용: 게이트 트리거
                 out_zone_file: str,          # ⚠️ 미사용 (Exit Conflict 삭제로 불필요)
                 danger_zone_file: str):      # ✅ 사용: Smart ACC
        super().__init__(f"driver_vehicle_{vehicle_id}")
        
        self.vid = int(vehicle_id)
        self.logical_id = int(vehicle_id)
        self.TOPIC = cav_topic(self.logical_id)
        self.PATH_FILENAME = path_filename
        
        # ====================================================
        # [DATA LOGGING SETUP] Optional - for debugging
        # ====================================================
        ENABLE_CSV_LOG = False  # Set True for debugging
        
        if ENABLE_CSV_LOG:
            self.log_dir = os.path.join(BASE_DIR, "csv_data")
            os.makedirs(self.log_dir, exist_ok=True)
            timestamp_str = datetime.now().strftime("%m%d_%H%M%S")
            self.csv_path = os.path.join(self.log_dir, f"veh_{self.vid}_{timestamp_str}.csv")
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.writer = csv.writer(self.csv_file)
            self.writer.writerow([
                "timestamp", "x", "y", "yaw", "v_current",
                "cte", "yaw_err", "look_ahead_dist",
                "target_x", "target_y", "target_local_x", "target_local_y",
                "kp", "ki", "kd", "k_cte", "mode_idx", "in_slow_zone",
                "steer_cmd", "accel_cmd"
            ])
            self.csv_flush_counter = 0
            self.get_logger().info(f"💾 Logging enabled: {self.csv_path}")
        else:
            self.csv_file = None
            self.writer = None
        
        # ====================================================
        # [QoS & Subscriptions]
        # ====================================================
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Path loading
        self.path_pts = load_path_points(self.PATH_FILENAME)
        self.path_x = [p[0] for p in self.path_pts]
        self.path_y = [p[1] for p in self.path_pts]
        
        if not self.path_pts:
            self.get_logger().error(f"❌ [Car{self.vid}] Path missing: {self.PATH_FILENAME}")
        
        # Roundabout zones
        # ⚠️ start_zone_file, out_zone_file: Exit Conflict 삭제로 미사용
        # ✅ start_trigger_file: 게이트 통과 트리거용 (모든 CAV)
        # ✅ danger_zone_file: Smart ACC용 (CAV 3,4 + 게이트 감지용 CAV 1,2)
        self.start_trigger_points = load_zone_from_csv(start_trigger_file)
        self.danger_zone_points = load_zone_from_csv(danger_zone_file)
        
        if self.vid in [3, 4] and self.danger_zone_points:
            self.get_logger().info(f"[V{self.vid:02d}] Smart ACC Logic Activated")
        
        # Publishers & Subscribers
        self.create_subscription(PoseStamped, self.TOPIC, self.pose_callback, qos_profile)
        self.accel_raw_pub = self.create_publisher(Accel, cav_accel_raw_topic(self.logical_id), 10)
        
        # HV subscriptions
        if self.start_trigger_points or self.danger_zone_points:
            self.create_subscription(PoseStamped, hv_topic("HV1"), self._callback_hv19, qos_profile)
            self.create_subscription(PoseStamped, hv_topic("HV2"), self._callback_hv20, qos_profile)
        
        # ====================================================
        # [STATE VARIABLES]
        # ====================================================
        # Pose
        self.curr_x, self.curr_y, self.curr_yaw = 0.0, 0.0, 0.0
        self.got_pose = False
        
        # Control
        self.current_vel_cmd = 0.15
        self.mode = "HARD"
        self.avg_steer_signed = 0.0
        
        # PID state
        self.prev_err = 0.0
        self.int_err = 0.0
        self.yaw_err_f = 0.0
        self.prev_yaw_err_f = 0.0
        self.dterm_f = 0.0
        self._skip_dterm = 0
        
        # Filter params (개선: D-term 필터 통일)
        self.LPF_TAU = 0.15      # Original: 0.18 → 0.15 (진동 완화)
        self.D_LPF_TAU = 0.15    # Original: 0.10 → 0.15 (D-term 필터 통일)
        
        # Path following
        self.old_nearest_idx = 0
        self.last_time = self.get_clock().now()
        
        # Roundabout state
        self.gate_released = False
        self.hv_trigger_hold = 0
        self.hv_danger_hold = 0
        
        # HV positions
        self.hv19_x, self.hv19_y = 0.0, 0.0
        self.hv20_x, self.hv20_y = 0.0, 0.0
        self.hv19_active = False
        self.hv20_active = False
        
        # Timer
        self.create_timer(TICK_RATE, self.drive_loop)
        
        self.get_logger().info(f"✅ [V{self.vid:02d}] Driver initialized")
    
    # ====================================================
    # [CALLBACKS]
    # ====================================================
    def pose_callback(self, msg):
        self.got_pose = True
        self.curr_yaw = float(msg.pose.orientation.z)
        self.curr_x = float(msg.pose.position.x) - (DIST_CENTER_TO_REAR * math.cos(self.curr_yaw))
        self.curr_y = float(msg.pose.position.y) - (DIST_CENTER_TO_REAR * math.sin(self.curr_yaw))
    
    def _callback_hv19(self, msg):
        self.hv19_x = msg.pose.position.x
        self.hv19_y = msg.pose.position.y
        self.hv19_active = True
    
    def _callback_hv20(self, msg):
        self.hv20_x = msg.pose.position.x
        self.hv20_y = msg.pose.position.y
        self.hv20_active = True
    
    # ====================================================
    # [UTILITY FUNCTIONS]
    # ====================================================
    def is_in_slow_zone(self, x, y):
        """Check if position is in slow zone (사지교차로 or 회전교차로)"""
        # Zone 1: 사지교차로 (그대로)
        x1_min, x1_max, y1_min, y1_max = SLOW_ZONES[0]
        in_zone1 = (x1_min <= x <= x1_max) and (y1_min <= y <= y1_max)
        
        # Zone 2: 회전교차로 (cutout 적용)
        x2_min, x2_max, y2_min, y2_max = SLOW_ZONES[1]
        in_big2 = (x2_min <= x <= x2_max) and (y2_min <= y <= y2_max)
        
        cx_min = SLOW2_CUTOUT["x_min"]
        cx_max = SLOW2_CUTOUT["x_max"]
        cy_min = SLOW2_CUTOUT["y_min"]
        cy_max = SLOW2_CUTOUT["y_max"]
        in_cutout = (cx_min <= x <= cx_max) and (cy_min <= y <= cy_max)
        
        in_zone2 = in_big2 and (not in_cutout)
        
        return in_zone1 or in_zone2
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi: 
            angle -= 2.0 * math.pi
        while angle < -math.pi: 
            angle += 2.0 * math.pi
        return angle
    
    def global_to_local(self, gx, gy, rx, ry, ryaw):
        """Transform global coordinates to local (vehicle frame)"""
        dx = gx - rx
        dy = gy - ry
        lx = dx * math.cos(ryaw) + dy * math.sin(ryaw)
        ly = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
        return lx, ly
    
    def _check_hv_in_zone(self, zone_points):
        """Check if any HV is in the given zone"""
        if not zone_points:
            return False
        
        if self.hv19_active:
            for hx, hy in zone_points:
                if math.hypot(hx - self.hv19_x, hy - self.hv19_y) < HV_DETECT_RADIUS:
                    return True
        
        if self.hv20_active:
            for hx, hy in zone_points:
                if math.hypot(hx - self.hv20_x, hy - self.hv20_y) < HV_DETECT_RADIUS:
                    return True
        
        return False
    
    def _check_hv_in_zone_hold(self, zone_points, hold_attr: str, hold_ticks: int = HV_HOLD_TICKS):
        """Check HV in zone with hysteresis (prevent flickering)"""
        hit = self._check_hv_in_zone(zone_points)
        
        cur = getattr(self, hold_attr, 0)
        if hit:
            cur = int(hold_ticks)
        else:
            cur = max(0, int(cur) - 1)
        
        setattr(self, hold_attr, cur)
        return cur > 0
    
    def _get_closest_hv_front(self):
        """Get distance to closest HV in front (for ACC)"""
        closest_dist = 999.0
        
        if self.hv19_active and self.hv19_y < self.curr_y:
            d = math.hypot(self.hv19_x - self.curr_x, self.hv19_y - self.curr_y)
            if d < closest_dist: 
                closest_dist = d
        
        if self.hv20_active and self.hv20_y < self.curr_y:
            d = math.hypot(self.hv20_x - self.curr_x, self.hv20_y - self.curr_y)
            if d < closest_dist: 
                closest_dist = d
        
        return closest_dist
    
    def _get_min_dist_to_zone(self, zone_points):
        """Get minimum distance to zone points"""
        min_d = 999.0
        if zone_points:
            for zx, zy in zone_points:
                d = math.hypot(zx - self.curr_x, zy - self.curr_y)
                if d < min_d: 
                    min_d = d
        return min_d
    
    # ====================================================
    # [MAIN CONTROL LOOP]
    # ====================================================
    def drive_loop(self):
        if not self.got_pose or not self.path_pts: 
            return
        
        # Time management
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        
        if dt <= 0.001 or dt > 0.05: 
            return
        
        # ================================================
        # [1] ROUNDABOUT GATE LOGIC (최우선)
        # ================================================
        dist_to_gate = math.hypot(self.curr_x - GATE_X, self.curr_y - GATE_Y)
        
        # Reset gate for next lap
        if dist_to_gate > GATE_RESET_DIST:
            self.gate_released = False
            self.hv_trigger_hold = 0
        
        # Default target velocity
        target_vel_req = MAX_SPEED
        gate_active = False  # 게이트 우선순위 플래그
        
        # Gate slow-down logic
        if dist_to_gate < GATE_SLOW_DIST:
            if not self.gate_released:
                # Wait for HV clearance
                target_vel_req = GATE_SLOW_VEL
                gate_active = True  # 다른 로직이 덮어쓰지 못하게
                
                # Release when HV passes trigger zone
                if self._check_hv_in_zone_hold(self.start_trigger_points, 'hv_trigger_hold'):
                    self.gate_released = True
                    target_vel_req = MAX_SPEED
                    gate_active = False
        
        # ================================================
        # [2] SMART ACC LOGIC (CAV 3, 4 only) - Gate가 아닐 때만
        # ================================================
        if not gate_active and self.vid in [3, 4] and self.danger_zone_points:
            if self._get_min_dist_to_zone(self.danger_zone_points) < ZONE_RADIUS:
                dist_hv = self._get_closest_hv_front()
                if dist_hv < 999.0:
                    dist_error = dist_hv - ACC_DIST_LIMIT
                    if dist_error < 0:
                        # Too close - slow down
                        target_vel_req = SLOW_VELOCITY
                    else:
                        # Safe distance - catch up
                        catch_up_vel = MAX_SPEED + (dist_error * ACC_P_GAIN)
                        target_vel_req = min(catch_up_vel, MAX_ACC_VELOCITY)
        
        # ================================================
        # [3] DETERMINE ZONE & PARAMS
        # ================================================
        in_zone = self.is_in_slow_zone(self.curr_x, self.curr_y)
        
        mode_idx = 2  # Default HARD
        if in_zone:
            params = SLOW_PARAMS
            self.mode = "HARD"
            self.avg_steer_signed = 0.0
            self.int_err = 0.0
            mode_idx = 3
        else:
            if self.mode == "HARD":
                params = HARD_PARAMS
                mode_idx = 2
            elif self.mode == "EASY":
                params = EASY_PARAMS
                mode_idx = 1
            else:
                params = STRAIGHT_PARAMS
                mode_idx = 0
        
        # ================================================
        # [4] PURE PURSUIT - FIND TARGET POINT
        # ================================================
        path_len = len(self.path_pts)
        min_d = float('inf')
        curr_idx = self.old_nearest_idx
        search_range = 50
        found_in_window = False
        
        # Search near previous index first
        for offset in range(-search_range, search_range):
            check_idx = (self.old_nearest_idx + offset) % path_len
            d = math.hypot(self.path_x[check_idx] - self.curr_x, 
                          self.path_y[check_idx] - self.curr_y)
            if d < min_d:
                min_d = d
                curr_idx = check_idx
                found_in_window = True
        
        # Full search if needed
        if not found_in_window or min_d > 5.0:
            min_d = float('inf')
            for i in range(path_len):
                d = math.hypot(self.path_x[i] - self.curr_x, 
                              self.path_y[i] - self.curr_y)
                if d < min_d: 
                    min_d = d
                    curr_idx = i
        
        self.old_nearest_idx = curr_idx
        
        # Path jump detection
        if min_d > 0.8:
            best_i = 0
            best_d = float("inf")
            for i in range(path_len):
                d = math.hypot(self.path_x[i] - self.curr_x, 
                              self.path_y[i] - self.curr_y)
                if d < best_d: 
                    best_d = d
                    best_i = i
            curr_idx = best_i
            self.old_nearest_idx = curr_idx
            self.int_err = 0.0
            self._skip_dterm = 1
            min_d = best_d
        
        # Dynamic look-ahead (speed-based)
        active_look_ahead = min(params["look_ahead"], self.current_vel_cmd * 0.45)
        
        # Find target point
        target_idx = curr_idx
        for i in range(path_len):
            idx = (curr_idx + i) % path_len
            d = math.hypot(self.path_x[idx] - self.curr_x, 
                          self.path_y[idx] - self.curr_y)
            if d >= active_look_ahead:
                target_idx = idx
                break
        
        tx, ty = self.path_x[target_idx], self.path_y[target_idx]
        
        # ================================================
        # [5] PID CONTROL - LATERAL
        # ================================================
        desired_yaw = math.atan2(ty - self.curr_y, tx - self.curr_x)
        yaw_err = self.normalize_angle(desired_yaw - self.curr_yaw)
        
        # Yaw error LPF
        alpha = dt / (self.LPF_TAU + dt)
        err_delta = self.normalize_angle(yaw_err - self.yaw_err_f)
        self.yaw_err_f = self.normalize_angle(self.yaw_err_f + alpha * err_delta)
        
        # CTE calculation
        path_dx = tx - self.path_x[curr_idx]
        path_dy = ty - self.path_y[curr_idx]
        
        if math.hypot(path_dx, path_dy) < 0.02:
            cte = 0.0
        else:
            car_dx = self.curr_x - self.path_x[curr_idx]
            car_dy = self.curr_y - self.path_y[curr_idx]
            cross_prod = path_dx * car_dy - path_dy * car_dx
            cte_sign = 1.0 if cross_prod > 0 else -1.0
            cte = min_d * cte_sign * params["k_cte"]
        
        # Integral term (with anti-windup)
        self.int_err = max(-2.0, min(2.0, self.int_err + yaw_err * dt))
        
        # PID terms
        p = params["kp"] * yaw_err
        i_term = params["ki"] * self.int_err
        
        # D-term with LPF
        if self._skip_dterm > 0:
            d_term_raw = 0.0
            self._skip_dterm -= 1
        else:
            d_error_f = self.normalize_angle(self.yaw_err_f - self.prev_yaw_err_f)
            d_term_raw = params["kd"] * d_error_f / dt
        
        self.prev_yaw_err_f = self.yaw_err_f
        d_alpha = dt / (self.D_LPF_TAU + dt)
        self.dterm_f = (1.0 - d_alpha) * self.dterm_f + d_alpha * d_term_raw
        d_term = self.dterm_f
        
        # Final steering command
        final_steer = max(-2.0, min(2.0, float(p + i_term + d_term + cte)))
        self.prev_err = yaw_err
        
        # ================================================
        # [6] MODE SWITCHING & SPEED SELECTION
        # ================================================
        if in_zone:
            # 교차로 안: SLOW_PARAMS 속도 사용
            target_v = SLOW_PARAMS["vel"]
        else:
            # 교차로 밖: 모드별 속도 결정
            # Steering filter for mode decision
            self.avg_steer_signed = 0.85 * self.avg_steer_signed + 0.15 * final_steer
            filter_val = abs(self.avg_steer_signed)
            
            next_mode = self.mode
            if abs(final_steer) > 0.90:
                next_mode = "HARD"
            else:
                if self.mode == "STRAIGHT":
                    if filter_val > 0.30: 
                        next_mode = "EASY"
                elif self.mode == "EASY":
                    if filter_val < 0.15: 
                        next_mode = "STRAIGHT"
                    elif filter_val > 0.80: 
                        next_mode = "HARD"
                elif self.mode == "HARD":
                    if filter_val < 0.70: 
                        next_mode = "EASY"
            
            # 개선: 모든 모드 전환 시 적분 리셋
            if self.mode != next_mode:
                self.int_err = 0.0
                self._skip_dterm = 1  # D-term도 한번 스킵
            
            self.mode = next_mode
            
            # Target velocity from mode
            if self.mode == "HARD": 
                target_v = HARD_PARAMS["vel"]
            elif self.mode == "EASY": 
                target_v = EASY_PARAMS["vel"]
            else: 
                target_v = STRAIGHT_PARAMS["vel"]
        
        # ✅ CRITICAL FIX: Gate/ACC 로직을 in_zone 밖으로 이동
        # 회전교차로 안팎 무관하게 Gate/ACC 우선 적용
        if target_vel_req < target_v:
            target_v = target_vel_req
        
        # ================================================
        # [7] SPEED RAMPING (Longitudinal)
        # ================================================
        if target_v > self.current_vel_cmd:
            self.current_vel_cmd = min(target_v, self.current_vel_cmd + ACCEL_LIMIT * dt)
        else:
            self.current_vel_cmd = max(target_v, self.current_vel_cmd - DECEL_LIMIT * dt)
        
        # ================================================
        # [8] PUBLISH COMMAND
        # ================================================
        cmd = Accel()
        cmd.linear.x = float(self.current_vel_cmd)
        cmd.angular.z = float(final_steer)  # ✅ 수정: steer → final_steer
        self.accel_raw_pub.publish(cmd)
        
        # ================================================
        # [9] DATA LOGGING (Optional)
        # ================================================
        if self.writer:
            lx, ly = self.global_to_local(tx, ty, self.curr_x, self.curr_y, self.curr_yaw)
            row_data = [
                time.time(), 
                round(self.curr_x, 4), round(self.curr_y, 4), round(self.curr_yaw, 4),
                round(self.current_vel_cmd, 3),
                round(cte, 4), round(yaw_err, 4), round(active_look_ahead, 3),
                round(tx, 4), round(ty, 4), round(lx, 4), round(ly, 4),
                params["kp"], params["ki"], params["kd"], params["k_cte"],
                mode_idx, 1 if in_zone else 0,
                round(final_steer, 4), round(self.current_vel_cmd, 3)
            ]
            self.writer.writerow(row_data)
            
            # Periodic flush (every 2 seconds)
            self.csv_flush_counter += 1
            if self.csv_flush_counter % 100 == 0:
                self.csv_file.flush()
    
    def destroy_node(self):
        """Cleanup on shutdown"""
        if self.csv_file:
            self.csv_file.close()
        super().destroy_node()

# ============================================================
# [NODE] Guardian Mux (Safety Controller)
# ============================================================
class GuardianMux(Node):
    def __init__(self):
        super().__init__("guardian_mux")
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            history=HistoryPolicy.KEEP_LAST, 
            depth=10
        )
        
        self.VEH_IDS = CAV_LOGICAL_IDS
        self.TOPICS = {vid: cav_topic(vid) for vid in self.VEH_IDS}
        
        # ================================================
        # [GUARDIAN PARAMS]
        # ================================================
        self.V_NOM = MAX_SPEED
        self.RANK_SPEEDS_3P = [MAX_SPEED, 0.9, 0.3, 0.3]
        self.RANK_SPEEDS_2P = [MAX_SPEED, 0.3]
        
        # Roundabout zones
        self.TOP_CENTER = (-2.3342, 2.3073)
        self.BOT_CENTER = (-2.3342, -2.3073)
        self.RADIUS = 1.5
        self.EXIT_RADIUS = 0.4
        self.APPROACH_N = 3
        self.EPS = 0.001
        self.HYSTERESIS_N = 5
        
        # Control params
        self.TICK = 0.02
        self.RAMP_DOWN_PER_SEC = 1.5
        self.RAMP_UP_PER_SEC = 0.60  # Original: 0.30 → 0.60 (가속 2배 빠르게)
        self.STOP_VELOCITY = 0.0
        self.MIN_SPEED = self.STOP_VELOCITY
        
        # Hold logic (깜빡임 방지)
        self.HOLD_TICKS = 20  # Original: 30 → 20
        self.hold_cnt = {vid: 0 for vid in self.VEH_IDS}
        self.hold_limit = {vid: None for vid in self.VEH_IDS}
        
        # ================================================
        # [STATE VARIABLES]
        # ================================================
        self.yaw = {vid: 0.0 for vid in self.VEH_IDS}
        self.pose = {vid: None for vid in self.VEH_IDS}
        self.raw = {vid: Accel() for vid in self.VEH_IDS}
        self.last_pose = {vid: None for vid in self.VEH_IDS}
        self.v_est = {vid: self.V_NOM for vid in self.VEH_IDS}
        
        # Zone states
        self.zones = {
            "TOP": self._make_zone_state(self.TOP_CENTER),
            "BOT": self._make_zone_state(self.BOT_CENTER),
        }
        
        self.cmd_limit = {vid: 99.0 for vid in self.VEH_IDS}
        self.tgt_limit = {vid: None for vid in self.VEH_IDS}
        
        # 4-Way Intersection
        self.FW_CENTER = (-2.3342, 0.0)
        self.FW_RADIUS = 2.2
        self.FW_EXIT_RADIUS = 0.4
        self.FW_HYSTERESIS_N = 10
        self.FW_APPROACH_N = 3
        self.FW_EPS = 0.001
        self.FW_V_NOM = MAX_SPEED
        self.FW_RANK_SPEEDS_2P = [MAX_SPEED, 0.3]
        self.FW_RANK_SPEEDS_3P = [MAX_SPEED, 0.9, 0.3, 0.3]
        
        self.fw = {
            "active": {vid: False for vid in self.VEH_IDS},
            "outside_ticks": {vid: 0 for vid in self.VEH_IDS},
            "prev_dist": {vid: None for vid in self.VEH_IDS},
            "approach_cnt": {vid: 0 for vid in self.VEH_IDS},
            "approaching": {vid: False for vid in self.VEH_IDS},
        }
        
        # Collision cases (direction-based)
        self.FW_CASES = [
            frozenset([(1, "N"), (2, "W"), (3, "E")]), 
            frozenset([(1, "S"), (2, "E"), (4, "W")]),
            frozenset([(1, "N"), (2, "W")]), 
            frozenset([(1, "N"), (2, "E")]),
            frozenset([(1, "N"), (3, "E")]), 
            frozenset([(2, "W"), (3, "E")]),
            frozenset([(1, "S"), (2, "W")]), 
            frozenset([(1, "S"), (2, "E")]),
            frozenset([(1, "S"), (4, "W")]), 
            frozenset([(2, "E"), (4, "W")])
        ]
        
        # Lap counter
        self.TARGET_LAPS = TARGET_LAPS
        self.LAP_ENTER_R = 0.35
        self.LAP_EXIT_R = 0.9
        self.MIN_LAP_DIST = 3.0
        self.MIN_LAP_TIME = 3.0
        
        self.start_point = {v: None for v in self.VEH_IDS}
        self.start_inited = {v: False for v in self.VEH_IDS}
        self.lap_cnt = {v: 0 for v in self.VEH_IDS}
        self.lap_armed = {v: False for v in self.VEH_IDS}
        self.last_lap_time = {v: None for v in self.VEH_IDS}
        self.dist_since_lap = {v: 0.0 for v in self.VEH_IDS}
        self.prev_for_dist = {v: None for v in self.VEH_IDS}
        
        # ================================================
        # [SUBSCRIPTIONS & PUBLISHERS]
        # ================================================
        for vid in self.VEH_IDS:
            topic = self.TOPICS[vid]
            self.create_subscription(PoseStamped, topic, self._make_pose_cb(vid), qos)
            self.create_subscription(Accel, cav_accel_raw_topic(vid), self._make_raw_cb(vid), 10)
        
        self.pub = {
            vid: self.create_publisher(Twist, cav_cmd_topic(vid), 10) 
            for vid in self.VEH_IDS
        }
        
        # Main tick
        self.create_timer(self.TICK, self.tick)
        
        self.get_logger().info("✅ Guardian Mux initialized")
    
    # ====================================================
    # [UTILITY FUNCTIONS]
    # ====================================================
    def _make_zone_state(self, c):
        return {
            "CENTER": c,
            "active": {v: False for v in self.VEH_IDS},
            "outside_ticks": {v: 0 for v in self.VEH_IDS},
            "prev_dist": {v: None for v in self.VEH_IDS},
            "approach_cnt": {v: 0 for v in self.VEH_IDS},
            "approaching": {v: False for v in self.VEH_IDS}
        }
    
    def _make_pose_cb(self, vid):
        def cb(msg):
            self.pose[vid] = (float(msg.pose.position.x), float(msg.pose.position.y))
            self.yaw[vid] = float(msg.pose.orientation.z)
            self._update_speed_est(vid, self.pose[vid])
        return cb
    
    def _make_raw_cb(self, vid):
        def cb(msg): 
            self.raw[vid] = msg
        return cb
    
    def _dist(self, p, c): 
        return math.hypot(p[0] - c[0], p[1] - c[1])
    
    def _in_round_box(self, p):
        """Check if in roundabout zone (with cutout)"""
        if not p:
            return False
        x, y = p
        
        x_min, x_max, y_min, y_max = SLOW_ZONES[1]
        in_big = (x_min <= x <= x_max) and (y_min <= y <= y_max)
        
        cx_min = float(SLOW2_CUTOUT["x_min"])
        cx_max = float(SLOW2_CUTOUT["x_max"])
        cy_min = float(SLOW2_CUTOUT["y_min"])
        cy_max = float(SLOW2_CUTOUT["y_max"])
        in_cut = (cx_min <= x <= cx_max) and (cy_min <= y <= cy_max)
        
        return in_big and (not in_cut)
    
    def _update_speed_est(self, vid, p):
        """Estimate vehicle speed from position"""
        prev = self.last_pose[vid]
        if prev is not None:
            d = math.hypot(p[0] - prev[0], p[1] - prev[1])
            v = d / self.TICK
            self.v_est[vid] = 0.35 * v + 0.65 * self.v_est[vid]
        self.last_pose[vid] = p
    
    def _apply_limit_ramp(self, vid, tgt, force_immediate=False, release_to=99.0):
        """Apply speed limit with ramping (개선: 가속 빠르게)"""
        if force_immediate:
            if tgt is None:
                self.cmd_limit[vid] = float(release_to)
            else:
                self.cmd_limit[vid] = max(self.MIN_SPEED, float(tgt))
            return
        
        cur = float(self.cmd_limit[vid])
        
        if cur > 50.0:
            cur = float(self.raw[vid].linear.x) if self.raw[vid] is not None else self.V_NOM
        
        step_down = self.RAMP_DOWN_PER_SEC * self.TICK
        step_up = self.RAMP_UP_PER_SEC * self.TICK  # 0.60 (2배 빠름)
        
        tgt2 = float(release_to) if tgt is None else float(tgt)
        
        if tgt2 > cur:
            cur = min(tgt2, cur + step_up)
        else:
            cur = max(tgt2, cur - step_down)
        
        self.cmd_limit[vid] = max(self.MIN_SPEED, float(cur))
    
    def _rank_by_ttc(self, zone_name, vids):
        """Rank vehicles by time-to-collision"""
        z = self.zones[zone_name]
        c = z["CENTER"]
        scored = []
        
        for vid in vids:
            p = self.pose[vid]
            if p is None: 
                continue
            dist = self._dist(p, c)
            v = max(0.05, float(self.v_est[vid]))
            ttc = dist / v
            scored.append((ttc, vid))
        
        scored.sort(key=lambda x: x[0])
        return [vid for _, vid in scored]
    
    def _fw_dist(self, vid):
        """Distance to 4-way intersection center"""
        p = self.pose[vid]
        return math.hypot(p[0] - self.FW_CENTER[0], p[1] - self.FW_CENTER[1]) if p else None
    
    def _fw_get_direction(self, vid):
        """Get approach direction for 4-way intersection"""
        p = self.pose[vid]
        if not p: 
            return None
        dx = p[0] - self.FW_CENTER[0]
        dy = p[1] - self.FW_CENTER[1]
        if abs(dx) > abs(dy): 
            return "E" if dx > 0 else "W"
        else: 
            return "N" if dy > 0 else "S"
    
    # ====================================================
    # [ZONE LOGIC]
    # ====================================================
    def _update_zone_flags(self, zone_name):
        """Update zone active/approaching flags"""
        z = self.zones[zone_name]
        c = z["CENTER"]
        
        for vid in self.VEH_IDS:
            p = self.pose[vid]
            if not p: 
                continue
            
            d = self._dist(p, c)
            
            # Active flag
            if d < self.RADIUS:
                z["active"][vid] = True
                z["outside_ticks"][vid] = 0
            elif d > self.EXIT_RADIUS:
                if z["active"][vid]:
                    z["outside_ticks"][vid] += 1
                    if z["outside_ticks"][vid] >= self.HYSTERESIS_N:
                        z["active"][vid] = False
                        z["outside_ticks"][vid] = 0
            
            # Approaching flag (distance decreasing)
            prev = z["prev_dist"][vid]
            if prev is None:
                z["prev_dist"][vid] = d
                z["approach_cnt"][vid] = 0
                z["approaching"][vid] = False
            else:
                if d < prev - self.EPS:
                    z["approach_cnt"][vid] += 1
                else:
                    z["approach_cnt"][vid] = 0
                
                z["approaching"][vid] = (z["approach_cnt"][vid] >= self.APPROACH_N)
                z["prev_dist"][vid] = d
    
    def _compute_zone_limits(self, zone_name):
        """Compute speed limits for zone (roundabout)"""
        z = self.zones[zone_name]
        in_eff = [v for v in self.VEH_IDS if (self.pose[v] and z["active"][v] and z["approaching"][v])]
        
        # Algo only activates for 2+ vehicles (exclude CAV1+2 pair)
        algo_on = (len(in_eff) >= 2) and not (set(in_eff) == {1, 2})
        
        limits = {v: None for v in self.VEH_IDS}
        if not algo_on: 
            return limits, in_eff, False
        
        rank = self._rank_by_ttc(zone_name, in_eff)
        n = len(rank)
        speeds = self.RANK_SPEEDS_2P if n == 2 else self.RANK_SPEEDS_3P
        
        # Special case: CAV1+2 first
        if n >= 3:
            if set(rank[:2]) == {1, 2}:
                speeds = [self.V_NOM, self.V_NOM] + speeds[2:]
        
        for i, vid in enumerate(rank):
            des = speeds[min(i, len(speeds) - 1)]
            limits[vid] = des if des < self.V_NOM else None
        
        return limits, in_eff, True
    
    # ====================================================
    # [LAP COUNTER]
    # ====================================================
    def _update_laps(self):
        """Update lap counter for all vehicles"""
        now = self.get_clock().now().nanoseconds / 1e9
        
        for vid in self.VEH_IDS:
            p = self.pose.get(vid)
            if p is None:
                continue
            
            # Initialize start point
            if not self.start_inited[vid]:
                self.start_point[vid] = (p[0], p[1])
                self.start_inited[vid] = True
                self.prev_for_dist[vid] = (p[0], p[1])
                self.dist_since_lap[vid] = 0.0
                self.lap_armed[vid] = False
                continue
            
            sp = self.start_point[vid]
            d = math.hypot(p[0] - sp[0], p[1] - sp[1])
            
            # Accumulate distance
            prev = self.prev_for_dist[vid]
            if prev is not None:
                self.dist_since_lap[vid] += math.hypot(p[0] - prev[0], p[1] - prev[1])
            self.prev_for_dist[vid] = (p[0], p[1])
            
            # Arm when leaving start area
            if d > self.LAP_EXIT_R:
                self.lap_armed[vid] = True
            
            # Count lap when returning to start area
            if self.lap_armed[vid] and d < self.LAP_ENTER_R:
                ok_time = True
                if self.last_lap_time[vid] is not None:
                    ok_time = (now - self.last_lap_time[vid]) >= self.MIN_LAP_TIME
                
                ok_dist = self.dist_since_lap[vid] >= self.MIN_LAP_DIST
                
                if ok_time and ok_dist:
                    self.lap_cnt[vid] += 1
                    self.last_lap_time[vid] = now
                    self.dist_since_lap[vid] = 0.0
                    self.lap_armed[vid] = False
                    
                    self.get_logger().info(
                        f"🏁 CAV{vid:02d} completed lap {self.lap_cnt[vid]}/{self.TARGET_LAPS}"
                    )
                    
                    # Check if all completed
                    if all(self.lap_cnt[v] >= self.TARGET_LAPS for v in self.VEH_IDS):
                        self.get_logger().info("✅ ALL VEHICLES COMPLETED 5 LAPS!")
    
    # ====================================================
    # [MAIN TICK]
    # ====================================================
    def tick(self):
        # Check if any vehicle has pose
        if all(self.pose[v] is None for v in self.VEH_IDS):
            return
        
        # Update lap counter
        self._update_laps()
        
        # Determine roundabout status per vehicle
        in_round = {}
        for vid in self.VEH_IDS:
            p = self.pose.get(vid)
            in_round[vid] = (p is not None) and self._in_round_box(p)
        
        # ================================================
        # [1] UPDATE ZONE LOGIC (Roundabout)
        # ================================================
        self._update_zone_flags("TOP")
        self._update_zone_flags("BOT")
        top_lim, top_eff, top_on = self._compute_zone_limits("TOP")
        bot_lim, bot_eff, bot_on = self._compute_zone_limits("BOT")
        
        # ================================================
        # [2] UPDATE 4-WAY INTERSECTION LOGIC
        # ================================================
        fw_lim = {v: None for v in self.VEH_IDS}
        
        for vid in self.VEH_IDS:
            # Skip if in roundabout
            if in_round.get(vid, False):
                self.fw["approach_cnt"][vid] = 0
                self.fw["approaching"][vid] = False
                self.fw["prev_dist"][vid] = None
                continue
            
            p = self.pose[vid]
            if not p:
                continue
            
            d = self._fw_dist(vid)
            if d is None:
                continue
            
            # Active flag
            if d < self.FW_RADIUS:
                self.fw["active"][vid] = True
                self.fw["outside_ticks"][vid] = 0
            elif d > self.FW_EXIT_RADIUS:
                if self.fw["active"][vid]:
                    self.fw["outside_ticks"][vid] += 1
                    if self.fw["outside_ticks"][vid] >= self.FW_HYSTERESIS_N:
                        self.fw["active"][vid] = False
                        self.fw["outside_ticks"][vid] = 0
            
            # Approaching flag
            prev = self.fw["prev_dist"][vid]
            if prev is None:
                self.fw["prev_dist"][vid] = d
                self.fw["approach_cnt"][vid] = 0
                self.fw["approaching"][vid] = False
            else:
                if d < prev - self.FW_EPS:
                    self.fw["approach_cnt"][vid] += 1
                else:
                    self.fw["approach_cnt"][vid] = 0
                
                self.fw["approaching"][vid] = (self.fw["approach_cnt"][vid] >= self.FW_APPROACH_N)
                self.fw["prev_dist"][vid] = d
        
        # Get effective vehicles (excluding those in roundabout)
        fw_eff = [
            v for v in self.VEH_IDS
            if self.pose[v]
            and self.fw["active"][v]
            and self.fw["approaching"][v]
            and (not in_round.get(v, False))
        ]
        
        # Match collision patterns
        fw_map = {}
        for v in fw_eff:
            dd = self._fw_get_direction(v)
            if dd:
                fw_map[v] = dd
        
        pairs = set((v, d) for v, d in fw_map.items())
        matched = None
        for c in self.FW_CASES:
            if c.issubset(pairs):
                matched = c
                break
        
        if matched:
            targs = [v for v, _ in matched]
            targs.sort(key=lambda v: self._fw_dist(v) or 1e9)
            n = len(targs)
            spd = self.FW_RANK_SPEEDS_2P if n == 2 else self.FW_RANK_SPEEDS_3P
            for i, vid in enumerate(targs):
                des = spd[min(i, len(spd) - 1)]
                fw_lim[vid] = des if des < self.FW_V_NOM else None
        
        # ================================================
        # [3] MERGE LIMITS & APPLY
        # ================================================
        for vid in self.VEH_IDS:
            # Collect all limits
            cands = []
            if top_lim.get(vid) is not None: 
                cands.append(float(top_lim[vid]))
            if bot_lim.get(vid) is not None: 
                cands.append(float(bot_lim[vid]))
            if fw_lim.get(vid) is not None: 
                cands.append(float(fw_lim[vid]))
            
            self.tgt_limit[vid] = min(cands) if cands else None
            
            # Hold logic (prevent flickering)
            if self.tgt_limit[vid] is not None:
                self.hold_limit[vid] = float(self.tgt_limit[vid])
                self.hold_cnt[vid] = self.HOLD_TICKS
            else:
                if self.hold_cnt[vid] > 0 and self.hold_limit[vid] is not None:
                    self.tgt_limit[vid] = float(self.hold_limit[vid])
                    self.hold_cnt[vid] -= 1
                else:
                    self.hold_limit[vid] = None
            
            # Roundabout: immediate release
            if in_round.get(vid, False):
                self.cmd_limit[vid] = 99.0
                continue
            
            # Apply ramp
            self._apply_limit_ramp(
                vid,
                self.tgt_limit[vid],
                force_immediate=False,
                release_to=self.V_NOM
            )
        
        # ================================================
        # [4] PUBLISH COMMANDS
        # ================================================
        for vid in self.VEH_IDS:
            src = self.raw[vid]
            
            raw_v = float(src.linear.x)
            steer = float(src.angular.z)
            
            lim = float(self.cmd_limit[vid])
            out_v = min(raw_v, lim)
            
            out = Twist()
            out.linear.x = float(out_v)
            out.angular.z = float(steer)
            self.pub[vid].publish(out)

# ============================================================
# [MAIN EXECUTION]
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    
    # Create driver nodes
    # 원본 RoundController 인자 순서:
    # (vid, path, start_zone, start_trigger, out_zone, danger_zone, ...)
    p = lambda name: os.path.join(PATH_DIR, name)
    drivers = [
        UnifiedVehicleDriver(
            vehicle_id=1, 
            path_filename=p("path3_1.json"),
            start_zone_file=None,                # 원본: None
            start_trigger_file=p("path_hv_3_1.csv"),
            out_zone_file=p("path3_1_out_zone.csv"),  # 원본: 있었으나 현재 미사용
            danger_zone_file=p("path_hv_3_2.csv")
        ),
        UnifiedVehicleDriver(
            vehicle_id=2,
            path_filename=p("path3_2.json"),
            start_zone_file=None,                # 원본: None
            start_trigger_file=p("path_hv_2_1.csv"),
            out_zone_file=p("path3_2_out_zone.csv"),  # 원본: 있었으나 현재 미사용
            danger_zone_file=p("path_hv_2_2.csv")
        ),
        UnifiedVehicleDriver(
            vehicle_id=3,
            path_filename=p("path3_3.json"),
            start_zone_file=None,                # 원본: None
            start_trigger_file=p("path_hv_2_1.csv"),
            out_zone_file=None,                  # 원본: None (CAV 3,4는 Smart ACC만)
            danger_zone_file=p("path_hv_2_2.csv")
        ),
        UnifiedVehicleDriver(
            vehicle_id=4,
            path_filename=p("path3_4.json"),
            start_zone_file=None,                # 원본: None
            start_trigger_file=p("path_hv_3_1.csv"),
            out_zone_file=None,                  # 원본: None
            danger_zone_file=p("path_hv_3_2.csv")
        ),
    ]
    
    # Create guardian mux
    guardian = GuardianMux()
    
    # Multi-threaded executor
    ex = MultiThreadedExecutor(num_threads=10)
    for d in drivers: 
        ex.add_node(d)
    ex.add_node(guardian)
    
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        for d in drivers: 
            d.destroy_node()
        guardian.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

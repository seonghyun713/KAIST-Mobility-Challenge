#!/usr/bin/env python3
import os
import math
import json
import csv

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Accel, PoseStamped, Twist
from round import VehicleController as RoundController


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DIR = os.path.join(BASE_DIR, "path")

# ============================================================
# [TOPIC CONFIG] Change only here at the venue
# ============================================================

# Logical IDs used inside the algorithm (DO NOT change)
CAV_LOGICAL_IDS = [1, 2, 3, 4]

# Map: logical CAV id -> real topic number at the venue
# Example: CAV1 uses /CAV_07, CAV2 uses /CAV_09, ...
CAV_TOPIC_NUM = {
    1: 28,
    2: 6,
    3: 7,
    4: 8,
}

# HV topics (role names -> actual topic name)
# If HV topic numbers change, edit here.
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

def cav_accel_round_raw_topic(logical_id: int) -> str:
    return f"{cav_topic(logical_id)}_accel_round_raw"

def hv_topic(role: str) -> str:
    return HV_TOPICS[role]




# ============================================================
# [1] Slow Zones Configuration (Fixed at 0.7 m/s)
# ============================================================
RAW_SLOW_ZONES = [
    (-1.0, -4.6,  2.0, -2.0), # 사지교차로 범위
    (-1.0,  2.7,  2.2, -1.1), # 회전교차로 범위
]

# Normalize coordinates (min, max) for safety
SLOW_ZONES = [
    (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
    for (x1, x2, y1, y2) in RAW_SLOW_ZONES
]

SLOW2_CUTOUT = {
    "x_min": -1.00,
    "x_max": 0.37,
    "y_min":  0.00,
    "y_max":  2.20,
}

SLOW_PARAMS = {
    "vel": 1.5,          
    "look_ahead": 0.50, 
    "kp": 6.0,          
    "ki": 0.045,
    "kd": 1.0,          
    "k_cte": 4.0
}
MAX_SPEED = 1.5

# ============================================================
# [2] Speed Profiles (3-Stage Transmission)
# ============================================================

# 1. Hard Curve (Low Speed, High Gain)
HARD_PARAMS = {
    "vel": 1.0,
    "look_ahead": 0.60,
    "kp": 6.0,
    "ki": 0.045,
    "kd": 1.0,
    "k_cte": 3.0
}

# 2. Easy Curve (Medium Speed)
EASY_PARAMS = {
    "vel": 1.3,
    "look_ahead": 0.65, 
    "kp": 6.0,
    "ki": 0.05,
    "kd": 1.0,
    "k_cte": 3.0
}

# 3. Straight (High Speed, Stability Focused)
STRAIGHT_PARAMS = {
    "vel": 1.8,
    "look_ahead": 1.2,  # Increased for high speed
    "kp": 2.0,          # Reduced to prevent oscillation
    "ki": 0.002,        # Minimize integral windup
    "kd": 2.5,          # Increased damping
    "k_cte": 1.0
}

# Vehicle Specs
WHEELBASE = 0.211
DIST_CENTER_TO_REAR = WHEELBASE / 2.0
TICK_RATE = 0.05      # 50 Hz
ACCEL_LIMIT = 3.0
DECEL_LIMIT = 1.5

# active lookahead min
MIN_LA = 0.5

# ============================================================
# [COMMON: File Loaders]
# ============================================================
def load_path_points(json_file):
    if not os.path.exists(json_file): return []
    with open(json_file, "r") as f: data = json.load(f)
    xs = data.get("x") or data.get("X")
    ys = data.get("y") or data.get("Y")
    if not xs or not ys: return []
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def load_zone_from_csv(filename):
    points = []
    if not filename or not os.path.exists(filename): return []
    try:
        with open(filename, "r") as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                if row and len(row) >= 2:
                    try: points.append((float(row[0]), float(row[1])))
                    except: continue
    except: pass
    return points


# ============================================================
# [NODE] Map Prediction Driver (Lateral Control)
# ============================================================
class MapPredictionDriver(Node):
    def __init__(self, vehicle_id: int, path_filename: str):
        super().__init__(f"driver_vehicle_{vehicle_id}")
        self.vid = int(vehicle_id)
        self.PATH_FILENAME = path_filename
        self.logical_id = int(vehicle_id)
        self.TOPIC = cav_topic(self.logical_id)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # =========================
        # LPF for D-term (yaw_err)
        # =========================
        self.yaw_err_f = 0.0        # filtered yaw error
        self.prev_yaw_err_f = 0.0   # prev filtered yaw error

        # 1차 LPF time constant (seconds) - 튜닝 파라미터
        # 0.10~0.30 정도부터 시작 추천 (TICK=0.05 기준)
        self.LPF_TAU = 0.18

        # (옵션) D-term 자체 LPF
        self.dterm_f = 0.0
        self.D_LPF_TAU = 0.10


        # Load Path
        self.path_pts = load_path_points(self.PATH_FILENAME)
        self.path_x = [p[0] for p in self.path_pts]
        self.path_y = [p[1] for p in self.path_pts]

        if not self.path_pts:
            self.get_logger().error(f"❌ [Car{self.vid}] Path missing: {self.PATH_FILENAME}")
        else:
            self.get_logger().info(f"✅ [Car{self.vid}] Loaded {len(self.path_pts)} pts (Ring Buffer Mode).")

        # ROS Comm
        self.create_subscription(PoseStamped, self.TOPIC, self.pose_callback, qos_profile)
        self.accel_raw_pub = self.create_publisher(Accel, cav_accel_raw_topic(self.logical_id), 10)

        # State Variables
        self.curr_x, self.curr_y, self.curr_yaw = 0.0, 0.0, 0.0
        self.got_pose = False

        
        self.prev_err = 0.0
        self.int_err = 0.0
        self.last_time = self.get_clock().now()
        self._skip_dterm = 0
        
        self.current_vel_cmd = 1.0
        self.mode = "HARD"
        self.avg_steer_signed = 0.0 
        self.log_counter = 0

        self.old_nearest_idx = 0

        self.create_timer(TICK_RATE, self.drive_loop)


    def pose_callback(self, msg):
        self.got_pose = True
        self.curr_yaw = float(msg.pose.orientation.z)
        # Convert Center-of-Mass to Rear-Axle position
        self.curr_x = float(msg.pose.position.x) - (DIST_CENTER_TO_REAR * math.cos(self.curr_yaw))
        self.curr_y = float(msg.pose.position.y) - (DIST_CENTER_TO_REAR * math.sin(self.curr_yaw))

    def is_in_slow_zone(self, x, y):
        # -------------------------
        # Slow zone 1 (기존 그대로)
        # -------------------------
        x1_min, x1_max, y1_min, y1_max = SLOW_ZONES[0]
        in_zone1 = (x1_min <= x <= x1_max) and (y1_min <= y <= y1_max)

        # -------------------------
        # Slow zone 2 (cutout 적용)
        # -------------------------
        x2_min, x2_max, y2_min, y2_max = SLOW_ZONES[1]
        in_big2 = (x2_min <= x <= x2_max) and (y2_min <= y <= y2_max)

        # 제거할 사각형 (좌표 기반)
        cx_min = SLOW2_CUTOUT["x_min"]
        cx_max = SLOW2_CUTOUT["x_max"]
        cy_min = SLOW2_CUTOUT["y_min"]
        cy_max = SLOW2_CUTOUT["y_max"]

        in_cutout = (cx_min <= x <= cx_max) and (cy_min <= y <= cy_max)

        in_zone2 = in_big2 and (not in_cutout)

        return in_zone1 or in_zone2

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def drive_loop(self):
        if not self.got_pose or not self.path_pts: return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0.001 or dt > 0.05: return

        # --------------------------------------------------------
        # [Step 1] Zone & Parameter Selection
        # --------------------------------------------------------
        in_zone = self.is_in_slow_zone(self.curr_x, self.curr_y)
        
        if in_zone:
            params = SLOW_PARAMS
            current_mode_str = "ZONE"
            self.mode = "HARD" 
            self.avg_steer_signed = 0.0
            self.int_err = 0.0
        else:
            if self.mode == "HARD": params = HARD_PARAMS
            elif self.mode == "EASY": params = EASY_PARAMS
            else: params = STRAIGHT_PARAMS
            current_mode_str = self.mode

        # --------------------------------------------------------
        # [Step 2] Pure Pursuit (Ring Buffer Search)
        # --------------------------------------------------------
        path_len = len(self.path_pts)
        min_d = float('inf')
        curr_idx = self.old_nearest_idx
        
        # Local search window (-50 to +50 points)
        search_range = 50 
        found_in_window = False

        for offset in range(-search_range, search_range):
            check_idx = (self.old_nearest_idx + offset) % path_len
            
            d = math.hypot(self.path_x[check_idx] - self.curr_x, self.path_y[check_idx] - self.curr_y)
            if d < min_d:
                min_d = d
                curr_idx = check_idx
                found_in_window = True

        # Global search fallback if off-track or initialized
        if not found_in_window or min_d > 5.0:
            min_d = float('inf')
            for i in range(path_len):
                d = math.hypot(self.path_x[i] - self.curr_x, self.path_y[i] - self.curr_y)
                if d < min_d: min_d = d; curr_idx = i
        
        self.old_nearest_idx = curr_idx

        # --------------------------------------------------------
        # ✅ [Jump Detection] off-track이면 글로벌 재탐색 + integrator reset + D-term skip
        # --------------------------------------------------------
        if min_d > 0.8:
            best_i = 0
            best_d = float("inf")
            for i in range(path_len):
                d = math.hypot(self.path_x[i] - self.curr_x, self.path_y[i] - self.curr_y)
                if d < best_d:
                    best_d = d
                    best_i = i

            curr_idx = best_i
            self.old_nearest_idx = curr_idx
            self.int_err = 0.0
            self._skip_dterm = 1
            min_d = best_d


        # Find Look-ahead Point
        active_look_ahead = min(params["look_ahead"], self.current_vel_cmd * 0.45)
        
        
        target_idx = curr_idx
        for i in range(path_len):
            idx = (curr_idx + i) % path_len
            d = math.hypot(self.path_x[idx] - self.curr_x, self.path_y[idx] - self.curr_y)
            if d >= active_look_ahead:
                target_idx = idx
                break
        
        tx, ty = self.path_x[target_idx], self.path_y[target_idx]

        # --------------------------------------------------------
        # [Step 3] Vector CTE + PID Controller
        # --------------------------------------------------------
        desired_yaw = math.atan2(ty - self.curr_y, tx - self.curr_x)
        yaw_err = self.normalize_angle(desired_yaw - self.curr_yaw)

        # --------------------------------------------------------
        # ✅ [LPF] Filter yaw_err to stabilize D-term against pose noise
        # --------------------------------------------------------
        alpha = dt / (self.LPF_TAU + dt)  # 1st-order LPF coefficient
        # angle wrapping 고려: 필터는 "차이"로 업데이트하는 게 안전
        err_delta = self.normalize_angle(yaw_err - self.yaw_err_f)
        self.yaw_err_f = self.normalize_angle(self.yaw_err_f + alpha * err_delta)


        # Calculate Cross Track Error (CTE)
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


        # # Calculate Cross Track Error (CTE) - tangent based (stable sign)
        # path_len = len(self.path_pts)
        # next_idx = (curr_idx + 1) % path_len

        # px, py = self.path_x[curr_idx], self.path_y[curr_idx]
        # nx, ny = self.path_x[next_idx], self.path_y[next_idx]

        # # # Tangent vector of path at curr_idx
        # t_x = nx - px
        # t_y = ny - py
        # t_norm = math.hypot(t_x, t_y)

        # if t_norm < 1e-6:
        #     cte = 0.0
        # else:
        #     # Vector from path point to car
        #     c_x = self.curr_x - px
        #     c_y = self.curr_y - py

        #     # signed distance from path (left/right)
        #     signed_cte = (t_x * c_y - t_y * c_x) / t_norm

        #     # clamp to suppress spikes from localization jitter
        #     cte_clip = 0.25 if self.mode in ["HARD","EASY"] else 0.6
        #     signed_cte = max(-cte_clip, min(cte_clip, signed_cte))


        #     cte = signed_cte * params["k_cte"]

        #     # ✅ 직진(STRATGT)에서만 CTE kill
        #     if self.mode == "STRAIGHT":
        #         cte = 0.0





        # PID Calculation
        self.int_err = max(-2.0, min(2.0, self.int_err + yaw_err * dt))
        
        p = params["kp"] * yaw_err
        i_term = params["ki"] * self.int_err
        if self._skip_dterm > 0:
            d_term_raw = 0.0
            self._skip_dterm -= 1
        else:
            d_error_f = self.normalize_angle(self.yaw_err_f - self.prev_yaw_err_f)
            d_term_raw = params["kd"] * d_error_f / dt

        self.prev_yaw_err_f = self.yaw_err_f

        # (옵션) D-term LPF 한번 더 (더 부드럽게)
        d_alpha = dt / (self.D_LPF_TAU + dt)
        self.dterm_f = (1.0 - d_alpha) * self.dterm_f + d_alpha * d_term_raw
        d_term = self.dterm_f




        final_steer = max(-2.0, min(2.0, float(p + i_term + d_term + cte)))
        self.prev_err = yaw_err

        # --------------------------------------------------------
        # [Step 4] Mode Switching & Speed Control
        # --------------------------------------------------------
        if in_zone:
            target_v = SLOW_PARAMS["vel"]
        else:
            # Low-pass filter on steering
            self.avg_steer_signed = 0.85 * self.avg_steer_signed + 0.15 * final_steer
            filter_val = abs(self.avg_steer_signed)

            next_mode = self.mode 
            if abs(final_steer) > 0.90:
                next_mode = "HARD"
                # self.avg_steer_signed 강제 주입 삭제

            else:
                if self.mode == "STRAIGHT":
                    if filter_val > 0.30: next_mode = "EASY"
                elif self.mode == "EASY":
                    if filter_val < 0.15: next_mode = "STRAIGHT"
                    elif filter_val > 0.80: next_mode = "HARD"
                elif self.mode == "HARD":
                    if filter_val < 0.70: next_mode = "EASY"
            
            # Anti-Windup on mode transition
            if (self.mode == "HARD" and next_mode == "EASY") or \
               (self.mode == "EASY" and next_mode == "STRAIGHT"):
                self.int_err = 0.0

            self.mode = next_mode
            if self.mode == "HARD": target_v = HARD_PARAMS["vel"]
            elif self.mode == "EASY": target_v = EASY_PARAMS["vel"]
            else: target_v = STRAIGHT_PARAMS["vel"]

        # Speed Ramp
        if target_v > self.current_vel_cmd:
            self.current_vel_cmd = min(target_v, self.current_vel_cmd + ACCEL_LIMIT * dt)
        else:
            self.current_vel_cmd = max(target_v, self.current_vel_cmd - DECEL_LIMIT * dt)

        cmd = Accel()
        cmd.linear.x = float(self.current_vel_cmd)
        cmd.angular.z = float(final_steer)
        self.accel_raw_pub.publish(cmd)


# ============================================================
# [NODE] Dual Zone Guardian Mux (Safety Controller)
# ============================================================
class Problem3DualZoneGuardianMux(Node):
    def __init__(self):
        super().__init__("problem3_dualzone_guardian_mux")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.VEH_IDS = CAV_LOGICAL_IDS
        self.TOPICS = {vid: cav_topic(vid) for vid in self.VEH_IDS}

        # Parameters
        self.V_NOM = MAX_SPEED
        self.RANK_SPEEDS_3P = [MAX_SPEED, 0.9, 0.3, 0.3]
        self.RANK_SPEEDS_2P = [MAX_SPEED, 0.3]

        self.TOP_CENTER = (-2.3342, 2.3073)
        self.BOT_CENTER = (-2.3342, -2.3073)
        self.RADIUS = 1.5
        self.EXIT_RADIUS = 0.4
        self.APPROACH_N = 3
        self.EPS = 0.001
        self.HYSTERESIS_N = 5

        self.TICK = 0.02
        self.RAMP_DOWN_PER_SEC = 1.5
        self.RAMP_UP_PER_SEC = 0.3
        self.STOP_VELOCITY = 0.0
        self.MIN_SPEED = self.STOP_VELOCITY

        # =========================
        # HOLD (감속 유지) 설정
        # =========================
        self.HOLD_TICKS = 12  # 감속 유지 시간 (tick 단위)
        self.hold_cnt = {vid: 0 for vid in self.VEH_IDS}
        self.hold_limit = {vid: None for vid in self.VEH_IDS}  # 마지막으로 걸린 제한값 저장


        self.yaw = {vid: 0.0 for vid in self.VEH_IDS}


        # States
        self.pose = {vid: None for vid in self.VEH_IDS}
        self.raw = {vid: Accel() for vid in self.VEH_IDS}
        self.round_raw = {vid: Accel() for vid in self.VEH_IDS}
        self.last_pose = {vid: None for vid in self.VEH_IDS}
        self.v_est = {vid: self.V_NOM for vid in self.VEH_IDS}

        self.zones = {
            "TOP": self._make_zone_state(self.TOP_CENTER),
            "BOT": self._make_zone_state(self.BOT_CENTER),
        }
        self.cmd_limit = {vid: 99.0 for vid in self.VEH_IDS}
        self.tgt_limit = {vid: None for vid in self.VEH_IDS}

        for vid in self.VEH_IDS:
            topic = self.TOPICS[vid]
            self.create_subscription(PoseStamped, topic, self._make_pose_cb(vid), qos)
            self.create_subscription(Accel, cav_accel_raw_topic(vid), self._make_raw_cb(vid), 10)
            self.create_subscription(Accel, cav_accel_round_raw_topic(vid), self._make_round_raw_cb(vid), 10)

        self.pub = {vid: self.create_publisher(Twist, cav_cmd_topic(vid), 10) for vid in self.VEH_IDS}
        self._log_counter = 0

        # 4-Way Intersection Settings
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
        # Defined Collision Pairs (Direction-based)
        self.FW_CASES = [
            frozenset([(1, "N"), (2, "W"), (3, "E")]), frozenset([(1, "S"), (2, "E"), (4, "W")]),
            frozenset([(1, "N"), (2, "W")]), frozenset([(1, "N"), (2, "E")]),
            frozenset([(1, "N"), (3, "E")]), frozenset([(2, "W"), (3, "E")]),
            frozenset([(1, "S"), (2, "W")]), frozenset([(1, "S"), (2, "E")]),
            frozenset([(1, "S"), (4, "W")]), frozenset([(2, "E"), (4, "W")])
        ]

        # Human Vehicle (HV) Safety Settings
        self.TARGET_VELOCITY = 0.5; self.ZONE_RADIUS = 0.20; self.HV_DETECT_RADIUS = 0.12; self.RESET_DISTANCE = 2.2
        self.hv19 = None; self.hv20 = None; self.hv19_active = False; self.hv20_active = False
        
        self.create_subscription(PoseStamped, hv_topic("HV1"), self._cb_hv19, qos)
        self.create_subscription(PoseStamped, hv_topic("HV2"), self._cb_hv20, qos)

        self.safety_cfg = {
            1: {
                "start_zone": [(0.5,-0.108333)]
                "start_trigger": load_zone_from_csv(os.path.join(PATH_DIR, "path_hv_3_1.csv")),
                "out_zone": load_zone_from_csv(os.path.join(PATH_DIR, "path3_1_out_zone.csv")),
                "danger_zone": load_zone_from_csv(os.path.join(PATH_DIR, "path_hv_3_2.csv")),
                "stop_logic_disabled": False
            },
            2: {
                "start_zone": (1.558333,1.2)]
                "start_trigger": load_zone_from_csv(os.path.join(PATH_DIR, "path_hv_2_1.csv")),
                "out_zone": load_zone_from_csv(os.path.join(PATH_DIR, "path3_2_out_zone.csv")),
                "danger_zone": load_zone_from_csv(os.path.join(PATH_DIR, "path_hv_2_2.csv")),
                "stop_logic_disabled": False
            },
            3: {
                "start_zone": [(1.308202,1.35)]
                "start_trigger": load_zone_from_csv(os.path.join(PATH_DIR, "path_hv_2_1.csv")),
                "out_zone": [],
                "danger_zone": [],
                "stop_logic_disabled": False
            },
            4: {
                "start_zone": [(0.35,-0.35833)]
                "start_trigger": load_zone_from_csv(os.path.join(PATH_DIR, "path_hv_3_1.csv")),
                "out_zone": [],
                "danger_zone": [],
                "stop_logic_disabled": False
            },
        }

        self.create_timer(self.TICK, self.tick)

        # =========================
        # LAP COUNTER (auto start point)
        # =========================
        self.TARGET_LAPS = 5

        self.LAP_ENTER_R = 0.35
        self.LAP_EXIT_R  = 0.9
        self.MIN_LAP_DIST = 3.0
        self.MIN_LAP_TIME = 3.0

        self.start_point = {v: None for v in self.VEH_IDS}      # (x,y) first pose
        self.start_inited = {v: False for v in self.VEH_IDS}

        self.lap_cnt = {v: 0 for v in self.VEH_IDS}
        self.lap_armed = {v: False for v in self.VEH_IDS}       # must go OUT once
        self.last_lap_time = {v: None for v in self.VEH_IDS}
        self.dist_since_lap = {v: 0.0 for v in self.VEH_IDS}
        self.prev_for_dist = {v: None for v in self.VEH_IDS}
        self.LAP_LOG_PERIOD = 20   # tick 기준 (20 * 0.05s = 1초)
        self._lap_log_cnt = 0
        




    # --- Callbacks & Helpers ---
    def _cb_hv19(self, msg): self.hv19 = (float(msg.pose.position.x), float(msg.pose.position.y)); self.hv19_active = True
    def _cb_hv20(self, msg): self.hv20 = (float(msg.pose.position.x), float(msg.pose.position.y)); self.hv20_active = True
    def _make_zone_state(self, c): return {"CENTER": c, "active": {v:False for v in self.VEH_IDS}, "outside_ticks": {v:0 for v in self.VEH_IDS}, "prev_dist": {v:None for v in self.VEH_IDS}, "approach_cnt": {v:0 for v in self.VEH_IDS}, "approaching": {v:False for v in self.VEH_IDS}}
    
    def _make_pose_cb(self, vid):
        def cb(msg):
            self.pose[vid] = (float(msg.pose.position.x), float(msg.pose.position.y))
            self.yaw[vid]  = float(msg.pose.orientation.z)  # 니네식 yaw
            self._update_speed_est(vid, self.pose[vid])
        return cb
    
    def _make_raw_cb(self, vid):
        def cb(msg): self.raw[vid] = msg
        return cb
    
    def _make_round_raw_cb(self, vid):
        def cb(msg):
            self.round_raw[vid] = msg
        return cb


    
    def _dist(self, p, c): return math.hypot(p[0] - c[0], p[1] - c[1])
    
    def _in_round_box(self, p):
        if not p:
            return False
        x, y = p

        # big round box (기존)
        x_min, x_max, y_min, y_max = SLOW_ZONES[1]   # (-0.5, 2.7, -1.4, 1.8) 와 동일
        in_big = (x_min <= x <= x_max) and (y_min <= y <= y_max)

        # cutout (좌표 기반)
        cx_min = float(SLOW2_CUTOUT["x_min"])
        cx_max = float(SLOW2_CUTOUT["x_max"])
        cy_min = float(SLOW2_CUTOUT["y_min"])
        cy_max = float(SLOW2_CUTOUT["y_max"])
        in_cut = (cx_min <= x <= cx_max) and (cy_min <= y <= cy_max)

        return in_big and (not in_cut)


    def _update_speed_est(self, vid, p):
        prev = self.last_pose[vid]
        if prev is not None:
            d = math.hypot(p[0]-prev[0], p[1]-prev[1]); v = d/self.TICK; self.v_est[vid] = 0.35*v + 0.65*self.v_est[vid]
        self.last_pose[vid] = p

    def _apply_limit_ramp(self, vid, tgt, force_immediate=False, release_to=99.0):
        # tgt: 제한값(예: 0.1, 0.5 ...) 또는 None(제한 해제)
        # release_to: None 해제 시 어디까지 풀어줄지 (99 or V_NOM)

        if force_immediate:
            # HV 같은 강제 상황은 즉시
            if tgt is None:
                self.cmd_limit[vid] = float(release_to)
            else:
                self.cmd_limit[vid] = max(self.MIN_SPEED, float(tgt))
            return

        cur = float(self.cmd_limit[vid])

        # 초기값(99)일 때는 현재 raw 속도에서 시작하게
        if cur > 50.0:
            cur = float(self.raw[vid].linear.x) if self.raw[vid] is not None else self.V_NOM

        step_down = self.RAMP_DOWN_PER_SEC * self.TICK
        step_up   = self.RAMP_UP_PER_SEC   * self.TICK

        tgt2 = float(release_to) if tgt is None else float(tgt)

        if tgt2 > cur:
            cur = min(tgt2, cur + step_up)
        else:
            cur = max(tgt2, cur - step_down)

        self.cmd_limit[vid] = max(self.MIN_SPEED, float(cur))

    def _rank_by_ttc(self, zone_name, vids):
        z = self.zones[zone_name]; c = z["CENTER"]; scored = []
        for vid in vids:
            p = self.pose[vid]
            if p is None: continue
            dist = self._dist(p, c); v = max(0.05, float(self.v_est[vid])); ttc = dist/v; scored.append((ttc, vid))
        scored.sort(key=lambda x: x[0])
        return [vid for _, vid in scored]

    def _fw_dist(self, vid):
        p = self.pose[vid]; return math.hypot(p[0]-self.FW_CENTER[0], p[1]-self.FW_CENTER[1]) if p else None
    
    def _fw_get_direction(self, vid):
        p = self.pose[vid]
        if not p: return None
        dx = p[0]-self.FW_CENTER[0]; dy = p[1]-self.FW_CENTER[1]
        if abs(dx)>abs(dy): return "E" if dx>0 else "W"
        else: return "N" if dy>0 else "S"

    # --- Zone Logic ---
    def _update_zone_flags(self, zone_name):
        z = self.zones[zone_name]; c = z["CENTER"]
        for vid in self.VEH_IDS:
            p = self.pose[vid]
            if not p: continue
            d = self._dist(p, c)
            if d < self.RADIUS: z["active"][vid]=True; z["outside_ticks"][vid]=0
            elif d > self.EXIT_RADIUS:
                if z["active"][vid]:
                    z["outside_ticks"][vid]+=1
                    if z["outside_ticks"][vid] >= self.HYSTERESIS_N: z["active"][vid]=False; z["outside_ticks"][vid]=0
            prev = z["prev_dist"][vid]
            if prev is None: z["prev_dist"][vid]=d; z["approach_cnt"][vid]=0; z["approaching"][vid]=False
            else:
                if d < prev - self.EPS: z["approach_cnt"][vid]+=1
                else: z["approach_cnt"][vid]=0
                z["approaching"][vid] = (z["approach_cnt"][vid] >= self.APPROACH_N); z["prev_dist"][vid]=d

    def _compute_zone_limits(self, zone_name):
        z = self.zones[zone_name]
        in_eff = [v for v in self.VEH_IDS if (self.pose[v] and z["active"][v] and z["approaching"][v])]
        algo_on = (len(in_eff)>=2) and not (set(in_eff)=={1,2})
        limits = {v:None for v in self.VEH_IDS}
        if not algo_on: return limits, in_eff, False
        rank = self._rank_by_ttc(zone_name, in_eff); n = len(rank)
        speeds = self.RANK_SPEEDS_2P if n==2 else self.RANK_SPEEDS_3P
        if n>=3:
            if set(rank[:2])=={1,2}: speeds = [self.V_NOM, self.V_NOM] + speeds[2:]
        for i, vid in enumerate(rank):
            des = speeds[min(i, len(speeds)-1)]; limits[vid] = des if des < self.V_NOM else None
        return limits, in_eff, True

    # --- HV Safety Logic ---
    def _min_dist_to_points(self, vid, pts):
        p = self.pose.get(vid)
        if not p or not pts: return 999.0
        md = 999.0
        for x,y in pts:
            d = math.hypot(x-p[0], y-p[1])
            if d<md: md=d
        return md
    
    def _hv_in_points(self, pts):
        if not pts: return False, None
        if self.hv19_active and self.hv19:
            for x,y in pts:
                if math.hypot(x-self.hv19[0], y-self.hv19[1]) < self.HV_DETECT_RADIUS: return True, 19
        if self.hv20_active and self.hv20:
            for x,y in pts:
                if math.hypot(x-self.hv20[0], y-self.hv20[1]) < self.HV_DETECT_RADIUS: return True, 20
        return False, None

    def _compute_hv_safety_limit(self, vid):
        cfg = self.safety_cfg.get(vid); 
        if not cfg: return None, False, None
        s_zone = cfg.get("start_zone", []); s_trig = cfg.get("start_trigger", [])
        d_start = self._min_dist_to_points(vid, s_zone)
        if cfg["stop_logic_disabled"]:
            if d_start > self.RESET_DISTANCE: cfg["stop_logic_disabled"] = False
        else:
            if d_start < self.ZONE_RADIUS:
                hit, w = self._hv_in_points(s_trig)
                if hit: cfg["stop_logic_disabled"] = True
                else: return self.STOP_VELOCITY, True, "START_WAIT"
        o_zone = cfg.get("out_zone", []); d_zone = cfg.get("danger_zone", [])
        if o_zone and d_zone:
            if self._min_dist_to_points(vid, o_zone) < self.ZONE_RADIUS:
                hit, w = self._hv_in_points(d_zone)
                if hit: return self.STOP_VELOCITY, True, f"EXIT_YIELD(HV_{w})"
        return None, False, None

    def _update_laps(self):
        now = self.get_clock().now().nanoseconds / 1e9

        for vid in self.VEH_IDS:
            p = self.pose.get(vid)
            if p is None:
                continue

            # (A) start point init (first pose)
            if not self.start_inited[vid]:
                self.start_point[vid] = (p[0], p[1])
                self.start_inited[vid] = True
                self.prev_for_dist[vid] = (p[0], p[1])
                self.dist_since_lap[vid] = 0.0
                self.lap_armed[vid] = False  # must go out first
                print(f"[LAP_INIT] CAV{vid} start_point=({p[0]:.2f},{p[1]:.2f})")
                continue

            sp = self.start_point[vid]
            d = math.hypot(p[0]-sp[0], p[1]-sp[1])

            # (B) accumulate distance (anti double count)
            prev = self.prev_for_dist[vid]
            if prev is not None:
                self.dist_since_lap[vid] += math.hypot(p[0]-prev[0], p[1]-prev[1])
            self.prev_for_dist[vid] = (p[0], p[1])

            # (C) re-arm once it leaves start area
            if d > self.LAP_EXIT_R:
                self.lap_armed[vid] = True

            # (D) count lap when it comes back in
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
                    print(f"[LAP] CAV{vid} -> {self.lap_cnt[vid]}/{self.TARGET_LAPS}")


    def _print_lap_status(self):
        msg = "[LAP_STATUS] "
        parts = []
        for vid in self.VEH_IDS:
            parts.append(f"CAV{vid:02d}: {self.lap_cnt[vid]} lap")
        msg += " | ".join(parts)
        print(msg)


    ########### 디버깅
    def _print_cav_status(self):
        lines = []
        for vid in self.VEH_IDS:
            p = self.pose.get(vid)
            if p is None:
                continue

            use_round = self._in_round_box(p)
            src = self.round_raw.get(vid) if use_round else self.raw.get(vid)
            if src is None:
                continue

            x, y = p
            yaw_pose = float(self.yaw.get(vid, 0.0))     # PoseStamped의 orientation.z
            steer_cmd = float(src.angular.z)             # 실제 사용 src 기준
            v_cmd = float(src.linear.x)                  # 실제 사용 src 기준
            v_est = float(self.v_est.get(vid, 0.0))

            cav_num = CAV_TOPIC_NUM[vid]
            lines.append(
                f"CAV{cav_num:02d} | v_cmd={v_cmd:4.2f} | v_est={v_est:4.2f} | "
                f"yaw={yaw_pose:+.2f} | steer={steer_cmd:+.2f} | x={x:+.2f} y={y:+.2f}"
                f"{' | ROUND' if use_round else ''}"
            )

        if lines:
            print("\n".join(lines))
            print("-" * 60)





    # --- Main Loop ---
    def tick(self):
         # ✅ 차량별 회전교차로 진입 여부 (전역 roundabout_on 삭제)
        in_round = {}
        for vid in self.VEH_IDS:
            p = self.pose.get(vid)
            in_round[vid] = (p is not None) and self._in_round_box(p)

        if all(self.pose[v] is None for v in self.VEH_IDS):
            return

        # 1. Update Zone Logic (Roundabout/Intersection)
        self._update_zone_flags("TOP")
        self._update_zone_flags("BOT")
        top_lim, top_eff, top_on = self._compute_zone_limits("TOP")
        bot_lim, bot_eff, bot_on = self._compute_zone_limits("BOT")

        # 2. Update 4-Way Intersection Logic  (✅ 차량별로 OFF)
        fw_lim = {v: None for v in self.VEH_IDS}
        fw_eff = []

        for vid in self.VEH_IDS:
            # ✅ 이 차량이 회전교차로 안이면 FW 로직만 스킵 (다른 차량은 계속)
            if in_round.get(vid, False):
                # (선택) FW 상태가 남아서 다음에 튀는 게 걱정되면 리셋도 가능:
                # self.fw["active"][vid] = False
                # self.fw["outside_ticks"][vid] = 0
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

            # --- active / hysteresis ---
            if d < self.FW_RADIUS:
                self.fw["active"][vid] = True
                self.fw["outside_ticks"][vid] = 0
            elif d > self.FW_EXIT_RADIUS:
                if self.fw["active"][vid]:
                    self.fw["outside_ticks"][vid] += 1
                    if self.fw["outside_ticks"][vid] >= self.FW_HYSTERESIS_N:
                        self.fw["active"][vid] = False
                        self.fw["outside_ticks"][vid] = 0

            # --- approaching (dist decreasing based) ---
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

        # ✅ fw_eff도 차량별 in_round 제외 (이중 안전장치)
        fw_eff = [
            v for v in self.VEH_IDS
            if self.pose[v]
            and self.fw["active"][v]
            and self.fw["approaching"][v]
            and (not in_round.get(v, False))
        ]

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

        # 3. HV Safety & Merge Limits
        hv_lim = {v:None for v in self.VEH_IDS}; hv_force = {v:False for v in self.VEH_IDS}; hv_r = {v:None for v in self.VEH_IDS}
        for vid in self.VEH_IDS:
            l, f, r = self._compute_hv_safety_limit(vid); hv_lim[vid]=l; hv_force[vid]=f; hv_r[vid]=r
        
        for vid in self.VEH_IDS:
            cands = []
            if top_lim.get(vid) is not None: cands.append(float(top_lim[vid]))
            if bot_lim.get(vid) is not None: cands.append(float(bot_lim[vid]))
            if fw_lim.get(vid) is not None: cands.append(float(fw_lim[vid]))
            if hv_lim.get(vid) is not None: cands.append(float(hv_lim[vid]))
            self.tgt_limit[vid] = min(cands) if cands else None
            
            
            # =========================
            # ✅ HOLD 적용 (깜빡임 방지)
            # =========================
            if hv_force[vid]:
                # HV 정지는 강제 즉시 적용이니까 hold 개념 필요 없음
                self.hold_cnt[vid] = 0
                self.hold_limit[vid] = None

            else:
                if self.tgt_limit[vid] is not None:
                    # 이번 tick에 제한이 새로 걸렸으면 "저장 + hold 충전"
                    self.hold_limit[vid] = float(self.tgt_limit[vid])
                    self.hold_cnt[vid] = self.HOLD_TICKS

                else:
                    # 제한이 없어졌더라도 hold가 남아있으면 이전 제한을 계속 유지
                    if self.hold_cnt[vid] > 0 and self.hold_limit[vid] is not None:
                        self.tgt_limit[vid] = float(self.hold_limit[vid])
                        self.hold_cnt[vid] -= 1
                    else:
                        self.hold_limit[vid] = None
            # 0) HV는 무조건 기존대로 (즉시)
            if hv_force[vid]:
                self._apply_limit_ramp(vid, self.tgt_limit[vid], force_immediate=True, release_to=99.0)
                continue

            # 1) 회전교차로는 무조건 즉시 풀기 (예전처럼)
            if in_round.get(vid, False):
                self.cmd_limit[vid] = 99.0
                continue

            # 2) 사지/합류만 램프 적용 (천천히 가속)
            self._apply_limit_ramp(
                vid,
                self.tgt_limit[vid],
                force_immediate=False,
                release_to=self.V_NOM
            )


        # 4. Publish Commands
        for vid in self.VEH_IDS:
            use_round = (self.pose[vid] is not None) and self._in_round_box(self.pose[vid])
            src = self.round_raw[vid] if use_round else self.raw[vid]

            raw_v = float(src.linear.x)
            steer = float(src.angular.z)

            lim = float(self.cmd_limit[vid])
            out_v = min(raw_v, lim)

            out = Twist()
            out.linear.x = float(out_v)
            out.angular.z = float(steer)
            self.pub[vid].publish(out)


        self._update_laps()
        self._lap_log_cnt += 1
        if self._lap_log_cnt % self.LAP_LOG_PERIOD == 0:
            self._print_lap_status()
            self._print_cav_status()





# ============================================================
# MAIN EXECUTION
# ============================================================
def main(args=None):
    rclpy.init(args=args)

    drivers = [MapPredictionDriver(i, os.path.join(PATH_DIR, f"path3_{i}.json")) for i in range(1,5)]

    # ✅ Round controllers (publish /CAV_xx_accel_round_raw)
    # start_zone/start_trigger/out_zone/danger_zone 경로는 round_main.py가 기대하는 인자 그대로 넣어야 함
    p = lambda name: os.path.join(PATH_DIR, name)
    round_nodes = [
        RoundController(
            1, p("path3_1.json"),[(0.5,-0.108333)] , p("path_hv_3_1.csv"),
            p("path3_1_out_zone.csv"), p("path_hv_3_2.csv"),
            pose_topic=cav_topic(1),
            pub_topic=cav_accel_round_raw_topic(1),
            hv1_topic=hv_topic("HV1"),
            hv2_topic=hv_topic("HV2"),
        ),
        RoundController(
            2, p("path3_2.json"),[(1.558333,1.2)] , p("path_hv_2_1.csv"),
            p("path3_2_out_zone.csv"), p("path_hv_2_2.csv"),
            pose_topic=cav_topic(2),
            pub_topic=cav_accel_round_raw_topic(2),
            hv1_topic=hv_topic("HV1"),
            hv2_topic=hv_topic("HV2"),
        ),
        RoundController(
            3, p("path3_3.json"),[(1.308202,1.35)] , p("path_hv_2_1.csv"),
            None, p("path_hv_2_1.csv"),
            pose_topic=cav_topic(3),
            pub_topic=cav_accel_round_raw_topic(3),
            hv1_topic=hv_topic("HV1"),
            hv2_topic=hv_topic("HV2"),
        ),
        RoundController(
            4, p("path3_4.json"),[(0.35,-0.35833)] , p("path_hv_3_1.csv"),
            None, p("path_hv_3_1.csv"),
            pose_topic=cav_topic(4),
            pub_topic=cav_accel_round_raw_topic(4),
            hv1_topic=hv_topic("HV1"),
            hv2_topic=hv_topic("HV2"),
        ),
    ]


    guardian = Problem3DualZoneGuardianMux()

    ex = MultiThreadedExecutor(num_threads=10)
    for d in drivers: ex.add_node(d)
    for r in round_nodes: ex.add_node(r)
    ex.add_node(guardian)

    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        for d in drivers: d.destroy_node()
        for r in round_nodes: r.destroy_node()
        guardian.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

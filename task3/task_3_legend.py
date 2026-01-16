#!/usr/bin/env python3
import os
import math
import json
import csv

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Accel, PoseStamped

# ============================================================
# [1] Slow Zones Configuration (Fixed at 0.7 m/s)
# ============================================================
RAW_SLOW_ZONES = [
    (-1.2, -3.5,  1.2, -1.2), # 4-Way Intersection
    ( 0.4,  2.7,  1.4, -1.4), # Roundabout
]

# Normalize coordinates (min, max) for safety
SLOW_ZONES = [
    (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
    for (x1, x2, y1, y2) in RAW_SLOW_ZONES
]

SLOW_PARAMS = {
    "vel": 0.7,          
    "look_ahead": 0.55, 
    "kp": 5.5,          
    "ki": 0.05,
    "kd": 1.2,          
    "k_cte": 4.0
}

# ============================================================
# [2] Speed Profiles (3-Stage Transmission)
# ============================================================

# 1. Hard Curve (Low Speed, High Gain)
HARD_PARAMS = {
    "vel": 0.5,
    "look_ahead": 0.45,
    "kp": 6.5,
    "ki": 0.045,
    "kd": 1.0,
    "k_cte": 5.0
}

# 2. Easy Curve (Medium Speed)
EASY_PARAMS = {
    "vel": 1.0,
    "look_ahead": 0.60, 
    "kp": 6.0,
    "ki": 0.05,
    "kd": 1.0,
    "k_cte": 4.0
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
TICK_RATE = 0.05
ACCEL_LIMIT = 3.0
DECEL_LIMIT = 3.0


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
        self.TOPIC = f"/CAV_{self.vid:02d}"

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

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
        self.accel_raw_pub = self.create_publisher(Accel, f"{self.TOPIC}_accel_raw", 10)

        # State Variables
        self.curr_x, self.curr_y, self.curr_yaw = 0.0, 0.0, 0.0
        self.got_pose = False
        
        self.prev_err = 0.0
        self.int_err = 0.0
        self.last_time = self.get_clock().now()
        
        self.current_vel_cmd = 0.5
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
        for (x_min, x_max, y_min, y_max) in SLOW_ZONES:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def drive_loop(self):
        if not self.got_pose or not self.path_pts: return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0.001 or dt > 0.1: return

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

        # Find Look-ahead Point
        active_look_ahead = max(params["look_ahead"], self.current_vel_cmd * 0.6)
        
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

        # Calculate Cross Track Error (CTE)
        path_dx = tx - self.path_x[curr_idx]
        path_dy = ty - self.path_y[curr_idx]
        
        if math.hypot(path_dx, path_dy) < 0.001:
            cte = 0.0
        else:
            car_dx = self.curr_x - self.path_x[curr_idx]
            car_dy = self.curr_y - self.path_y[curr_idx]
            cross_prod = path_dx * car_dy - path_dy * car_dx
            cte_sign = 1.0 if cross_prod > 0 else -1.0
            cte = min_d * cte_sign * params["k_cte"]

        # PID Calculation
        self.int_err = max(-1.0, min(1.0, self.int_err + yaw_err * dt))
        
        p = params["kp"] * yaw_err
        i_term = params["ki"] * self.int_err
        d_error = self.normalize_angle(yaw_err - self.prev_err)
        d_term = params["kd"] * d_error / dt

        final_steer = max(-1.0, min(1.0, float(p + i_term + d_term + cte)))
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
                self.avg_steer_signed = 0.7 if final_steer > 0 else -0.7
            else:
                if self.mode == "STRGT":
                    if filter_val > 0.30: next_mode = "EASY"
                elif self.mode == "EASY":
                    if filter_val < 0.15: next_mode = "STRGT"
                    elif filter_val > 0.80: next_mode = "HARD"
                elif self.mode == "HARD":
                    if filter_val < 0.70: next_mode = "EASY"
            
            # Anti-Windup on mode transition
            if (self.mode == "HARD" and next_mode == "EASY") or \
               (self.mode == "EASY" and next_mode == "STRGT"):
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

        self.log_counter += 1
        if self.log_counter % 20 == 0:
            print(f"[C{self.vid}] Pos:({self.curr_x:.2f}, {self.curr_y:.2f}) | Mode:[{current_mode_str}] | Vel:{self.current_vel_cmd:.2f} | Steer:{final_steer:.2f}")


# ============================================================
# [NODE] Dual Zone Guardian Mux (Safety Controller)
# ============================================================
class Problem3DualZoneGuardianMux(Node):
    def __init__(self):
        super().__init__("problem3_dualzone_guardian_mux")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.VEH_IDS = [1, 2, 3, 4]
        self.TOPICS = {vid: f"/CAV_{vid:02d}" for vid in self.VEH_IDS}

        # Parameters
        self.V_NOM = 0.7
        self.RANK_SPEEDS_3P = [0.7, 0.45, 0.2, 0.2]
        self.RANK_SPEEDS_2P = [0.7, 0.2]

        self.TOP_CENTER = (-2.3342, 2.3073)
        self.BOT_CENTER = (-2.3342, -2.3073)
        self.RADIUS = 1.5
        self.EXIT_RADIUS = 0.4
        self.APPROACH_N = 2
        self.EPS = 0.001
        self.HYSTERESIS_N = 5

        self.TICK = 0.05
        self.RAMP_DOWN_PER_SEC = 3.0
        self.RAMP_UP_PER_SEC = 0.10
        self.STOP_VELOCITY = 0.02
        self.MIN_SPEED = self.STOP_VELOCITY

        # States
        self.pose = {vid: None for vid in self.VEH_IDS}
        self.raw = {vid: Accel() for vid in self.VEH_IDS}
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
            self.create_subscription(Accel, f"{topic}_accel_raw", self._make_raw_cb(vid), 10)
            
        self.pub = {vid: self.create_publisher(Accel, f"{self.TOPICS[vid]}_accel", 10) for vid in self.VEH_IDS}
        self._log_counter = 0

        # 4-Way Intersection Settings
        self.FW_CENTER = (-2.3342, 0.0)
        self.FW_RADIUS = 2.2
        self.FW_EXIT_RADIUS = 0.4
        self.FW_HYSTERESIS_N = 10
        self.FW_APPROACH_N = 2
        self.FW_EPS = 0.001
        self.FW_V_NOM = 0.7
        self.FW_RANK_SPEEDS_2P = [0.7, 0.2]
        self.FW_RANK_SPEEDS_3P = [0.7, 0.45, 0.2, 0.1]
        self.fw = {
            "active": {vid: False for vid in self.VEH_IDS},
            "outside_ticks": {vid: 0 for vid in self.VEH_IDS},
            "prev_dist": {vid: None for vid in self.VEH_IDS},
            "approach_cnt": {vid: 0 for vid in self.VEH_IDS},
            "approaching": {vid: False for vid in self.VEH_IDS},
        }
        # Defined Collision Pairs (Direction-based)
        self.FW_CASES = [
            frozenset([(1, "N"), (2, "W")]), frozenset([(1, "N"), (2, "E")]),
            frozenset([(1, "N"), (3, "E")]), frozenset([(2, "W"), (3, "E")]),
            frozenset([(1, "S"), (2, "W")]), frozenset([(1, "S"), (2, "E")]),
            frozenset([(1, "S"), (4, "W")]), frozenset([(2, "E"), (4, "W")]),
            frozenset([(1, "N"), (2, "W"), (3, "E")]), frozenset([(1, "S"), (2, "E"), (4, "W")]),
        ]

        # Human Vehicle (HV) Safety Settings
        self.TARGET_VELOCITY = 0.7; self.ZONE_RADIUS = 0.25; self.HV_DETECT_RADIUS = 0.10; self.RESET_DISTANCE = 2.2
        self.hv19 = None; self.hv20 = None; self.hv19_active = False; self.hv20_active = False
        
        self.create_subscription(PoseStamped, "/HV_19", self._cb_hv19, qos)
        self.create_subscription(PoseStamped, "/HV_20", self._cb_hv20, qos)

        self.safety_cfg = {
            1: { "start_zone": load_zone_from_csv("tool/path/path3_1_zone.csv"), "start_trigger": load_zone_from_csv("tool/path/path_hv_3_1.csv"), "out_zone": load_zone_from_csv("tool/path/path3_1_out_zone.csv"), "danger_zone": load_zone_from_csv("tool/path/path_hv_3_2.csv"), "stop_logic_disabled": False},
            2: { "start_zone": load_zone_from_csv("tool/path/path3_2_zone.csv"), "start_trigger": load_zone_from_csv("tool/path/path_hv_2_1.csv"), "out_zone": load_zone_from_csv("tool/path/path3_2_out_zone.csv"), "danger_zone": load_zone_from_csv("tool/path/path_hv_2_2.csv"), "stop_logic_disabled": False},
            3: { "start_zone": load_zone_from_csv("tool/path/path3_3_zone.csv"), "start_trigger": load_zone_from_csv("tool/path/path_hv_2_1.csv"), "out_zone": [], "danger_zone": [], "stop_logic_disabled": False},
            4: { "start_zone": load_zone_from_csv("tool/path/path3_4_zone.csv"), "start_trigger": load_zone_from_csv("tool/path/path_hv_3_1.csv"), "out_zone": [], "danger_zone": [], "stop_logic_disabled": False},
        }
        self.create_timer(self.TICK, self.tick)

    # --- Callbacks & Helpers ---
    def _cb_hv19(self, msg): self.hv19 = (float(msg.pose.position.x), float(msg.pose.position.y)); self.hv19_active = True
    def _cb_hv20(self, msg): self.hv20 = (float(msg.pose.position.x), float(msg.pose.position.y)); self.hv20_active = True
    def _make_zone_state(self, c): return {"CENTER": c, "active": {v:False for v in self.VEH_IDS}, "outside_ticks": {v:0 for v in self.VEH_IDS}, "prev_dist": {v:None for v in self.VEH_IDS}, "approach_cnt": {v:0 for v in self.VEH_IDS}, "approaching": {v:False for v in self.VEH_IDS}}
    
    def _make_pose_cb(self, vid):
        def cb(msg): self.pose[vid] = (float(msg.pose.position.x), float(msg.pose.position.y)); self._update_speed_est(vid, self.pose[vid])
        return cb
    
    def _make_raw_cb(self, vid):
        def cb(msg): self.raw[vid] = msg
        return cb
    
    def _dist(self, p, c): return math.hypot(p[0] - c[0], p[1] - c[1])
    
    def _update_speed_est(self, vid, p):
        prev = self.last_pose[vid]
        if prev is not None:
            d = math.hypot(p[0]-prev[0], p[1]-prev[1]); v = d/self.TICK; self.v_est[vid] = 0.35*v + 0.65*self.v_est[vid]
        self.last_pose[vid] = p

    def _apply_limit_ramp(self, vid, tgt, force_immediate=False):
        if tgt is None: self.cmd_limit[vid] = 99.0; return
        if force_immediate: self.cmd_limit[vid] = max(self.MIN_SPEED, float(tgt)); return
        cur = self.cmd_limit[vid]
        if cur > 50: cur = float(self.raw[vid].linear.x) if self.raw[vid] is not None else self.V_NOM
        step_down = self.RAMP_DOWN_PER_SEC * self.TICK; step_up = self.RAMP_UP_PER_SEC * self.TICK
        if tgt > cur: cur = min(tgt, cur + step_up)
        else: cur = max(tgt, cur - step_down)
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

    # --- Main Loop ---
    def tick(self):
        if all(self.pose[v] is None for v in self.VEH_IDS): return
        
        # 1. Update Zone Logic (Roundabout/Intersection)
        self._update_zone_flags("TOP"); self._update_zone_flags("BOT")
        top_lim, top_eff, top_on = self._compute_zone_limits("TOP")
        bot_lim, bot_eff, bot_on = self._compute_zone_limits("BOT")
        
        # 2. Update 4-Way Intersection Logic
        for vid in self.VEH_IDS:
            p = self.pose[vid]; 
            if not p: continue
            d = self._fw_dist(vid)
            if d is None: continue
            if d < self.FW_RADIUS: self.fw["active"][vid]=True; self.fw["outside_ticks"][vid]=0
            elif d > self.FW_EXIT_RADIUS:
                if self.fw["active"][vid]:
                    self.fw["outside_ticks"][vid]+=1
                    if self.fw["outside_ticks"][vid]>=self.FW_HYSTERESIS_N: self.fw["active"][vid]=False
            prev = self.fw["prev_dist"][vid]
            if prev is None: self.fw["prev_dist"][vid]=d; self.fw["approaching"][vid]=False
            else:
                if d < prev - self.FW_EPS: self.fw["approach_cnt"][vid]+=1
                else: self.fw["approach_cnt"][vid]=0
                self.fw["approaching"][vid] = (self.fw["approach_cnt"][vid]>=self.FW_APPROACH_N); self.fw["prev_dist"][vid]=d
        
        fw_eff = [v for v in self.VEH_IDS if self.pose[v] and self.fw["active"][v] and self.fw["approaching"][v]]
        fw_map = {}
        for v in fw_eff:
            dd = self._fw_get_direction(v)
            if dd: fw_map[v]=dd
        pairs = set((v, d) for v,d in fw_map.items())
        matched = None
        for c in self.FW_CASES:
            if c.issubset(pairs): matched=c; break
        fw_lim = {v:None for v in self.VEH_IDS}
        if matched:
            targs = [v for v,_ in matched]
            targs.sort(key=lambda v: self._fw_dist(v) or 1e9)
            n = len(targs); spd = self.FW_RANK_SPEEDS_2P if n==2 else self.FW_RANK_SPEEDS_3P
            for i, vid in enumerate(targs):
                des = spd[min(i, len(spd)-1)]; fw_lim[vid] = des if des < self.FW_V_NOM else None

        # 3. HV Safety & Merge Limits
        hv_lim = {v:None for v in self.VEH_IDS}; hv_force = {v:False for v in self.VEH_IDS}; hv_r = {v:None for v in self.VEH_IDS}
        for vid in self.VEH_IDS:
            l, f, r = self._compute_hv_safety_limit(vid); hv_lim[vid]=l; hv_force[vid]=f; hv_r[vid]=r
        
        for vid in self.VEH_IDS:
            cands = []
            if top_lim.get(vid): cands.append(float(top_lim[vid]))
            if bot_lim.get(vid): cands.append(float(bot_lim[vid]))
            if fw_lim.get(vid): cands.append(float(fw_lim[vid]))
            if hv_lim.get(vid) is not None: cands.append(float(hv_lim[vid]))
            self.tgt_limit[vid] = min(cands) if cands else None
            self._apply_limit_ramp(vid, self.tgt_limit[vid], force_immediate=hv_force[vid])

        # 4. Publish Commands
        for vid in self.VEH_IDS:
            raw_v = float(self.raw[vid].linear.x); lim = float(self.cmd_limit[vid]); out_v = min(raw_v, lim)
            out = Accel(); out.linear.x = float(out_v); out.angular.z = float(self.raw[vid].angular.z)
            self.pub[vid].publish(out)
        
        self._log_counter += 1
        if self._log_counter % 20 == 0:
            print(f"[GUARD] TOP:{top_eff} BOT:{bot_eff} FW:{fw_eff}")
            for vid in self.VEH_IDS:
                if hv_r[vid]: print(f"  > CAV{vid} SAFETY: {hv_r[vid]}")

# ============================================================
# MAIN EXECUTION
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    drivers = [MapPredictionDriver(i, f"tool/path/path3_{i}.json") for i in range(1, 5)]
    guardian = Problem3DualZoneGuardianMux()
    ex = MultiThreadedExecutor(num_threads=10)
    for d in drivers: ex.add_node(d)
    ex.add_node(guardian)
    try: ex.spin()
    except KeyboardInterrupt: pass
    finally:
        ex.shutdown(); [d.destroy_node() for d in drivers]; guardian.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
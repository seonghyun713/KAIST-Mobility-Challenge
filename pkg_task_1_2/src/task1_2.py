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
# PATH BASE (For Docker and Pkg)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DIR = os.path.join(BASE_DIR, "path")


# ============================================================
# IO helpers
# ============================================================
def load_dz_points(csv_file: str):
    pts = []
    if not os.path.exists(csv_file):
        return pts
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                pts.append((float(row[0]), float(row[1])))
            except:
                pass
    return pts

def load_path_points(json_file: str):
    if not os.path.exists(json_file):
        return []
    with open(json_file, "r") as f:
        data = json.load(f)

    xs = data.get("x") or data.get("X")
    ys = data.get("y") or data.get("Y")
    if not xs or not ys:
        return []
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


# ============================================================
# Geometry helpers
# ============================================================
def min_dist_to_points(x, y, pts):
    if not pts:
        return float("inf")
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
        if det < 1e-12:
            continue
        t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / det))
        qx, qy = ax + t * vx, ay + t * vy
        d2 = (px - qx)**2 + (py - qy)**2
        if d2 < best_d2:
            best_d2 = d2
            best_s = s_cum[i] + t * math.sqrt(det)
    return best_s

def dist2_point_to_polyline(px, py, pts):
    best = float("inf")
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        vv = vx * vx + vy * vy
        if vv <= 1e-12:
            continue
        t = (wx * vx + wy * vy) / vv
        if t < 0.0: t = 0.0
        elif t > 1.0: t = 1.0
        qx = ax + t * vx
        qy = ay + t * vy
        d2 = (px - qx) ** 2 + (py - qy) ** 2
        if d2 < best:
            best = d2
    return best

def normalize_angle(a):
    while a > math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


# ============================================================
# Global Settings
# ============================================================
WHEELBASE = 0.211
DIST_CENTER_TO_REAR = WHEELBASE / 2.0
TICK = 0.05
TARGET_VELOCITY = 0.9
MINSPEED = 0.0

# guardian이 계산한 속도 제한(최종 속도 상한)
GUARDIAN_LIMITS = {1: 99.0, 2: 99.0}


# ============================================================
# Hyperparameters for ZonePriorityDriver
# ============================================================

# 1. Hard Curve (Low Speed, High Gain)
HARD_PARAMS = {
    "vel": 0.6,
    "look_ahead": 0.53,
    "kp": 6.0,
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
    "vel": 1.2,
    "look_ahead": 1.2,  # Increased for high speed
    "kp": 2.0,          # Reduced to prevent oscillation
    "ki": 0.002,        # Minimize integral windup
    "kd": 2.5,          # Increased damping
    "k_cte": 1.0
}

ACCEL_LIMIT = 2.0
DECEL_LIMIT = 3.5

class ZonePriorityDriver(Node):
    def __init__(self, vehicle_id: int):
        super().__init__(f'zone_driver_{vehicle_id}')
        self.vehicle_id = int(vehicle_id)

        if self.vehicle_id == 1:
            self.MY_PATH_FILE = os.path.join(PATH_DIR, "path1_1.json")
            self.MY_TOPIC = "/CAV_01"
        else:
            self.MY_PATH_FILE = os.path.join(PATH_DIR, "path1_2.json")
            self.MY_TOPIC = "/CAV_02"

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.accel_raw_pub = self.create_publisher(Accel, f"{self.MY_TOPIC}_accel_raw", 10)
        self.create_subscription(PoseStamped, self.MY_TOPIC, self.pose_callback, qos)

        self.path_pts = load_path_points(self.MY_PATH_FILE)
        self.path_x = [p[0] for p in self.path_pts]
        self.path_y = [p[1] for p in self.path_pts]

        if not self.path_pts:
            self.get_logger().error(f"[CAV{self.vehicle_id:02d}] 경로 파일 없음: {self.MY_PATH_FILE}")

        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0
        self.got_pose = False

        self.prev_err = 0.0
        self.int_err = 0.0
        self.last_time = self.get_clock().now()

        self.current_vel_cmd = 0.5
        self.mode = "HARD"
        self.avg_steer_signed = 0.0

        # jump detection
        self.old_nearest_idx = 0
        self._skip_dterm = 0

        self.create_timer(TICK, self.drive_loop)

    def pose_callback(self, msg: PoseStamped):
        self.got_pose = True
        self.curr_yaw = float(msg.pose.orientation.z)
        self.curr_x = float(msg.pose.position.x) - (DIST_CENTER_TO_REAR * math.cos(self.curr_yaw))
        self.curr_y = float(msg.pose.position.y) - (DIST_CENTER_TO_REAR * math.sin(self.curr_yaw))

    def _choose_params(self):
        if self.mode == "HARD":
            return HARD_PARAMS
        elif self.mode == "EASY":
            return EASY_PARAMS
        else:
            return STRAIGHT_PARAMS

    def _find_nearest_idx_windowed(self, search_range=50):
        path_len = len(self.path_pts)
        min_d = float("inf")
        curr_idx = self.old_nearest_idx
        found = False

        for offset in range(-search_range, search_range + 1):
            idx = (self.old_nearest_idx + offset) % path_len
            d = math.hypot(self.path_x[idx] - self.curr_x, self.path_y[idx] - self.curr_y)
            if d < min_d:
                min_d = d
                curr_idx = idx
                found = True

        if (not found) or (min_d > 5.0):
            min_d = float("inf")
            for i in range(path_len):
                d = math.hypot(self.path_x[i] - self.curr_x, self.path_y[i] - self.curr_y)
                if d < min_d:
                    min_d = d
                    curr_idx = i

        self.old_nearest_idx = curr_idx
        return curr_idx, min_d

    def drive_loop(self):
        if not self.got_pose or not self.path_pts:
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0.001 or dt > 0.1:
            return

        params = self._choose_params()
        path_len = len(self.path_pts)

        # Step 1) nearest index (windowed)
        curr_idx, min_d = self._find_nearest_idx_windowed(search_range=50)

        # Step 1-1) Jump detection + relocalize + D-term skip
        if min_d > 0.8:
            # 글로벌 재탐색
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

        # Step 2) lookahead
        active_look_ahead = max(params["look_ahead"], self.current_vel_cmd * 0.6)

        target_idx = curr_idx
        for k in range(1, path_len + 1):
            idx = (curr_idx + k) % path_len
            d = math.hypot(self.path_x[idx] - self.curr_x, self.path_y[idx] - self.curr_y)
            if d >= active_look_ahead:
                target_idx = idx
                break

        tx, ty = self.path_x[target_idx], self.path_y[target_idx]

        # Step 3) yaw error
        desired_yaw = math.atan2(ty - self.curr_y, tx - self.curr_x)
        yaw_err = normalize_angle(desired_yaw - self.curr_yaw)

        # Step 3-1) vector CTE
        path_dx = tx - self.path_x[curr_idx]
        path_dy = ty - self.path_y[curr_idx]
        if math.hypot(path_dx, path_dy) < 1e-6:
            cte = 0.0
        else:
            car_dx = self.curr_x - self.path_x[curr_idx]
            car_dy = self.curr_y - self.path_y[curr_idx]
            cross = path_dx * car_dy - path_dy * car_dx
            cte_sign = 1.0 if cross > 0 else -1.0
            cte = min_d * cte_sign * params["k_cte"]

        # Step 3-2) PID
        self.int_err = max(-1.0, min(1.0, self.int_err + yaw_err * dt))
        p = params["kp"] * yaw_err
        i_term = params["ki"] * self.int_err

        if self._skip_dterm > 0:
            d_term = 0.0
            self._skip_dterm -= 1
        else:
            d_term = params["kd"] * normalize_angle(yaw_err - self.prev_err) / dt

        final_steer = max(-1.0, min(1.0, float(p + i_term + d_term + cte)))
        self.prev_err = yaw_err

        # Step 4) 모드 전환
        self.avg_steer_signed = 0.85 * self.avg_steer_signed + 0.15 * final_steer
        filter_val = abs(self.avg_steer_signed)

        next_mode = self.mode
        if abs(final_steer) > 0.90:
            next_mode = "HARD"
            self.avg_steer_signed = 0.7 if final_steer > 0 else -0.7
        else:
            if self.mode == "STRGT":
                if filter_val > 0.30:
                    next_mode = "EASY"
            elif self.mode == "EASY":
                if filter_val < 0.15:
                    next_mode = "STRGT"
                elif filter_val > 0.80:
                    next_mode = "HARD"
            elif self.mode == "HARD":
                if filter_val < 0.70:
                    next_mode = "EASY"

        # Anti-windup on upshift
        if (self.mode == "HARD" and next_mode == "EASY") or (self.mode == "EASY" and next_mode == "STRGT"):
            self.int_err = 0.0

        self.mode = next_mode

        # Step 5) target_v
        if self.mode == "HARD":
            target_v = HARD_PARAMS["vel"]
        elif self.mode == "EASY":
            target_v = EASY_PARAMS["vel"]
        else:
            target_v = STRAIGHT_PARAMS["vel"]

        # Step 6) 속도 램프 + 최종 캡(TARGET_VELOCITY)
        if target_v > self.current_vel_cmd:
            self.current_vel_cmd = min(target_v, self.current_vel_cmd + ACCEL_LIMIT * dt)
        else:
            self.current_vel_cmd = max(target_v, self.current_vel_cmd - DECEL_LIMIT * dt)

        v_cmd = float(min(self.current_vel_cmd, TARGET_VELOCITY))

        cmd = Accel()
        cmd.linear.x = v_cmd
        cmd.angular.z = float(final_steer)
        self.accel_raw_pub.publish(cmd)


# ============================================================
# Guardian Mux for All Zones
# ============================================================
class AllZonesGuardianMux(Node):
    def __init__(self):
        super().__init__("all_zones_guardian_mux")

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)

        # ---------- pose ----------
        self.p1 = None
        self.p2 = None
        self.create_subscription(PoseStamped, "/CAV_01", self.cb1_pose, qos)
        self.create_subscription(PoseStamped, "/CAV_02", self.cb2_pose, qos)

        # ---------- raw accel from drivers ----------
        self.raw1 = Accel()
        self.raw2 = Accel()
        self.create_subscription(Accel, "/CAV_01_accel_raw", self.cb1_raw, 10)
        self.create_subscription(Accel, "/CAV_02_accel_raw", self.cb2_raw, 10)

        # ---------- final accel publishers ----------
        self.pub1 = self.create_publisher(Accel, "/CAV_01_accel", 10)
        self.pub2 = self.create_publisher(Accel, "/CAV_02_accel", 10)

        # ====================================================
        # upper/right/rotary 파일들
        # ====================================================
        self.DZ_RIGHT    = os.path.join(PATH_DIR, "right_1_2.csv")
        self.DZ_UPPER_V1 = os.path.join(PATH_DIR, "upper_dz_v1.csv")
        self.DZ_UPPER_V2 = os.path.join(PATH_DIR, "upper_dz_v2.csv")
        self.DZ_ROTARY_V1 = os.path.join(PATH_DIR, "path1_1_rotary.csv")
        self.DZ_ZONE_V2   = os.path.join(PATH_DIR, "path1_2_zone.csv")

        self.dz_right = load_dz_points(self.DZ_RIGHT)
        self.dz_upper_v1 = load_dz_points(self.DZ_UPPER_V1)
        self.dz_upper_v2 = load_dz_points(self.DZ_UPPER_V2)
        self.dz_rotary_v1 = load_dz_points(self.DZ_ROTARY_V1)
        self.dz_zone_v2 = load_dz_points(self.DZ_ZONE_V2)

        # upper arc-length paths
        self.path1_pts = load_path_points(os.path.join(PATH_DIR, "path1_1.json"))
        self.path2_pts = load_path_points(os.path.join(PATH_DIR, "path1_2.json"))
        self.path1_s = build_cumulative_s(self.path1_pts) if self.path1_pts else []
        self.path2_s = build_cumulative_s(self.path2_pts) if self.path2_pts else []

        # upper merge point
        upper_all = self.dz_upper_v1 + self.dz_upper_v2
        self.upper_merge = None
        self.s_merge_1 = None
        self.s_merge_2 = None
        if upper_all and self.path1_pts and self.path2_pts:
            best_score = float("inf")
            best_pt = None
            for zx, zy in upper_all:
                score = dist2_point_to_polyline(zx, zy, self.path1_pts) + dist2_point_to_polyline(zx, zy, self.path2_pts)
                if score < best_score:
                    best_score = score
                    best_pt = (zx, zy)
            self.upper_merge = best_pt
            mx, my = best_pt
            self.s_merge_1 = project_point_to_polyline_s(mx, my, self.path1_pts, self.path1_s)
            self.s_merge_2 = project_point_to_polyline_s(mx, my, self.path2_pts, self.path2_s)

        # ====================================================
        # upper/right/rotary 파라미터
        # ====================================================
        self.NEAR_TOL_RIGHT, self.SAFETY_DIST_RIGHT = 0.55, 1.0
        self.NEAR_TOL_UPPER, self.SAFETY_DIST_UPPER = 0.8, 1.2
        self.UPPER_RELEASE_DIST = 2.2
        self.NEAR_TOL_ROTARY = 0.4

        self.UPPER_TIE_EPS = 0.10
        self.UPPER_TIE_YIELD_ID = 1

        self.ACCEL_YIELD = 0.20
        self.ACCEL_RESUME = TARGET_VELOCITY
        self.RAMP_PER_SEC = 2.0
        self.TICK = 0.05
        self.MIN_SPEED = MINSPEED
        self.RESUME_PULSE_TICKS = int(1.5 / self.TICK)

        self.cmd_speed = {1: 99.0, 2: 99.0}
        self.target_speed = {1: None, 2: None}
        self.resume_ticks_left = {1: 0, 2: 0}
        self.upper_lock_yield_id = None

        # ==========================
        # 스타일 FW 파라미터 (Task3 참고)
        # ==========================
        self.VEH_IDS = [1, 2]
        self.FW_CENTER = (-2.3351, 0.0)

        self.FW_RADIUS = 2.35
        self.FW_EXIT_RADIUS = 1.1
        self.FW_HYSTERESIS_N = 5
        self.FW_APPROACH_N = 2
        self.FW_EPS = 0.001

        self.FW_V_NOM = TARGET_VELOCITY

        # 우선순위별 목표 속도
        self.FW_RANK_SPEEDS_2P = [TARGET_VELOCITY, 0.15]

        # 2대 케이스
        self.FW_CASES_12 = [
            frozenset([(1, "N"), (2, "W")]), frozenset([(1, "N"), (2, "E")]),
            frozenset([(1, "S"), (2, "W")]), frozenset([(1, "S"), (2, "E")]),
        ]

        self.fw = {
            "active": {vid: False for vid in self.VEH_IDS},
            "outside_ticks": {vid: 0 for vid in self.VEH_IDS},
            "prev_dist": {vid: None for vid in self.VEH_IDS},
            "approach_cnt": {vid: 0 for vid in self.VEH_IDS},
            "approaching": {vid: False for vid in self.VEH_IDS},
        }

        # speed estimator (TTC ranking)
        self.last_pose = {1: None, 2: None}
        self.v_est = {1: TARGET_VELOCITY, 2: TARGET_VELOCITY}

        # ==========================
        # 로그용 상태 저장
        # ==========================
        self._tick_count = 0
        self._last_zone_state = {}
        self._last_pub_state = None

        self.create_timer(self.TICK, self.tick)
        self.get_logger().info("✅ GuardianMux started: raw->final accel (upper/right/rotary + FW intersection)")

    # ---------- callbacks ----------
    def cb1_pose(self, msg):
        self.p1 = (msg.pose.position.x, msg.pose.position.y)
        self._update_speed_est(1, self.p1)

    def cb2_pose(self, msg):
        self.p2 = (msg.pose.position.x, msg.pose.position.y)
        self._update_speed_est(2, self.p2)

    def cb1_raw(self, msg):
        self.raw1 = msg

    def cb2_raw(self, msg):
        self.raw2 = msg

    # ---------- speed estimator ----------
    def _update_speed_est(self, vid, curr_p):
        if self.last_pose[vid] is not None:
            prev = self.last_pose[vid]
            dist = math.hypot(curr_p[0] - prev[0], curr_p[1] - prev[1])
            v = dist / self.TICK
            self.v_est[vid] = 0.35 * v + 0.65 * self.v_est[vid]
        self.last_pose[vid] = curr_p

    # ---------- ramped limiter ----------
    def _apply_limit(self, vid, tgt):
        if tgt is None:
            self.cmd_speed[vid] = 99.0
            GUARDIAN_LIMITS[vid] = 99.0
            return

        cur = self.cmd_speed[vid]
        if cur > 50:
            cur = self.FW_V_NOM
        step = self.RAMP_PER_SEC * self.TICK
        if tgt > cur:
            cur = min(tgt, cur + step)
        else:
            cur = max(tgt, cur - step)
        cur = max(self.MIN_SPEED, cur)

        self.cmd_speed[vid] = cur
        GUARDIAN_LIMITS[vid] = cur

    def _start_resume(self, vid):
        self.resume_ticks_left[vid] = self.RESUME_PULSE_TICKS
        self.target_speed[vid] = self.ACCEL_RESUME

    def _log_zone(self, zone_name: str, state_tuple, text: str):
        prev = self._last_zone_state.get(zone_name)
        if prev != state_tuple:
            print(text)
            self._last_zone_state[zone_name] = state_tuple

    # =====================================
    # FW intersection helpers (Task 3 참고)
    # =====================================
    def _fw_dist(self, vid):
        p = self.p1 if vid == 1 else self.p2
        if not p:
            return None
        return math.hypot(p[0] - self.FW_CENTER[0], p[1] - self.FW_CENTER[1])

    def _fw_get_direction(self, vid):
        p = self.p1 if vid == 1 else self.p2
        if not p:
            return None
        dx = p[0] - self.FW_CENTER[0]
        dy = p[1] - self.FW_CENTER[1]
        if abs(dx) > abs(dy):
            return "E" if dx > 0 else "W"
        else:
            return "N" if dy > 0 else "S"

    def _fw_update_flags(self):
        for vid in self.VEH_IDS:
            d = self._fw_dist(vid)
            if d is None:
                continue

            # active hysteresis
            if d < self.FW_RADIUS:
                self.fw["active"][vid] = True
                self.fw["outside_ticks"][vid] = 0
            elif d > self.FW_EXIT_RADIUS:
                if self.fw["active"][vid]:
                    self.fw["outside_ticks"][vid] += 1
                    if self.fw["outside_ticks"][vid] >= self.FW_HYSTERESIS_N:
                        self.fw["active"][vid] = False
                        self.fw["outside_ticks"][vid] = 0

            # approaching 판단
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

    def _fw_compute_limits(self):
        """2대 전용: 케이스 매칭되면 가까운 놈 우선, 뒤는 감속"""
        self._fw_update_flags()

        fw_eff = []
        for vid in self.VEH_IDS:
            if self.fw["active"][vid] and self.fw["approaching"][vid]:
                fw_eff.append(vid)

        lim = {1: None, 2: None}
        if len(fw_eff) < 2:
            self._log_zone("FW", ("IDLE", tuple(fw_eff)), f"✅ [FW] idle eff={fw_eff}")
            return lim, fw_eff

        d1 = self._fw_get_direction(1)
        d2 = self._fw_get_direction(2)
        if (d1 is None) or (d2 is None):
            return lim, fw_eff

        pairs = frozenset([(1, d1), (2, d2)])
        matched = None
        for c in self.FW_CASES_12:
            if c.issubset(pairs):
                matched = c
                break

        if not matched:
            self._log_zone("FW", ("IDLE_CASE", d1, d2), f"✅ [FW] no-case d1={d1} d2={d2}")
            return lim, fw_eff

        # priority: 더 가까운(=dist 작은) 차량
        targs = [1, 2]
        targs.sort(key=lambda v: self._fw_dist(v) if self._fw_dist(v) is not None else 1e9)

        spd = self.FW_RANK_SPEEDS_2P
        for i, vid in enumerate(targs):
            des = spd[min(i, len(spd)-1)]
            if des < self.FW_V_NOM:
                lim[vid] = des

        self._log_zone(
            "FW",
            ("ACTIVE", d1, d2, tuple(targs)),
            f"🛑 [FW ACTIVE] d1={d1} d2={d2} priority=CAV{targs[0]} yield=CAV{targs[1]} lim={lim}"
        )
        return lim, fw_eff

    def _infer_mode(self, steer):
        s = abs(steer)
        if s > 0.80:
            return "HARD"
        elif s > 0.30:
            return "EASY"
        else:
            return "STRGT"


    # ============================================================
    # main tick
    # ============================================================
    def tick(self):
        if self.p1 is None or self.p2 is None:
            return

        self._tick_count += 1
        x1, y1 = self.p1
        x2, y2 = self.p2
        d12 = math.hypot(x1 - x2, y1 - y2)

        # resume pulses
        for vid in (1, 2):
            if self.resume_ticks_left[vid] > 0:
                self._apply_limit(vid, self.target_speed[vid])
                self.resume_ticks_left[vid] -= 1
                if self.resume_ticks_left[vid] <= 0:
                    self.target_speed[vid] = None

        # default: no limit
        limit_candidates = {1: [None], 2: [None]}

        # ---------------- RIGHT (2 우선 => 1 감속) ----------------
        if self.dz_right:
            n1 = (min_dist_to_points(x1, y1, self.dz_right) <= self.NEAR_TOL_RIGHT)
            n2 = (min_dist_to_points(x2, y2, self.dz_right) <= self.NEAR_TOL_RIGHT)
            if n1 and n2 and d12 <= self.SAFETY_DIST_RIGHT:
                limit_candidates[1].append(self.ACCEL_YIELD)
                self._log_zone("RIGHT", ("ACTIVE", 2, 1), f"🛑 [RIGHT ACTIVE] priority=CAV2 yield=CAV1 d12={d12:.2f}")
            else:
                self._log_zone("RIGHT", ("IDLE",), "✅ [RIGHT] idle")

        # ---------------- UPPER ----------------
        if self.upper_merge and self.dz_upper_v1 and self.dz_upper_v2 and (self.s_merge_1 is not None) and (self.s_merge_2 is not None):
            n1 = (min_dist_to_points(x1, y1, self.dz_upper_v1) <= self.NEAR_TOL_UPPER)
            n2 = (min_dist_to_points(x2, y2, self.dz_upper_v2) <= self.NEAR_TOL_UPPER)

            if self.upper_lock_yield_id is not None:
                if (not (n1 and n2)) or (d12 > self.UPPER_RELEASE_DIST):
                    yid = self.upper_lock_yield_id
                    self.upper_lock_yield_id = None
                    self._log_zone("UPPER", ("UNLOCK",), "✅ [UPPER] unlock/clear")
                    if self.target_speed[yid] == self.ACCEL_YIELD and self.resume_ticks_left[yid] == 0:
                        self._start_resume(yid)
                else:
                    limit_candidates[self.upper_lock_yield_id].append(self.ACCEL_YIELD)
                    priority = 3 - self.upper_lock_yield_id
                    self._log_zone("UPPER", ("LOCK", priority, self.upper_lock_yield_id),
                                   f"🛑 [UPPER LOCK] priority=CAV{priority} yield=CAV{self.upper_lock_yield_id} d12={d12:.2f}")

            elif n1 and n2 and d12 <= self.SAFETY_DIST_UPPER:
                s1 = project_point_to_polyline_s(x1, y1, self.path1_pts, self.path1_s)
                s2 = project_point_to_polyline_s(x2, y2, self.path2_pts, self.path2_s)
                rem1 = max(0.0, self.s_merge_1 - s1)
                rem2 = max(0.0, self.s_merge_2 - s2)

                if rem1 < rem2 - self.UPPER_TIE_EPS:
                    yield_id = 2
                elif rem2 < rem1 - self.UPPER_TIE_EPS:
                    yield_id = 1
                else:
                    yield_id = self.UPPER_TIE_YIELD_ID

                self.upper_lock_yield_id = yield_id
                limit_candidates[yield_id].append(self.ACCEL_YIELD)

                priority = 3 - yield_id
                self._log_zone(
                    "UPPER",
                    ("ACTIVE", priority, yield_id),
                    f"🛑 [UPPER ACTIVE] priority=CAV{priority} yield=CAV{yield_id} rem1={rem1:.2f} rem2={rem2:.2f} d12={d12:.2f}"
                )
            else:
                self._log_zone("UPPER", ("IDLE",), "✅ [UPPER] idle")

        # ---------------- ROTARY ----------------
        if self.dz_rotary_v1 and self.dz_zone_v2:
            in_rotary = (min_dist_to_points(x1, y1, self.dz_rotary_v1) <= self.NEAR_TOL_ROTARY)
            in_zone   = (min_dist_to_points(x2, y2, self.dz_zone_v2) <= self.NEAR_TOL_ROTARY)
            if in_rotary and in_zone:
                limit_candidates[2].append(self.ACCEL_YIELD)
                self._log_zone("ROTARY", ("ACTIVE", 1, 2), "🛑 [ROTARY ACTIVE] priority=CAV1 yield=CAV2")
            else:
                self._log_zone("ROTARY", ("IDLE",), "✅ [ROTARY] idle")

        # ---------------- FW INTERSECTION ----------------
        fw_lim, fw_eff = self._fw_compute_limits()
        if fw_lim.get(1) is not None:
            limit_candidates[1].append(float(fw_lim[1]))
        if fw_lim.get(2) is not None:
            limit_candidates[2].append(float(fw_lim[2]))

        # choose min-speed among candidates
        final_limits = {}
        for vid in (1, 2):
            vals = [v for v in limit_candidates[vid] if v is not None]
            final_limits[vid] = min(vals) if vals else None

        # apply limiter with ramp
        self.target_speed[1] = final_limits[1]
        self.target_speed[2] = final_limits[2]
        self._apply_limit(1, self.target_speed[1])
        self._apply_limit(2, self.target_speed[2])

        # publish FINAL accel: steer from raw, speed=min(raw_speed, guardian_limit)
        v1 = min(float(self.raw1.linear.x), float(GUARDIAN_LIMITS[1]))
        v2 = min(float(self.raw2.linear.x), float(GUARDIAN_LIMITS[2]))

        out1 = Accel()
        out1.linear.x = v1
        out1.angular.z = float(self.raw1.angular.z)
        self.pub1.publish(out1)

        out2 = Accel()
        out2.linear.x = v2
        out2.angular.z = float(self.raw2.angular.z)
        self.pub2.publish(out2)

        # ---------- 로그 출력 ----------
        if self._tick_count % int(1.0 / self.TICK) == 0:
            mode1 = self._infer_mode(self.raw1.angular.z)
            mode2 = self._infer_mode(self.raw2.angular.z)

            print(
                f"CAV 01 : {mode1:<6} | {v1:>5.2f} m/s | steer {self.raw1.angular.z:>6.3f} | "
                f"pose ({self.p1[0]:>6.2f}, {self.p1[1]:>6.2f})"
            )
            print(
                f"CAV 02 : {mode2:<6} | {v2:>5.2f} m/s | steer {self.raw2.angular.z:>6.3f} | "
                f"pose ({self.p2[0]:>6.2f}, {self.p2[1]:>6.2f})"
            )
            print("-" * 72)


# ============================================================
# main
# ============================================================
def main():
    rclpy.init(args=None)

    driver1 = ZonePriorityDriver(1)
    driver2 = ZonePriorityDriver(2)
    guardian_mux = AllZonesGuardianMux()

    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(driver1)
    ex.add_node(driver2)
    ex.add_node(guardian_mux)

    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        driver1.destroy_node()
        driver2.destroy_node()
        guardian_mux.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

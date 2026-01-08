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
# [Geometry helpers]
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

# ============================================================
# [Global Settings]
# ============================================================
TARGET_VELOCITY = 0.48
LOOK_AHEAD_DISTANCE = 0.37
WHEELBASE = 0.211
DIST_CENTER_TO_REAR = WHEELBASE / 2.0

Kp, Ki, Kd = 6.0, 0.055, 1.0
K_cte = 5.0

# guardian이 계산한 속도 제한(최종 속도 상한)
GUARDIAN_LIMITS = {1: 99.0, 2: 99.0}

# ============================================================
# [Driver] publish to *_accel_raw ONLY
# ============================================================
class ZonePriorityDriver(Node):
    def __init__(self, vehicle_id: int):
        super().__init__(f'zone_driver_{vehicle_id}')
        self.vehicle_id = int(vehicle_id)

        if self.vehicle_id == 1:
            self.MY_PATH_FILE = 'path1_1_xplus0p2.json'
            self.MY_TOPIC = '/CAV_01'
        else:
            self.MY_PATH_FILE = 'path1_2_xplus0p2.json'
            self.MY_TOPIC = '/CAV_02'

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)

        # ✅ raw만 publish
        self.accel_raw_pub = self.create_publisher(Accel, f'{self.MY_TOPIC}_accel_raw', 10)
        self.create_subscription(PoseStamped, self.MY_TOPIC, self.pose_callback, qos)

        self.path_points = load_path_points(self.MY_PATH_FILE)
        self.path_x = [p[0] for p in self.path_points]
        self.path_y = [p[1] for p in self.path_points]

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.is_pose_received = False

        self.prev_error = 0.0
        self.integral_error = 0.0
        self.last_time = self.get_clock().now()

        self.create_timer(0.05, self.drive_callback)
        self.log_counter = 0

    def pose_callback(self, msg):
        self.is_pose_received = True

        # 명세: orientation.z == yaw
        raw_yaw = msg.pose.orientation.z
        self.current_yaw = raw_yaw

        # center -> rear 보정
        self.current_x = msg.pose.position.x - (DIST_CENTER_TO_REAR * math.cos(self.current_yaw))
        self.current_y = msg.pose.position.y - (DIST_CENTER_TO_REAR * math.sin(self.current_yaw))

    def drive_callback(self):
        if not self.is_pose_received or not self.path_x:
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0.001:
            return
        if dt > 0.1:
            dt = 0.1

        # nearest index
        min_dist = float('inf')
        current_idx = 0
        for i, (px, py) in enumerate(zip(self.path_x, self.path_y)):
            d = math.hypot(px - self.current_x, py - self.current_y)
            if d < min_dist:
                min_dist = d
                current_idx = i

        # look ahead
        target_idx = current_idx
        for i in range(current_idx, len(self.path_x)):
            if math.hypot(self.path_x[i]-self.current_x, self.path_y[i]-self.current_y) >= LOOK_AHEAD_DISTANCE:
                target_idx = i
                break
        tx, ty = self.path_x[target_idx], self.path_y[target_idx]

        # steering PID-ish
        desired_yaw = math.atan2(ty - self.current_y, tx - self.current_x)
        yaw_err = desired_yaw - self.current_yaw
        while yaw_err > math.pi: yaw_err -= 2*math.pi
        while yaw_err < -math.pi: yaw_err += 2*math.pi

        self.integral_error = max(-1.0, min(1.0, self.integral_error + yaw_err*dt))
        p = Kp * yaw_err
        i = Ki * self.integral_error
        d = Kd * (yaw_err - self.prev_error) / dt
        cte = min_dist * K_cte * (-1 if yaw_err < 0 else 1)

        steer = max(-1.0, min(1.0, p + i + d + cte))
        self.prev_error = yaw_err

        # ✅ 속도는 raw에서 “기본 목표속도”만
        cmd = Accel()
        cmd.linear.x = float(TARGET_VELOCITY)
        cmd.angular.z = float(steer)
        self.accel_raw_pub.publish(cmd)

 
# ============================================================
# [Guardian + Mux] subscribes *_accel_raw, publishes final *_accel
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

        # ---------- final accel publishers (ONLY THIS NODE publishes these) ----------
        self.pub1 = self.create_publisher(Accel, "/CAV_01_accel", 10)
        self.pub2 = self.create_publisher(Accel, "/CAV_02_accel", 10)

        # ====================================================
        # 기존 upper/right/rotary 파일들 (xplus0p2)
        # ====================================================
        self.DZ_RIGHT = "right_1_2_xplus0p2.csv"
        self.DZ_UPPER_V1 = "upper_dz_v1_xplus0p2.csv"
        self.DZ_UPPER_V2 = "upper_dz_v2_xplus0p2.csv"
        self.DZ_ROTARY_V1 = "path1_1_rotary_xplus0p2.csv"
        self.DZ_ZONE_V2   = "path1_2_zone_xplus0p2.csv"

        self.dz_right = load_dz_points(self.DZ_RIGHT)
        self.dz_upper_v1 = load_dz_points(self.DZ_UPPER_V1)
        self.dz_upper_v2 = load_dz_points(self.DZ_UPPER_V2)
        self.dz_rotary_v1 = load_dz_points(self.DZ_ROTARY_V1)
        self.dz_zone_v2 = load_dz_points(self.DZ_ZONE_V2)

        # upper arc-length paths
        self.path1_pts = load_path_points("path1_1_xplus0p2.json")
        self.path2_pts = load_path_points("path1_2_xplus0p2.json")
        self.path1_s = build_cumulative_s(self.path1_pts) if self.path1_pts else []
        self.path2_s = build_cumulative_s(self.path2_pts) if self.path2_pts else []

        # upper merge point (두 경로에 동시에 가장 가까운 점)
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
        # upper/right/rotary 파라미터(너가 쓰던 것 유지)
        # ====================================================
        self.NEAR_TOL_RIGHT, self.SAFETY_DIST_RIGHT = 0.55, 1.0
        self.NEAR_TOL_UPPER, self.SAFETY_DIST_UPPER = 0.8, 1.2
        self.UPPER_RELEASE_DIST = 2.2
        self.NEAR_TOL_ROTARY = 0.4

        self.UPPER_TIE_EPS = 0.10
        self.UPPER_TIE_YIELD_ID = 1

        self.ACCEL_YIELD = 0.20
        self.ACCEL_RESUME = TARGET_VELOCITY
        self.RAMP_PER_SEC = 0.45
        self.TICK = 0.05
        self.MIN_SPEED = 0.2
        self.RESUME_PULSE_TICKS = int(1.5 / self.TICK)

        self.cmd_speed = {1: 99.0, 2: 99.0}       # current limited speed
        self.target_speed = {1: None, 2: None}    # desired limit or None(=no limit)
        self.resume_ticks_left = {1: 0, 2: 0}
        self.upper_lock_yield_id = None

        # ====================================================
        # ✅ 사지교차로(너가 준 최신 IntersectionGuardian 로직) -> speed limit로 적용
        # ====================================================
        self.CENTER = (-2.3351, 0.0)

        self.RADIUS = 1.8333
        self.CORE_RADIUS = 1.05
        self.EXIT_RADIUS = 1.8333

        self.V_NOM = TARGET_VELOCITY
        self.D_SAFE = 1.5
        self.D_CLEAR = 1.2

        self.HYSTERESIS_N = 10

        self.last_pose = {1: None, 2: None}
        self.current_speeds = {1: self.V_NOM, 2: self.V_NOM}
        self.prev_dist_c = {1: 0.0, 2: 0.0}
        self.leader_id = None
        self.exit_counter = 0

        # ==========================
        # ✅ [추가] 로그용 상태 저장
        # ==========================
        self._tick_count = 0
        self._last_zone_state = {}   # zone -> tuple
        self._last_pub_state = None  # (v1,v2,lim1,lim2)
        self._last_leader = None

        # loop
        self.create_timer(self.TICK, self.tick)
        self.get_logger().info("✅ GuardianMux started: raw->final accel (upper/right/rotary + intersection)")

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

    # ---------- speed estimator (intersection uses this) ----------
    def _update_speed_est(self, vid, curr_p):
        if self.last_pose[vid] is not None:
            prev = self.last_pose[vid]
            dist = math.hypot(curr_p[0] - prev[0], curr_p[1] - prev[1])
            v_est = dist / self.TICK
            self.current_speeds[vid] = 0.3 * v_est + 0.7 * self.current_speeds[vid]
        self.last_pose[vid] = curr_p

    # ---------- ramped limiter ----------
    def _apply_limit(self, vid, tgt):
        # tgt: None => no limit (99)
        if tgt is None:
            self.cmd_speed[vid] = 99.0
            GUARDIAN_LIMITS[vid] = 99.0
            return

        cur = self.cmd_speed[vid]
        if cur > 50:
            cur = self.V_NOM
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

    def _cancel_resume(self, vid):
        self.resume_ticks_left[vid] = 0

    # ---------------------------
    # ✅ [추가] 로그 중복 방지 함수
    # ---------------------------
    def _log_zone(self, zone_name: str, state_tuple, text: str):
        prev = self._last_zone_state.get(zone_name)
        if prev != state_tuple:
            print(text)
            self._last_zone_state[zone_name] = state_tuple

    # ============================================================
    # ✅ intersection logic => returns (limit_v1, limit_v2) or (None,None)
    # ============================================================
    def _intersection_limits(self):
        if self.p1 is None or self.p2 is None:
            return (None, None)

        (x1, y1) = self.p1
        (x2, y2) = self.p2

        dist_c1 = math.hypot(x1 - self.CENTER[0], y1 - self.CENTER[1])
        dist_c2 = math.hypot(x2 - self.CENTER[0], y2 - self.CENTER[1])
        dist_between = math.hypot(x1 - x2, y1 - y2)

        target_v1 = self.V_NOM
        target_v2 = self.V_NOM

        # leader 선택
        if self.leader_id is None:
            if dist_c1 < self.RADIUS and dist_c2 < self.RADIUS:
                
                if abs(dist_c1 - dist_c2) < self.RADIUS * 0.1:
                    self.leader_id = 1
                else:
                    self.leader_id = 1 if dist_c1 < dist_c2 else 2
                self.exit_counter = 0

                # ✅ leader 바뀔 때 로그
                if self._last_leader != self.leader_id:
                    print(f"🏁 [INTERSECTION] priority 선정: CAV{self.leader_id} (dist_c1={dist_c1:.2f}, dist_c2={dist_c2:.2f})")
                    self._last_leader = self.leader_id

        if self.leader_id is not None:
            d_l = dist_c1 if self.leader_id == 1 else dist_c2
            prev_d_l = self.prev_dist_c[self.leader_id]

            if d_l > prev_d_l and d_l > self.EXIT_RADIUS:
                self.exit_counter += 1
            else:
                self.exit_counter = 0

            collision_active = (dist_c1 < self.RADIUS and dist_c2 < self.RADIUS and dist_between < self.D_CLEAR)
            core_active = (dist_c1 < self.CORE_RADIUS and dist_c2 < self.CORE_RADIUS)
            slow_active = collision_active or core_active

            follower = 2 if self.leader_id == 1 else 1
            if slow_active and self.exit_counter < self.HYSTERESIS_N:
                ratio = max(0.2, min(1.0, dist_between / self.D_SAFE))
                v_slow = self.V_NOM * ratio

                if self.leader_id == 1:
                    target_v2 = v_slow
                else:
                    target_v1 = v_slow

                # ✅ slow_active 상태 로그(변할 때만)
                self._log_zone(
                    "INTERSECTION",
                    ("ACTIVE", self.leader_id, follower),
                    f"🛑 [INTERSECTION ACTIVE] priority=CAV{self.leader_id} yield=CAV{follower} "
                    f"dist_c1={dist_c1:.2f} dist_c2={dist_c2:.2f} d12={dist_between:.2f} v_slow={v_slow:.2f}"
                )
            else:
                # slow 아닌 상태면 idle 로그
                self._log_zone("INTERSECTION", ("IDLE", self.leader_id), f"✅ [INTERSECTION] idle (priority=CAV{self.leader_id})")

            if self.exit_counter >= self.HYSTERESIS_N:
                print("✅ [INTERSECTION] priority reset (exit confirmed)")
                self.leader_id = None
                self.exit_counter = 0
                self._last_leader = None

        self.prev_dist_c[1] = dist_c1
        self.prev_dist_c[2] = dist_c2

        lim1 = target_v1 if target_v1 < self.V_NOM else None
        lim2 = target_v2 if target_v2 < self.V_NOM else None
        return (lim1, lim2)

    # ============================================================
    # main tick: compute all limits, choose min, publish final accel
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

        # ---------------- UPPER (split DZ, arc-length, toggle 방지) ----------------
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

        # ---------------- ROTARY (in_rotary & in_zone => 2 감속) ----------------
        if self.dz_rotary_v1 and self.dz_zone_v2:
            in_rotary = (min_dist_to_points(x1, y1, self.dz_rotary_v1) <= self.NEAR_TOL_ROTARY)
            in_zone   = (min_dist_to_points(x2, y2, self.dz_zone_v2) <= self.NEAR_TOL_ROTARY)
            if in_rotary and in_zone:
                limit_candidates[2].append(self.ACCEL_YIELD)
                self._log_zone("ROTARY", ("ACTIVE", 1, 2), "🛑 [ROTARY ACTIVE] priority=CAV1 yield=CAV2")
            else:
                self._log_zone("ROTARY", ("IDLE",), "✅ [ROTARY] idle")

        # ---------------- INTERSECTION (사지교차로) ----------------
        lim1_i, lim2_i = self._intersection_limits()
        if lim1_i is not None:
            limit_candidates[1].append(lim1_i)
        if lim2_i is not None:
            limit_candidates[2].append(lim2_i)

        # choose min-speed among candidates (ignore None)
        final_limits = {}
        for vid in (1, 2):
            vals = [v for v in limit_candidates[vid] if v is not None]
            final_limits[vid] = min(vals) if vals else None

        # apply limiter with ramp
        self.target_speed[1] = final_limits[1]
        self.target_speed[2] = final_limits[2]
        self._apply_limit(1, self.target_speed[1])
        self._apply_limit(2, self.target_speed[2])

        # publish FINAL accel: take steering from raw, speed=min(raw_speed, guardian_limit)
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

        # ====================================================
        # ✅ [추가] 최종 출력 속도/리밋 로그
        # - 너무 자주 찍히지 않게: 1초마다 또는 상태 변화 시
        # ====================================================
        state = (round(v1, 3), round(v2, 3), round(GUARDIAN_LIMITS[1], 3), round(GUARDIAN_LIMITS[2], 3))
        if (self._tick_count % int(1.0 / self.TICK) == 0) or (self._last_pub_state != state):
            print(f"[FINAL] out_v1={v1:.3f} out_v2={v2:.3f} | limit1={GUARDIAN_LIMITS[1]:.3f} limit2={GUARDIAN_LIMITS[2]:.3f}")
            self._last_pub_state = state


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


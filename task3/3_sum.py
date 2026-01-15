#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import json
import os
import math
from geometry_msgs.msg import Accel, PoseStamped

# ============================================================
# [COMMON: path loader]
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


# ============================================================
# [Driver params] - 속도 0.7 고정
# ============================================================
FIXED_VELOCITY = 0.7

CURVE_PARAMS = {
    "vel": FIXED_VELOCITY,
    "look_ahead": 0.52,
    "kp": 6.0,
    "ki": 0.05,
    "kd": 1.0,
    "k_cte": 4.0
}

STRAIGHT_PARAMS = {
    "vel": FIXED_VELOCITY,
    "look_ahead": 0.52,
    "kp": 6.0,
    "ki": 0.05,
    "kd": 1.0,
    "k_cte": 4.0
}

WHEELBASE = 0.211
DIST_CENTER_TO_REAR = WHEELBASE / 2.0
TICK_RATE = 0.05

ACCEL_LIMIT = 0.8
DECEL_LIMIT = 3.0

class MapPredictionDriver(Node):

    
    def __init__(self, vehicle_id: int, path_filename: str):
        super().__init__(f'driver_vehicle_{vehicle_id}')
        self.vid = int(vehicle_id)
        self.last_idx = 0  # 인덱스 연속성 유지용
        self.PATH_FILENAME = path_filename
        self.TOPIC = f"/CAV_{self.vid:02d}"

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.path_pts = load_path_points(self.PATH_FILENAME)
        self.path_x = [p[0] for p in self.path_pts]
        self.path_y = [p[1] for p in self.path_pts]

        if not self.path_pts:
            self.get_logger().error(f"❌ [Car{self.vid}] Path missing: {self.PATH_FILENAME}")
        else:
            self.get_logger().info(f"✅ [Car{self.vid}] Path loaded: {self.PATH_FILENAME}")

        self.create_subscription(PoseStamped, self.TOPIC, self.pose_callback, qos_profile)
        self.accel_raw_pub = self.create_publisher(Accel, f"{self.TOPIC}_accel_raw", 10)

        self.curr_x, self.curr_y, self.curr_yaw = 0.0, 0.0, 0.0
        self.got_pose = False
        self.prev_err, self.int_err = 0.0, 0.0
        self.last_time = self.get_clock().now()
        self.current_vel_cmd = FIXED_VELOCITY
        self.create_timer(TICK_RATE, self.drive_loop)
        
        



    def pose_callback(self, msg):
        self.got_pose = True
        self.curr_yaw = float(msg.pose.orientation.z)
        self.curr_x = float(msg.pose.position.x) - (DIST_CENTER_TO_REAR * math.cos(self.curr_yaw))
        self.curr_y = float(msg.pose.position.y) - (DIST_CENTER_TO_REAR * math.sin(self.curr_yaw))

    def get_road_curvature(self, current_idx):
        num_pts = len(self.path_x)
        idx_now = current_idx
        idx_near = (current_idx + 50) % num_pts
        idx_far = (current_idx + 100) % num_pts

        dx1, dy1 = self.path_x[idx_near] - self.path_x[idx_now], self.path_y[idx_near] - self.path_y[idx_now]
        dx2, dy2 = self.path_x[idx_far] - self.path_x[idx_near], self.path_y[idx_far] - self.path_y[idx_near]
        
        diff = math.atan2(dy2, dx2) - math.atan2(dy1, dx1)
        while diff > math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        return abs(diff)
    def find_global_best_index(self):
        min_d = float('inf')
        best_idx = 0
        for idx in range(len(self.path_pts)):
            d = math.hypot(self.path_x[idx] - self.curr_x, self.path_y[idx] - self.curr_y)
            if d < min_d:
                min_d, best_idx = d, idx
        return best_idx

    def drive_loop(self):
        if not self.got_pose or not self.path_pts:
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0.001: return

        # ============================================================
        # [신규 추가] 위치 도약 감지 및 강제 재동기화 로직
        # ============================================================
        # 현재 저장된 인덱스(last_idx)와 실제 위치 간의 거리 계산
        actual_dist = math.hypot(self.path_x[self.last_idx] - self.curr_x, 
                                 self.path_y[self.last_idx] - self.curr_y)

        # 0.8m 이상 차이 나면 수동 이동으로 간주 (임계값은 환경에 따라 조정)
        if actual_dist > 0.8:
            self.get_logger().warn(f"⚠️ [Car{self.vid}] Jump Detected! Global re-localization triggered.")
            self.last_idx = self.find_global_best_index()
            self.int_err = 0.0  # 적분 오차 초기화 (급회전 방지)
            self.prev_err = 0.0 # 이전 오차 초기화
        # ============================================================

        num_pts = len(self.path_x)
        
        # 1) 현재 위치에서 가장 가까운 인덱스 탐색 (Circular)
        min_d = float('inf')
        best_idx = self.last_idx
        search_range = 50
        for i in range(self.last_idx - search_range, self.last_idx + search_range):
            idx = i % num_pts
            d = math.hypot(self.path_x[idx] - self.curr_x, self.path_y[idx] - self.curr_y)
            if d < min_d:
                min_d, best_idx = d, idx
        
        self.last_idx = best_idx
        curr_idx = best_idx
        # 2) 파라미터 결정
        road_curve_amount = self.get_road_curvature(curr_idx)
        if road_curve_amount < 0.15 and min_d < 0.4:
            params = STRAIGHT_PARAMS
            self.mode = "STRGT"
        else:
            params = CURVE_PARAMS
            self.mode = "CURVE"

        # 3) Look-ahead target 탐색
        target_idx = curr_idx
        for i in range(1, num_pts):
            idx = (curr_idx + i) % num_pts
            d = math.hypot(self.path_x[idx] - self.curr_x, self.path_y[idx] - self.curr_y)
            # 반드시 위에서 정의된 params를 사용해야 함
            if d >= params["look_ahead"]:
                target_idx = idx
                break

        tx, ty = self.path_x[target_idx], self.path_y[target_idx]
        desired_yaw = math.atan2(ty - self.curr_y, tx - self.curr_x)
        yaw_err = desired_yaw - self.curr_yaw
        while yaw_err > math.pi: yaw_err -= 2 * math.pi
        while yaw_err < -math.pi: yaw_err += 2 * math.pi

        # 4) PID 및 조향 계산
        self.int_err = max(-1.0, min(1.0, self.int_err + yaw_err * dt))
        p_term = params["kp"] * yaw_err
        i_term = params["ki"] * self.int_err
        d_term = params["kd"] * (yaw_err - self.prev_err) / dt
        cte_term = min_d * params["k_cte"] * (-1.0 if yaw_err < 0 else 1.0)

        final_steer = max(-1.0, min(1.0, float(p_term + i_term + d_term + cte_term)))
        self.prev_err = yaw_err

        # 5) 명령 발행
        cmd = Accel()
        cmd.linear.x = float(self.current_vel_cmd)
        cmd.angular.z = float(final_steer)
        self.accel_raw_pub.publish(cmd)



# ============================================================
# [Guardian + Mux]  (TOP + BOTTOM + CROSS) 
# ============================================================

import math
from geometry_msgs.msg import PoseStamped, Accel
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class Problem3DualZoneGuardianMux(Node):
    def __init__(self):
        super().__init__("problem3_dualzone_guardian_mux")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ---------------- vehicles / topics ----------------
        self.VEH_IDS = [1, 2, 3, 4]
        self.TOPICS = {vid: f"/CAV_{vid:02d}" for vid in self.VEH_IDS}

        # ---------------- speed policy ----------------
        # 드라이버 raw 속도(직진/회전 상관없이) = 0.7
        self.V_NOM = 0.7

        # 3대 이상: 1등/2등/3등/그외
        self.RANK_SPEEDS_3P = [0.7, 0.45, 0.2, 0.2]

        # 2대: 1등/2등
        self.RANK_SPEEDS_2P = [0.7, 0.2]

        # ---------------- zone geometry ----------------
        # TOP center 
        self.TOP_CENTER = (-2.3342, 2.3073)

        # BOTTOM center
        self.BOT_CENTER = (-2.3342, -2.3073)

        # 충돌 구역 반경 
        self.RADIUS = 1.6
        self.EXIT_RADIUS = 0.4

        # approaching 판단
        self.APPROACH_N = 2
        self.EPS = 0.001

        # leaving hysteresis ticks 
        self.HYSTERESIS_N = 5

        # ---------------- timing / ramp ----------------
        self.TICK = 0.05

        # ✅ 램프를 상/하 다르게: 감속은 빠르게, 가속(제한 해제)은 느리게
        self.RAMP_DOWN_PER_SEC = 3.0   # 제한값 내려갈 때 빠르게
        self.RAMP_UP_PER_SEC = 0.30    # 제한값 올라갈 때 느리게
        self.MIN_SPEED = 0.05

        # ---------------- state ----------------
        self.pose = {vid: None for vid in self.VEH_IDS}
        self.raw = {vid: Accel() for vid in self.VEH_IDS}

        # speed estimate (for TTC)
        self.last_pose = {vid: None for vid in self.VEH_IDS}
        self.v_est = {vid: self.V_NOM for vid in self.VEH_IDS}

        # per-zone state
        self.zones = {
            "TOP": self._make_zone_state(self.TOP_CENTER),
            "BOT": self._make_zone_state(self.BOT_CENTER),
        }

        # limiter
        self.cmd_limit = {vid: 99.0 for vid in self.VEH_IDS}
        self.tgt_limit = {vid: None for vid in self.VEH_IDS}

        # ---------------- sub/pub ----------------
        for vid in self.VEH_IDS:
            topic = self.TOPICS[vid]
            self.create_subscription(PoseStamped, topic, self._make_pose_cb(vid), qos)
            self.create_subscription(Accel, f"{topic}_accel_raw", self._make_raw_cb(vid), 10)

        self.pub = {vid: self.create_publisher(Accel, f"{self.TOPICS[vid]}_accel", 10)
                    for vid in self.VEH_IDS}

        self._log_counter = 0
        self.create_timer(self.TICK, self.tick)
        self.get_logger().info("✅ DualZone GuardianMux: 0.7-fixed, approach-based, 2p/3p speeds, ignore {1,2}-only")
        
        # ============================================================
        # [FOUR-WAY INTERSECTION] (사지교차로) settings
        # ============================================================

        # 사지교차로 중심점 
        self.FW_CENTER = (-2.3342, 0.0)

        # 사지교차로 반경 
        self.FW_RADIUS = 2.0
        self.FW_EXIT_RADIUS = 0.4
        self.FW_HYSTERESIS_N = 10  

        # 접근판정(거리 감소) 파라미터
        self.FW_APPROACH_N = 2
        self.FW_EPS = 0.001

        # 사지교차로 속도 정책 
        self.FW_V_NOM = 0.7
        self.FW_RANK_SPEEDS_2P = [0.7, 0.1]           # 2대일 때
        self.FW_RANK_SPEEDS_3P = [0.7, 0.4, 0.1, 0.1] # 3대 이상일 때(꼴찌는 0.3 유지)

        # 사지교차로 상태(active/approaching/hysteresis)
        self.fw = {
            "active": {vid: False for vid in self.VEH_IDS},
            "outside_ticks": {vid: 0 for vid in self.VEH_IDS},
            "prev_dist": {vid: None for vid in self.VEH_IDS},
            "approach_cnt": {vid: 0 for vid in self.VEH_IDS},
            "approaching": {vid: False for vid in self.VEH_IDS},
        }

        # ✅ 10가지 케이스(부분집합 매칭)
        # - case는 (vid, dir) 튜플들의 집합
        self.FW_CASES = [
            # --- 2대 ---
            frozenset([(1,'N'), (2,'W')]),
            frozenset([(1,'N'), (2,'E')]),
            frozenset([(1,'N'), (3,'E')]),
            frozenset([(2,'W'), (3,'E')]),
            frozenset([(1,'S'), (2,'W')]),
            frozenset([(1,'S'), (2,'E')]),
            frozenset([(1,'S'), (4,'W')]),
            frozenset([(2,'E'), (4,'W')]),

            # --- 3대 ---
            frozenset([(1,'N'), (2,'W'), (3,'E')]),
            frozenset([(1,'S'), (2,'E'), (4,'W')]),
        ]

    # ============================================================
    # zone state container
    # ============================================================
    def _make_zone_state(self, center_xy):
        return {
            "CENTER": center_xy,
            "active": {vid: False for vid in self.VEH_IDS},
            "outside_ticks": {vid: 0 for vid in self.VEH_IDS},

            # approach detection
            "prev_dist": {vid: None for vid in self.VEH_IDS},
            "approach_cnt": {vid: 0 for vid in self.VEH_IDS},
            "approaching": {vid: False for vid in self.VEH_IDS},
        }

    # ============================================================
    # callbacks
    # ============================================================
    def _make_pose_cb(self, vid):
        def cb(msg):
            p = (msg.pose.position.x, msg.pose.position.y)
            self.pose[vid] = p
            self._update_speed_est(vid, p)
        return cb

    def _make_raw_cb(self, vid):
        def cb(msg):
            self.raw[vid] = msg
        return cb

    # ============================================================
    # helpers
    # ============================================================
    def _dist(self, p, c):
        return math.hypot(p[0] - c[0], p[1] - c[1])

    def _update_speed_est(self, vid, p):
        prev = self.last_pose[vid]
        if prev is not None:
            d = math.hypot(p[0] - prev[0], p[1] - prev[1])
            v = d / self.TICK
            self.v_est[vid] = 0.35 * v + 0.65 * self.v_est[vid]
        self.last_pose[vid] = p

    def _apply_limit_ramp(self, vid, tgt):
        if tgt is None:
            self.cmd_limit[vid] = 99.0
            return

        cur = self.cmd_limit[vid]
        # 초기 99 상태면 "현재 주행속도 기준"으로 시작
        if cur > 50:
            cur = float(self.raw[vid].linear.x) if self.raw[vid] is not None else self.V_NOM

        step_down = self.RAMP_DOWN_PER_SEC * self.TICK
        step_up   = self.RAMP_UP_PER_SEC * self.TICK

        # tgt > cur : 제한 완화(가속) → 천천히
        # tgt < cur : 제한 강화(감속) → 빠르게
        if tgt > cur:
            cur = min(tgt, cur + step_up)
        else:
            cur = max(tgt, cur - step_down)

        self.cmd_limit[vid] = max(self.MIN_SPEED, float(cur))

    def _rank_by_ttc(self, zone_name, vids):
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
        p = self.pose[vid]
        if p is None:
            return None
        return math.hypot(p[0] - self.FW_CENTER[0], p[1] - self.FW_CENTER[1])

    def _fw_get_direction(self, vid):
        """
        사지교차로 진입 방향 판별: 'N','S','E','W'
    -   중심점 기준으로 어느 쪽에 있나
        """
        p = self.pose[vid]
        if p is None:
            return None
        dx = p[0] - self.FW_CENTER[0]
        dy = p[1] - self.FW_CENTER[1]
        if abs(dx) > abs(dy):
            return 'E' if dx > 0 else 'W'
        else:
            return 'N' if dy > 0 else 'S'


    # ============================================================
    # per-zone update: active + approaching
    # ============================================================
    def _update_zone_flags(self, zone_name):
        z = self.zones[zone_name]
        c = z["CENTER"]

        for vid in self.VEH_IDS:
            p = self.pose[vid]
            if p is None:
                continue

            d = self._dist(p, c)

            # ---- active hysteresis ----
            if d < self.RADIUS:
                z["active"][vid] = True
                z["outside_ticks"][vid] = 0
            elif d > self.EXIT_RADIUS:
                if z["active"][vid]:
                    z["outside_ticks"][vid] += 1
                    if z["outside_ticks"][vid] >= self.HYSTERESIS_N:
                        z["active"][vid] = False
                        z["outside_ticks"][vid] = 0

            # ---- approach detection ----
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

    # ============================================================
    # compute per-zone limits
    # ============================================================
    def _compute_zone_limits(self, zone_name):
        z = self.zones[zone_name]

        # effective in-zone: active AND approaching
        in_eff = [vid for vid in self.VEH_IDS
                  if (self.pose[vid] is not None) and z["active"][vid] and z["approaching"][vid]]

        # algo ON: >=2 vehicles, but ignore only {1,2}
        algo_on = (len(in_eff) >= 2) and not (set(in_eff) == {1, 2})

        limits = {vid: None for vid in self.VEH_IDS}
        if not algo_on:
            return limits, in_eff, False

        # rank by TTC
        rank_list = self._rank_by_ttc(zone_name, in_eff)
        n = len(rank_list)

        speeds = self.RANK_SPEEDS_2P if n == 2 else self.RANK_SPEEDS_3P
        
        if n >= 3:
            top2 = set(rank_list[:2])
            if top2 == {1,2}:
                speeds = [self.V_NOM, self.V_NOM] + speeds[2:]

        for i, vid in enumerate(rank_list):
            desired = speeds[min(i, len(speeds) - 1)]
            # 1등(0.7)은 굳이 제한 걸 필요 없음
            limits[vid] = desired if desired < self.V_NOM else None

        return limits, in_eff, True

    # ============================================================
    # main tick
    # ============================================================
    def tick(self):
        if all(self.pose[vid] is None for vid in self.VEH_IDS):
            return

        # 1) update flags
        self._update_zone_flags("TOP")
        self._update_zone_flags("BOT")

        # 2) compute limits
        top_limits, top_eff, top_on = self._compute_zone_limits("TOP")
        bot_limits, bot_eff, bot_on = self._compute_zone_limits("BOT")
        # ============================================================
        # [FOUR-WAY] update flags (active + approaching)
        # ============================================================
        for vid in self.VEH_IDS:
            p = self.pose[vid]
            if p is None:
                continue

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

             # approaching (distance decreasing)
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

           # ============================================================
           # [FOUR-WAY] effective in-zone set (active AND approaching)
           # ============================================================
        fw_in_eff = [
            vid for vid in self.VEH_IDS
            if (self.pose[vid] is not None)
            and self.fw["active"][vid]
            and self.fw["approaching"][vid]
        ] 

         # ============================================================
         # [FOUR-WAY] CASE SUBSET MATCH
         # - 케이스+@ (추가 차량) 있어도 케이스 차량만 추출해서 적용
         # ============================================================
        fw_dir_map = {}
        for vid in fw_in_eff:
            ddir = self._fw_get_direction(vid)
            if ddir is not None:
                fw_dir_map[vid] = ddir

        active_pairs = set((vid, ddir) for vid, ddir in fw_dir_map.items())

        matched_case = None
        for case in self.FW_CASES:
            if case.issubset(active_pairs):
                matched_case = case
                break

        fw_targets = []
        if matched_case is not None:
            fw_targets = [vid for (vid, _) in matched_case]
            
        # ============================================================
        # [FOUR-WAY] compute limits for fw_targets only
        # ============================================================
        fw_limits = {vid: None for vid in self.VEH_IDS}

        if fw_targets:
            # 대상 차량 수
            n = len(fw_targets)

            # ✅ TTC 대신 "거리" 기준으로 가까운 차량을 우선(속도 동일 0.7 가정이면 거리=우선순위)
            fw_targets_sorted = sorted(
                fw_targets,
                key=lambda vid: (self._fw_dist(vid) if self._fw_dist(vid) is not None else 1e9)
            )

            speeds = self.FW_RANK_SPEEDS_2P if n == 2 else self.FW_RANK_SPEEDS_3P

            for i, vid in enumerate(fw_targets_sorted):
                desired = speeds[min(i, len(speeds) - 1)]
                # raw보다 작은 속도 제한만 의미 있으므로 limit로 적용
                fw_limits[vid] = desired if desired < self.FW_V_NOM else None


        # 3) merge limits (min across zones)
        for vid in self.VEH_IDS:
            cands = []
            if top_limits.get(vid) is not None:
                cands.append(float(top_limits[vid]))
            if bot_limits.get(vid) is not None:
                cands.append(float(bot_limits[vid]))
                # ✅ add four-way limits
            if fw_limits.get(vid) is not None:
                cands.append(float(fw_limits[vid]))
            self.tgt_limit[vid] = min(cands) if cands else None

        # 4) apply ramp
        for vid in self.VEH_IDS:
            self._apply_limit_ramp(vid, self.tgt_limit[vid])

        # 5) publish FINAL accel
        for vid in self.VEH_IDS:
            raw_v = float(self.raw[vid].linear.x)
            lim = float(self.cmd_limit[vid])
            out_v = min(raw_v, lim)

            out = Accel()
            out.linear.x = float(out_v)
            out.angular.z = float(self.raw[vid].angular.z)
            self.pub[vid].publish(out)

        # 6) logs
        self._log_counter += 1
        if self._log_counter % 20 == 0:
            print(f"[GUARD] TOP_on={top_on} eff={top_eff} | BOT_on={bot_on} eff={bot_eff}")
            for vid in self.VEH_IDS:
                print(f"  {vid}: raw={float(self.raw[vid].linear.x):.2f} "
                      f"lim={float(self.cmd_limit[vid]):.2f} out={min(float(self.raw[vid].linear.x), float(self.cmd_limit[vid])):.2f}")


# ============================================================
# main
# ============================================================
def main(args=None):
    rclpy.init(args=args)

    drivers = [MapPredictionDriver(i, f"path3_{i}.json") for i in range(1, 5)]

    guardian = Problem3DualZoneGuardianMux()

    ex = MultiThreadedExecutor(num_threads=10)
    for d in drivers: ex.add_node(d)
    
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


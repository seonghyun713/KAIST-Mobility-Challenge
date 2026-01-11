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
# [Driver params]  (너가 올린 코드 그대로)
# ============================================================
CURVE_PARAMS = {
    "vel": 0.58,
    "look_ahead": 0.38,
    "kp": 6.3,
    "ki": 0.05,
    "kd": 1.0,
    "k_cte": 5.0
}

STRAIGHT_PARAMS = {
    #"vel": 0.7,
    #"look_ahead": 0.42,
    #"kp": 4.0,
    #"ki": 0.1,
    #"kd": 1.5,
    #"k_cte": 0.8
    "vel": 0.7,
    "look_ahead": 0.38,
    "kp": 6.3,
    "ki": 0.05,
    "kd": 1.0,
    "k_cte": 5.0
}

WHEELBASE = 0.211
DIST_CENTER_TO_REAR = WHEELBASE / 2.0
TICK_RATE = 0.05

ACCEL_LIMIT = 0.8
DECEL_LIMIT = 3.0


# ============================================================
# [Driver] publish to *_accel_raw ONLY  (4 vehicles)
# ============================================================
class MapPredictionDriver(Node):
    def __init__(self, vehicle_id: int, path_filename: str):
        super().__init__(f'driver_vehicle_{vehicle_id}')
        self.vid = int(vehicle_id)

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
            self.get_logger().error(f"❌ [Car{self.vid}] Path missing/empty: {self.PATH_FILENAME}")
        else:
            self.get_logger().info(f"✅ [Car{self.vid}] Path loaded: {self.PATH_FILENAME} ({len(self.path_pts)} pts)")

        self.create_subscription(PoseStamped, self.TOPIC, self.pose_callback, qos_profile)

        # ✅ raw only
        self.accel_raw_pub = self.create_publisher(Accel, f"{self.TOPIC}_accel_raw", 10)

        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0
        self.got_pose = False

        self.prev_err = 0.0
        self.int_err = 0.0
        self.last_time = self.get_clock().now()

        self.current_vel_cmd = 0.58
        self.mode = "CURVE"

        self.log_counter = 0
        self.create_timer(TICK_RATE, self.drive_loop)

    def pose_callback(self, msg):
        self.got_pose = True
        self.curr_yaw = float(msg.pose.orientation.z)
        self.curr_x = float(msg.pose.position.x) - (DIST_CENTER_TO_REAR * math.cos(self.curr_yaw))
        self.curr_y = float(msg.pose.position.y) - (DIST_CENTER_TO_REAR * math.sin(self.curr_yaw))

    def get_road_curvature(self, current_idx):
        idx_now = current_idx
        idx_near = min(len(self.path_x) - 1, current_idx + 50)
        idx_far  = min(len(self.path_x) - 1, current_idx + 100)

        if idx_near == idx_far:
            return 0.0

        dx1 = self.path_x[idx_near] - self.path_x[idx_now]
        dy1 = self.path_y[idx_near] - self.path_y[idx_now]
        angle1 = math.atan2(dy1, dx1)

        dx2 = self.path_x[idx_far] - self.path_x[idx_near]
        dy2 = self.path_y[idx_far] - self.path_y[idx_near]
        angle2 = math.atan2(dy2, dx2)

        diff = abs(angle1 - angle2)
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return abs(diff)

    def drive_loop(self):
        if (not self.got_pose) or (not self.path_pts):
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0.001:
            return
        if dt > 0.1:
            dt = 0.1

        # 1) closest point
        min_d = float('inf')
        curr_idx = 0
        for i, (px, py) in enumerate(zip(self.path_x, self.path_y)):
            d = math.hypot(px - self.curr_x, py - self.curr_y)
            if d < min_d:
                min_d = d
                curr_idx = i

        # 2) mode decision (map-based)
        road_curve_amount = self.get_road_curvature(curr_idx)
        if road_curve_amount < 0.15 and min_d < 0.4:
            self.mode = "STRGT"
            params = STRAIGHT_PARAMS
        else:
            self.mode = "CURVE"
            params = CURVE_PARAMS

        # 3) lookahead target
        target_idx = curr_idx
        for i in range(curr_idx, len(self.path_x)):
            d = math.hypot(self.path_x[i] - self.curr_x, self.path_y[i] - self.curr_y)
            if d >= params["look_ahead"]:
                target_idx = i
                break

        tx, ty = self.path_x[target_idx], self.path_y[target_idx]
        desired_yaw = math.atan2(ty - self.curr_y, tx - self.curr_x)
        yaw_err = desired_yaw - self.curr_yaw

        while yaw_err > math.pi:
            yaw_err -= 2 * math.pi
        while yaw_err < -math.pi:
            yaw_err += 2 * math.pi

        self.int_err = max(-1.0, min(1.0, self.int_err + yaw_err * dt))

        p = params["kp"] * yaw_err
        i_term = params["ki"] * self.int_err
        d_term = params["kd"] * (yaw_err - self.prev_err) / dt
        cte = min_d * params["k_cte"] * (-1.0 if yaw_err < 0 else 1.0)

        raw_steer = p + i_term + d_term + cte
        final_steer = max(-1.0, min(1.0, float(raw_steer)))
        self.prev_err = yaw_err

        # 4) speed with accel/decel limits
        target_v = float(params["vel"])
        if target_v > self.current_vel_cmd:
            self.current_vel_cmd = min(target_v, self.current_vel_cmd + ACCEL_LIMIT * dt)
        else:
            self.current_vel_cmd = max(target_v, self.current_vel_cmd - DECEL_LIMIT * dt)

        cmd = Accel()
        cmd.linear.x = float(self.current_vel_cmd)
        cmd.angular.z = float(final_steer)  # ✅ 반드시 float
        self.accel_raw_pub.publish(cmd)

        self.log_counter += 1
        if self.log_counter % 40 == 0:
            print(f"[Car{self.vid} {self.mode}] Vel:{self.current_vel_cmd:.2f} | RoadCurve:{road_curve_amount:.3f} | DistErr:{min_d:.3f}")


# ============================================================
# [Guardian + Mux] for Problem 3 (TOP + BOTTOM)
# ✅ 요구사항 전부 포함 + (고속 1.8 대응) PRE_SLOW_RADIUS 추가
# ============================================================
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

        # ---------------- speeds ----------------
        # V_NOM: 주행코드에서 나올 수 있는 최대 속도(직진 1.8) 기준으로 "제한 걸지 말아야 하는 값"
        self.V_NOM = 0.7

        # (A) PRE_SLOW: 감속반경에 들어오면 일단 0.58로 캡(고속으로 지나가서 감속이 늦는 문제 해결)
        self.ENTER_CAP_SPEED = 0.58

        # (B) CONFLICT: 충돌반경 안에서 2대 이상일 때 랭킹 속도
        self.RANK_SPEEDS_3P = [0.58, 0.44, 0.30, 0.10]   # 3대 이상
        self.RANK_SPEEDS_2P = [0.58, 0.30]              # 2대

        # ---------------- geometry (TOP/BOT) ----------------
        self.TOP_CENTER = (-2.3342, 2.3073)
        self.BOT_CENTER = (-2.3342, -2.3073)  # x축 대칭

        # ✅ 네가 "안정적"이라고 한 값
        self.CONFLICT_RADIUS = 1.8
        self.EXIT_RADIUS = 2.0   # 유지 요청

        # ✅ 고속(1.8) 대응: 충돌반경보다 큰 감속반경(여기서 미리 0.58로 캡)
        #    필요하면 2.6~3.2 정도로 더 키워도 됨.
        self.PRE_SLOW_RADIUS = 2.8

        # approaching 판정
        self.APPROACH_N = 2
        self.EPS = 0.01

        # leaving hysteresis ticks
        self.HYSTERESIS_N = 5

        # limiter ramp (lim이 바뀌는 건 이 램프 때문임)
        self.TICK = 0.05
        self.RAMP_DOWN_PER_SEC = 4.0
        self.RAMP_UP_PER_SEC = 0.60
        self.MIN_SPEED = 0.05

        # ---------------- state ----------------
        self.pose = {vid: None for vid in self.VEH_IDS}
        self.raw = {vid: Accel() for vid in self.VEH_IDS}

        # speed estimate (TTC 용) - raw 속도가 변해도 dist/vel로 대충 순위는 잡힘
        self.last_pose = {vid: None for vid in self.VEH_IDS}
        self.v_est = {vid: self.V_NOM for vid in self.VEH_IDS}

        # limiter
        self.cmd_limit = {vid: 99.0 for vid in self.VEH_IDS}
        self.tgt_limit = {vid: None for vid in self.VEH_IDS}

        # zones
        # ✅ TOP: 직진차=3, (요구사항 유지) 3이 중심점 지나면 1 감속 즉시 해제(2와 무관)
        # ✅ BOT: 직진차=4, 동일 로직(4 지나면 2 감속 해제)로 대칭 적용
        self.zones = {
            "TOP": self._make_zone_state(self.TOP_CENTER, straight_id=3, release_target=1),
            "BOT": self._make_zone_state(self.BOT_CENTER, straight_id=4, release_target=2),
        }

        # ---------------- sub/pub ----------------
        for vid in self.VEH_IDS:
            topic = self.TOPICS[vid]
            self.create_subscription(PoseStamped, topic, self._make_pose_cb(vid), qos)
            self.create_subscription(Accel, f"{topic}_accel_raw", self._make_raw_cb(vid), 10)

        self.pub = {
            vid: self.create_publisher(Accel, f"{self.TOPICS[vid]}_accel", 10)
            for vid in self.VEH_IDS
        }

        self._log_counter = 0
        self.create_timer(self.TICK, self.tick)
        self.get_logger().info("✅ P3 DualZone GuardianMux ready (TOP+BOT, approaching-only, {1,2}-only OFF, 2p/3p split, pass-center release, PRE_SLOW_RADIUS added)")

    # ============================================================
    # zone state
    # ============================================================
    def _make_zone_state(self, center_xy, straight_id, release_target):
        return {
            "CENTER": center_xy,
            "STRAIGHT_ID": int(straight_id),
            "RELEASE_TARGET": int(release_target),

            # active (within conflict radius) + hysteresis
            "active": {vid: False for vid in self.VEH_IDS},
            "outside_ticks": {vid: 0 for vid in self.VEH_IDS},

            # approach detection
            "prev_dist": {vid: None for vid in self.VEH_IDS},
            "approach_cnt": {vid: 0 for vid in self.VEH_IDS},
            "approaching": {vid: False for vid in self.VEH_IDS},

            # pass-center detection for straight vehicle
            "straight_min_dist": float("inf"),
            "straight_passed": False,
        }

    # ============================================================
    # callbacks
    # ============================================================
    def _make_pose_cb(self, vid):
        def cb(msg):
            p = (float(msg.pose.position.x), float(msg.pose.position.y))
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
        if cur > 50:
            cur = self.V_NOM

        step_down = self.RAMP_DOWN_PER_SEC * self.TICK
        step_up = self.RAMP_UP_PER_SEC * self.TICK
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

    # ============================================================
    # update per-zone flags (approaching + active/hysteresis + pass)
    # ============================================================
    def _update_zone_flags(self, zone_name):
        z = self.zones[zone_name]
        c = z["CENTER"]

        # --- update approach for all vehicles ---
        for vid in self.VEH_IDS:
            p = self.pose[vid]
            if p is None:
                continue

            d = self._dist(p, c)
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

            # --- active (conflict radius) ---
            # ✅ "멀어지면 OUT" 요구사항 반영:
            # - d < CONFLICT_RADIUS 이면 active = True
            # - d > CONFLICT_RADIUS 이면 나가면 해제
            # - 또한 "approaching=False(멀어지는 중)" 이고 d > EXIT_RADIUS 이면 빠르게 해제 가능
            if d < self.CONFLICT_RADIUS:
                z["active"][vid] = True
                z["outside_ticks"][vid] = 0
            else:
                if z["active"][vid]:
                    # 밖으로 나간 경우
                    z["outside_ticks"][vid] += 1
                    if z["outside_ticks"][vid] >= self.HYSTERESIS_N:
                        z["active"][vid] = False
                        z["outside_ticks"][vid] = 0

            # 추가 OUT 보강: "멀어지면 zone OUT" (EXIT_RADIUS=0.4 유지)
            if z["active"][vid] and (not z["approaching"][vid]) and (d > self.EXIT_RADIUS):
                z["outside_ticks"][vid] += 1
                if z["outside_ticks"][vid] >= self.HYSTERESIS_N:
                    z["active"][vid] = False
                    z["outside_ticks"][vid] = 0

        # --- pass-center detection for straight vehicle ---
        sid = z["STRAIGHT_ID"]
        pS = self.pose.get(sid, None)
        if pS is None:
            z["straight_min_dist"] = float("inf")
            z["straight_passed"] = False
        else:
            dS = self._dist(pS, c)
            if dS < z["straight_min_dist"]:
                z["straight_min_dist"] = dS
            else:
                # 최소거리 찍고 다시 멀어지기 시작하면 passed
                if (z["straight_min_dist"] < float("inf")) and (dS > z["straight_min_dist"] + 0.02):
                    z["straight_passed"] = True

            # straight가 zone에서 완전히 나가면 리셋
            if not z["active"].get(sid, False):
                z["straight_min_dist"] = float("inf")
                z["straight_passed"] = False

    # ============================================================
    # compute limits for one zone:
    # - pre-slow cap targets (PRE_SLOW_RADIUS)
    # - conflict ranking limits (CONFLICT_RADIUS)
    # ============================================================
    def _compute_zone_effects(self, zone_name):
        z = self.zones[zone_name]
        c = z["CENTER"]

        # (1) pre-slow effective: within PRE_SLOW_RADIUS AND approaching
        pre_eff = []
        for vid in self.VEH_IDS:
            p = self.pose[vid]
            if p is None:
                continue
            if (self._dist(p, c) < self.PRE_SLOW_RADIUS) and z["approaching"][vid]:
                pre_eff.append(vid)

        # ✅ {1,2}만 pre 영역에 있는 경우는 "아예 무시" (불필요 감속 방지)
        if set(pre_eff) == {1, 2}:
            pre_eff = []

        pre_caps = set(pre_eff)  # 한 대만 들어와도 캡(고속 대응)

        # (2) conflict effective: active(conflict radius) AND approaching
        in_eff = [vid for vid in self.VEH_IDS
                  if (self.pose[vid] is not None) and z["active"][vid] and z["approaching"][vid]]

        # ✅ 알고리즘 ON 조건: 2대 이상 + (1,2만은 제외)
        algo_on = (len(in_eff) >= 2) and not (set(in_eff) == {1, 2})

        limits = {vid: None for vid in self.VEH_IDS}
        if not algo_on:
            # 그래도 pre_caps는 캡으로 적용될 수 있음
            return limits, pre_caps

        # rank by TTC
        rank_list = self._rank_by_ttc(zone_name, in_eff)
        n = len(rank_list)

        speeds = self.RANK_SPEEDS_2P if n == 2 else self.RANK_SPEEDS_3P

        # map
        rank_speed_map = {}
        for i, vid in enumerate(rank_list):
            rank_speed_map[vid] = float(speeds[min(i, len(speeds) - 1)])

        # ✅ "직진차가 중심점 지나면 (2와 무관) release_target 즉시 해제" 유지
        if z["straight_passed"]:
            rt = z["RELEASE_TARGET"]
            sid = z["STRAIGHT_ID"]
            # straight가 rt보다 우선(앞)일 때만 해제(원래 요구)
            if (sid in rank_list) and (rt in rank_list) and (rank_list.index(sid) < rank_list.index(rt)):
                rank_speed_map[rt] = self.V_NOM  # 제한 없음으로 만들어버림

        # apply limiter only if < V_NOM
        for vid in in_eff:
            desired = rank_speed_map.get(vid, self.V_NOM)
            limits[vid] = desired if desired < self.V_NOM else None

        return limits, pre_caps

    # ============================================================
    # main tick
    # ============================================================
    def tick(self):
        if all(self.pose[vid] is None for vid in self.VEH_IDS):
            return

        # 1) update zone flags
        self._update_zone_flags("TOP")
        self._update_zone_flags("BOT")

        # 2) per-zone effects
        top_limits, top_caps = self._compute_zone_effects("TOP")
        bot_limits, bot_caps = self._compute_zone_effects("BOT")

        caps_all = set()
        caps_all |= top_caps
        caps_all |= bot_caps

        # 3) merge: min of (top limit, bot limit, ENTER_CAP if in caps)
        for vid in self.VEH_IDS:
            cands = []

            if top_limits.get(vid) is not None:
                cands.append(float(top_limits[vid]))
            if bot_limits.get(vid) is not None:
                cands.append(float(bot_limits[vid]))

            # pre-slow cap
            if vid in caps_all:
                cands.append(float(self.ENTER_CAP_SPEED))

            self.tgt_limit[vid] = min(cands) if cands else None

        # 4) ramp limiter
        for vid in self.VEH_IDS:
            self._apply_limit_ramp(vid, self.tgt_limit[vid])

        # 5) publish final
        for vid in self.VEH_IDS:
            raw_v = float(self.raw[vid].linear.x)
            lim = float(self.cmd_limit[vid])
            out_v = min(raw_v, lim)

            out = Accel()
            out.linear.x = float(out_v)
            out.angular.z = float(self.raw[vid].angular.z)  # raw steering 그대로
            self.pub[vid].publish(out)

        # 6) log
        self._log_counter += 1
        if self._log_counter % 20 == 0:
            def eff_list(zone_name):
                z = self.zones[zone_name]
                in_eff = [vid for vid in self.VEH_IDS
                          if (self.pose[vid] is not None) and z["active"][vid] and z["approaching"][vid]]
                return in_eff

            top_eff = eff_list("TOP")
            bot_eff = eff_list("BOT")
            print(f"[P3] TOP_eff={top_eff} BOT_eff={bot_eff} caps={sorted(list(caps_all))}")
            for vid in self.VEH_IDS:
                raw_v = float(self.raw[vid].linear.x)
                lim = float(self.cmd_limit[vid])
                out_v = min(raw_v, lim)
                print(f"  {vid}: raw={raw_v:.2f} lim={lim:.2f} out={out_v:.2f}")


# ============================================================
# main
# ============================================================
def main(args=None):
    rclpy.init(args=args)

    # ✅ 여기 경로파일만 네 환경에 맞게 지정하면 됨
    # Problem3라면 보통 path3_*.json, Problem1/2라면 path1_*.json
    drivers = [
        MapPredictionDriver(1, "path3_1.json"),
        MapPredictionDriver(2, "path3_2.json"),
        MapPredictionDriver(3, "path3_3.json"),
        MapPredictionDriver(4, "path3_4.json"),
    ]

    guardian = Problem3DualZoneGuardianMux()

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


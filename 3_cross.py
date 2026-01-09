#!/usr/bin/env python3
import os
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
def load_path_points(json_file: str):
    if not os.path.exists(json_file):
        return []
    with open(json_file, "r") as f:
        data = json.load(f)

    pts = []
    if isinstance(data, dict):
        xs = data.get("x") or data.get("X")
        ys = data.get("y") or data.get("Y")
        if xs and ys:
            for x, y in zip(xs, ys):
                pts.append((float(x), float(y)))
    return pts


# ============================================================
# [Global Settings]
# ============================================================
TARGET_VELOCITY = 0.48
LOOK_AHEAD_DISTANCE = 0.37
WHEELBASE = 0.211
DIST_CENTER_TO_REAR = WHEELBASE / 2.0

Kp, Ki, Kd = 6.0, 0.055, 1.0
K_cte = 5.0


# ============================================================
# [Driver] publish to *_accel_raw ONLY
# ============================================================
class ZonePriorityDriver(Node):
    def __init__(self, vehicle_id: int):
        super().__init__(f'zone_driver_{vehicle_id}')
        self.vehicle_id = int(vehicle_id)

        self.MY_PATH_FILE = f'path3_{self.vehicle_id}.json'
        self.MY_TOPIC = f'/CAV_{self.vehicle_id:02d}'

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.accel_raw_pub = self.create_publisher(Accel, f'{self.MY_TOPIC}_accel_raw', 10)
        self.create_subscription(PoseStamped, self.MY_TOPIC, self.pose_callback, qos)

        self.path_points = load_path_points(self.MY_PATH_FILE)
        self.path_x = [p[0] for p in self.path_points]
        self.path_y = [p[1] for p in self.path_points]

        if not self.path_x:
            self.get_logger().error(f"[Car{self.vehicle_id}] Path file missing/empty: {self.MY_PATH_FILE}")
        else:
            self.get_logger().info(f"[Car{self.vehicle_id}] Path loaded: {self.MY_PATH_FILE} ({len(self.path_x)} pts)")

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
        raw_yaw = msg.pose.orientation.z  # spec: orientation.z == yaw

        self.current_yaw = raw_yaw
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

        min_dist = float('inf')
        current_idx = 0
        for i, (px, py) in enumerate(zip(self.path_x, self.path_y)):
            d = math.hypot(px - self.current_x, py - self.current_y)
            if d < min_dist:
                min_dist = d
                current_idx = i

        target_idx = current_idx
        for i in range(current_idx, len(self.path_x)):
            if math.hypot(self.path_x[i] - self.current_x, self.path_y[i] - self.current_y) >= LOOK_AHEAD_DISTANCE:
                target_idx = i
                break

        tx, ty = self.path_x[target_idx], self.path_y[target_idx]

        desired_yaw = math.atan2(ty - self.current_y, tx - self.current_x)
        yaw_err = desired_yaw - self.current_yaw
        while yaw_err > math.pi:
            yaw_err -= 2 * math.pi
        while yaw_err < -math.pi:
            yaw_err += 2 * math.pi

        self.integral_error = max(-1.0, min(1.0, self.integral_error + yaw_err * dt))
        p = Kp * yaw_err
        i = Ki * self.integral_error
        d = Kd * (yaw_err - self.prev_error) / dt
        cte = min_dist * K_cte * (-1 if yaw_err < 0 else 1)

        steer = max(-1.0, min(1.0, p + i + d + cte))
        self.prev_error = yaw_err

        cmd = Accel()
        cmd.linear.x = float(TARGET_VELOCITY)
        cmd.angular.z = float(steer)
        self.accel_raw_pub.publish(cmd)

        self.log_counter += 1
        if self.log_counter % 40 == 0:
            print(f"[Driver{self.vehicle_id}] raw V:{TARGET_VELOCITY:.2f}, steer:{steer:.2f}")


# ============================================================
# [GuardianMux] Top + Bottom using shared logic
# ============================================================
class Problem3DualGuardianMux(Node):
    def __init__(self):
        super().__init__("problem3_dual_guardian_mux")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.VEH_IDS = [1, 2, 3, 4]
        self.TOPICS = {vid: f"/CAV_{vid:02d}" for vid in self.VEH_IDS}

        # ---------------- tuning (네가 안정적이라 한 값) ----------------
        self.V_NOM = 0.48

        self.RADIUS = 1.60
        self.EXIT_RADIUS = 0.40
        self.HYSTERESIS_N = 5

        self.TICK = 0.05
        self.RAMP_PER_SEC = 0.70
        self.MIN_SPEED = 0.05

        self.APPROACH_EPS = 0.01
        self.APPROACH_N = 2

        # 3대 이상: 1/2/3/4등 속도
        self.RANK_SPEEDS_3P = [0.48, 0.34, 0.20, 0.10]
        # 2대: 1등 0.48, 2등 0.20
        self.RANK_SPEEDS_2P = [0.48, 0.20]

        # ---------------- TOP / BOTTOM zones ----------------
        center_top = (-2.3342, 2.3073)
        center_bottom = (center_top[0], -center_top[1])  # x축 대칭

        self.ZONES = {
            "TOP": {
                "center": center_top,
                "straight_id": 3,   # 위쪽 직진차
                "min_dist": float("inf"),
                "passed": False,
            },
            "BOT": {
                "center": center_bottom,
                "straight_id": 4,   # 아래쪽 직진차
                "min_dist": float("inf"),
                "passed": False,
            }
        }

        # ---------------- state ----------------
        self.pose = {vid: None for vid in self.VEH_IDS}
        self.raw = {vid: Accel() for vid in self.VEH_IDS}

        # “반경/히스테리시스”는 zone별로 따로 관리해야 함
        self.active = {z: {vid: False for vid in self.VEH_IDS} for z in self.ZONES.keys()}
        self.outside_ticks = {z: {vid: 0 for vid in self.VEH_IDS} for z in self.ZONES.keys()}

        # approaching도 zone별
        self.prev_dist = {z: {vid: None for vid in self.VEH_IDS} for z in self.ZONES.keys()}
        self.appr_count = {z: {vid: 0 for vid in self.VEH_IDS} for z in self.ZONES.keys()}
        self.approaching = {z: {vid: False for vid in self.VEH_IDS} for z in self.ZONES.keys()}

        # speed estimate(TTC)
        self.last_pose = {vid: None for vid in self.VEH_IDS}
        self.v_est = {vid: self.V_NOM for vid in self.VEH_IDS}

        # limiter (전체에서 최종 하나)
        self.cmd_limit = {vid: 99.0 for vid in self.VEH_IDS}
        self.tgt_limit = {vid: None for vid in self.VEH_IDS}

        # sub/pub
        for vid in self.VEH_IDS:
            topic = self.TOPICS[vid]
            self.create_subscription(PoseStamped, topic, self._make_pose_cb(vid), qos)
            self.create_subscription(Accel, f"{topic}_accel_raw", self._make_raw_cb(vid), 10)

        self.pub = {vid: self.create_publisher(Accel, f"{self.TOPICS[vid]}_accel", 10)
                    for vid in self.VEH_IDS}

        self._log_counter = 0
        self.create_timer(self.TICK, self.tick)
        self.get_logger().info(
            f"✅ Dual GuardianMux ready: TOP(center={center_top}, straight=3) + BOT(center={center_bottom}, straight=4)"
        )

    # ---------- callbacks ----------
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

        step = self.RAMP_PER_SEC * self.TICK
        if tgt > cur:
            cur = min(tgt, cur + step)
        else:
            cur = max(tgt, cur - step)

        self.cmd_limit[vid] = max(self.MIN_SPEED, cur)

    # ---------- per-zone geometry ----------
    def _dist_to_center(self, zone_name, p):
        cx, cy = self.ZONES[zone_name]["center"]
        return math.hypot(p[0] - cx, p[1] - cy)

    def _update_active_flags(self, zone_name):
        for vid in self.VEH_IDS:
            p = self.pose[vid]
            if p is None:
                continue
            d = self._dist_to_center(zone_name, p)

            if d < self.RADIUS:
                self.active[zone_name][vid] = True
                self.outside_ticks[zone_name][vid] = 0
            elif d > self.EXIT_RADIUS:
                if self.active[zone_name][vid]:
                    self.outside_ticks[zone_name][vid] += 1
                    if self.outside_ticks[zone_name][vid] >= self.HYSTERESIS_N:
                        self.active[zone_name][vid] = False
                        self.outside_ticks[zone_name][vid] = 0

    def _update_approaching_flags(self, zone_name):
        for vid in self.VEH_IDS:
            p = self.pose[vid]
            if p is None:
                self.approaching[zone_name][vid] = False
                self.appr_count[zone_name][vid] = 0
                self.prev_dist[zone_name][vid] = None
                continue

            d = self._dist_to_center(zone_name, p)
            pd = self.prev_dist[zone_name][vid]

            if pd is None:
                self.appr_count[zone_name][vid] = 0
                self.approaching[zone_name][vid] = True
            else:
                if d < pd - self.APPROACH_EPS:
                    self.appr_count[zone_name][vid] = min(self.APPROACH_N, self.appr_count[zone_name][vid] + 1)
                elif d > pd + self.APPROACH_EPS:
                    self.appr_count[zone_name][vid] = 0
                self.approaching[zone_name][vid] = (self.appr_count[zone_name][vid] >= self.APPROACH_N)

            self.prev_dist[zone_name][vid] = d

    def _rank_by_ttc(self, zone_name, vids):
        scored = []
        for vid in vids:
            p = self.pose[vid]
            if p is None:
                continue
            dist = self._dist_to_center(zone_name, p)
            v = max(0.05, float(self.v_est[vid]))
            ttc = dist / v
            scored.append((ttc, vid))
        scored.sort(key=lambda x: x[0])
        return [vid for _, vid in scored] if scored else list(vids)

    def _update_straight_passed(self, zone_name):
        sid = self.ZONES[zone_name]["straight_id"]
        if self.pose[sid] is None:
            self.ZONES[zone_name]["min_dist"] = float("inf")
            self.ZONES[zone_name]["passed"] = False
            return

        d = self._dist_to_center(zone_name, self.pose[sid])
        if d < self.ZONES[zone_name]["min_dist"]:
            self.ZONES[zone_name]["min_dist"] = d
        else:
            if d > self.ZONES[zone_name]["min_dist"] + 0.02:
                self.ZONES[zone_name]["passed"] = True

        # sid가 해당 zone에서 벗어나면 리셋
        if not self.active[zone_name].get(sid, False):
            self.ZONES[zone_name]["min_dist"] = float("inf")
            self.ZONES[zone_name]["passed"] = False

    # ---------- compute limits for one zone ----------
    def _zone_limits(self, zone_name):
        # active + approaching으로 in_zone 구성
        in_zone = [vid for vid in self.VEH_IDS
                   if self.active[zone_name][vid] and self.pose[vid] is not None and self.approaching[zone_name][vid]]

        # zone 단위로 “1&2만 들어오면 OFF”
        algo_on = (len(in_zone) >= 2) and not (set(in_zone) == {1, 2})

        # zone별 제한 결과(None = 제한 없음)
        zone_limit = {vid: None for vid in self.VEH_IDS}

        if not algo_on:
            return algo_on, in_zone, [], zone_limit

        rank_list = self._rank_by_ttc(zone_name, in_zone)

        if len(in_zone) == 2:
            speeds = self.RANK_SPEEDS_2P[:]     # [0.48, 0.20]
        else:
            speeds = self.RANK_SPEEDS_3P[:]     # [0.48, 0.34, 0.20, 0.10]

        rank_speed_map = {}
        for i, vid in enumerate(rank_list):
            sp = speeds[min(i, len(speeds) - 1)]
            rank_speed_map[vid] = sp

        # ✅ “직진차가 1보다 우선이고, 직진차가 center 통과하면 1은 즉시 해제”
        sid = self.ZONES[zone_name]["straight_id"]
        if self.ZONES[zone_name]["passed"] and (1 in rank_list) and (sid in rank_list):
            if rank_list.index(sid) < rank_list.index(1):
                rank_speed_map[1] = self.V_NOM

        # limit로 변환
        for vid in in_zone:
            desired = rank_speed_map.get(vid, self.V_NOM)
            zone_limit[vid] = desired if desired < self.V_NOM else None

        return algo_on, in_zone, rank_list, zone_limit

    # ---------- main tick ----------
    def tick(self):
        if all(self.pose[vid] is None for vid in self.VEH_IDS):
            return

        # zone 상태 업데이트
        for zn in self.ZONES.keys():
            self._update_active_flags(zn)
            self._update_approaching_flags(zn)
            self._update_straight_passed(zn)

        # zone별 제한 계산
        zone_infos = {}
        for zn in self.ZONES.keys():
            algo_on, in_zone, rank_list, zlim = self._zone_limits(zn)
            zone_infos[zn] = (algo_on, in_zone, rank_list, zlim)

        # 차량별 최종 제한: zone들에서 나온 제한 중 "가장 느린(최소)" 선택
        for vid in self.VEH_IDS:
            candidates = []
            for zn in self.ZONES.keys():
                v = zone_infos[zn][3].get(vid, None)
                if v is not None:
                    candidates.append(v)
            self.tgt_limit[vid] = min(candidates) if candidates else None

        # limiter ramp 적용
        for vid in self.VEH_IDS:
            self._apply_limit_ramp(vid, self.tgt_limit[vid])

        # publish final accel
        for vid in self.VEH_IDS:
            raw_v = float(self.raw[vid].linear.x)
            lim = float(self.cmd_limit[vid])
            out_v = min(raw_v, lim)

            out = Accel()
            out.linear.x = float(out_v)
            out.angular.z = float(self.raw[vid].angular.z)
            self.pub[vid].publish(out)

        # logs
        self._log_counter += 1
        if self._log_counter % 20 == 0:
            print("========== [P3 DUAL] ==========")
            for zn in self.ZONES.keys():
                algo_on, in_zone, rank_list, _ = zone_infos[zn]
                iz = ",".join(str(v) for v in in_zone) if in_zone else "-"
                rk = ",".join(str(v) for v in rank_list) if rank_list else "-"
                sid = self.ZONES[zn]["straight_id"]
                passed = self.ZONES[zn]["passed"]
                print(f"[{zn}] algo={'ON' if algo_on else 'OFF'} in_zone={iz} rank={rk} straight={sid} passed={passed}")

            for vid in self.VEH_IDS:
                raw_v = float(self.raw[vid].linear.x)
                lim = float(self.cmd_limit[vid])
                out_v = min(raw_v, lim)
                tgt = self.tgt_limit[vid]
                tgt_s = f"{tgt:.2f}" if tgt is not None else "-"
                print(f"  car{vid} raw={raw_v:.2f} lim={lim:.2f} out={out_v:.2f} tgt={tgt_s}")


# ============================================================
# main
# ============================================================
def main():
    rclpy.init(args=None)

    nodes = [
        ZonePriorityDriver(1),
        ZonePriorityDriver(2),
        ZonePriorityDriver(3),
        ZonePriorityDriver(4),
        Problem3DualGuardianMux(),
    ]

    ex = MultiThreadedExecutor(num_threads=8)
    for n in nodes:
        ex.add_node(n)

    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        for n in nodes:
            n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


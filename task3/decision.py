#!/usr/bin/env python3
import os
import math

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Accel, PoseStamped

from control import MapPredictionDriver


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_CONFIG = {
    1: "/home/dongminkim/Desktop/Mobility_Challenge_Simulator/src/central_control/task3/path/path3_1.json",
    2: "/home/dongminkim/Desktop/Mobility_Challenge_Simulator/src/central_control/task3/path/path3_2.json",
    3: "/home/dongminkim/Desktop/Mobility_Challenge_Simulator/src/central_control/task3/path/path3_3.json",
    4: "/home/dongminkim/Desktop/Mobility_Challenge_Simulator/src/central_control/task3/path/path3_4.json",
}

LOG_EVERY_N = 20  # 20틱=1초 (TICK=0.05 기준), 1이면 매틱 출력


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
        self.V_NOM = 0.7
        self.ENTER_CAP_SPEED = 0.58
        self.RANK_SPEEDS_3P = [0.58, 0.44, 0.30, 0.10]
        self.RANK_SPEEDS_2P = [0.58, 0.30]

        # ---------------- geometry (TOP/BOT) ----------------
        self.TOP_CENTER = (-2.3342, 2.3073)
        self.BOT_CENTER = (-2.3342, -2.3073)

        self.CONFLICT_RADIUS = 1.8
        self.EXIT_RADIUS = 2.0
        self.PRE_SLOW_RADIUS = 2.8

        # approaching
        self.APPROACH_N = 2
        self.EPS = 0.01

        # leaving hysteresis ticks
        self.HYSTERESIS_N = 5

        # limiter ramp
        self.TICK = 0.05
        self.RAMP_DOWN_PER_SEC = 4.0
        self.RAMP_UP_PER_SEC = 0.60
        self.MIN_SPEED = 0.05

        # ---------------- state ----------------
        # pose: (x, y, yaw)
        self.pose = {vid: None for vid in self.VEH_IDS}
        self.raw = {vid: Accel() for vid in self.VEH_IDS}

        # speed estimate (TTC)
        self.last_xy = {vid: None for vid in self.VEH_IDS}
        self.v_est = {vid: self.V_NOM for vid in self.VEH_IDS}

        # limiter
        self.cmd_limit = {vid: 99.0 for vid in self.VEH_IDS}
        self.tgt_limit = {vid: None for vid in self.VEH_IDS}

        # zones
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

        # logging
        self._log_counter = 0

        self.create_timer(self.TICK, self.tick)

    # ============================================================
    # zone state
    # ============================================================
    def _make_zone_state(self, center_xy, straight_id, release_target):
        return {
            "CENTER": center_xy,
            "STRAIGHT_ID": int(straight_id),
            "RELEASE_TARGET": int(release_target),

            "active": {vid: False for vid in self.VEH_IDS},
            "outside_ticks": {vid: 0 for vid in self.VEH_IDS},

            "prev_dist": {vid: None for vid in self.VEH_IDS},
            "approach_cnt": {vid: 0 for vid in self.VEH_IDS},
            "approaching": {vid: False for vid in self.VEH_IDS},

            "straight_min_dist": float("inf"),
            "straight_passed": False,
        }

    # ============================================================
    # callbacks
    # ============================================================
    def _make_pose_cb(self, vid):
        def cb(msg: PoseStamped):
            x = float(msg.pose.position.x)
            y = float(msg.pose.position.y)
            yaw = float(msg.pose.orientation.z)  # 너 코드 기준 유지

            self.pose[vid] = (x, y, yaw)
            self._update_speed_est(vid, (x, y))
        return cb

    def _make_raw_cb(self, vid):
        def cb(msg: Accel):
            self.raw[vid] = msg
        return cb

    # ============================================================
    # helpers
    # ============================================================
    @staticmethod
    def _dist_xy(pxy, cxy):
        return math.hypot(pxy[0] - cxy[0], pxy[1] - cxy[1])

    def _update_speed_est(self, vid, xy):
        prev = self.last_xy[vid]
        if prev is not None:
            d = math.hypot(xy[0] - prev[0], xy[1] - prev[1])
            v = d / self.TICK
            self.v_est[vid] = 0.35 * v + 0.65 * self.v_est[vid]
        self.last_xy[vid] = xy

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
            dist = self._dist_xy((p[0], p[1]), c)
            v = max(0.05, float(self.v_est[vid]))
            ttc = dist / v
            scored.append((ttc, vid))

        scored.sort(key=lambda x: x[0])
        return [vid for _, vid in scored]

    # ============================================================
    # update per-zone flags
    # ============================================================
    def _update_zone_flags(self, zone_name):
        z = self.zones[zone_name]
        c = z["CENTER"]

        for vid in self.VEH_IDS:
            p = self.pose[vid]
            if p is None:
                continue

            d = self._dist_xy((p[0], p[1]), c)
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

            if d < self.CONFLICT_RADIUS:
                z["active"][vid] = True
                z["outside_ticks"][vid] = 0
            else:
                if z["active"][vid]:
                    z["outside_ticks"][vid] += 1
                    if z["outside_ticks"][vid] >= self.HYSTERESIS_N:
                        z["active"][vid] = False
                        z["outside_ticks"][vid] = 0

            if z["active"][vid] and (not z["approaching"][vid]) and (d > self.EXIT_RADIUS):
                z["outside_ticks"][vid] += 1
                if z["outside_ticks"][vid] >= self.HYSTERESIS_N:
                    z["active"][vid] = False
                    z["outside_ticks"][vid] = 0

        # pass-center detection for straight vehicle
        sid = z["STRAIGHT_ID"]
        pS = self.pose.get(sid, None)
        if pS is None:
            z["straight_min_dist"] = float("inf")
            z["straight_passed"] = False
        else:
            dS = self._dist_xy((pS[0], pS[1]), c)
            if dS < z["straight_min_dist"]:
                z["straight_min_dist"] = dS
            else:
                if (z["straight_min_dist"] < float("inf")) and (dS > z["straight_min_dist"] + 0.02):
                    z["straight_passed"] = True

            if not z["active"].get(sid, False):
                z["straight_min_dist"] = float("inf")
                z["straight_passed"] = False

    # ============================================================
    # compute zone effects
    # ============================================================
    def _compute_zone_effects(self, zone_name):
        z = self.zones[zone_name]
        c = z["CENTER"]

        # pre-slow caps
        pre_eff = []
        for vid in self.VEH_IDS:
            p = self.pose[vid]
            if p is None:
                continue
            if (self._dist_xy((p[0], p[1]), c) < self.PRE_SLOW_RADIUS) and z["approaching"][vid]:
                pre_eff.append(vid)

        if set(pre_eff) == {1, 2}:
            pre_eff = []

        pre_caps = set(pre_eff)

        # conflict effective
        in_eff = []
        for vid in self.VEH_IDS:
            if self.pose[vid] is None:
                continue
            if z["active"][vid] and z["approaching"][vid]:
                in_eff.append(vid)

        algo_on = (len(in_eff) >= 2) and not (set(in_eff) == {1, 2})

        limits = {vid: None for vid in self.VEH_IDS}
        if not algo_on:
            return limits, pre_caps

        rank_list = self._rank_by_ttc(zone_name, in_eff)
        n = len(rank_list)
        speeds = self.RANK_SPEEDS_2P if n == 2 else self.RANK_SPEEDS_3P

        rank_speed_map = {}
        for i, vid in enumerate(rank_list):
            rank_speed_map[vid] = float(speeds[min(i, len(speeds) - 1)])

        # pass-center release
        if z["straight_passed"]:
            rt = z["RELEASE_TARGET"]
            sid = z["STRAIGHT_ID"]
            if (sid in rank_list) and (rt in rank_list) and (rank_list.index(sid) < rank_list.index(rt)):
                rank_speed_map[rt] = self.V_NOM

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

        self._update_zone_flags("TOP")
        self._update_zone_flags("BOT")

        top_limits, top_caps = self._compute_zone_effects("TOP")
        bot_limits, bot_caps = self._compute_zone_effects("BOT")

        caps_all = set()
        caps_all |= top_caps
        caps_all |= bot_caps

        for vid in self.VEH_IDS:
            cands = []
            if top_limits.get(vid) is not None:
                cands.append(float(top_limits[vid]))
            if bot_limits.get(vid) is not None:
                cands.append(float(bot_limits[vid]))
            if vid in caps_all:
                cands.append(float(self.ENTER_CAP_SPEED))
            self.tgt_limit[vid] = min(cands) if cands else None

        for vid in self.VEH_IDS:
            self._apply_limit_ramp(vid, self.tgt_limit[vid])

        # publish final
        for vid in self.VEH_IDS:
            raw_v = float(self.raw[vid].linear.x)
            lim = float(self.cmd_limit[vid])
            out_v = min(raw_v, lim)

            out = Accel()
            out.linear.x = float(out_v)
            out.angular.z = float(self.raw[vid].angular.z)
            self.pub[vid].publish(out)

        # 4-vehicles group log, then '---'
        self._log_counter += 1
        if LOG_EVERY_N > 0 and (self._log_counter % LOG_EVERY_N == 0):
            for vid in self.VEH_IDS:
                path_name = os.path.basename(PATH_CONFIG[vid])
                linear = float(self.raw[vid].linear.x)
                yaw = 0.0
                if self.pose[vid] is not None:
                    yaw = float(self.pose[vid][2])
                print(f"CAV{vid} | {path_name} | Linear:{linear:.2f} | Yaw:{yaw:.3f}")
            print("---")


def main(args=None):
    rclpy.init(args=args)

    drivers = [
        MapPredictionDriver(1, PATH_CONFIG[1]),
        MapPredictionDriver(2, PATH_CONFIG[2]),
        MapPredictionDriver(3, PATH_CONFIG[3]),
        MapPredictionDriver(4, PATH_CONFIG[4]),
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

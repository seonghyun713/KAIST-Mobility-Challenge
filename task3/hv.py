#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped

TICK = 0.1

# 너 decision.py에서 쓰던 값 그대로
TOP_CENTER = (-2.3342,  2.3073)
BOT_CENTER = (-2.3342, -2.3073)

PRE_SLOW_RADIUS  = 2.8
CONFLICT_RADIUS  = 1.8

CAV_IDS = [1, 2, 3, 4]
HV_IDS  = [19, 20]

CAV_TOPICS = {vid: f"/CAV_{vid:02d}" for vid in CAV_IDS}
HV_TOPICS  = {19: "/HV_19", 20: "/HV_20"}   # ✅ 너가 말한 그대로


def dist_xy(p, c):
    return math.hypot(p[0] - c[0], p[1] - c[1])


class HVWatch(Node):
    def __init__(self):
        super().__init__("hv_watch")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.cav_pose = {vid: None for vid in CAV_IDS}
        self.hv_pose  = {hid: None for hid in HV_IDS}

        for vid in CAV_IDS:
            self.create_subscription(PoseStamped, CAV_TOPICS[vid], self._mk_cav_cb(vid), qos)

        for hid in HV_IDS:
            self.create_subscription(PoseStamped, HV_TOPICS[hid], self._mk_hv_cb(hid), qos)

        self.create_timer(TICK, self.tick)

    def _mk_cav_cb(self, vid):
        def cb(msg: PoseStamped):
            self.cav_pose[vid] = (
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.orientation.z),  # 너 기준 yaw=z
            )
        return cb

    def _mk_hv_cb(self, hid):
        def cb(msg: PoseStamped):
            self.hv_pose[hid] = (
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.orientation.z),
            )
        return cb

    def _zone_str(self, p):
        dT = dist_xy(p, TOP_CENTER)
        dB = dist_xy(p, BOT_CENTER)
        near = "TOP" if dT < dB else "BOT"
        dmin = min(dT, dB)

        flag = ""
        if dmin < CONFLICT_RADIUS:
            flag = "IN"
        elif dmin < PRE_SLOW_RADIUS:
            flag = "PRE"

        return near, dmin, flag

    def tick(self):
        hv_lines = []
        for hid in HV_IDS:
            p = self.hv_pose[hid]
            if p is None:
                hv_lines.append(f"HV{hid}:None")
                continue
            near, dmin, flag = self._zone_str(p)
            hv_lines.append(f"HV{hid}:{near} d={dmin:.2f} {flag}")

        cav_lines = []
        for vid in CAV_IDS:
            p = self.cav_pose[vid]
            if p is None:
                cav_lines.append(f"CAV{vid}:None")
                continue
            near, dmin, flag = self._zone_str(p)
            cav_lines.append(f"CAV{vid}:{near} d={dmin:.2f} {flag}")

        print(" | ".join(hv_lines))
        print(" | ".join(cav_lines))
        print("---")


def main():
    rclpy.init()
    node = HVWatch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

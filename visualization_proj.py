#!/usr/bin/env python3
import json
import math
from collections import deque
from typing import Dict, List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker


# ----------------------------
# Config
# ----------------------------
PATH_FILES = [
    '/home/dongminkim/Desktop/Mobility_Challenge_Simulator/src/central_control/path_2_1.json',
    '/home/dongminkim/Desktop/Mobility_Challenge_Simulator/src/central_control/path_2_2.json',
    '/home/dongminkim/Desktop/Mobility_Challenge_Simulator/src/central_control/path_2_3.json',
    '/home/dongminkim/Desktop/Mobility_Challenge_Simulator/src/central_control/path_2_4.json'
]

LANE_COUNT_FOR_PROJ = 4      # project onto lane 1~3 only
HV_IDS = list(range(19, 37)) # /HV_19 ... /HV_36

FRAME_ID = "map"

# path markers
PATH_LINE_WIDTH = 0.03

# hv / proj markers
POINT_P_SCALE = 0.08
POINT_Q_SCALE = 0.06
LINE_PQ_WIDTH = 0.02

# local search
NEAREST_WINDOW = 250         # path ds=0.05 -> 250 ~ 12.5m
GLOBAL_WINDOW = 10**9

# recovery
RECOVER_DIST = 0.6           # if local dist > this -> redo global

# trail (accumulated best-Q)
TRAIL_ENABLE = False
TRAIL_MAXLEN = 2500          # points per HV
TRAIL_LINE_WIDTH = 0.03
TRAIL_Z = 0.06
TRAIL_MAX_DIST = 0.5         # do NOT append if best_dist > this (prevents crazy far dot)

# debug
DEBUG_ENABLE = True
DEBUG_HV_ID = 21
DEBUG_PRINT_PERIOD = 0.5     # seconds
JUMP_WARN_DIST = 1.5         # warn if best_dist > this

# republish paths so RViz late-join shows them
PATH_REPUB_PERIOD = 1.0      # seconds


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ----------------------------
# HV State
# ----------------------------
class HVState:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.valid = False

        # per-lane hint cache: lane_idx -> seg_i
        self.seg_hint: Dict[int, int] = {}


# ----------------------------
# Main Node
# ----------------------------
class ProjectionVizNode(Node):
    def __init__(self):
        super().__init__('projection_viz_node')

        # Subscribers (HV only)
        self.hvs: Dict[int, HVState] = {hv_id: HVState() for hv_id in HV_IDS}
        self.hv_subs = []
        for hv_id in HV_IDS:
            topic = f'/HV_{hv_id}'
            sub = self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, hv_id=hv_id: self.hv_pose_callback(hv_id, msg),
                qos_profile_sensor_data
            )
            self.hv_subs.append(sub)

        # Publishers (RViz markers)
        self.path_pub = self.create_publisher(Marker, '/viz/paths', 10)
        self.viz_pub = self.create_publisher(Marker, '/viz/projection', 50)

        # Load paths + preprocess arc-length
        self.paths = [self.load_path(p) for p in PATH_FILES]
        for p in self.paths:
            p['s'] = self.preprocess_path(p['x'], p['y'])

        # path repub
        self.last_path_pub = 0.0

        # trail buffers per HV
        self.trail: Dict[int, deque] = {hv_id: deque(maxlen=TRAIL_MAXLEN) for hv_id in HV_IDS}

        # debug throttling
        self.last_debug_print = 0.0

        # loop timer
        self.dt = 0.1
        self.timer = self.create_timer(self.dt, self.loop)

        self.get_logger().info("ProjectionVizNode started (best-lane only + optional trail).")

    # ---------- Path loading ----------
    def load_path(self, file: str) -> Dict[str, List[float]]:
        with open(file, 'r') as f:
            data = json.load(f)
        xs = list(map(float, data['X']))
        ys = list(map(float, data['Y']))
        if len(xs) != len(ys) or len(xs) < 2:
            raise RuntimeError(f'Bad path file: {file}')
        return {'x': xs, 'y': ys}

    def preprocess_path(self, xs: List[float], ys: List[float]) -> List[float]:
        s = [0.0]
        for i in range(1, len(xs)):
            s.append(s[-1] + math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1]))
        return s

    # ---------- HV callback ----------
    def hv_pose_callback(self, hv_id: int, msg: PoseStamped):
        hv = self.hvs[hv_id]
        hv.x = msg.pose.position.x
        hv.y = msg.pose.position.y
        hv.valid = True

    # ---------- Segment projection ----------
    def project_to_lane(
        self,
        lane_idx: int,
        x: float,
        y: float,
        hint: int = 0,
        window: int = NEAREST_WINDOW
    ) -> Tuple[float, float, int, float, float, float, float]:
        """
        Returns:
          s_proj, d_proj, seg_i, u, dist, qx, qy
        """
        px = self.paths[lane_idx]['x']
        py = self.paths[lane_idx]['y']
        ps = self.paths[lane_idx]['s']

        n = len(px)
        if n < 2:
            return 0.0, 0.0, 0, 0.0, 1e9, 0.0, 0.0

        hint = int(clamp(hint, 0, n - 2))

        a = max(0, hint - window)
        b = min(n - 1, hint + window + 1)  # ensure i+1 valid

        best_dist2 = 1e30
        best_i = a
        best_u = 0.0
        best_s = ps[a]
        best_d = 0.0
        best_qx = px[a]
        best_qy = py[a]

        for i in range(a, b - 1):
            x1, y1 = px[i], py[i]
            x2, y2 = px[i + 1], py[i + 1]

            vx = x2 - x1
            vy = y2 - y1
            seg_len2 = vx * vx + vy * vy
            if seg_len2 < 1e-12:
                continue

            wx = x - x1
            wy = y - y1
            u = (wx * vx + wy * vy) / seg_len2
            u = clamp(u, 0.0, 1.0)

            qx = x1 + u * vx
            qy = y1 + u * vy

            dx = x - qx
            dy = y - qy
            dist2 = dx * dx + dy * dy

            if dist2 < best_dist2:
                best_dist2 = dist2
                best_i = i
                best_u = u
                best_qx = qx
                best_qy = qy

                ds_seg = ps[i + 1] - ps[i]
                best_s = ps[i] + u * ds_seg

                cross = vx * dy - vy * dx
                mag = math.sqrt(dist2)
                best_d = mag if cross > 0.0 else -mag

        return best_s, best_d, best_i, best_u, abs(best_d), best_qx, best_qy

    # ---------- Robust wrapper (global init + recovery) ----------
    def robust_project(
        self,
        hv: HVState,
        lane_idx: int,
        x: float,
        y: float
    ) -> Tuple[float, float, int, float, float, float, float, str]:
        """
        Returns:
          s, d, seg_i, u, dist, qx, qy, mode
        mode: "global_init" | "local" | "recover_global"
        """
        hint_opt: Optional[int] = hv.seg_hint.get(lane_idx, None)

        # 1) 초기에는 global scan
        if hint_opt is None:
            s, d, seg_i, u, dist, qx, qy = self.project_to_lane(
                lane_idx, x, y, hint=0, window=GLOBAL_WINDOW
            )
            hv.seg_hint[lane_idx] = seg_i
            return s, d, seg_i, u, dist, qx, qy, "global_init"

        # 2) 평소 local scan
        hint = hint_opt
        s, d, seg_i, u, dist, qx, qy = self.project_to_lane(
            lane_idx, x, y, hint=hint, window=NEAREST_WINDOW
        )
        mode = "local"

        # 3) 이상치면 global로 회복
        if dist > RECOVER_DIST:
            s2, d2, seg_i2, u2, dist2, qx2, qy2 = self.project_to_lane(
                lane_idx, x, y, hint=0, window=GLOBAL_WINDOW
            )
            if dist2 < dist:
                s, d, seg_i, u, dist, qx, qy = s2, d2, seg_i2, u2, dist2, qx2, qy2
                mode = "recover_global"

        hv.seg_hint[lane_idx] = seg_i
        return s, d, seg_i, u, dist, qx, qy, mode

    # ---------- RViz: publish paths ----------
    def publish_paths(self):
        now = self.get_clock().now().to_msg()

        colors = [
            (1.0, 0.0, 0.0),  # lane 1
            (0.0, 1.0, 0.0),  # lane 2
            (0.0, 0.0, 1.0),  # lane 3
            (1.0, 1.0, 1.0),  # ref
        ]

        for lane_idx, path in enumerate(self.paths):
            m = Marker()
            m.header.frame_id = FRAME_ID
            m.header.stamp = now
            m.ns = "lane_path"
            m.id = lane_idx
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = PATH_LINE_WIDTH

            r, g, b = colors[lane_idx] if lane_idx < len(colors) else (1.0, 1.0, 1.0)
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 1.0

            m.points = []
            for x, y in zip(path['x'], path['y']):
                p = Point()
                p.x = x
                p.y = y
                p.z = 0.0
                m.points.append(p)

            self.path_pub.publish(m)

    # ---------- RViz: publish current HV + current best Q + current line ----------
    def publish_current_best(self, hv_id: int, lane_idx: int, hv_x: float, hv_y: float, qx: float, qy: float):
        now = self.get_clock().now().to_msg()

        lane_colors = [
            (1.0, 0.4, 0.4),  # lane 1 tint
            (0.4, 1.0, 0.4),  # lane 2 tint
            (0.4, 0.4, 1.0),  # lane 3 tint
        ]
        r, g, b = lane_colors[lane_idx] if lane_idx < 3 else (1.0, 1.0, 1.0)

        # HV point P (id fixed per HV)
        mp = Marker()
        mp.header.frame_id = FRAME_ID
        mp.header.stamp = now
        mp.ns = "hv_point"
        mp.id = hv_id
        mp.type = Marker.SPHERE
        mp.action = Marker.ADD
        mp.pose.position.x = hv_x
        mp.pose.position.y = hv_y
        mp.pose.position.z = 0.10
        mp.scale.x = mp.scale.y = mp.scale.z = POINT_P_SCALE
        mp.color.r = 1.0
        mp.color.g = 0.0
        mp.color.b = 0.0
        mp.color.a = 1.0
        self.viz_pub.publish(mp)

        # Current best projection Q (id fixed per HV)
        mq = Marker()
        mq.header.frame_id = FRAME_ID
        mq.header.stamp = now
        mq.ns = "best_proj_point"
        mq.id = hv_id
        mq.type = Marker.SPHERE
        mq.action = Marker.ADD
        mq.pose.position.x = qx
        mq.pose.position.y = qy
        mq.pose.position.z = 0.10
        mq.scale.x = mq.scale.y = mq.scale.z = POINT_Q_SCALE
        mq.color.r = r
        mq.color.g = g
        mq.color.b = b
        mq.color.a = 1.0
        self.viz_pub.publish(mq)

        # Line P->Q (id fixed per HV)
        ml = Marker()
        ml.header.frame_id = FRAME_ID
        ml.header.stamp = now
        ml.ns = "best_proj_line"
        ml.id = hv_id
        ml.type = Marker.LINE_LIST
        ml.action = Marker.ADD
        ml.scale.x = LINE_PQ_WIDTH
        ml.color.r = r
        ml.color.g = g
        ml.color.b = b
        ml.color.a = 1.0

        p1 = Point()
        p1.x = hv_x
        p1.y = hv_y
        p1.z = 0.05
        p2 = Point()
        p2.x = qx
        p2.y = qy
        p2.z = 0.05
        ml.points = [p1, p2]
        self.viz_pub.publish(ml)

    # ---------- RViz: publish trail (LINE_STRIP) ----------
    def publish_trail(self, hv_id: int, lane_idx: int):
        if not TRAIL_ENABLE:
            return
        pts = self.trail[hv_id]
        if len(pts) < 2:
            return

        now = self.get_clock().now().to_msg()

        lane_colors = [
            (1.0, 0.4, 0.4),  # lane 1 tint
            (0.4, 1.0, 0.4),  # lane 2 tint
            (0.4, 0.4, 1.0),  # lane 3 tint
        ]
        r, g, b = lane_colors[lane_idx] if lane_idx < 3 else (1.0, 1.0, 1.0)

        m = Marker()
        m.header.frame_id = FRAME_ID
        m.header.stamp = now
        m.ns = "best_proj_trail"
        m.id = hv_id
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = TRAIL_LINE_WIDTH
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 1.0
        m.points = []

        for (x, y) in pts:
            p = Point()
            p.x = x
            p.y = y
            p.z = TRAIL_Z
            m.points.append(p)

        self.viz_pub.publish(m)

    # ---------- Debug text marker ----------
    def publish_debug_text(self, hv_id: int, x: float, y: float, text: str):
        now = self.get_clock().now().to_msg()

        mt = Marker()
        mt.header.frame_id = FRAME_ID
        mt.header.stamp = now
        mt.ns = "debug_text"
        mt.id = hv_id
        mt.type = Marker.TEXT_VIEW_FACING
        mt.action = Marker.ADD
        mt.pose.position.x = x
        mt.pose.position.y = y
        mt.pose.position.z = 0.35
        mt.scale.z = 0.18
        mt.color.r = 1.0
        mt.color.g = 1.0
        mt.color.b = 1.0
        mt.color.a = 1.0
        mt.text = text
        self.viz_pub.publish(mt)

    # ---------- Loop ----------
    def loop(self):
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # republish paths periodically (RViz late join safe)
        if (now_sec - self.last_path_pub) > PATH_REPUB_PERIOD:
            self.publish_paths()
            self.last_path_pub = now_sec

        for hv_id, hv in self.hvs.items():
            if not hv.valid:
                continue

            # compute projections for lane 1~3
            lane_results = []
            for lane_idx in range(LANE_COUNT_FOR_PROJ):
                s, d, seg_i, u, dist, qx, qy, mode = self.robust_project(hv, lane_idx, hv.x, hv.y)
                lane_results.append((dist, lane_idx, qx, qy, s, d, seg_i, mode))

            # choose best lane only
            lane_results.sort(key=lambda t: t[0])
            best_dist, best_lane, qx, qy, s, d, seg_i, mode = lane_results[0]

            # warn if something is clearly wrong
            if best_dist > JUMP_WARN_DIST:
                self.get_logger().warn(
                    f"[JUMP?] HV{hv_id} pos=({hv.x:.2f},{hv.y:.2f}) best=L{best_lane+1} dist={best_dist:.2f} mode={mode}"
                )

            # publish current best markers
            self.publish_current_best(hv_id, best_lane, hv.x, hv.y, qx, qy)

            # trail: append only if reasonable
            if TRAIL_ENABLE and best_dist <= TRAIL_MAX_DIST:
                self.trail[hv_id].append((qx, qy))
            # publish trail (best lane color)
            self.publish_trail(hv_id, best_lane)

            # debug only for one HV
            if DEBUG_ENABLE and (DEBUG_HV_ID is not None) and hv_id == DEBUG_HV_ID:
                if (now_sec - self.last_debug_print) > DEBUG_PRINT_PERIOD:
                    msg = " | ".join(
                        [f"L{lr[1]+1}:d={lr[0]:.3f},seg={lr[6]},mode={lr[7]}"
                         for lr in lane_results]
                    )
                    self.get_logger().info(f"HV{hv_id} @ ({hv.x:.2f},{hv.y:.2f}) -> {msg}")
                    self.last_debug_print = now_sec

                text = f"HV{hv_id}  best=L{best_lane+1}\n" \
                       f"dist={best_dist:.2f}  mode={mode}\n" \
                       f"seg={seg_i}  s={s:.1f}"
                self.publish_debug_text(hv_id, hv.x, hv.y, text)


def main():
    rclpy.init()
    node = ProjectionVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

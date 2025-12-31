#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import csv
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Accel
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker

from common import quat_to_yaw, clamp

Point = Tuple[float, float]

WAYPOINT_DIR = os.path.expanduser("~/Desktop/Mobility_Challenge_Simulator/tool/waypoint")

# ========= 특정 경로만 =========
ROUTE = [21, 51, 46, 40, 63, 34, 27, 31, 1, 3, 7, 9, 56, 59, 18, 21]

# ========= 주행/제어 =========
CONTROL_DT = 0.05  # 20Hz

# 속도 (일단 튜닝 쉬우라고 고정)
V_CMD = 0.35

# 조향(각속도) 한계: "회전이 약함" -> 여기 올려야 함
MAX_WZ = 4.5  # 기존 2.0은 코너에서 부족할 확률 큼

# Lookahead (기본은 v 기반, 코너에서 자동으로 더 줄임)
L_BASE = 1.0
L_GAIN = 2.0
L_MIN = 0.35
L_MAX = 3.0

# 코너 판별(heading error alpha) 기준
ALPHA_SHARP = 0.55     # rad (~31.5deg) 이 이상이면 "급코너" 취급
L_SHARP_SCALE = 0.55   # 급코너면 lookahead를 이 비율로 줄임(더 잘 감김)

# closest 탐색/리셋
SEARCH_WINDOW = 400
RESET_DIST = 6.0

# goal stop (루프면 보통 False)
STOP_AT_GOAL = False
GOAL_TOL = 1.0

# ========= 데이터 정책 =========
STRICT_NO_INTERPOLATION = True
AUTO_DROP_NON_FINITE = True
PRINT_ROUTE_FILE_CHECK = True

# ========= RViz 디버그 =========
PUBLISH_DEBUG = True
DEBUG_PUB_HZ = 5.0              # 경로/마커 publish 주기(Hz)
TRAJ_MAX_POINTS = 4000          # 실제 궤적 path 최대 점수(너무 많으면 RViz 무거움)


# --------------------------- helpers ---------------------------
def route_segment_files(route: List[int]) -> List[str]:
    return [f"{route[i]}_{route[i+1]}.csv" for i in range(len(route) - 1)]


def is_finite_point(p: Point) -> bool:
    return math.isfinite(p[0]) and math.isfinite(p[1])


def sanitize_polyline(points: List[Point], name: str) -> List[Point]:
    cleaned: List[Point] = []
    dropped = 0
    for p in points:
        if not is_finite_point(p):
            dropped += 1
            continue
        if cleaned and abs(cleaned[-1][0] - p[0]) < 1e-9 and abs(cleaned[-1][1] - p[1]) < 1e-9:
            continue
        cleaned.append(p)

    if dropped > 0:
        print(f"[WARN] {name}: dropped {dropped} non-finite points (nan/inf).")

    if len(cleaned) < 2:
        raise RuntimeError(f"{name}: too few valid points after cleanup: {len(cleaned)}")

    return cleaned


def load_waypoint_csv_robust(csv_path: str, name: str) -> List[Point]:
    pts: List[Point] = []
    bad_rows = 0
    non_finite = 0

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                bad_rows += 1
                continue
            try:
                x = float(row[0])
                y = float(row[1])
            except Exception:
                bad_rows += 1
                continue

            if (not math.isfinite(x)) or (not math.isfinite(y)):
                non_finite += 1
                if AUTO_DROP_NON_FINITE:
                    continue
                raise RuntimeError(f"{name}: contains non-finite row {(x, y)}")

            pts.append((x, y))

    if PRINT_ROUTE_FILE_CHECK and (bad_rows > 0 or non_finite > 0):
        msg = f"[WARN] {name}: bad_rows={bad_rows}, non_finite={non_finite}"
        if AUTO_DROP_NON_FINITE:
            msg += " (auto-dropped non-finite)"
        print(msg)

    if len(pts) == 0:
        raise RuntimeError(f"{name}: empty or unreadable (0 valid points)")

    return sanitize_polyline(pts, name)


def verify_route_files(waypoint_dir: str, route: List[int]) -> None:
    segs = route_segment_files(route)

    missing = []
    for fname in segs:
        if not os.path.exists(os.path.join(waypoint_dir, fname)):
            missing.append(fname)

    if missing:
        print("==== MISSING ROUTE SEGMENTS ====")
        for m in missing:
            print(f"- {m}")
        raise SystemExit("Missing required segment CSV(s). Fix files or ROUTE.")

    if PRINT_ROUTE_FILE_CHECK:
        print("==== ROUTE FILE CHECK (route-only) ====")
        for fname in segs:
            fpath = os.path.join(waypoint_dir, fname)
            _ = load_waypoint_csv_robust(fpath, fname)
        print("==== ROUTE FILE CHECK DONE ====")


def build_path_route_only(waypoint_dir: str, route: List[int]) -> List[Point]:
    final_path: List[Point] = []

    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        csv_name = f"{a}_{b}.csv"
        csv_path = os.path.join(waypoint_dir, csv_name)

        if not os.path.exists(csv_path):
            if STRICT_NO_INTERPOLATION:
                raise RuntimeError(f"Missing segment CSV: {csv_name} (STRICT)")
            raise RuntimeError(f"Missing segment CSV: {csv_name}")

        wp = load_waypoint_csv_robust(csv_path, csv_name)

        if final_path and abs(final_path[-1][0] - wp[0][0]) < 1e-6 and abs(final_path[-1][1] - wp[0][1]) < 1e-6:
            wp = wp[1:]

        final_path.extend(wp)

    return sanitize_polyline(final_path, "built_path_route_only")


# --------------------------- Node ---------------------------
class Task11PurePursuit(Node):
    def __init__(self, path: List[Point], is_loop: bool):
        super().__init__("task1_1_pure_pursuit")

        self.declare_parameter("pose_topic", "/Ego_pose")
        self.declare_parameter("accel_topic", "/Accel")
        self.pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        self.accel_topic = self.get_parameter("accel_topic").get_parameter_value().string_value

        self.path = path
        self.is_loop = is_loop

        self.v_cmd = float(V_CMD)
        self.max_wz = float(MAX_WZ)

        self.l_base = float(L_BASE)
        self.l_gain = float(L_GAIN)
        self.l_min = float(L_MIN)
        self.l_max = float(L_MAX)

        self.stop_at_goal = bool(STOP_AT_GOAL)
        self.goal_tol = float(GOAL_TOL)

        self.closest_idx = 0
        self.last_pose: Optional[PoseStamped] = None

        self.frame_id: Optional[str] = None  # pose에서 가져옴

        # QoS
        pose_qos = QoSProfile(depth=10)
        pose_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        pose_qos.durability = DurabilityPolicy.VOLATILE

        cmd_qos = QoSProfile(depth=10)
        cmd_qos.reliability = ReliabilityPolicy.RELIABLE
        cmd_qos.durability = DurabilityPolicy.VOLATILE

        self.sub_pose = self.create_subscription(PoseStamped, self.pose_topic, self.cb_pose, pose_qos)
        self.pub_accel = self.create_publisher(Accel, self.accel_topic, cmd_qos)

        # Debug publishers
        self.pub_ref_path: Optional[rclpy.publisher.Publisher] = None
        self.pub_traj_path: Optional[rclpy.publisher.Publisher] = None
        self.pub_target_marker: Optional[rclpy.publisher.Publisher] = None

        self.ref_path_msg: Optional[Path] = None
        self.traj_path_msg: Optional[Path] = None
        self.traj_points = 0

        if PUBLISH_DEBUG:
            self.pub_ref_path = self.create_publisher(Path, "/debug/ref_path", 1)
            self.pub_traj_path = self.create_publisher(Path, "/debug/traj_path", 1)
            self.pub_target_marker = self.create_publisher(Marker, "/debug/target_point", 1)
            self.debug_timer = self.create_timer(1.0 / max(1e-3, DEBUG_PUB_HZ), self.publish_debug)

        self.timer = self.create_timer(CONTROL_DT, self.control_step)

        self.get_logger().info(f"Pose topic: {self.pose_topic}, Accel topic: {self.accel_topic}")
        self.get_logger().info(f"Loaded path points: {len(self.path)} | loop={self.is_loop}")
        self.get_logger().info(
            f"Tuning: V_CMD={self.v_cmd:.2f}, MAX_WZ={self.max_wz:.2f}, "
            f"L=[{self.l_min:.2f}..{self.l_max:.2f}], L_BASE={self.l_base:.2f}, L_GAIN={self.l_gain:.2f}"
        )

        # target debug
        self.last_target: Optional[Point] = None

    def cb_pose(self, msg: PoseStamped):
        self.last_pose = msg
        if self.frame_id is None:
            fid = msg.header.frame_id.strip()
            self.frame_id = fid if fid else "map"
            # 최초 frame_id 확보되면 ref_path 메시지 준비
            if PUBLISH_DEBUG:
                self.ref_path_msg = self.make_ref_path_msg(self.frame_id)
                self.traj_path_msg = Path()
                self.traj_path_msg.header.frame_id = self.frame_id

    def compute_lookahead(self, v: float, alpha: float) -> float:
        # 기본: v 기반
        L = self.l_base + self.l_gain * max(0.0, v)
        L = clamp(L, self.l_min, self.l_max)

        # 급코너면 lookahead 줄여서 더 잘 "감기게"
        if abs(alpha) >= ALPHA_SHARP:
            L = max(self.l_min, L * L_SHARP_SCALE)

        return L

    def publish_cmd(self, vx: float, wz: float):
        msg = Accel()
        msg.linear.x = float(vx)
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(wz)
        self.pub_accel.publish(msg)

    def dist2_idx(self, x: float, y: float, idx: int) -> float:
        px, py = self.path[idx]
        return (px - x) ** 2 + (py - y) ** 2

    def update_closest_idx(self, x: float, y: float) -> int:
        n = len(self.path)
        if n == 0:
            return 0

        cur_d = math.sqrt(self.dist2_idx(x, y, self.closest_idx))
        if cur_d > RESET_DIST:
            best_i = 0
            best_d2 = float("inf")
            for i in range(n):
                d2 = self.dist2_idx(x, y, i)
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i
            self.closest_idx = best_i
            return best_i

        half = SEARCH_WINDOW // 2
        lo = max(0, self.closest_idx - half)
        hi = min(n, self.closest_idx + half)

        best_i = self.closest_idx
        best_d2 = float("inf")
        for i in range(lo, hi):
            d2 = self.dist2_idx(x, y, i)
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        self.closest_idx = best_i
        return best_i

    def find_target_index_arclength(self, start_idx: int, lookahead: float) -> int:
        n = len(self.path)
        if n == 0:
            return 0
        if lookahead <= 1e-6:
            return start_idx

        acc = 0.0
        i = start_idx
        max_steps = n if self.is_loop else (n - 1 - start_idx)
        steps = 0

        while steps < max_steps:
            j = (i + 1) % n if self.is_loop else (i + 1)
            x0, y0 = self.path[i]
            x1, y1 = self.path[j]
            seg = math.hypot(x1 - x0, y1 - y0)
            if math.isfinite(seg) and seg > 1e-9:
                acc += seg
            i = j
            steps += 1
            if acc >= lookahead:
                return i

        return i

    # ---------- RViz debug ----------
    def make_ref_path_msg(self, frame_id: str) -> Path:
        msg = Path()
        msg.header.frame_id = frame_id
        # timestamp는 publish할 때마다 현재 시간으로 갱신
        poses: List[PoseStamped] = []
        for (x, y) in self.path:
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            poses.append(ps)
        msg.poses = poses
        return msg

    def append_traj(self, pose: PoseStamped):
        if self.traj_path_msg is None:
            return
        ps = PoseStamped()
        ps.header.frame_id = self.traj_path_msg.header.frame_id
        ps.pose = pose.pose
        self.traj_path_msg.poses.append(ps)
        self.traj_points += 1

        # 너무 길어지면 앞부분 버림
        if self.traj_points > TRAJ_MAX_POINTS:
            drop = self.traj_points - TRAJ_MAX_POINTS
            self.traj_path_msg.poses = self.traj_path_msg.poses[drop:]
            self.traj_points = TRAJ_MAX_POINTS

    def publish_debug(self):
        if not PUBLISH_DEBUG:
            return
        if self.last_pose is None or self.frame_id is None:
            return

        now = self.get_clock().now().to_msg()

        # ref path
        if self.pub_ref_path is not None and self.ref_path_msg is not None:
            self.ref_path_msg.header.stamp = now
            # 각 pose도 stamp 찍으면 RViz에서 더 안정적임
            for ps in self.ref_path_msg.poses:
                ps.header.stamp = now
            self.pub_ref_path.publish(self.ref_path_msg)

        # traj path
        if self.pub_traj_path is not None and self.traj_path_msg is not None:
            self.traj_path_msg.header.stamp = now
            self.append_traj(self.last_pose)
            for ps in self.traj_path_msg.poses[-50:]:  # 최근 것만 stamp 갱신해도 충분
                ps.header.stamp = now
            self.pub_traj_path.publish(self.traj_path_msg)

        # target marker
        if self.pub_target_marker is not None and self.last_target is not None:
            mk = Marker()
            mk.header.frame_id = self.frame_id
            mk.header.stamp = now
            mk.ns = "pp_target"
            mk.id = 0
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.pose.position.x = float(self.last_target[0])
            mk.pose.position.y = float(self.last_target[1])
            mk.pose.position.z = 0.2
            mk.pose.orientation.w = 1.0
            mk.scale.x = 0.6
            mk.scale.y = 0.6
            mk.scale.z = 0.6
            mk.color.a = 1.0
            mk.color.r = 1.0
            mk.color.g = 0.2
            mk.color.b = 0.2
            self.pub_target_marker.publish(mk)

    # ---------- control ----------
    def control_step(self):
        if self.last_pose is None or not self.path:
            return

        p = self.last_pose.pose.position
        q = self.last_pose.pose.orientation
        x, y = float(p.x), float(p.y)
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
            return

        if self.stop_at_goal and (not self.is_loop):
            gx, gy = self.path[-1]
            if math.hypot(gx - x, gy - y) <= self.goal_tol:
                self.publish_cmd(0.0, 0.0)
                self.timer.cancel()
                self.get_logger().info("Reached goal. Stop.")
                return

        ci = self.update_closest_idx(x, y)

        # 1차로 "현재 lookahead" 계산 전에 임시 target(가까운 점 다음)으로 alpha를 대충 잡아도 되지만,
        # 여기서는 그냥 lookahead를 v 기반으로 먼저 대충 만들고 target 찾은 후 alpha로 다시 보정한다.
        L0 = clamp(self.l_base + self.l_gain * max(0.0, self.v_cmd), self.l_min, self.l_max)
        ti0 = self.find_target_index_arclength(ci, L0)
        tx0, ty0 = self.path[ti0]

        angle_to_target0 = math.atan2(ty0 - y, tx0 - x)
        alpha0 = math.atan2(math.sin(angle_to_target0 - yaw), math.cos(angle_to_target0 - yaw))

        # 코너면 lookahead 축소해서 다시 target 재선정
        L = self.compute_lookahead(self.v_cmd, alpha0)
        ti = self.find_target_index_arclength(ci, L)
        tx, ty = self.path[ti]
        self.last_target = (tx, ty)

        angle_to_target = math.atan2(ty - y, tx - x)
        alpha = math.atan2(math.sin(angle_to_target - yaw), math.cos(angle_to_target - yaw))

        # Pure Pursuit
        kappa = 2.0 * math.sin(alpha) / max(1e-3, L)
        wz = self.v_cmd * kappa

        # 회전 약하면 결국 여기서 잘려나가거나 kappa가 작음
        wz = clamp(wz, -self.max_wz, self.max_wz)

        self.publish_cmd(self.v_cmd, wz)


# --------------------------- main ---------------------------
def main():
    # ROUTE에 필요한 파일만 체크/로드
    verify_route_files(WAYPOINT_DIR, ROUTE)

    path = build_path_route_only(WAYPOINT_DIR, ROUTE)
    is_loop = (len(ROUTE) >= 2 and ROUTE[0] == ROUTE[-1])

    rclpy.init()
    node = Task11PurePursuit(path, is_loop=is_loop)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_cmd(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

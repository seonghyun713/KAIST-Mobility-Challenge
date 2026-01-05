#!/usr/bin/env python3
import os
import math
from enum import Enum, auto
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Accel

from projection2_1 import ProjectionManager


# ===================== CONFIG =====================
ALLOWED_LANES = [0, 1, 2]
START_LANE = 1

LANE_ABS_D_GATE = 0.35
EGO_ABS_D_WARN = 0.60

# ---- Triggers (same lane) ----
TRIGGER_FRONT_DS = 1.0          # 앞차 가까움
TRIGGER_BACK_DS  = 1.0          # 뒷차 가까움
BACK_FAST_DV     = 0.20          # 뒷차가 내 속도보다 이만큼 빠르면 "급접근"으로 간주
HV_WARMUP = 0.30

# ---- Safety gap on target lane ----
SAFE_FRONT_DS = 1.00
SAFE_BACK_DS  = 1.30

# ---- FSM timing ----
COOLDOWN_TIME = 1.0
MIN_HOLD_TIME = 0.6
LC_DONE_ABS_D = 0.18

# ---- Control ----
SPEED_CMD = 0.5
LOOKAHEAD_DIST = 0.30
LOOKAHEAD_V_GAIN = 0.5
LOOKAHEAD_MAX = 0.70
STEER_GAIN = 1.0
CTE_GAIN = 0.5
STEER_FILTER_ALPHA = 0.7
OMEGA_MAX = 6.0

LOG_EVERY_N_TICKS = 10


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class State(Enum):
    KEEP = auto()
    CHANGE = auto()


class DecisionLaneChange(Node):
    def __init__(self):
        super().__init__("decision_lane_change")

        self.get_logger().info(f"[LC] ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '(unset)')}")
        self.pm = ProjectionManager()

        self.EGO_POSE_TOPIC = "/CAV_01"
        self.EGO_ACCEL_TOPIC = "/CAV_01_accel"
        self.HV_TOPIC_FMT = "/HV_{}"
        self.HV_IDS = list(range(19, 37))

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE

        self.create_subscription(PoseStamped, self.EGO_POSE_TOPIC, self.pm.update_ego_pose, qos)
        for hv_id in self.HV_IDS:
            self.create_subscription(
                PoseStamped,
                self.HV_TOPIC_FMT.format(hv_id),
                lambda msg, hid=hv_id: self.pm.update_hv_pose(hid, msg),
                qos,
            )

        self.ctrl_pub = self.create_publisher(Accel, self.EGO_ACCEL_TOPIC, qos)

        if self.pm.ego_seg_hint is None:
            self.pm.ego_seg_hint = {}
        for ln in ALLOWED_LANES:
            self.pm.ego_seg_hint.setdefault(ln, 0)

        self.state = State.KEEP
        self.active_lane = START_LANE
        self.target_lane: Optional[int] = None
        self.t_last_change = 0.0
        self.t_last_switch = 0.0

        self.prev_omega = 0.0

        self.tick_count = 0
        self.create_timer(0.05, self.tick)

        self.get_logger().info(f"[LC] started | active_lane={self.active_lane} allowed={ALLOWED_LANES}")

    # ---------------- time ----------------
    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def cooldown_ok(self) -> bool:
        return (self.now() - self.t_last_change) > COOLDOWN_TIME

    def hold_ok(self) -> bool:
        return (self.now() - self.t_last_switch) > MIN_HOLD_TIME

    # ---------------- projections ----------------
    def project_ego(self) -> Tuple[float, float, int, float, float]:
        ego = self.pm.ego
        hint = int(self.pm.ego_seg_hint.get(self.active_lane, 0))
        a_s, a_d, a_seg_i, a_u, a_abs_d = self.pm.project_to_lane(
            self.active_lane, ego.x, ego.y, hint=hint, window=None
        )
        self.pm.ego_seg_hint[self.active_lane] = a_seg_i
        return a_s, a_d, a_seg_i, a_u, a_abs_d

    def project_hv_best_lane(self, hv_id: int):
        hv = self.pm.hvs[hv_id]
        if not hv.valid:
            return None

        hint_map = hv.seg_hint
        for ln in ALLOWED_LANES:
            hint_map.setdefault(ln, 0)

        best = None
        for ln in ALLOWED_LANES:
            hint = int(hint_map.get(ln, 0))
            s, d, seg_i, u, abs_d = self.pm.project_to_lane(ln, hv.x, hv.y, hint=hint, window=None)
            if best is None or abs_d < best[-1]:
                best = (ln, s, d, seg_i, u, abs_d)

        if best is not None:
            ln, _, _, seg_i, _, _ = best
            hint_map[ln] = seg_i
        return best  # (lane, s, d, seg_i, u, abs_d)

    # ---------------- traffic query ----------------
    def closest_front(self, lane_idx: int, ego_s: float):
        best = None
        for hv_id in self.HV_IDS:
            proj = self.project_hv_best_lane(hv_id)
            if proj is None:
                continue
            ln, hv_s, _, _, _, abs_d = proj
            if ln != lane_idx or abs_d > LANE_ABS_D_GATE:
                continue
            ds = hv_s - ego_s
            if ds <= 0:
                continue
            if best is None or ds < best[1]:
                best = (hv_id, ds)
        return best  # (id, ds)

    def closest_back(self, lane_idx: int, ego_s: float):
        best = None
        for hv_id in self.HV_IDS:
            proj = self.project_hv_best_lane(hv_id)
            if proj is None:
                continue
            ln, hv_s, _, _, _, abs_d = proj
            if ln != lane_idx or abs_d > LANE_ABS_D_GATE:
                continue
            ds = ego_s - hv_s
            if ds <= 0:
                continue
            if best is None or ds < best[1]:
                best = (hv_id, ds)
        return best  # (id, ds)

    def lane_is_safe(self, lane_idx: int, ego_s: float) -> bool:
        f = self.closest_front(lane_idx, ego_s)
        b = self.closest_back(lane_idx, ego_s)
        if f and f[1] < SAFE_FRONT_DS:
            return False
        if b and b[1] < SAFE_BACK_DS:
            return False
        return True

    def choose_target_lane(self, ego_s: float) -> Optional[int]:
        candidates = [ln for ln in ALLOWED_LANES if ln != self.active_lane]
        candidates.sort(reverse=True)  # 오른쪽 우선
        for ln in candidates:
            if self.lane_is_safe(ln, ego_s):
                return ln
        return None

    # ---------------- decision triggers ----------------
    def back_is_fast_and_close(self, back_info) -> bool:
        """back_info=(hv_id, ds_back). speed valid하면 dv로 fast 판단, 아니면 거리만으로 판단."""
        if back_info is None:
            return False
        hv_id, ds = back_info
        if ds > TRIGGER_BACK_DS:
            return False

        ego_v = self.pm.ego.speed
        if self.pm.hv_speed_valid(hv_id, T_WARMUP=HV_WARMUP):
            hv_v = self.pm.estimate_hv_speed(hv_id)
            return hv_v > ego_v + BACK_FAST_DV
        else:
            # 속도 못 믿겠으면 "가까움"만으로도 트리거 (보수적으로)
            return True

    # ---------------- s->point ----------------
    def point_on_lane_by_s(self, lane_idx: int, target_s: float, hint_i: int):
        px = self.pm.paths[lane_idx]["x"]
        py = self.pm.paths[lane_idx]["y"]
        ps = self.pm.paths[lane_idx]["s"]
        n = len(px)

        target_s = clamp(target_s, ps[0], ps[-1])
        i = int(clamp(hint_i, 0, n - 2))

        if ps[i] > target_s:
            while i > 0 and ps[i] > target_s:
                i -= 1
        else:
            while i < n - 2 and ps[i + 1] < target_s:
                i += 1

        s0, s1 = ps[i], ps[i + 1]
        if (s1 - s0) < 1e-9:
            return px[i], py[i], i

        u = clamp((target_s - s0) / (s1 - s0), 0.0, 1.0)
        tx = px[i] + u * (px[i + 1] - px[i])
        ty = py[i] + u * (py[i + 1] - py[i])
        return tx, ty, i

    # ---------------- control ----------------
    def compute_omega(self, lane_idx: int, a_s: float, a_d: float, a_seg_i: int) -> float:
        ego = self.pm.ego
        v = max(0.05, ego.speed)

        Ld = max(LOOKAHEAD_DIST, v * LOOKAHEAD_V_GAIN)
        Ld = min(Ld, LOOKAHEAD_MAX)

        tx, ty, _ = self.point_on_lane_by_s(lane_idx, a_s + Ld, hint_i=a_seg_i)

        dx, dy = tx - ego.x, ty - ego.y
        yaw = ego.yaw

        local_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy
        curvature = 2.0 * local_y / (Ld ** 2)

        steer = (curvature * STEER_GAIN) + (a_d * CTE_GAIN)
        raw_omega = v * steer

        omega = (STEER_FILTER_ALPHA * raw_omega) + ((1.0 - STEER_FILTER_ALPHA) * self.prev_omega)
        self.prev_omega = omega
        return clamp(omega, -OMEGA_MAX, OMEGA_MAX)

    # ---------------- FSM ----------------
    def try_start_lane_change(self, reason: str, ego_s: float):
        tgt = self.choose_target_lane(ego_s)
        if tgt is None:
            self.t_last_change = self.now()
            return

        self.target_lane = tgt
        self.active_lane = tgt
        self.pm.ego_seg_hint.setdefault(self.active_lane, self.pm.ego_seg_hint.get(self.active_lane, 0))

        self.state = State.CHANGE
        self.t_last_change = self.now()
        self.t_last_switch = self.now()
        self.get_logger().info(f"[FSM] KEEP->CHANGE | {reason} | switch->{tgt}")

    def fsm_update(self, ego_s: float, ego_abs_d: float):
        if self.state == State.KEEP:
            if not self.cooldown_ok():
                return

            front = self.closest_front(self.active_lane, ego_s)
            back  = self.closest_back(self.active_lane, ego_s)

            front_close = (front is not None and front[1] < TRIGGER_FRONT_DS)
            back_threat = self.back_is_fast_and_close(back)

            if front_close:
                self.try_start_lane_change(reason=f"front_close ds={front[1]:.2f} hv={front[0]}", ego_s=ego_s)
            elif back_threat:
                self.try_start_lane_change(reason=f"back_threat ds={back[1]:.2f} hv={back[0]}", ego_s=ego_s)

        elif self.state == State.CHANGE:
            if self.target_lane is None:
                self.state = State.KEEP
                return

            if self.hold_ok() and (abs(ego_abs_d) < LC_DONE_ABS_D) and (self.active_lane == self.target_lane):
                self.get_logger().info(f"[FSM] CHANGE->KEEP | done absd={ego_abs_d:.2f} lane={self.active_lane}")
                self.target_lane = None
                self.state = State.KEEP
                self.t_last_change = self.now()

    # ---------------- main tick ----------------
    def tick(self):
        self.tick_count += 1

        ego = self.pm.ego
        if not ego.valid:
            if self.tick_count % LOG_EVERY_N_TICKS == 0:
                self.get_logger().warn("[LC] ego invalid (no pose yet)")
            return

        a_s, a_d, a_seg_i, _, a_abs_d = self.project_ego()

        if a_abs_d > EGO_ABS_D_WARN and self.tick_count % LOG_EVERY_N_TICKS == 0:
            self.get_logger().warn(f"[LC] ego absd large on lane={self.active_lane}: absd={a_abs_d:.2f}")

        if self.tick_count % LOG_EVERY_N_TICKS == 0:
            f = self.closest_front(self.active_lane, a_s)
            b = self.closest_back(self.active_lane, a_s)
            fi = "none" if f is None else f"id={f[0]} ds={f[1]:.2f}"
            bi = "none" if b is None else f"id={b[0]} ds={b[1]:.2f}"
            self.get_logger().info(
                f"[LC] state={self.state.name} lane={self.active_lane} tgt={self.target_lane} "
                f"s={a_s:.2f} d={a_d:.2f} absd={a_abs_d:.2f} v={ego.speed:.2f} yaw={ego.yaw:.3f} "
                f"front({fi}) back({bi})"
            )

        # decision
        self.fsm_update(a_s, a_abs_d)

        # control
        omega = self.compute_omega(self.active_lane, a_s, a_d, a_seg_i)
        cmd = Accel()
        cmd.linear.x = float(SPEED_CMD)
        cmd.angular.z = float(omega)
        self.ctrl_pub.publish(cmd)


def main():
    rclpy.init()
    node = DecisionLaneChange()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# import os
# import math
# from enum import Enum, auto
# from typing import Optional, Tuple

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
# from geometry_msgs.msg import PoseStamped, Accel

# from projection2_1 import ProjectionManager


# # ===================== CONFIG =====================
# # lane index convention (ProjectionManager가 이렇게 로드한다고 가정):
# # 0: path_2_1.json
# # 1: path_2_2.json
# # 2: path_2_3.json
# # 3: path_2_4.json  <-- CAV 진입 금지
# FORBIDDEN_LANES = {3}
# ALLOWED_LANES = [ln for ln in [0, 1, 2, 3] if ln not in FORBIDDEN_LANES]
# START_LANE = 1

# LANE_ABS_D_GATE = 0.35
# EGO_ABS_D_WARN = 0.60

# # ---- Trigger (same lane only, distance only) ----
# TRIGGER_NEAR_DS = 1.0  # 같은 차선 위에서 ds < 1.0m 이면 차선 변경

# # ---- Safety gap on target lane ----
# SAFE_FRONT_DS = 1.00
# SAFE_BACK_DS  = 1.30

# # ---- FSM timing ----
# COOLDOWN_TIME = 1.0
# MIN_HOLD_TIME = 0.6
# LC_DONE_ABS_D = 0.18

# # ---- Control (유지) ----
# SPEED_CMD = 0.5
# LOOKAHEAD_DIST = 0.30
# LOOKAHEAD_V_GAIN = 0.5
# LOOKAHEAD_MAX = 0.70
# STEER_GAIN = 1.0
# CTE_GAIN = 0.5
# STEER_FILTER_ALPHA = 0.7
# OMEGA_MAX = 6.0

# LOG_EVERY_N_TICKS = 10


# def clamp(v: float, lo: float, hi: float) -> float:
#     return lo if v < lo else hi if v > hi else v


# class State(Enum):
#     KEEP = auto()
#     CHANGE = auto()


# class DecisionLaneChange(Node):
#     def __init__(self):
#         super().__init__("decision_lane_change")

#         self.get_logger().info(f"[LC] ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '(unset)')}")
#         self.pm = ProjectionManager()

#         self.EGO_POSE_TOPIC = "/CAV_01"
#         self.EGO_ACCEL_TOPIC = "/CAV_01_accel"
#         self.HV_TOPIC_FMT = "/HV_{}"
#         self.HV_IDS = list(range(19, 37))

#         qos = QoSProfile(depth=10)
#         qos.reliability = ReliabilityPolicy.BEST_EFFORT
#         qos.durability = DurabilityPolicy.VOLATILE

#         self.create_subscription(PoseStamped, self.EGO_POSE_TOPIC, self.pm.update_ego_pose, qos)
#         for hv_id in self.HV_IDS:
#             self.create_subscription(
#                 PoseStamped,
#                 self.HV_TOPIC_FMT.format(hv_id),
#                 lambda msg, hid=hv_id: self.pm.update_hv_pose(hid, msg),
#                 qos,
#             )

#         self.ctrl_pub = self.create_publisher(Accel, self.EGO_ACCEL_TOPIC, qos)

#         # ego seg hint init
#         if self.pm.ego_seg_hint is None:
#             self.pm.ego_seg_hint = {}
#         for ln in ALLOWED_LANES:
#             self.pm.ego_seg_hint.setdefault(ln, 0)

#         self.state = State.KEEP
#         self.active_lane = START_LANE
#         self.target_lane: Optional[int] = None
#         self.t_last_change = 0.0
#         self.t_last_switch = 0.0

#         self.prev_omega = 0.0
#         self.tick_count = 0

#         # debug: 특정 HV(예: 33) 집중 로그 보고싶으면 여기에 넣어
#         self.DEBUG_HV_ID = 33

#         self.create_timer(0.05, self.tick)
#         self.get_logger().info(
#             f"[LC] started | active_lane={self.active_lane} allowed={ALLOWED_LANES} forbidden={sorted(FORBIDDEN_LANES)}"
#         )

#     # ---------------- time ----------------
#     def now(self) -> float:
#         return self.get_clock().now().nanoseconds * 1e-9

#     def cooldown_ok(self) -> bool:
#         return (self.now() - self.t_last_change) > COOLDOWN_TIME

#     def hold_ok(self) -> bool:
#         return (self.now() - self.t_last_switch) > MIN_HOLD_TIME

#     # ---------------- projections ----------------
#     def project_ego(self) -> Tuple[float, float, int, float, float]:
#         ego = self.pm.ego
#         hint = int(self.pm.ego_seg_hint.get(self.active_lane, 0))
#         a_s, a_d, a_seg_i, a_u, a_abs_d = self.pm.project_to_lane(
#             self.active_lane, ego.x, ego.y, hint=hint, window=None
#         )
#         self.pm.ego_seg_hint[self.active_lane] = a_seg_i
#         return a_s, a_d, a_seg_i, a_u, a_abs_d

#     def project_hv_on_lane(self, hv_id: int, lane_idx: int):
#         """
#         ★ 핵심 수정점:
#         - HV를 best-lane으로 고르지 않고, '지정한 lane_idx'에만 투영한다.
#         - "같은 차선 위" 판단이 흔들리지 않게 만듦.
#         """
#         hv = self.pm.hvs[hv_id]
#         if not hv.valid:
#             return None

#         if hv.seg_hint is None:
#             hv.seg_hint = {}
#         hv.seg_hint.setdefault(lane_idx, 0)

#         hint = int(hv.seg_hint.get(lane_idx, 0))
#         s, d, seg_i, u, abs_d = self.pm.project_to_lane(lane_idx, hv.x, hv.y, hint=hint, window=None)
#         hv.seg_hint[lane_idx] = seg_i
#         return (s, d, seg_i, u, abs_d)

#     # ---------------- traffic query (ACTIVE LANE ONLY) ----------------
#     def closest_front(self, lane_idx: int, ego_s: float):
#         best = None
#         for hv_id in self.HV_IDS:
#             proj = self.project_hv_on_lane(hv_id, lane_idx)
#             if proj is None:
#                 continue
#             hv_s, _, _, _, abs_d = proj
#             if abs_d > LANE_ABS_D_GATE:
#                 continue
#             ds = hv_s - ego_s
#             if ds <= 0:
#                 continue
#             if best is None or ds < best[1]:
#                 best = (hv_id, ds)
#         return best  # (id, ds_front)

#     def closest_back(self, lane_idx: int, ego_s: float):
#         best = None
#         for hv_id in self.HV_IDS:
#             proj = self.project_hv_on_lane(hv_id, lane_idx)
#             if proj is None:
#                 continue
#             hv_s, _, _, _, abs_d = proj
#             if abs_d > LANE_ABS_D_GATE:
#                 continue
#             ds = ego_s - hv_s
#             if ds <= 0:
#                 continue
#             if best is None or ds < best[1]:
#                 best = (hv_id, ds)
#         return best  # (id, ds_back)

#     def lane_is_safe(self, lane_idx: int, ego_s: float) -> bool:
#         # 안전판단도 "그 lane_idx 기준으로만 투영" (best-lane 금지)
#         f = self.closest_front(lane_idx, ego_s)
#         b = self.closest_back(lane_idx, ego_s)
#         if f and f[1] < SAFE_FRONT_DS:
#             return False
#         if b and b[1] < SAFE_BACK_DS:
#             return False
#         return True

#     def choose_target_lane(self, ego_s: float) -> Optional[int]:
#         candidates = [ln for ln in ALLOWED_LANES if ln != self.active_lane and ln not in FORBIDDEN_LANES]
#         candidates.sort(reverse=True)  # 오른쪽 우선
#         for ln in candidates:
#             if self.lane_is_safe(ln, ego_s):
#                 return ln
#         return None

#     # ---------------- s->point ----------------
#     def point_on_lane_by_s(self, lane_idx: int, target_s: float, hint_i: int):
#         px = self.pm.paths[lane_idx]["x"]
#         py = self.pm.paths[lane_idx]["y"]
#         ps = self.pm.paths[lane_idx]["s"]
#         n = len(px)

#         target_s = clamp(target_s, ps[0], ps[-1])
#         i = int(clamp(hint_i, 0, n - 2))

#         if ps[i] > target_s:
#             while i > 0 and ps[i] > target_s:
#                 i -= 1
#         else:
#             while i < n - 2 and ps[i + 1] < target_s:
#                 i += 1

#         s0, s1 = ps[i], ps[i + 1]
#         if (s1 - s0) < 1e-9:
#             return px[i], py[i], i

#         u = clamp((target_s - s0) / (s1 - s0), 0.0, 1.0)
#         tx = px[i] + u * (px[i + 1] - px[i])
#         ty = py[i] + u * (py[i + 1] - py[i])
#         return tx, ty, i

#     # ---------------- control (유지) ----------------
#     def compute_omega(self, lane_idx: int, a_s: float, a_d: float, a_seg_i: int) -> float:
#         ego = self.pm.ego
#         v = max(0.05, ego.speed)

#         Ld = max(LOOKAHEAD_DIST, v * LOOKAHEAD_V_GAIN)
#         Ld = min(Ld, LOOKAHEAD_MAX)

#         tx, ty, _ = self.point_on_lane_by_s(lane_idx, a_s + Ld, hint_i=a_seg_i)

#         dx, dy = tx - ego.x, ty - ego.y
#         yaw = ego.yaw

#         local_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy
#         curvature = 2.0 * local_y / (Ld ** 2)

#         steer = (curvature * STEER_GAIN) + (a_d * CTE_GAIN)
#         raw_omega = v * steer

#         omega = (STEER_FILTER_ALPHA * raw_omega) + ((1.0 - STEER_FILTER_ALPHA) * self.prev_omega)
#         self.prev_omega = omega
#         return clamp(omega, -OMEGA_MAX, OMEGA_MAX)

#     # ---------------- FSM ----------------
#     def try_start_lane_change(self, reason: str, ego_s: float):
#         tgt = self.choose_target_lane(ego_s)
#         if tgt is None:
#             self.t_last_change = self.now()
#             return

#         self.target_lane = tgt
#         self.active_lane = tgt
#         self.pm.ego_seg_hint.setdefault(self.active_lane, self.pm.ego_seg_hint.get(self.active_lane, 0))

#         self.state = State.CHANGE
#         self.t_last_change = self.now()
#         self.t_last_switch = self.now()
#         self.get_logger().info(f"[FSM] KEEP->CHANGE | {reason} | switch->{tgt}")

#     def fsm_update(self, ego_s: float, ego_abs_d: float):
#         if self.state == State.KEEP:
#             if not self.cooldown_ok():
#                 return

#             front = self.closest_front(self.active_lane, ego_s)
#             back  = self.closest_back(self.active_lane, ego_s)

#             # 규칙: 같은 차선 위 + 0.5m 이내면 차선 변경 (상대속도 사용 X)
#             front_near = (front is not None and front[1] < TRIGGER_NEAR_DS)
#             back_near  = (back  is not None and back[1]  < TRIGGER_NEAR_DS)

#             if front_near:
#                 self.try_start_lane_change(
#                     reason=f"near(front) ds={front[1]:.2f} hv={front[0]}",
#                     ego_s=ego_s
#                 )
#             elif back_near:
#                 self.try_start_lane_change(
#                     reason=f"near(back) ds={back[1]:.2f} hv={back[0]}",
#                     ego_s=ego_s
#                 )

#         elif self.state == State.CHANGE:
#             if self.target_lane is None:
#                 self.state = State.KEEP
#                 return

#             if self.hold_ok() and (abs(ego_abs_d) < LC_DONE_ABS_D) and (self.active_lane == self.target_lane):
#                 self.get_logger().info(f"[FSM] CHANGE->KEEP | done absd={ego_abs_d:.2f} lane={self.active_lane}")
#                 self.target_lane = None
#                 self.state = State.KEEP
#                 self.t_last_change = self.now()

#     # ---------------- main tick ----------------
#     def tick(self):
#         self.tick_count += 1

#         ego = self.pm.ego
#         if not ego.valid:
#             if self.tick_count % LOG_EVERY_N_TICKS == 0:
#                 self.get_logger().warn("[LC] ego invalid (no pose yet)")
#             return

#         a_s, a_d, a_seg_i, _, a_abs_d = self.project_ego()

#         if a_abs_d > EGO_ABS_D_WARN and self.tick_count % LOG_EVERY_N_TICKS == 0:
#             self.get_logger().warn(f"[LC] ego absd large on lane={self.active_lane}: absd={a_abs_d:.2f}")

#         # ---- Debug: 특정 HV가 왜 로그에 안 뜨는지 확인 ----
#         # (33이랑 22 케이스 진단용)
#         if self.tick_count % LOG_EVERY_N_TICKS == 0:
#             dbg = self.project_hv_on_lane(self.DEBUG_HV_ID, self.active_lane)
#             if dbg is not None:
#                 hv_s_dbg, _, _, _, absd_dbg = dbg
#                 ds_dbg = hv_s_dbg - a_s
#                 self.get_logger().info(
#                     f"[DBG] hv={self.DEBUG_HV_ID} on active_lane={self.active_lane}: ds={ds_dbg:.2f} absd={absd_dbg:.2f}"
#                 )
#             else:
#                 self.get_logger().info(f"[DBG] hv={self.DEBUG_HV_ID} invalid/no pose")

#         if self.tick_count % LOG_EVERY_N_TICKS == 0:
#             f = self.closest_front(self.active_lane, a_s)
#             b = self.closest_back(self.active_lane, a_s)
#             fi = "none" if f is None else f"id={f[0]} ds={f[1]:.2f}"
#             bi = "none" if b is None else f"id={b[0]} ds={b[1]:.2f}"
#             self.get_logger().info(
#                 f"[LC] state={self.state.name} lane={self.active_lane} tgt={self.target_lane} "
#                 f"s={a_s:.2f} d={a_d:.2f} absd={a_abs_d:.2f} v={ego.speed:.2f} yaw={ego.yaw:.3f} "
#                 f"front({fi}) back({bi})"
#             )

#         # decision
#         self.fsm_update(a_s, a_abs_d)

#         # control (유지)
#         omega = self.compute_omega(self.active_lane, a_s, a_d, a_seg_i)
#         cmd = Accel()
#         cmd.linear.x = float(SPEED_CMD)
#         cmd.angular.z = float(omega)
#         self.ctrl_pub.publish(cmd)


# def main():
#     rclpy.init()
#     node = DecisionLaneChange()
#     try:
#         rclpy.spin(node)
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()

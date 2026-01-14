import math
from enum import Enum, auto
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data

from projection import ProjectionManager
from std_msgs.msg import Int32


LANE_ABS_D_GATE = 0.15
FRONT_DS_MAX = 10.0
BACK_DS_MAX = 10.0
INF = float('inf')

# --- Circular ROI (meters): ignore too far vehicles ---
ROI_RADIUS = 6.0  # 반지름 임계값

# FSM Parameters
FRONT_CLOSE   = 1.0   # trigger: front too close
BACK_CLOSE    = 1.0   # trigger: back too close
BACK_FAST_DV  = 0.2   # trigger: back approaching fast (dv >)
COOLDOWN_SEC  = 2.0

# safety check for target lane
SAFE_FRONT_GAP = 1.2  # 앞 여유
SAFE_BACK_GAP  = 1.2  # 뒤 여유 (절대값)

ALLOWED_LANES = [0, 1, 2]

# --- ZONE lane ids (projection only, NOT real lanes) ---
LANE_CHANGE_ZONE = 3
NOT_LANE_CHANGE_ZONE = 4

ZONE_ABS_D_GATE = 0.000001

# behavior
BEHAVIOR_GO = 0
BEHAVIOR_SLOW = 1


class LCState(Enum):
    KEEP_LANE = auto()
    PREPARE_LC = auto()
    LANE_CHANGE = auto()


class PerceptionDebugNode(Node):
    def __init__(self):
        super().__init__('perception_debug')
        self.proj = ProjectionManager()

        # FSM state
        self.state = LCState.KEEP_LANE
        self.target_lane: Optional[int] = None
        self.last_lc_time = 0.0
        self.cooldown = COOLDOWN_SEC

        # speed estimation (s-dot)
        self.prev_time: Optional[float] = None
        self.prev_cav_s: Optional[float] = None
        self.prev_hv_s: Dict[int, float] = {}
        self.ego_v = 0.0
        self.hv_v: Dict[int, float] = {}

        # lane snapshot (front/back per lane)
        self.lane_front_back: Dict[int, Tuple[Optional[int], float, Optional[int], float]] = {}

        # ego s for each lane (IMPORTANT for cross-lane ds)
        self.cav_s_by_lane: Dict[int, float] = {}

        # zone & behavior
        self.in_lc_zone = False
        self.in_kl_zone = False
        self.behavior = BEHAVIOR_GO

        # Subscribers
        self.create_subscription(PoseStamped, '/CAV_01',
                                 self.cav_callback,
                                 qos_profile_sensor_data)

        for i in range(19, 37):
            topic = f'/HV_{i}'
            self.create_subscription(
                PoseStamped, topic,
                lambda msg, hv_id=i: self.hv_callback(msg, hv_id),
                10
            )

        self.create_timer(0.1, self.tick)

        # publisher
        self.pub_target_lane = self.create_publisher(Int32, '/cav/target_lane', 10)
        self.pub_behavior    = self.create_publisher(Int32, '/cav/behavior', 10)
        self.pub_fsm_state = self.create_publisher(Int32, '/cav/fsm_state', 10)



    def cav_callback(self, msg: PoseStamped):
        self.proj.update_ego_pose(msg)

    def hv_callback(self, msg: PoseStamped, hv_id: int):
        self.proj.update_hv_pose(hv_id, msg)

    def build_lane_front_back(self, cav_x: float, cav_y: float):
        lane_fb: Dict[int, Tuple[Optional[int], float, Optional[int], float]] = {}
        r2 = ROI_RADIUS * ROI_RADIUS

        for lane in ALLOWED_LANES:
            front_ds = float('inf')
            front_id = None
            back_ds = -float('inf')
            back_id = None

            cav_s_lane = self.cav_s_by_lane.get(lane, None)
            if cav_s_lane is None:
                lane_fb[lane] = (None, float('inf'), None, -float('inf'))
                continue

            for hv_id, hv in self.proj.hvs.items():
                if not hv.valid:
                    continue

                dx = hv.x - cav_x
                dy = hv.y - cav_y
                if (dx * dx + dy * dy) > r2:
                    continue

                if hv.lane != lane:
                    continue
                if hv.abs_d > LANE_ABS_D_GATE:
                    continue

                ds = hv.s - cav_s_lane

                if ds > FRONT_DS_MAX:
                    continue
                if ds < -BACK_DS_MAX:
                    continue

                if ds > 0 and ds < front_ds:
                    front_ds = ds
                    front_id = hv_id
                if ds < 0 and ds > back_ds:
                    back_ds = ds
                    back_id = hv_id

            lane_fb[lane] = (front_id, front_ds, back_id, back_ds)

        self.lane_front_back = lane_fb

    def get_front_back_in_lane(self, lane: int, cav_s: float) -> Tuple[Optional[int], float, Optional[int], float]:
        if lane in self.lane_front_back:
            return self.lane_front_back[lane]
        return None, float('inf'), None, -float('inf')

    def lane_is_safe(self, lane: int, cav_s: float) -> bool:
        fid, fds, bid, bds = self.get_front_back_in_lane(lane, cav_s)
        front_ok = (fid is None) or (fds >= SAFE_FRONT_GAP)
        back_ok = (bid is None) or (abs(bds) >= SAFE_BACK_GAP)
        return front_ok and back_ok

    def update_speed_estimates(self, cav_s: float, front_id: Optional[int], back_id: Optional[int]):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_time is None:
            self.prev_time = now
            self.prev_cav_s = cav_s
            return

        dt = max(1e-3, now - self.prev_time)
        self.ego_v = (cav_s - self.prev_cav_s) / dt
        self.prev_cav_s = cav_s
        self.prev_time = now

    def fsm_step(self, cav_lane: int, cav_s: float):
        now = self.get_clock().now().nanoseconds * 1e-9

        # --- NOT LANE CHANGE ZONE : absolute KEEP ---
        if self.in_kl_zone:
            self.state = LCState.KEEP_LANE
            self.target_lane = cav_lane
            self.behavior = BEHAVIOR_GO
            return

        # --- LANE CHANGE ZONE : forced LC ---
        if self.in_lc_zone:
            if cav_lane == 2:
                if self.lane_is_safe(1, cav_s):
                    self.state = LCState.LANE_CHANGE
                    self.target_lane = 1
                    self.behavior = BEHAVIOR_GO
                else:
                    self.state = LCState.KEEP_LANE
                    self.target_lane = 2
                    self.behavior = BEHAVIOR_SLOW
            return

        # --- normal FSM ---
        front_id, front_ds, back_id, back_ds = self.get_front_back_in_lane(cav_lane, cav_s)

        front_close = (front_id is not None) and (0.0 < front_ds < FRONT_CLOSE)
        back_close = (back_id is not None) and (-BACK_CLOSE < back_ds < 0.0)

        if self.state == LCState.KEEP_LANE:
            self.target_lane = cav_lane
            if front_close or back_close:
                self.state = LCState.PREPARE_LC

        elif self.state == LCState.PREPARE_LC:
            for nl in [cav_lane - 1, cav_lane + 1]:
                if nl in ALLOWED_LANES and self.lane_is_safe(nl, cav_s):
                    self.state = LCState.LANE_CHANGE
                    self.target_lane = nl
                    self.last_lc_time = now
                    return
            self.state = LCState.KEEP_LANE
            self.target_lane = cav_lane

        elif self.state == LCState.LANE_CHANGE:
            if cav_lane == self.target_lane:
                self.state = LCState.KEEP_LANE

    def tick(self):
        if not self.proj.ego.valid:
            return

        cav_x = self.proj.ego.x
        cav_y = self.proj.ego.y

        self.cav_s_by_lane = {}
        best_lane = None
        best_abs_d = INF
        best_tuple = None

        for lane in ALLOWED_LANES:
            s, d, i, u, abs_d, qx, qy, mode = self.proj.robust_project(self.proj.ego, lane, cav_x, cav_y)
            self.cav_s_by_lane[lane] = s
            if abs_d < best_abs_d:
                best_abs_d = abs_d
                best_lane = lane
                best_tuple = (s, d)

        cav_lane = best_lane
        cav_s, cav_d = best_tuple

        self.proj.ego.lane = cav_lane
        self.proj.ego.s = cav_s
        self.proj.ego.d = cav_d
        self.proj.ego.abs_d = best_abs_d

        for hv_id, hv in self.proj.hvs.items():
            if not hv.valid:
                continue
            best_l = None
            best_a = INF
            for lane in ALLOWED_LANES:
                s, d, i, u, abs_d, *_ = self.proj.robust_project(hv, lane, hv.x, hv.y)
                if abs_d < best_a:
                    best_a = abs_d
                    best_l = lane
                    hv.s = s
                    hv.d = d
            hv.lane = best_l
            hv.abs_d = best_a

        self.build_lane_front_back(cav_x, cav_y)

        # --- zone projection (CAV only) ---
        s, d, i, u, abs_d, *_ = self.proj.robust_project(self.proj.ego, LANE_CHANGE_ZONE, cav_x, cav_y)
        self.in_lc_zone = abs_d <= ZONE_ABS_D_GATE

        s, d, i, u, abs_d, *_ = self.proj.robust_project(self.proj.ego, NOT_LANE_CHANGE_ZONE, cav_x, cav_y)
        self.in_kl_zone = abs_d <= ZONE_ABS_D_GATE

        self.behavior = BEHAVIOR_GO
        self.fsm_step(cav_lane, cav_s)
        
        # tick()에서 build_lane_front_back() 이후에 추가
        front_id, front_ds, back_id, back_ds = \
            self.get_front_back_in_lane(cav_lane, cav_s)
        
        msg_fsm = Int32()
        msg_fsm.data = int(self.state.value)
        self.pub_fsm_state.publish(msg_fsm)



        # Debug
        self.get_logger().info(
            f'[PERCEPTION]\n'
            f'  CAV lane={cav_lane} s={cav_s:.2f} d={cav_d:.2f} v={self.ego_v:.2f}\n'
            f'  FRONT: id={front_id} ds={front_ds if front_id is not None else "none"}\n'
            f'  BACK : id={back_id} ds={back_ds if back_id is not None else "none"}\n'
            f'  FSM  : state={self.state.name} target_lane={self.target_lane}\n'
            f'  Pose : X={cav_x:.2f} Y={cav_y:.2f}'
        )

        # Lane-wise debug
        lane_debug_lines = []
        for lane in ALLOWED_LANES:
            fid, fds, bid, bds = self.lane_front_back.get(lane, (None, INF, None, -INF))
            fds_str = f'{fds:.2f}' if fid is not None else 'none'
            bds_str = f'{bds:.2f}' if bid is not None else 'none'
            tag = '<- CURRENT' if lane == cav_lane else ''
            lane_debug_lines.append(
                f'  Lane {lane}: '
                f'FRONT(id={fid}, ds={fds_str}) | '
                f'BACK(id={bid}, ds={bds_str}) {tag}'
            )

        self.get_logger().info('[LANE SNAPSHOT]\n' + '\n'.join(lane_debug_lines))



        # publish
        msg_lane = Int32()
        msg_lane.data = int(self.target_lane if self.target_lane is not None else cav_lane)
        self.pub_target_lane.publish(msg_lane)

        msg_beh = Int32()
        msg_beh.data = self.behavior
        self.pub_behavior.publish(msg_beh)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionDebugNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from geometry_msgs.msg import PoseStamped, Accel


########### Constants
PATH_FILES = [
    '/home/dongminkim/Desktop/KAIST-Mobility-Challenge-H6/task2/paths/path_2_1.json',
    '/home/dongminkim/Desktop/KAIST-Mobility-Challenge-H6/task2/paths/path_2_2.json',
    '/home/dongminkim/Desktop/KAIST-Mobility-Challenge-H6/task2/paths/path_2_3.json',
    '/home/dongminkim/Desktop/KAIST-Mobility-Challenge-H6/task2/paths/path_2_4.json',
    '/home/dongminkim/Desktop/KAIST-Mobility-Challenge-H6/task2/paths/not_lane_change.json',
    '/home/dongminkim/Desktop/KAIST-Mobility-Challenge-H6/task2/paths/lane_change.json',
]
LANE_COUNT = 4  # IMPORTANT: project onto all 4 lanes


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


########### Vehicle States
@dataclass
class EgoState:
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    valid: bool = False
    prev_time: Optional[float] = None
    prev_x: Optional[float] = None
    prev_y: Optional[float] = None
    
    # Projected 
    lane: int = -1
    s: float = 0.0
    d: float = 0.0
    abs_d: float = 0.0

    # per-lane hint cache: lane_idx -> seg_i
    seg_hint: Optional[Dict[int, int]] = None

    def __post_init__(self):
        if self.seg_hint is None:
            self.seg_hint = {}



@dataclass
class HVState:
    # current
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0

    # Projected
    lane: int = -1
    s: float = 0.0
    d: float = 0.0
    abs_d: float = 0.0

    # prev (for speed estimation)
    prev_time: Optional[float] = None
    prev_x: Optional[float] = None
    prev_y: Optional[float] = None

    # first observation time (for warm-up)
    first_time: Optional[float] = None

    valid: bool = False

    # per-lane hint cache: lane_idx -> seg_i
    seg_hint: Optional[Dict[int, int]] = None

    def __post_init__(self):
        if self.seg_hint is None:
            self.seg_hint = {}


########## Path + Projection Manager
class ProjectionManager:
    def __init__(
        self,
        hv_ids=range(19, 37),
        nearest_window: int = 200,
        recover_dist: float = 0.6,
    ):
        self.NEAREST_WINDOW = nearest_window
        self.RECOVER_DIST = recover_dist
        self.GLOBAL_WINDOW = 10**9

        # Load paths
        self.paths = [self.load_path(p) for p in PATH_FILES]
        for p in self.paths:
            p['s'] = self.preprocess_path(p['x'], p['y'])

        # Ego + HVs
        self.ego = EgoState()
        self.hvs: Dict[int, HVState] = {i: HVState() for i in hv_ids}

        # Ego hint per lane
        self.ego_seg_hint: Dict[int, int] = {}

    ########## Path
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
            s.append(s[-1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
        return s

    ########## Ego Pose
    def update_ego_pose(self, msg: PoseStamped) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.position.x
        y = msg.pose.position.y

        if self.ego.valid:
            self.ego.prev_time = self.ego.t
            self.ego.prev_x = self.ego.x
            self.ego.prev_y = self.ego.y

            dt = t - self.ego.prev_time
            if dt > 1e-3:
                dx = x - self.ego.prev_x
                dy = y - self.ego.prev_y
                self.ego.speed = math.hypot(dx, dy) / dt

        self.ego.t = t
        self.ego.x = x
        self.ego.y = y
        self.ego.yaw = msg.pose.orientation.z
        self.ego.valid = True

    ########## HV Pose
    def update_hv_pose(self, hv_id: int, msg: PoseStamped) -> None:
        hv = self.hvs[hv_id]

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.position.x
        y = msg.pose.position.y

        # first observation
        if not hv.valid:
            hv.first_time = t
        else:
            hv.prev_time = hv.t
            hv.prev_x = hv.x
            hv.prev_y = hv.y

        hv.t = t
        hv.x = x
        hv.y = y
        hv.valid = True

    ########## HV speed estimate (DO NOT TRUST BLINDLY)
    def estimate_hv_speed(self, hv_id: int) -> float:
        hv = self.hvs[hv_id]
        if not hv.valid or hv.prev_time is None:
            return 0.0
        dt = hv.t - hv.prev_time
        if dt <= 1e-3:
            return 0.0
        dx = hv.x - (hv.prev_x if hv.prev_x is not None else hv.x)
        dy = hv.y - (hv.prev_y if hv.prev_y is not None else hv.y)
        return math.hypot(dx, dy) / dt

    ########## HV speed warm-up guard (IMPORTANT)
    def hv_speed_valid(self, hv_id: int, T_WARMUP: float = 0.3) -> bool:
        hv = self.hvs[hv_id]
        if not hv.valid or hv.first_time is None:
            return False
        return (hv.t - hv.first_time) >= T_WARMUP

    ########## Projection core
    def project_to_lane_q(
        self,
        lane_idx: int,
        x: float,
        y: float,
        hint: int = 0,
        window: Optional[int] = None
    ) -> Tuple[float, float, int, float, float, float, float]:
        """
        기존 project_to_lane과 동일한데, projection foot-point(qx,qy)까지 리턴한다.
        returns: (s, d, seg_i, u, abs_d, qx, qy)
        """
        if window is None:
            window = self.NEAREST_WINDOW

        px = self.paths[lane_idx]['x']
        py = self.paths[lane_idx]['y']
        ps = self.paths[lane_idx]['s']

        n = len(px)
        hint = int(clamp(hint, 0, n - 2))
        a = max(0, hint - window)
        b = min(n - 1, hint + window + 1)

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
            u = clamp((wx * vx + wy * vy) / seg_len2, 0.0, 1.0)

            qx = x1 + u * vx
            qy = y1 + u * vy

            dx = x - qx
            dy = y - qy
            dist2 = dx * dx + dy * dy

            if dist2 < best_dist2:
                best_dist2 = dist2
                best_i = i
                best_u = u
                best_s = ps[i] + u * (ps[i + 1] - ps[i])
                cross = vx * dy - vy * dx
                mag = math.sqrt(dist2)
                best_d = mag if cross > 0 else -mag
                best_qx = qx
                best_qy = qy

        return best_s, best_d, best_i, best_u, abs(best_d), best_qx, best_qy


    def robust_project(
        self,
        veh,               # EgoState or HVState (seg_hint dict가 있어야 함)
        lane_idx: int,
        x: float,
        y: float,
    ) -> Tuple[float, float, int, float, float, float, float, str]:

        hint_opt = veh.seg_hint.get(lane_idx, None)

        # 1) global init
        if hint_opt is None:
            s, d, seg_i, u, abs_d, qx, qy = self.project_to_lane_q(
                lane_idx, x, y, hint=0, window=self.GLOBAL_WINDOW
            )
            veh.seg_hint[lane_idx] = seg_i
            return s, d, seg_i, u, abs_d, qx, qy, "global_init"

        # 2) local scan
        s, d, seg_i, u, abs_d, qx, qy = self.project_to_lane_q(
            lane_idx, x, y, hint=hint_opt, window=self.NEAREST_WINDOW
        )
        mode = "local"

        # 3) recover
        if abs_d > self.RECOVER_DIST:
            s2, d2, seg_i2, u2, abs_d2, qx2, qy2 = self.project_to_lane_q(
                lane_idx, x, y, hint=0, window=self.GLOBAL_WINDOW
            )
            if abs_d2 < abs_d:
                s, d, seg_i, u, abs_d, qx, qy = s2, d2, seg_i2, u2, abs_d2, qx2, qy2
                mode = "recover_global"

        veh.seg_hint[lane_idx] = seg_i
        return s, d, seg_i, u, abs_d, qx, qy, mode

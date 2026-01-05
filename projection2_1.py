import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from geometry_msgs.msg import PoseStamped, Accel


########### Constants
PATH_FILES = [
    '/home/dongminkim/Desktop/Mobility_Challenge_Simulator/src/central_control/path_2_1.json',
    '/home/dongminkim/Desktop/Mobility_Challenge_Simulator/src/central_control/path_2_2.json',
    '/home/dongminkim/Desktop/Mobility_Challenge_Simulator/src/central_control/path_2_3.json',
    '/home/dongminkim/Desktop/Mobility_Challenge_Simulator/src/central_control/path_2_4.json'
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



@dataclass
class HVState:
    # current
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0

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
    def project_to_lane(
        self,
        lane_idx: int,
        x: float,
        y: float,
        hint: int = 0,
        window: Optional[int] = None
    ) -> Tuple[float, float, int, float, float]:
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

        return best_s, best_d, best_i, best_u, abs(best_d)

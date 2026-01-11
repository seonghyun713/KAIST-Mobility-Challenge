# ========================= control.py =========================
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import json
import os
import math
from geometry_msgs.msg import Accel, PoseStamped



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


CURVE_PARAMS = {
    "vel": 0.58,
    "look_ahead": 0.38,
    "kp": 6.3,
    "ki": 0.05,
    "kd": 1.0,
    "k_cte": 5.0
}

STRAIGHT_PARAMS = {
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


class MapPredictionDriver(Node):
    def __init__(self, vehicle_id: int, path_filename: str):
        super().__init__(f"driver_vehicle_{int(vehicle_id)}")
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

        self.create_subscription(PoseStamped, self.TOPIC, self.pose_callback, qos_profile)
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

        self.create_timer(TICK_RATE, self.drive_loop)

        # nearest search hint
        self.last_nearest_idx = 0
        self.HINT_WINDOW = 150  # 안정적으로 따라가게 윈도우로 탐색

    def pose_callback(self, msg: PoseStamped):
        self.got_pose = True
        self.curr_yaw = float(msg.pose.orientation.z)
        self.curr_x = float(msg.pose.position.x) - (DIST_CENTER_TO_REAR * math.cos(self.curr_yaw))
        self.curr_y = float(msg.pose.position.y) - (DIST_CENTER_TO_REAR * math.sin(self.curr_yaw))

    def get_road_curvature(self, current_idx: int) -> float:
        idx_now = current_idx
        idx_near = min(len(self.path_x) - 1, current_idx + 50)
        idx_far = min(len(self.path_x) - 1, current_idx + 100)
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

    def nearest_index_windowed(self):
        n = len(self.path_x)
        if n == 0:
            return 0, float("inf")

        a = max(0, self.last_nearest_idx - self.HINT_WINDOW)
        b = min(n - 1, self.last_nearest_idx + self.HINT_WINDOW)

        best_d = float("inf")
        best_i = self.last_nearest_idx

        for i in range(a, b + 1):
            d = math.hypot(self.path_x[i] - self.curr_x, self.path_y[i] - self.curr_y)
            if d < best_d:
                best_d = d
                best_i = i

        self.last_nearest_idx = best_i
        return best_i, best_d

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

        curr_idx, min_d = self.nearest_index_windowed()

        road_curve_amount = self.get_road_curvature(curr_idx)
        if road_curve_amount < 0.15 and min_d < 0.4:
            self.mode = "STRGT"
            params = STRAIGHT_PARAMS
        else:
            self.mode = "CURVE"
            params = CURVE_PARAMS

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

        target_v = float(params["vel"])
        if target_v > self.current_vel_cmd:
            self.current_vel_cmd = min(target_v, self.current_vel_cmd + ACCEL_LIMIT * dt)
        else:
            self.current_vel_cmd = max(target_v, self.current_vel_cmd - DECEL_LIMIT * dt)

        cmd = Accel()
        cmd.linear.x = float(self.current_vel_cmd)
        cmd.angular.z = float(final_steer)
        self.accel_raw_pub.publish(cmd)

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import json
import os
import math
from geometry_msgs.msg import Accel, PoseStamped

# ============================================================
# [파라미터 세트] (네 값 그대로)
# ============================================================
CURVE_PARAMS = {
    "vel": 0.58,
    "look_ahead": 0.38,
    "kp": 6.3,
    "ki": 0.05,
    "kd": 1.0,
    "k_cte": 5.0
}

STRAIGHT_PARAMS = {
    "vel": 1.8,
    "look_ahead": 0.47,
    "kp": 4.0,
    "ki": 0.005,
    "kd": 1.5,
    "k_cte": 0.8
}

WHEELBASE = 0.211
DIST_CENTER_TO_REAR = WHEELBASE / 2.0
TICK_RATE = 0.05

ACCEL_LIMIT = 0.8
DECEL_LIMIT = 3.0


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


class MapPredictionDriver(Node):
    def __init__(self, vehicle_id: int):
        super().__init__(f'driver_vehicle_{vehicle_id}')
        self.vehicle_id = int(vehicle_id)

        self.PATH_FILENAME = f'path1_{self.vehicle_id}.json'   # ✅ 차량별 경로
        self.VEHICLE_TOPIC_NAME = f'/CAV_{self.vehicle_id:02d}' # ✅ 차량별 토픽

        print(f"\n [차량 {self.vehicle_id}] 지도 기반 예지 주행 (Map Prediction)")
        print(f"   - 목표: 직진 1.8m/s / 커브 0.58m/s")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.path_pts = load_path_points(self.PATH_FILENAME)
        self.path_x = [p[0] for p in self.path_pts]
        self.path_y = [p[1] for p in self.path_pts]

        if not self.path_pts:
            self.get_logger().error(f"❌ 경로 파일 없음: {self.PATH_FILENAME}")
        else:
            self.get_logger().info(f"✅ Path loaded: {self.PATH_FILENAME} ({len(self.path_pts)} pts)")

        # ✅ Pose는 그대로 구독
        self.create_subscription(PoseStamped, self.VEHICLE_TOPIC_NAME, self.pose_callback, qos_profile)

        # ✅ 중요: Driver는 raw로만 publish (Guardian이 최종 accel publish)
        self.accel_publisher = self.create_publisher(
            Accel, f'{self.VEHICLE_TOPIC_NAME}_accel_raw', 10
        )

        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0
        self.got_pose = False

        self.prev_err = 0.0
        self.int_err = 0.0
        self.last_time = self.get_clock().now()

        self.current_vel_cmd = 0.48
        self.mode = "CURVE"

        self.log_counter = 0
        self.create_timer(TICK_RATE, self.drive_loop)

    def pose_callback(self, msg):
        self.got_pose = True
        self.curr_yaw = msg.pose.orientation.z
        self.curr_x = msg.pose.position.x - (DIST_CENTER_TO_REAR * math.cos(self.curr_yaw))
        self.curr_y = msg.pose.position.y - (DIST_CENTER_TO_REAR * math.sin(self.curr_yaw))

    def get_road_curvature(self, current_idx):
        idx_now = current_idx
        idx_near = min(len(self.path_x) - 1, current_idx + 50)
        idx_far  = min(len(self.path_x) - 1, current_idx + 100)
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

    def drive_loop(self):
        if not self.got_pose or not self.path_pts:
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0.001:
            return
        if dt > 0.1:
            dt = 0.1

        # Step 1) closest
        min_d = float('inf')
        curr_idx = 0
        for i, (px, py) in enumerate(zip(self.path_x, self.path_y)):
            d = math.hypot(px - self.curr_x, py - self.curr_y)
            if d < min_d:
                min_d = d
                curr_idx = i

        # Step 2) mode decision
        road_curve_amount = self.get_road_curvature(curr_idx)
        if road_curve_amount < 0.15 and min_d < 0.4:
            self.mode = "STRGT"
            params = STRAIGHT_PARAMS
        else:
            self.mode = "CURVE"
            params = CURVE_PARAMS

        # Step 3) look-ahead
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
        cte = min_d * params["k_cte"] * (-1 if yaw_err < 0 else 1)

        raw_steer = p + i_term + d_term + cte
        final_steer = max(-1.0, min(1.0, raw_steer))
        self.prev_err = yaw_err

        # Step 4) speed ramp
        target_v = float(params["vel"])
        if target_v > self.current_vel_cmd:
            self.current_vel_cmd = min(target_v, self.current_vel_cmd + ACCEL_LIMIT * dt)
        else:
            self.current_vel_cmd = max(target_v, self.current_vel_cmd - DECEL_LIMIT * dt)

        cmd = Accel()
        cmd.linear.x = float(self.current_vel_cmd)
        cmd.angular.z = float(final_steer)
        self.accel_publisher.publish(cmd)

        self.log_counter += 1
        if self.log_counter % 20 == 0:
            print(f"[Car{self.vehicle_id} {self.mode}] Vel:{self.current_vel_cmd:.2f} | RoadCurve:{road_curve_amount:.3f} | DistErr:{min_d:.3f}")


def main(args=None):
    rclpy.init(args=args)

    drivers = [MapPredictionDriver(i) for i in [1, 2, 3, 4]]

    ex = MultiThreadedExecutor(num_threads=8)
    for n in drivers:
        ex.add_node(n)

    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        for n in drivers:
            n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

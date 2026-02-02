#!/usr/bin/env python3
import os
import math
import json
import torch
import torch.nn as nn
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Accel, PoseStamped, Twist
from round import VehicleController as RoundController 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DIR = os.path.join(BASE_DIR, "path")

# ================================
# [Time Attack 튜닝]
# ================================
BOOST_SPEED_RATIO = 1.0  # 안전하게 1.0부터 시작해서 1.1, 1.2로 올려보세요!
BASE_SPEED = 1.5         # 학습 기준 속도
LOOK_AHEAD_DISTS = [0.6, 1.2, 1.8] # 3개의 눈

# [설정] 차량별 토픽 매핑
CAV_TOPIC_NUM = { 1: 4, 2: 6, 3: 10, 4: 28 }
HV_TOPICS = { "HV1": "/HV_19", "HV2": "/HV_20" }

def cav_topic(logical_id: int) -> str: return f"/CAV_{int(CAV_TOPIC_NUM[logical_id]):02d}"
def cav_accel_raw_topic(logical_id: int) -> str: return f"{cav_topic(logical_id)}_accel_raw"
def cav_accel_round_raw_topic(logical_id: int) -> str: return f"{cav_topic(logical_id)}_accel_round_raw"
def hv_topic(role: str) -> str: return HV_TOPICS[role]

# ------------------------------
# 모델 클래스 (학습 코드와 동일)
# ------------------------------
class SmartControlNet(nn.Module):
    def __init__(self):
        super(SmartControlNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

# ------------------------------
# 유틸리티
# ------------------------------
def load_path_points(json_file):
    if not os.path.exists(json_file): return []
    with open(json_file, "r") as f: data = json.load(f)
    xs = data.get("x") or data.get("X"); ys = data.get("y") or data.get("Y")
    return list(zip(xs, ys)) if xs and ys else []

def load_zone_from_csv(filename):
    points = []
    if not os.path.exists(filename): return []
    try:
        import csv
        with open(filename, "r") as f:
            reader = csv.reader(f); next(reader, None)
            for row in reader:
                if len(row)>=2: points.append((float(row[0]), float(row[1])))
    except: pass
    return points

# ------------------------------
# [핵심] Smart AI Driver Node
# ------------------------------
class SmartNeuralDriver(Node):
    def __init__(self, vehicle_id, path_filename):
        super().__init__(f"driver_vehicle_{vehicle_id}")
        self.vid = int(vehicle_id)
        self.logical_id = int(vehicle_id)
        
        # 모델 로드
        self.device = torch.device("cpu")
        self.model = SmartControlNet().to(self.device)
        self.model_path = os.path.join(BASE_DIR, "weights_smart", f"model_veh{self.vid}.pth")
        
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            self.get_logger().info(f"✅ [Car {self.vid}] Smart Driver Loaded!")
        else:
            self.get_logger().error(f"❌ Model missing: {self.model_path}")

        # Path Load & ROS Setup
        self.path_pts = load_path_points(path_filename)
        self.path_x = [p[0] for p in self.path_pts]
        self.path_y = [p[1] for p in self.path_pts]
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(PoseStamped, cav_topic(self.logical_id), self.pose_callback, qos)
        self.accel_raw_pub = self.create_publisher(Accel, cav_accel_raw_topic(self.logical_id), 10)
        
        self.curr_x = 0.0; self.curr_y = 0.0; self.curr_yaw = 0.0; self.got_pose = False
        self.current_vel_cmd = 1.0; self.old_nearest_idx = 0
        
        # Zone 정보 (감속용)
        self.SLOW_ZONES = [(-0.5, -4.6, 2.0, -2.0), (-0.5, 2.7, 2.0, -1.4)]
        self.SLOW_ZONES = [(min(x1,x2), max(x1,x2), min(y1,y2), max(y1,y2)) for x1,x2,y1,y2 in self.SLOW_ZONES]

        self.create_timer(0.02, self.drive_loop)

    def pose_callback(self, msg):
        self.got_pose = True
        self.curr_yaw = float(msg.pose.orientation.z)
        self.curr_x = float(msg.pose.position.x) - (0.1055 * math.cos(self.curr_yaw)) # Rear Axle
        self.curr_y = float(msg.pose.position.y) - (0.1055 * math.sin(self.curr_yaw))

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle
        
    def global_to_local(self, gx, gy, rx, ry, ryaw):
        dx = gx - rx; dy = gy - ry
        lx = dx * math.cos(ryaw) + dy * math.sin(ryaw)
        ly = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
        return lx, ly

    def is_in_zone(self):
        for (x1, x2, y1, y2) in self.SLOW_ZONES:
            if x1 <= self.curr_x <= x2 and y1 <= self.curr_y <= y2: return True
        return False

    def drive_loop(self):
        if not self.got_pose or not self.path_pts: return

        # 1. Nearest Point 찾기
        path_len = len(self.path_pts)
        min_d = float('inf'); curr_idx = self.old_nearest_idx
        # 최적화 탐색 (-50 ~ +50)
        for offset in range(-50, 50):
            idx = (self.old_nearest_idx + offset) % path_len
            d = math.hypot(self.path_x[idx]-self.curr_x, self.path_y[idx]-self.curr_y)
            if d < min_d: min_d = d; curr_idx = idx
        
        # 놓쳤으면 전체 탐색
        if min_d > 5.0:
            min_d = float('inf')
            for i in range(path_len):
                d = math.hypot(self.path_x[i]-self.curr_x, self.path_y[i]-self.curr_y)
                if d < min_d: min_d = d; curr_idx = i
        
        self.old_nearest_idx = curr_idx

        # 2. [핵심] 3개의 눈(Look-ahead Points) 찾기
        local_ys = []
        for target_dist in LOOK_AHEAD_DISTS:
            t_idx = curr_idx
            for i in range(path_len):
                next_idx = (curr_idx + i) % path_len
                p = self.path_pts[next_idx]
                if math.hypot(p[0]-self.curr_x, p[1]-self.curr_y) >= target_dist:
                    t_idx = next_idx; break
            
            tx, ty = self.path_pts[t_idx]
            lx, ly = self.global_to_local(tx, ty, self.curr_x, self.curr_y, self.curr_yaw)
            local_ys.append(ly)

        # 3. 에러 계산 (CTE, YawErr)
        # Yaw Error는 1.2m 앞(중거리) 경로의 방향과 비교
        idx_mid = (curr_idx + 10) % path_len # 대략
        tx_mid, ty_mid = self.path_pts[idx_mid]
        desired_yaw = math.atan2(ty_mid - self.curr_y, tx_mid - self.curr_x)
        yaw_err = self.normalize_angle(desired_yaw - self.curr_yaw)
        # CTE는 가장 가까운 점 기준 (local y 사용이 더 정확함)
        # 로컬 좌표계에서 ly가 곧 CTE에 근사함 (직선 근사 시)
        cte = min_d if local_ys[0] > 0 else -min_d

        # 4. 모델 추론
        # Input: [x, y, yaw, v, cte, yaw_err, ly1, ly2, ly3]
        inp = torch.tensor([
            self.curr_x, self.curr_y, self.curr_yaw,
            min(self.current_vel_cmd, BASE_SPEED), # 속도 속이기 (안정적 추론 유도)
            cte, yaw_err,
            local_ys[0], local_ys[1], local_ys[2]
        ], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred_steer = self.model(inp).item()

        # 5. [Time Attack] 속도 제어
        # 곡률 기반 자동 감속: ly_far(1.8m 앞)가 많이 꺾여있으면 감속
        in_zone = self.is_in_zone()
        curvature = abs(local_ys[2]) 
        
        if in_zone:
            target_v = 1.5 # Zone에서는 안전 속도
        else:
            if curvature > 0.5: target_v = 1.2 * BOOST_SPEED_RATIO # 급커브
            elif curvature > 0.2: target_v = 1.6 * BOOST_SPEED_RATIO # 완만
            else: target_v = 1.8 * BOOST_SPEED_RATIO # 직선

        # 6. 명령 발행
        if target_v > self.current_vel_cmd:
            self.current_vel_cmd = min(target_v, self.current_vel_cmd + 0.8 * 0.02)
        else:
            self.current_vel_cmd = max(target_v, self.current_vel_cmd - 1.5 * 0.02)

        cmd = Accel()
        cmd.linear.x = float(self.current_vel_cmd)
        cmd.angular.z = float(max(-2.0, min(2.0, pred_steer)))
        self.accel_raw_pub.publish(cmd)

# ============================================================
# [GUARDIAN] MUX (기존과 동일 구조 - 간소화 버전)
# ============================================================
class Problem3DualZoneGuardianMux(Node):
    def __init__(self):
        super().__init__("problem3_dualzone_guardian_mux")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.VEH_IDS = [1, 2, 3, 4]
        self.pub = {vid: self.create_publisher(Twist, f"/CAV_{int(CAV_TOPIC_NUM[vid]):02d}/cmd_vel", 10) for vid in self.VEH_IDS}
        self.raw = {vid: Accel() for vid in self.VEH_IDS}
        self.round_raw = {vid: Accel() for vid in self.VEH_IDS}
        self.pose = {vid: None for vid in self.VEH_IDS}
        
        for vid in self.VEH_IDS:
            topic = f"/CAV_{int(CAV_TOPIC_NUM[vid]):02d}"
            self.create_subscription(PoseStamped, topic, self._make_pose_cb(vid), qos)
            self.create_subscription(Accel, f"{topic}_accel_raw", self._make_raw_cb(vid), 10)
            self.create_subscription(Accel, f"{topic}_accel_round_raw", self._make_round_raw_cb(vid), 10)
        self.create_timer(0.02, self.tick)

    def _make_pose_cb(self, vid):
        def cb(msg): self.pose[vid] = (msg.pose.position.x, msg.pose.position.y)
        return cb
    def _make_raw_cb(self, vid):
        def cb(msg): self.raw[vid] = msg
        return cb
    def _make_round_raw_cb(self, vid):
        def cb(msg): self.round_raw[vid] = msg
        return cb

    def _in_round_box(self, p):
        if not p: return False
        return (-0.5 <= p[0] <= 2.7) and (-1.4 <= p[1] <= 2.0)

    def tick(self):
        for vid in self.VEH_IDS:
            p = self.pose[vid]
            # MUX: 회전교차로 안에 있으면 RoundController, 아니면 AI
            use_round = self._in_round_box(p)
            src = self.round_raw[vid] if use_round else self.raw[vid]
            
            cmd = Twist()
            cmd.linear.x = float(src.linear.x)
            cmd.angular.z = float(src.angular.z)
            self.pub[vid].publish(cmd)

# ============================================================
# MAIN
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    drivers = [SmartNeuralDriver(i, os.path.join(PATH_DIR, f"path3_{i}.json")) for i in range(1,5)]
    
    p = lambda name: os.path.join(PATH_DIR, name)
    round_nodes = [
        RoundController(1, p("path3_1.json"), p("path3_1_zone.csv"), p("path_hv_3_1.csv"), p("path3_1_out_zone.csv"), p("path_hv_3_2.csv"), cav_topic(1), cav_accel_round_raw_topic(1), hv_topic("HV1"), hv_topic("HV2")),
        RoundController(2, p("path3_2.json"), p("path3_2_zone.csv"), p("path_hv_2_1.csv"), p("path3_2_out_zone.csv"), p("path_hv_2_2.csv"), cav_topic(2), cav_accel_round_raw_topic(2), hv_topic("HV1"), hv_topic("HV2")),
        RoundController(3, p("path3_3.json"), p("path3_3_zone.csv"), p("path_hv_2_1.csv"), None, p("path_hv_2_1.csv"), cav_topic(3), cav_accel_round_raw_topic(3), hv_topic("HV1"), hv_topic("HV2")),
        RoundController(4, p("path3_4.json"), p("path3_4_zone.csv"), p("path_hv_3_1.csv"), None, p("path_hv_3_1.csv"), cav_topic(4), cav_accel_round_raw_topic(4), hv_topic("HV1"), hv_topic("HV2")),
    ]
    guardian = Problem3DualZoneGuardianMux()
    
    ex = MultiThreadedExecutor(num_threads=12)
    for n in drivers + round_nodes + [guardian]: ex.add_node(n)
    
    try: ex.spin()
    except KeyboardInterrupt: pass
    finally:
        ex.shutdown()
        for n in drivers + round_nodes + [guardian]: n.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

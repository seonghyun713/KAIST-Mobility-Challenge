#!/usr/bin/env python3
import sys
import os
import math
import csv
import json
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Accel, PoseStamped

# 터미널 출력 즉시 확인
sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# [설정] 파라미터
# ============================================================
TARGET_VELOCITY = 0.7       # 주행 속도
STOP_VELOCITY   = 0.02       # 정지 속도
ZONE_RADIUS     = 0.20      # CAV가 구역(Start/Out)에 있는지 판단하는 반경
HV_DETECT_RADIUS = 0.10     # HV가 경로(CSV) 위에 있는지 판단하는 반경 (40cm)

# 리셋 거리 (Start Zone 로직용)
RESET_DISTANCE  = 2.2       

CTRL_PARAMS = {
    "look_ahead": 0.38, 
    "kp": 6.0, "ki": 0.05, "kd": 1.0, 
    "k_cte": 4.0
}

# ============================================================
# [파일 로드 함수]
# ============================================================
def load_path_from_json(filename):
    if not filename or not os.path.exists(filename): return []
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        xs = data.get("x") or data.get("X")
        ys = data.get("y") or data.get("Y")
        if not xs or not ys: return []
        return [(float(x), float(y)) for x, y in zip(xs, ys)]
    except: return []

def load_zone_from_csv(filename):
    points = []
    if not filename or not os.path.exists(filename): return []
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and len(row) >= 2:
                    try: points.append((float(row[0]), float(row[1])))
                    except: continue
    except: pass
    return points

# ============================================================
# [차량 제어 클래스]
# ============================================================
class VehicleController(Node):
    def __init__(self, vehicle_id, path_file, start_zone, start_trigger, out_zone=None, danger_zone=None):
        super().__init__(f"drive_node_v{vehicle_id:02d}")
        
        self.vid = vehicle_id
        self.id_str = f"{vehicle_id:02d}"
        
        # 1. 파일 로드
        self.path = load_path_from_json(path_file)
        
        # (A) 출발/정지 로직용 파일
        self.start_zone_points = load_zone_from_csv(start_zone)
        self.start_trigger_points = load_zone_from_csv(start_trigger)
        
        # (B) 합류/충돌 방지 로직용 파일 (추가됨)
        self.out_zone_points = load_zone_from_csv(out_zone)
        self.danger_zone_points = load_zone_from_csv(danger_zone)

        if self.start_trigger_points:
            print(f"[INFO] V{self.id_str} Start Trigger Loaded")
        if self.out_zone_points and self.danger_zone_points:
            print(f"[INFO] V{self.id_str} Exit Conflict Logic Activated")

        # 2. 상태 변수
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0
        
        # 로직 제어 (Start Zone용)
        self.stop_logic_disabled = False 
        
        # HV 좌표
        self.hv19_x, self.hv19_y = 0.0, 0.0
        self.hv20_x, self.hv20_y = 0.0, 0.0
        self.hv19_active = False 
        self.hv20_active = False

        self.prev_error = 0.0
        self.integral_error = 0.0
        self.is_connected = False
        
        # 3. 통신
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.topic_pose = f"/CAV_{self.id_str}"
        self.sub_pose = self.create_subscription(PoseStamped, self.topic_pose, self._callback_pose, qos)
        self.pub_cmd  = self.create_publisher(Accel, f"/CAV_{self.id_str}_accel", 10)
        
        # HV 구독 (트리거 또는 위험구역 파일이 하나라도 있으면 구독)
        if self.start_trigger_points or self.danger_zone_points:
            self.sub_hv19 = self.create_subscription(PoseStamped, "/HV_19", self._callback_hv19, qos)
            self.sub_hv20 = self.create_subscription(PoseStamped, "/HV_20", self._callback_hv20, qos)

    def _callback_hv19(self, msg):
        self.hv19_x = msg.pose.position.x; self.hv19_y = msg.pose.position.y; self.hv19_active = True
    def _callback_hv20(self, msg):
        self.hv20_x = msg.pose.position.x; self.hv20_y = msg.pose.position.y; self.hv20_active = True

    # --- 유틸리티: 특정 포인트 집합과 거리 체크 ---
    def _check_hv_in_zone(self, zone_points):
        """ HV19 또는 HV20이 주어진 zone_points 안에 있는지 확인 """
        if not zone_points: return False
        
        # Check HV19
        if self.hv19_active:
            for hx, hy in zone_points:
                if math.hypot(hx - self.hv19_x, hy - self.hv19_y) < HV_DETECT_RADIUS:
                    return True
        # Check HV20
        if self.hv20_active:
            for hx, hy in zone_points:
                if math.hypot(hx - self.hv20_x, hy - self.hv20_y) < HV_DETECT_RADIUS:
                    return True
        return False

    def _get_min_dist(self, zone_points):
        """ 내 차량과 zone_points 사이의 최소 거리 """
        min_d = 999.0
        if zone_points:
            for zx, zy in zone_points:
                d = math.hypot(zx - self.curr_x, zy - self.curr_y)
                if d < min_d: min_d = d
        return min_d

    # --- 메인 루프 ---
    def _callback_pose(self, msg):
        if not self.is_connected:
            self.is_connected = True
            print(f"[LINK] V{self.id_str} Connected!")

        self.curr_x = msg.pose.position.x
        self.curr_y = msg.pose.position.y
        self.curr_yaw = msg.pose.orientation.z 
        
        if not self.path: return

        # =========================================================
        # [속도 결정 로직]
        # =========================================================
        final_vel = TARGET_VELOCITY # 기본값

        # ---------------------------------------------------------
        # [Logic 1] Start Zone (정지 -> Latch 출발 -> 리셋)
        # ---------------------------------------------------------
        dist_to_start = self._get_min_dist(self.start_zone_points)

        if self.stop_logic_disabled:
            # 면제 상태: 리셋 거리 벗어났는지 체크
            if dist_to_start > RESET_DISTANCE:
                self.stop_logic_disabled = False
                # print(f"[RESET] V{self.id_str} Start-Logic Reset", end='\r')
        else:
            # 감시 상태: Start Zone 진입 체크
            if dist_to_start < ZONE_RADIUS:
                final_vel = STOP_VELOCITY # 일단 정지
                
                # HV 트리거 체크
                if self._check_hv_in_zone(self.start_trigger_points):
                    final_vel = TARGET_VELOCITY
                    self.stop_logic_disabled = True # 출발! (Latch ON)
                    print(f"[START] V{self.id_str} Green Light! (Start Zone)", end='\r')
                else:
                    print(f"[WAIT] V{self.id_str} Waiting at Start Zone...   ", end='\r')

        # ---------------------------------------------------------
        # [Logic 2] Exit/Conflict Zone (합류 구간 충돌 방지)
        # ★ 주의: Start 로직에서 가라고 했어도(TARGET), 여기가 위험하면 멈춰야 함 (STOP 덮어쓰기)
        # ---------------------------------------------------------
        if self.out_zone_points and self.danger_zone_points:
            dist_to_out = self._get_min_dist(self.out_zone_points)
            
            # 내가 합류 구간(Out Zone)에 있는가?
            if dist_to_out < ZONE_RADIUS:
                # HV가 위험 구간(Danger Zone)에 있는가?
                if self._check_hv_in_zone(self.danger_zone_points):
                    final_vel = STOP_VELOCITY # 위험! 정지
                    print(f"[YIELD] V{self.id_str} Waiting for HV at Exit/Merge...!!!", end='\r')
                else:
                    # HV가 지나갔으면 (그리고 Start 로직이 막지 않는다면) 주행
                    # 여기서 굳이 TARGET_VELOCITY로 강제할 필요 없음 (Start 로직 결과 유지)
                    pass

        # 3. 주행 제어
        self._control_vehicle(final_vel)

    def _control_vehicle(self, velocity):
        # Pure Pursuit
        min_dist = 1e9
        idx = 0
        path_len = len(self.path)
        for i in range(path_len):
            px, py = self.path[i]
            d = math.hypot(px - self.curr_x, py - self.curr_y)
            if d < min_dist: min_dist = d; idx = i
        
        target_idx = idx
        for i in range(idx, path_len):
            if math.hypot(self.path[i][0] - self.curr_x, self.path[i][1] - self.curr_y) >= CTRL_PARAMS["look_ahead"]:
                target_idx = i; break
        
        tx, ty = self.path[target_idx]
        desired_yaw = math.atan2(ty - self.curr_y, tx - self.curr_x)
        yaw_err = desired_yaw - self.curr_yaw
        while yaw_err > math.pi: yaw_err -= 2 * math.pi
        while yaw_err < -math.pi: yaw_err += 2 * math.pi

        dt = 0.1
        self.integral_error = max(-1.0, min(1.0, self.integral_error + yaw_err * dt))
        p = CTRL_PARAMS["kp"] * yaw_err
        i = CTRL_PARAMS["ki"] * self.integral_error
        d = CTRL_PARAMS["kd"] * (yaw_err - self.prev_error) / dt
        cte = min_dist * CTRL_PARAMS["k_cte"] * (-1.0 if yaw_err < 0 else 1.0)
        
        steer = max(-1.0, min(1.0, float(p + i + d + cte)))
        self.prev_error = yaw_err

        cmd = Accel()
        cmd.linear.x = float(velocity)
        cmd.angular.z = float(steer)
        self.pub_cmd.publish(cmd)

# ============================================================
# [메인 실행]
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    
    # -------------------------------------------------------------
    # [차량 설정 Config]
    # 형식: (ID, Path, Start_Zone, Start_Trigger, Out_Zone, Danger_Zone)
    # Out/Danger가 없으면 None
    # -------------------------------------------------------------
    cars_config = [
        # CAV 1: Start(3_1_zone, HV_3_1) + Exit(3_1_out, HV_3_2)
        (1, 'path3_1.json', 'path3_1_zone.csv', 'path_hv_3_1.csv', 
            'path3_1_out_zone.csv', 'path_hv_3_2.csv'), 
        
        # CAV 2: Start(3_2_zone, HV_2_1) + Exit(3_2_out, HV_2_2)
        (2, 'path3_2.json', 'path3_2_zone.csv', 'path_hv_2_1.csv', 
            'path3_2_out_zone.csv', 'path_hv_2_2.csv'), 
        
        # CAV 3, 4: Start 기능만 있음 (Out/Danger는 없음)
        (3, 'path3_3.json', 'path3_3_zone.csv', 'path_hv_2_1.csv', None, None), 
        (4, 'path3_4.json', 'path3_4_zone.csv', 'path_hv_3_1.csv', None, None), 
    ]

    executor = MultiThreadedExecutor()
    nodes = []

    print("=== FINAL SYSTEM: Full Safety Logic (Start & Exit) ===")
    print(f" - Start Reset Dist: {RESET_DISTANCE}m")
    print(f" - Exit Safety Logic: Active for CAV 1 & 2")
    
    for vid, p_file, s_zone, s_trig, o_zone, d_zone in cars_config:
        node = VehicleController(vid, p_file, s_zone, s_trig, o_zone, d_zone)
        nodes.append(node)
        executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        for n in nodes: 
            stop = Accel(); stop.linear.x = 0.0
            n.pub_cmd.publish(stop)
            n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

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
TARGET_VELOCITY = 0.7       # 기본 주행 속도
CRAWL_VELOCITY  = 0.28      # 서행 속도
STOP_VELOCITY   = -0.01     # 정지 속도
BOOST_VELOCITY  = 2.0       # 탈출 속도

SLOW_VELOCITY   = 0.2       
MAX_ACC_VELOCITY = 2.0      

#######################################################
ZONE_RADIUS      = 0.3       # 최종 정지 구역 (30cm 안으로 들어와야 브레이크 작동)
BRAKING_DISTANCE = 0.8      # 감속 시작 거리 (80cm, 약 차체 2대 거리부터 감속)

MIN_APPROACH_VEL = 0.5      # 정지 직전 최소 진입 속도 (너무 느리지 않게)
HARD_BRAKE_VEL   = -0.2      # 확실한 제동을 위한 역전압 (차량에 따라 조절)
#######################################################
HV_DETECT_RADIUS = 0.12     # HV 감지 반경 (트리거용)

# 출발 시 안전 거리 (43cm)
START_SAFETY_DIST = 0.43    

# 가속도 제한
ACCEL_STEP      = 0.1       

# ACC 거리 유지 파라미터 (주행 중)
ACC_DIST_LIMIT  = 0.42      
ACC_P_GAIN      = 2.5       

ACC_EXIT_BOOST_VEL  = 1.8   # 탈출 직후 최고속도
ACC_EXIT_BOOST_DIST = 0.009   # 몇 m 동안 유지할지 

# 리셋 거리 (다음 바퀴 준비용)
RESET_DISTANCE  = 2.2       

# CAV 01, 02 출구 가속 거리
# HV: 0.5, 1.0, 1.5 의 경우 0.48 이었음
EXIT_BOOST_DIST = 0.48
CTRL_PARAMS = {
    "look_ahead": 0.50, 
    "kp": 6.0, 
    "ki": 0.05, 
    "kd": 1.0, 
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
    def __init__(
        self,
        vehicle_id: int,
        path_file: str,
        start_zone: str,
        start_trigger: str,
        out_zone: str = None,
        danger_zone: str = None,
        # ✅ 추가: 토픽을 밖(task3.py)에서 주입받는다
        pose_topic: str = None,
        hv1_topic: str = None,
        hv2_topic: str = None,
        pub_topic: str = None,
    ):
        super().__init__(f"drive_node_v{vehicle_id:02d}")

        self.vid = int(vehicle_id)
        self.id_str = f"{self.vid:02d}"
        
        #브레이크 타이머 변수 추가
        self.brake_tick_count = 0        # 브레이크 밟은 횟수 카운트
        self.MAX_BRAKE_TICKS  = 10       # 10번 루프 동안만 브레이크 (약 0.5초)
        
        # 1. 파일 로드
        self.path = load_path_from_json(path_file)
        self.start_zone_points = load_zone_from_csv(start_zone)
        self.start_trigger_points = load_zone_from_csv(start_trigger)
        self.out_zone_points = load_zone_from_csv(out_zone)
        self.danger_zone_points = load_zone_from_csv(danger_zone)

        # 로그
        if self.vid in [3, 4] and self.danger_zone_points:
            print(f"[INFO] V{self.id_str} Smart ACC Logic Activated")
        elif self.out_zone_points:
            print(f"[INFO] V{self.id_str} Exit Logic Activated")

        # 2. 상태 변수
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0
        
        self.current_cmd_vel = 0.0 
        self.stop_logic_disabled = False 
        
        # ★ [신규] 출발 시 39cm 거리 유지 기능을 켤지 말지 결정하는 플래그
        self.is_start_gap_check_active = False

        # 상태 관리
        self.is_in_out_zone = False     
        self.boost_active = False       
        self.boost_start_pos = (0, 0)   

        # HV 상태
        self.hv19_x, self.hv19_y = 0.0, 0.0
        self.hv20_x, self.hv20_y = 0.0, 0.0
        self.hv19_active = False 
        self.hv20_active = False

        self.prev_error = 0.0
        self.integral_error = 0.0
        self.is_connected = False
        
        # 3. 통신
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)

        # ✅ defaults (혹시라도 안 넣으면 기존 포맷으로 fallback)
        if pose_topic is None:
            pose_topic = f"/CAV_{self.id_str}"
        if pub_topic is None:
            pub_topic = f"/CAV_{self.id_str}_accel_round_raw"
        if hv1_topic is None:
            hv1_topic = "/HV_19"
        if hv2_topic is None:
            hv2_topic = "/HV_20"

        self.topic_pose = pose_topic
        self.sub_pose = self.create_subscription(PoseStamped, self.topic_pose, self._callback_pose, qos)

        self.pub_cmd = self.create_publisher(Accel, pub_topic, 10)

        # ACC edge boost states (CAV 3,4)
        self.acc_zone_prev_in = False
        self.acc_exit_boost_active = False
        self.acc_exit_boost_start_pos = (0.0, 0.0)

        if self.start_trigger_points or self.danger_zone_points:
            self.sub_hv19 = self.create_subscription(PoseStamped, hv1_topic, self._callback_hv19, qos)
            self.sub_hv20 = self.create_subscription(PoseStamped, hv2_topic, self._callback_hv20, qos)

    def _callback_hv19(self, msg):
        self.hv19_x = msg.pose.position.x; self.hv19_y = msg.pose.position.y; self.hv19_active = True
    def _callback_hv20(self, msg):
        self.hv20_x = msg.pose.position.x; self.hv20_y = msg.pose.position.y; self.hv20_active = True

    # --- 유틸리티 ---
    def _check_hv_in_zone(self, zone_points):
        if not zone_points: return False
        if self.hv19_active:
            for hx, hy in zone_points:
                if math.hypot(hx - self.hv19_x, hy - self.hv19_y) < HV_DETECT_RADIUS: return True
        if self.hv20_active:
            for hx, hy in zone_points:
                if math.hypot(hx - self.hv20_x, hy - self.hv20_y) < HV_DETECT_RADIUS: return True
        return False

    def _get_min_dist(self, zone_points):
        min_d = 999.0
        if zone_points:
            for zx, zy in zone_points:
                d = math.hypot(zx - self.curr_x, zy - self.curr_y)
                if d < min_d: min_d = d
        return min_d

    def _get_closest_hv_front(self):
        closest_dist = 999.0
        if self.hv19_active and self.hv19_y < self.curr_y:
            d = math.hypot(self.hv19_x - self.curr_x, self.hv19_y - self.curr_y)
            if d < closest_dist: closest_dist = d
        if self.hv20_active and self.hv20_y < self.curr_y:
            d = math.hypot(self.hv20_x - self.curr_x, self.hv20_y - self.curr_y)
            if d < closest_dist: closest_dist = d
        return closest_dist

    # --- 메인 루프 ---
    def _callback_pose(self, msg):
        if not self.is_connected:
            self.is_connected = True
            print(f"[LINK] V{self.id_str} Connected!")

        self.curr_x = msg.pose.position.x
        self.curr_y = msg.pose.position.y
        self.curr_yaw = msg.pose.orientation.z 
        
        if not self.path: return

        # [Default] 기본은 0.7
        target_vel_req = TARGET_VELOCITY 

        # ---------------------------------------------------------
        # [Logic 1] Start Zone (정지 및 39cm 유지 로직)
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # [Logic 1] Smart Stop (타이머 브레이크)
        # ---------------------------------------------------------
        dist_to_start = self._get_min_dist(self.start_zone_points)

        # 1-1. 리셋 로직 (구역을 벗어나면 카운터도 초기화해야 다시 쓸 수 있음)
        if dist_to_start > RESET_DISTANCE:
            self.stop_logic_disabled = False
            self.is_start_gap_check_active = False
            self.brake_tick_count = 0  # [리셋] 다음 바퀴를 위해 초기화

        # 1-2. 정지 제어
        if not self.stop_logic_disabled:
            
            # (A) 정지 구역 진입 (브레이크 구간)
            if dist_to_start < ZONE_RADIUS:
                
                # 정해진 횟수만큼만 브레이크(-0.2)를 밟음
                if self.brake_tick_count < self.MAX_BRAKE_TICKS:
                    target_vel_req = HARD_BRAKE_VEL  # -0.2 (역전압)
                    self.brake_tick_count += 1       # 카운트 증가
                    print(f"[BRAKE] Kicking Reverse! ({self.brake_tick_count}/{self.MAX_BRAKE_TICKS})", end='\r')
                
                # 횟수가 차면 브레이크를 떼고 0.0 (중립/정지) 유지
                else:
                    target_vel_req = 0.0             # 완전 정지

                # 출발 트리거 체크 (출발하면 카운터 리셋 필요 없음, 어차피 멀어지니까)
                if self._check_hv_in_zone(self.start_trigger_points):
                    self.stop_logic_disabled = True
                    self.is_start_gap_check_active = True
                    target_vel_req = TARGET_VELOCITY
            
            # (B) 감속 구간
            elif dist_to_start < BRAKING_DISTANCE:
                # 감속 중에는 브레이크 카운터를 쓰지 않음 (0으로 유지)
                self.brake_tick_count = 0 
                
                # 감속 로직 
                ratio = (dist_to_start - ZONE_RADIUS) / (BRAKING_DISTANCE - ZONE_RADIUS)
                ramp_vel = MIN_APPROACH_VEL + (TARGET_VELOCITY - MIN_APPROACH_VEL) * ratio
                target_vel_req = ramp_vel
                
            # (C) 주행 구간
            else:
                self.brake_tick_count = 0 # 확실하게 0으로 유지

        # ---------------------------------------------------------
        # [Logic 2] Exit Conflict (CAV 1, 2)
        # ---------------------------------------------------------
        if self.out_zone_points and self.danger_zone_points:
            dist_to_out = self._get_min_dist(self.out_zone_points)
            
            # (A) 구역 내부
            if dist_to_out < ZONE_RADIUS:
                self.is_in_out_zone = True
                self.boost_active = False

                if self._check_hv_in_zone(self.danger_zone_points):
                    target_vel_req = STOP_VELOCITY 
                else:
                    target_vel_req = CRAWL_VELOCITY 

            # (B) 구역 탈출
            else:
                if self.is_in_out_zone: 
                    self.is_in_out_zone = False
                    self.boost_active = True
                    self.boost_start_pos = (self.curr_x, self.curr_y)

                if self.boost_active:
                    target_vel_req = BOOST_VELOCITY
                    dist_boosted = math.hypot(self.curr_x - self.boost_start_pos[0], 
                                              self.curr_y - self.boost_start_pos[1])
                    if dist_boosted > EXIT_BOOST_DIST: 
                        self.boost_active = False

        # ---------------------------------------------------------
        # [Logic 3] Smart ACC (CAV 3, 4)
        # ---------------------------------------------------------
        # if self.vid in [3, 4] and self.danger_zone_points:
        #     if self._get_min_dist(self.danger_zone_points) < ZONE_RADIUS:
        #         dist_hv = self._get_closest_hv_front()
        #         if dist_hv < 999.0:
        #             dist_error = dist_hv - ACC_DIST_LIMIT
        #             if dist_error < 0: 
        #                 target_vel_req = SLOW_VELOCITY
        #             else: 
        #                 if target_vel_req > STOP_VELOCITY:
        #                     catch_up_vel = TARGET_VELOCITY + (dist_error * ACC_P_GAIN)
        #                     target_vel_req = min(catch_up_vel, MAX_ACC_VELOCITY)
        #     else:
        #         if target_vel_req > STOP_VELOCITY:
        #             target_vel_req = TARGET_VELOCITY

        # ---------------------------------------------------------
        # [Logic 3] Smart ACC (CAV 3, 4) + Exit Boost
        # ---------------------------------------------------------
        if self.vid in [3, 4] and self.danger_zone_points:

            in_acc_zone = (self._get_min_dist(self.danger_zone_points) < ZONE_RADIUS)

            # (1) zone -> 밖으로 나가는 "엣지"에서 부스트 시작
            if (self.acc_zone_prev_in is True) and (in_acc_zone is False):
                self.acc_exit_boost_active = True
                self.acc_exit_boost_start_pos = (self.curr_x, self.curr_y)

            self.acc_zone_prev_in = in_acc_zone

            # (2) 부스트가 켜져있으면: 일정 거리까지 최고속도 유지
            if self.acc_exit_boost_active:
                target_vel_req = ACC_EXIT_BOOST_VEL
                dist_boosted = math.hypot(self.curr_x - self.acc_exit_boost_start_pos[0],
                                        self.curr_y - self.acc_exit_boost_start_pos[1])
                if dist_boosted > ACC_EXIT_BOOST_DIST:
                    self.acc_exit_boost_active = False

            # (3) 부스트가 꺼져있으면: 기존 Smart ACC 동작
            else:
                if in_acc_zone:
                    dist_hv = self._get_closest_hv_front()
                    if dist_hv < 999.0:
                        dist_error = dist_hv - ACC_DIST_LIMIT
                        if dist_error < 0:
                            target_vel_req = SLOW_VELOCITY
                        else:
                            if target_vel_req > STOP_VELOCITY:
                                catch_up_vel = TARGET_VELOCITY + (dist_error * ACC_P_GAIN)
                                target_vel_req = min(catch_up_vel, MAX_ACC_VELOCITY)
                else:
                    if target_vel_req > STOP_VELOCITY:
                        target_vel_req = TARGET_VELOCITY


        # 주행 제어 호출
        self._control_vehicle(target_vel_req)

    def _control_vehicle(self, target_vel):
        if target_vel > self.current_cmd_vel:
            self.current_cmd_vel += ACCEL_STEP
            if self.current_cmd_vel > target_vel:
                self.current_cmd_vel = target_vel
        else:
            self.current_cmd_vel = target_vel

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
        cmd.linear.x = float(self.current_cmd_vel)
        cmd.angular.z = float(steer)
        self.pub_cmd.publish(cmd)
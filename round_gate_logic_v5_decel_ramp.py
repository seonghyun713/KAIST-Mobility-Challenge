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
TARGET_VELOCITY = 1.3       # 기본 주행 속도
CRAWL_VELOCITY  = 0.15      # 서행 속도
STOP_VELOCITY   = -0.02       # 정지 속도
BOOST_VELOCITY  = 2.0       # 탈출 속도

SLOW_VELOCITY   = 0.2       
MAX_ACC_VELOCITY = 2.0      

ZONE_RADIUS     = 0.3      # 구역 반경 (38cm)
HV_DETECT_RADIUS = 0.12     # HV 감지 반경 (트리거용)

# HV 감지 히스테리시스(깜빡임 방지): True가 한번 뜨면 HOLD_TICKS 동안 True 유지
HV_HOLD_TICKS = 5   # tick=0.05s 기준 약 0.25초

# 감가속도 제한
ACCEL_STEP      = 0.3      
DECEL_STEP      = 0.1       
ACC_DIST_LIMIT  = 0.6      
ACC_P_GAIN      = 2.5       

# 리셋 거리 (다음 바퀴 준비용)
RESET_DISTANCE  = 2.2       

MIN_LA = 0.5
LA_GAIN = 0.45
     # 속도 스케일 (v * 0.45)


# CAV 01, 02 출구 가속 거리
# HV: 0.5, 1.0, 1.5 의 경우 0.48 이었음
EXIT_BOOST_DIST = 0.5
CTRL_PARAMS = {
    "look_ahead": 1.2, 
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
    def __init__(self, vehicle_id, path_file, start_zone, start_trigger, out_zone, danger_zone,pose_topic=None,pub_topic=None,hv1_topic="/HV_19",hv2_topic="/HV_20"):
        super().__init__(f"drive_node_v{vehicle_id:02d}")
        
        self.vid = vehicle_id
        self.id_str = f"{vehicle_id:02d}"
        
        # 1. 파일 로드
        self.path = load_path_from_json(path_file)
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
        
        self.gate_released = False
        
        # 상태 관리
        self.is_in_out_zone = False     
        self.boost_active = False       
        self.boost_start_pos = (0, 0)   

        # HV 상태
        self.hv19_x, self.hv19_y = 0.0, 0.0
        self.hv20_x, self.hv20_y = 0.0, 0.0
        self.hv19_active = False 
        self.hv20_active = False

        # HV 감지 hold 카운터(깜빡임 방지)
        self.hv_trigger_hold = 0
        self.hv_danger_hold = 0
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.is_connected = False
        
        # 3. 통신
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.topic_pose = f"/CAV_{self.id_str}"
        self.sub_pose = self.create_subscription(PoseStamped, self.topic_pose, self._callback_pose, qos)
        # 변경
        self.pub_cmd  = self.create_publisher(Accel, f"/CAV_{self.id_str}_accel_round_raw", 10)

        if self.start_trigger_points or self.danger_zone_points:
            self.sub_hv19 = self.create_subscription(PoseStamped, "/HV_19", self._callback_hv19, qos)
            self.sub_hv20 = self.create_subscription(PoseStamped, "/HV_20", self._callback_hv20, qos)

    def _callback_hv19(self, msg):
        self.hv19_x = msg.pose.position.x; self.hv19_y = msg.pose.position.y; self.hv19_active = True
    def _callback_hv20(self, msg):
        self.hv20_x = msg.pose.position.x; self.hv20_y = msg.pose.position.y; self.hv20_active = True

    # --- 유틸리티 ---
    def _check_hv_in_zone(self, zone_points):
        if not zone_points:
            return False

        if self.hv19_active:
            for hx, hy in zone_points:
                if math.hypot(hx - self.hv19_x, hy - self.hv19_y) < HV_DETECT_RADIUS:
                    return True

        if self.hv20_active:
            for hx, hy in zone_points:
                if math.hypot(hx - self.hv20_x, hy - self.hv20_y) < HV_DETECT_RADIUS:
                    return True

        return False


    def _check_hv_in_zone_hold(self, zone_points, hold_attr: str, hold_ticks: int = HV_HOLD_TICKS):
        """HV가 zone_points에 감지되면 hold_ticks 동안 True를 유지(깜빡임 방지)."""
        hit = self._check_hv_in_zone(zone_points)

        cur = getattr(self, hold_attr, 0)
        if hit:
            cur = int(hold_ticks)
        else:
            cur = max(0, int(cur) - 1)

        setattr(self, hold_attr, cur)
        return cur > 0

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

        # [Default] 기본은 0.6
        target_vel_req = TARGET_VELOCITY 

        # ---------------------------------------------------------
        # [Logic 1] 서행 zone
        # ---------------------------------------------------------
        GATE_X, GATE_Y = 1.7, 0.0
        GATE_SLOW_DIST = 2.1
        GATE_RESET_DIST = 2.5
        GATE_SLOW_VEL = 0.05

        dist_to_gate = math.hypot(self.curr_x - GATE_X, self.curr_y - GATE_Y)

        # 1-1 멀티 랩 리셋: 게이트에서 충분히 멀어지면 다음 바퀴 준비
        if dist_to_gate > GATE_RESET_DIST:
            self.gate_released = False
            self.hv_trigger_hold = 0

        # 기본 속도
        target_vel_req = TARGET_VELOCITY

        # 1-2 로직: 게이트 근처에서만 "눈치보기"
        if dist_to_gate < GATE_SLOW_DIST:
            if not self.gate_released:
                # 기본은 0.1로 눈치보기
                target_vel_req = GATE_SLOW_VEL

                # HV가 트리거존에 있으면 통과 허가 + 원래 속도 복귀
                if self._check_hv_in_zone_hold(self.start_trigger_points, 'hv_trigger_hold'):
                    self.gate_released = True
                    target_vel_req = TARGET_VELOCITY
            else:
                # 이미 통과 허가 난 상태면, 게이트 안에서도 속도 떨어뜨리지 않음
                pass

        # # ---------------------------------------------------------
        # # [Logic 2] Exit Conflict (CAV 1, 2)
        # # ---------------------------------------------------------
        # if self.out_zone_points and self.danger_zone_points:
        #     dist_to_out = self._get_min_dist(self.out_zone_points)
            
        #     # (A) 구역 내부
        #     if dist_to_out < ZONE_RADIUS:
        #         self.is_in_out_zone = True
        #         self.boost_active = False

        #         if self._check_hv_in_zone_hold(self.danger_zone_points, 'hv_danger_hold'):
        #             target_vel_req = STOP_VELOCITY 
        #         else:
        #             target_vel_req = CRAWL_VELOCITY 

        #     # (B) 구역 탈출
        #     else:
        #         if self.is_in_out_zone: 
        #             self.is_in_out_zone = False
        #             self.boost_active = True
        #             self.boost_start_pos = (self.curr_x, self.curr_y)

        #         if self.boost_active:
        #             target_vel_req = BOOST_VELOCITY
        #             dist_boosted = math.hypot(self.curr_x - self.boost_start_pos[0], 
        #                                       self.curr_y - self.boost_start_pos[1])
        #             if dist_boosted > EXIT_BOOST_DIST: 
        #                 self.boost_active = False

        # ---------------------------------------------------------
        # [Logic 3] Smart ACC (CAV 3, 4)
        # ---------------------------------------------------------
        if self.vid in [3, 4] and self.danger_zone_points:
            if self._get_min_dist(self.danger_zone_points) < ZONE_RADIUS:
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
                # 게이트 서행 중이면 target_vel_req를 유지
                if not (dist_to_gate < GATE_SLOW_DIST and not self.gate_released):
                    if target_vel_req > STOP_VELOCITY:
                        target_vel_req = TARGET_VELOCITY



        # 주행 제어 호출
        self._control_vehicle(target_vel_req)

    def _control_vehicle(self, target_vel):
        # 속도 램프(가속/감속 모두 단계적으로 적용)
        if target_vel > self.current_cmd_vel:
            self.current_cmd_vel += ACCEL_STEP
            if self.current_cmd_vel > target_vel:
                self.current_cmd_vel = target_vel
        elif target_vel < self.current_cmd_vel:
            self.current_cmd_vel -= DECEL_STEP
            if self.current_cmd_vel < target_vel:
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
        
        # ✅ speed-based dynamic lookahead
        active_look_ahead = max(
            MIN_LA,
            min(CTRL_PARAMS["look_ahead"], self.current_cmd_vel * LA_GAIN)
        )

        target_idx = idx
        for i in range(idx, path_len):
            if math.hypot(self.path[i][0] - self.curr_x,
                        self.path[i][1] - self.curr_y) >= active_look_ahead:
                target_idx = i
                break

        
        tx, ty = self.path[target_idx]
        desired_yaw = math.atan2(ty - self.curr_y, tx - self.curr_x)
        yaw_err = desired_yaw - self.curr_yaw
        while yaw_err > math.pi: yaw_err -= 2 * math.pi
        while yaw_err < -math.pi: yaw_err += 2 * math.pi

        dt = 0.1
        self.integral_error = max(-2.0, min(2.0, self.integral_error + yaw_err * dt))
        p = CTRL_PARAMS["kp"] * yaw_err
        i = CTRL_PARAMS["ki"] * self.integral_error
        d = CTRL_PARAMS["kd"] * (yaw_err - self.prev_error) / dt
        cte = min_dist * CTRL_PARAMS["k_cte"] * (-1.0 if yaw_err < 0 else 1.0)
        
        steer = max(-2.0, min(2.0, float(p + i + d + cte)))
        self.prev_error = yaw_err

        cmd = Accel()
        cmd.linear.x = float(self.current_cmd_vel)
        cmd.angular.z = float(steer)
        self.pub_cmd.publish(cmd)

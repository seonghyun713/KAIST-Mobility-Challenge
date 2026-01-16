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
TARGET_VELOCITY = 0.7       # 목표 주행 속도 (가속 시)
SLOW_VELOCITY   = 0.3       # ACC 감속 속도 (거리 < 40cm)
STOP_VELOCITY   = 0.01      # 정지 속도
ZONE_RADIUS     = 0.30      # 구역 감지 반경
HV_DETECT_RADIUS = 0.10     # HV 경로 감지 반경

# 가속도 제한 (Soft Start)
ACCEL_STEP      = 0.04      

# ACC 거리 기준 (40cm)
ACC_DIST_LIMIT  = 0.40

# 리셋 거리
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
        self.start_zone_points = load_zone_from_csv(start_zone)
        self.start_trigger_points = load_zone_from_csv(start_trigger)
        self.out_zone_points = load_zone_from_csv(out_zone)
        # danger_zone은 CAV 3,4에게는 'ACC 감지 구간(HV경로)' 역할을 함
        self.danger_zone_points = load_zone_from_csv(danger_zone)

        # 로그
        if self.vid in [3, 4] and self.danger_zone_points:
            print(f"[INFO] V{self.id_str} ACC Logic (Y-Check) Activated")
        elif self.out_zone_points:
            print(f"[INFO] V{self.id_str} Exit Conflict Logic Activated")

        # 2. 상태 변수
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0
        
        # Soft Start용 현재 명령 속도
        self.current_cmd_vel = 0.0 
        
        self.stop_logic_disabled = False 
        
        # HV 상태
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
        
        if self.start_trigger_points or self.danger_zone_points:
            self.sub_hv19 = self.create_subscription(PoseStamped, "/HV_19", self._callback_hv19, qos)
            self.sub_hv20 = self.create_subscription(PoseStamped, "/HV_20", self._callback_hv20, qos)

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
        """ 
        내 앞(Y값이 더 작은 쪽)에 있는 가장 가까운 HV 거리 반환
        조건: HV_Y < My_Y
        """
        closest_dist = 999.0
        
        # HV 19 체크 (Y좌표 비교)
        if self.hv19_active and self.hv19_y < self.curr_y:
            d = math.hypot(self.hv19_x - self.curr_x, self.hv19_y - self.curr_y)
            if d < closest_dist: closest_dist = d
            
        # HV 20 체크 (Y좌표 비교)
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

        # [Default] 기본은 0.7 (구역 밖이거나 특별한 일 없으면)
        target_vel_req = TARGET_VELOCITY 

        # ---------------------------------------------------------
        # [Logic 1] Start Zone (공통)
        # ---------------------------------------------------------
        dist_to_start = self._get_min_dist(self.start_zone_points)

        if self.stop_logic_disabled:
            if dist_to_start > RESET_DISTANCE:
                self.stop_logic_disabled = False
        else:
            if dist_to_start < ZONE_RADIUS:
                target_vel_req = STOP_VELOCITY 
                if self._check_hv_in_zone(self.start_trigger_points):
                    target_vel_req = TARGET_VELOCITY
                    self.stop_logic_disabled = True
                    print(f"[START] V{self.id_str} Soft Start...", end='\r')
                else:
                    print(f"[WAIT] V{self.id_str} Waiting...", end='\r')

        # ---------------------------------------------------------
        # [Logic 2] Exit Conflict (CAV 1, 2)
        # ---------------------------------------------------------
        if self.vid in [1, 2] and self.out_zone_points and self.danger_zone_points:
            if self._get_min_dist(self.out_zone_points) < ZONE_RADIUS:
                if self._check_hv_in_zone(self.danger_zone_points):
                    target_vel_req = STOP_VELOCITY 
                    print(f"[YIELD] V{self.id_str} Stop (Yield)", end='\r')

        # ---------------------------------------------------------
        # [Logic 3] ACC (CAV 3, 4) - 요청하신 기능
        # 조건: 내가 path_hv 안에 있고 + HV가 내 앞에 있을 때
        # ---------------------------------------------------------
        if self.vid in [3, 4] and self.danger_zone_points:
            # 1. 내가 HV 경로 구역 안에 있는가?
            if self._get_min_dist(self.danger_zone_points) < ZONE_RADIUS:
                
                # 2. 내 앞(Y가 작은 쪽)에 HV가 있는가?
                dist_hv = self._get_closest_hv_front()
                
                if dist_hv < 999.0: # 앞에 HV가 있음
                    if dist_hv < ACC_DIST_LIMIT: # < 40cm
                        target_vel_req = SLOW_VELOCITY # 감속
                        print(f"[ACC] Too Close ({dist_hv:.2f}m) -> Slow", end='\r')
                    else: # > 40cm
                        # 정지 상태가 아니라면(Logic 1이 잡고 있지 않다면) 가속
                        if target_vel_req > STOP_VELOCITY:
                            target_vel_req = TARGET_VELOCITY # 가속(원래 속도)
                            print(f"[ACC] Clear ({dist_hv:.2f}m) -> Accel", end='\r')
                else:
                    # 구역 안이지만 앞에 차가 없음 -> 원래 속도
                    pass
            else:
                # 3. 구역을 빠져나옴 -> 원래 속도 0.7 (Logic 1이 잡지 않는 한)
                if target_vel_req > STOP_VELOCITY:
                    target_vel_req = TARGET_VELOCITY

        # 주행 제어 호출
        self._control_vehicle(target_vel_req)

    def _control_vehicle(self, target_vel):
        # Soft Start (가속 제한)
        if target_vel > self.current_cmd_vel:
            self.current_cmd_vel += ACCEL_STEP
            if self.current_cmd_vel > target_vel:
                self.current_cmd_vel = target_vel
        else:
            self.current_cmd_vel = target_vel

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
        cmd.linear.x = float(self.current_cmd_vel)
        cmd.angular.z = float(steer)
        self.pub_cmd.publish(cmd)

# ============================================================
# [메인 실행]
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    
    # (ID, Path, Start_Zone, Start_Trigger, Out_Zone, Danger_Zone)
    cars_config = [
        (1, 'path3_1.json', 'path3_1_zone.csv', 'path_hv_3_1.csv', 'path3_1_out_zone.csv', 'path_hv_3_2.csv'), 
        (2, 'path3_2.json', 'path3_2_zone.csv', 'path_hv_2_1.csv', 'path3_2_out_zone.csv', 'path_hv_2_2.csv'), 
        (3, 'path3_3.json', 'path3_3_zone.csv', 'path_hv_2_1.csv', None, 'path_hv_2_1.csv'), 
        (4, 'path3_4.json', 'path3_4_zone.csv', 'path_hv_3_1.csv', None, 'path_hv_3_1.csv'), 
    ]

    executor = MultiThreadedExecutor()
    nodes = []

    print("=== FINAL SYSTEM: Complete Safety & ACC Logic ===")
    
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

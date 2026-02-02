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
TARGET_VELOCITY = 0.6       # 기본 주행 속도
CRAWL_VELOCITY  = 0.15      # 서행 속도
STOP_VELOCITY   = -0.0001       # 정지 속도
BOOST_VELOCITY  = 1.5       # 탈출 속도

SLOW_VELOCITY   = 0.15
MAX_ACC_VELOCITY = 1.5

ZONE_RADIUS     = 0.15      # 구역 반경
HV_DETECT_RADIUS = 0.12     # HV 감지 반경 (트리거용)

# ============================================================
# [정밀 정지 튜닝] StartZone 정지 보정 + 감속 램프
# ============================================================
STOP_LEAD_DIST = 0.30       # STOP 앞당기기
STOP_RAMP_DIST = 0.80       # 감속 램프 길이
STOP_MIN_VEL   = 0.01       # 정지 직전 최소 크리핑 속도

# ============================================================
# [정지 대기 중 "잡아두기(브레이크 흉내)" 설정 + 안전장치]
# ============================================================
HOLD_BRAKE_VEL = -0.03          # HV 트리거 대기 중 계속 보낼 속도(약한 역토크)
HOLD_MAX_MOVE_DIST = 0.5       # 안전장치: 대기 시작점 대비 50cm 이상 움직이면 0으로 풀기

# 출발 시 안전 거리
START_SAFETY_DIST = 0.5

# 가속/감속 스텝 제한 (명령 램프)
ACCEL_STEP      = 0.05
DECEL_STEP      = 0.15

# ACC 거리 유지 파라미터 (주행 중)
ACC_DIST_LIMIT  = 0.5
ACC_P_GAIN      = 2.5

# 리셋 거리 (다음 바퀴 준비용)
RESET_DISTANCE  = 2.2

# CAV 01, 02 출구 가속 거리
EXIT_BOOST_DIST = 0.48

CTRL_PARAMS = {
    "look_ahead": 0.4,
    "kp": 5.0,
    "ki": 0.05,
    "kd": 1.0,
    "k_cte": 4.0
}

# ============================================================
# [파일 로드 함수]
# ============================================================
def load_path_from_json(filename):
    if not filename or not os.path.exists(filename):
        return []
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        xs = data.get("x") or data.get("X")
        ys = data.get("y") or data.get("Y")
        if not xs or not ys:
            return []
        return [(float(x), float(y)) for x, y in zip(xs, ys)]
    except:
        return []

def load_zone_from_csv(filename):
    points = []
    if not filename or not os.path.exists(filename):
        return []
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and len(row) >= 2:
                    try:
                        points.append((float(row[0]), float(row[1])))
                    except:
                        continue
    except:
        pass
    return points

# ============================================================
# [차량 제어 클래스]
# ============================================================
class VehicleController(Node):
    def __init__(
        self,
        vehicle_id,
        path_file,
        start_zone,
        start_trigger,
        out_zone=None,
        danger_zone=None,
        pose_topic=None,
        pub_topic=None,
        hv1_topic="/HV_19",
        hv2_topic="/HV_20",
    ):

        super().__init__(f"drive_node_v{vehicle_id:02d}")

        
        self.vid = vehicle_id
        self.id_str = f"{vehicle_id:02d}"

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

        # ★ 출발 직후 gap 체크
        self.is_start_gap_check_active = False

        # ★ 대기 중 hold(-0.01) 안전장치용 (StartZone)
        self.wait_hold_active = False
        self.wait_hold_start_pos = (0.0, 0.0)

        # ★ out zone에서 HV 때문에 멈춰 있을 때 hold(-0.01) 안전장치용
        self.out_hold_active = False
        self.out_hold_start_pos = (0.0, 0.0)

        # 상태 관리
        self.is_in_out_zone = False
        self.boost_active = False
        self.boost_start_pos = (0.0, 0.0)

        # HV 상태
        self.hv19_x, self.hv19_y = 0.0, 0.0
        self.hv20_x, self.hv20_y = 0.0, 0.0
        self.hv19_active = False
        self.hv20_active = False

        self.prev_error = 0.0
        self.integral_error = 0.0
        self.is_connected = False

        # 3. 통신
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.topic_pose = pose_topic if pose_topic else f"/CAV_{self.id_str}"
        self.sub_pose = self.create_subscription(
            PoseStamped,
            self.topic_pose,
            self._callback_pose,
            qos
        )

        pub_t = pub_topic if pub_topic else f"/CAV_{self.id_str}_accel_round_raw"
        self.pub_cmd = self.create_publisher(Accel, pub_t, 10)


        if self.start_trigger_points or self.danger_zone_points:
            self.sub_hv19 = self.create_subscription(
                PoseStamped, hv1_topic, self._callback_hv19, qos
            )
            self.sub_hv20 = self.create_subscription(
                PoseStamped, hv2_topic, self._callback_hv20, qos
            )
        
        # =========================
        # [LOG] HV in/out 상태 추적용
        # =========================
        self.prev_hv_in_start_trigger = False
        self.prev_hv_in_danger_zone = False

        self.last_hv_log_t = 0.0      # 주기 로그용(너무 많이 찍히는 것 방지)
        self.hv_log_period = 0.5      # 0.2s마다 한 번 (원하면 0.5s 등으로)


    def _hv_in_zone_detail(self, zone_points):
        """
        return:
          (in_zone: bool, hv_id: str or None, hv_x: float, hv_y: float, min_dist: float)
        """
        if not zone_points:
            return (False, None, None, None, 999.0)

        best = (999.0, None, None, None)  # (dist, hv_id, x, y)

        if self.hv19_active:
            for zx, zy in zone_points:
                d = math.hypot(zx - self.hv19_x, zy - self.hv19_y)
                if d < best[0]:
                    best = (d, "HV19", self.hv19_x, self.hv19_y)

        if self.hv20_active:
            for zx, zy in zone_points:
                d = math.hypot(zx - self.hv20_x, zy - self.hv20_y)
                if d < best[0]:
                    best = (d, "HV20", self.hv20_x, self.hv20_y)

        min_d, hv_id, hx, hy = best
        in_zone = (min_d < HV_DETECT_RADIUS)
        return (in_zone, hv_id, hx, hy, min_d)



    def _callback_hv19(self, msg):
        self.hv19_x = msg.pose.position.x
        self.hv19_y = msg.pose.position.y
        self.hv19_active = True

    def _callback_hv20(self, msg):
        self.hv20_x = msg.pose.position.x
        self.hv20_y = msg.pose.position.y
        self.hv20_active = True

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

    def _get_min_dist(self, zone_points):
        min_d = 999.0
        if zone_points:
            for zx, zy in zone_points:
                d = math.hypot(zx - self.curr_x, zy - self.curr_y)
                if d < min_d:
                    min_d = d
        return min_d

    def _get_closest_hv_front(self):
        closest_dist = 999.0
        if self.hv19_active and self.hv19_y < self.curr_y:
            d = math.hypot(self.hv19_x - self.curr_x, self.hv19_y - self.curr_y)
            if d < closest_dist:
                closest_dist = d
        if self.hv20_active and self.hv20_y < self.curr_y:
            d = math.hypot(self.hv20_x - self.curr_x, self.hv20_y - self.curr_y)
            if d < closest_dist:
                closest_dist = d
        return closest_dist

    # --- 메인 루프 ---
    def _callback_pose(self, msg):
        if not self.is_connected:
            self.is_connected = True
            print(f"[LINK] V{self.id_str} Connected!")

        self.curr_x = msg.pose.position.x
        self.curr_y = msg.pose.position.y
        self.curr_yaw = msg.pose.orientation.z

        if not self.path:
            return

        # [Default]
        target_vel_req = TARGET_VELOCITY

        # ---------------------------------------------------------
        # [Logic 1] Start Zone (정지 + 39cm gap 유지) + 정밀 정지(리드 + 램프)
        # ---------------------------------------------------------
        dist_to_start = self._get_min_dist(self.start_zone_points)

        # 1-1. 멀티 랩 리셋
        if dist_to_start > RESET_DISTANCE:
            self.stop_logic_disabled = False
            self.is_start_gap_check_active = False
            self.wait_hold_active = False
            self.out_hold_active = False

        # 1-2. StartZone 접근/정지 제어 (stop_logic_disabled가 False일 때만)
        if not self.stop_logic_disabled:
            stop_zero_dist  = ZONE_RADIUS + STOP_LEAD_DIST
            stop_ramp_start = stop_zero_dist + STOP_RAMP_DIST

            if dist_to_start < stop_zero_dist:
                # ★ StartZone 안(정지 구간) : HV 트리거 오기 전까지 -0.01로 잡아두기(+안전장치)
                if self._check_hv_in_zone(self.start_trigger_points):
                    # 트리거 들어오면 출발 허가
                    self.stop_logic_disabled = True
                    self.is_start_gap_check_active = True
                    self.wait_hold_active = False
                    target_vel_req = TARGET_VELOCITY
                    print(f"[START] V{self.id_str} Triggered! Checking Gap...", end='\r')
                else:
                    # 트리거 없음: hold 시작점 저장(최초 1회)
                    if not self.wait_hold_active:
                        self.wait_hold_active = True
                        self.wait_hold_start_pos = (self.curr_x, self.curr_y)

                    # 안전장치: 대기 시작점 대비 일정 거리 이상 움직이면 0으로 풀어버림
                    move_dist = math.hypot(self.curr_x - self.wait_hold_start_pos[0],
                                           self.curr_y - self.wait_hold_start_pos[1])
                    if move_dist > HOLD_MAX_MOVE_DIST:
                        target_vel_req = STOP_VELOCITY
                        print(f"[WAIT] V{self.id_str} HOLD LIMIT (>{HOLD_MAX_MOVE_DIST*100:.0f}cm) -> 0 ", end='\r')
                    else:
                        target_vel_req = HOLD_BRAKE_VEL
                        print(f"[WAIT] V{self.id_str} Holding with {HOLD_BRAKE_VEL}... ", end='\r')

            elif dist_to_start < stop_ramp_start:
                # (B) 감속 램프 구간: 거리 기반으로 TARGET -> STOP_MIN_VEL로 서서히 감소
                ratio = (dist_to_start - stop_zero_dist) / STOP_RAMP_DIST
                ratio = max(0.0, min(1.0, ratio))
                ratio_smooth = ratio**2  # 곡선형 램프(부드럽게)
                target_vel_req = STOP_MIN_VEL + (TARGET_VELOCITY - STOP_MIN_VEL) * ratio_smooth

            else:
                # (A) 멀리 있으면 정상 주행
                target_vel_req = TARGET_VELOCITY

        # 1-3. 출발 직후 39cm 거리 유지 로직
        if self.is_start_gap_check_active:
            if dist_to_start > ZONE_RADIUS:
                self.is_start_gap_check_active = False
                print(f"[GAP] Zone Exited (>ZONE_RADIUS). Gap Logic OFF    ", end='\r')
            else:
                dist_hv = self._get_closest_hv_front()
                if dist_hv < START_SAFETY_DIST:
                    target_vel_req = STOP_VELOCITY
                    print(f"[GAP] Too Close ({dist_hv:.2f}m < {START_SAFETY_DIST:.2f}m) WAIT  ", end='\r')
                else:
                    pass

        # ---------------------------------------------------------
        # [Logic 2] Exit Conflict (CAV 1, 2)  (※ HV 있을 때 -0.01 hold 추가)
        # ---------------------------------------------------------
        if self.out_zone_points and self.danger_zone_points:
            dist_to_out = self._get_min_dist(self.out_zone_points)

            # (A) 구역 내부
            if dist_to_out < ZONE_RADIUS:
                self.is_in_out_zone = True
                self.boost_active = False

                if self._check_hv_in_zone(self.danger_zone_points):
                    # HV가 위험구역에 있는 동안: -0.01로 계속 홀드(+안전장치)
                    if not self.out_hold_active:
                        self.out_hold_active = True
                        self.out_hold_start_pos = (self.curr_x, self.curr_y)

                    move_dist = math.hypot(
                        self.curr_x - self.out_hold_start_pos[0],
                        self.curr_y - self.out_hold_start_pos[1]
                    )

                    if move_dist > HOLD_MAX_MOVE_DIST:
                        target_vel_req = STOP_VELOCITY
                        print(f"[OUT] V{self.id_str} HOLD LIMIT (>{HOLD_MAX_MOVE_DIST*100:.0f}cm) -> 0 ", end='\r')
                    else:
                        target_vel_req = HOLD_BRAKE_VEL
                        print(f"[OUT] V{self.id_str} Holding with {HOLD_BRAKE_VEL}... ", end='\r')
                else:
                    # HV가 빠져나갔으면: 홀드 상태 해제하고 서행
                    self.out_hold_active = False
                    target_vel_req = CRAWL_VELOCITY

            # (B) 구역 탈출
            else:
                if self.is_in_out_zone:
                    self.is_in_out_zone = False
                    self.boost_active = True
                    self.boost_start_pos = (self.curr_x, self.curr_y)
                    self.out_hold_active = False  # out-zone hold 해제

                if self.boost_active:
                    target_vel_req = BOOST_VELOCITY
                    dist_boosted = math.hypot(
                        self.curr_x - self.boost_start_pos[0],
                        self.curr_y - self.boost_start_pos[1]
                    )
                    if dist_boosted > EXIT_BOOST_DIST:
                        self.boost_active = False

        # ---------------------------------------------------------
        # [Logic 3] Smart ACC (CAV 3, 4)  (※ Exit Boost 제거 버전)
        # ---------------------------------------------------------
        if self.vid in [3, 4] and self.danger_zone_points:
            in_acc_zone = (self._get_min_dist(self.danger_zone_points) < ZONE_RADIUS)

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

        # ==========================================================
        # [LOG] HV pose & Zone in/out 이벤트 로그 (제어 publish 전/후 상관없지만, 여기서는 publish 전 추천)
        # ==========================================================
        now_t = self.get_clock().now().nanoseconds * 1e-9

        # 1) Start Trigger Zone in/out
        hv_in_trig, hv_id_trig, hx_trig, hy_trig, d_trig = self._hv_in_zone_detail(self.start_trigger_points)
        if hv_in_trig != self.prev_hv_in_start_trigger:
            state = "IN" if hv_in_trig else "OUT"
            print(f"[HV-{state}] V{self.id_str} StartTrigger {hv_id_trig} pose=({hx_trig:.2f},{hy_trig:.2f}) min_d={d_trig:.3f}")
            self.prev_hv_in_start_trigger = hv_in_trig

        # 2) Danger Zone in/out
        hv_in_danger, hv_id_d, hx_d, hy_d, d_d = self._hv_in_zone_detail(self.danger_zone_points)
        if hv_in_danger != self.prev_hv_in_danger_zone:
            state = "IN" if hv_in_danger else "OUT"
            print(f"[HV-{state}] V{self.id_str} DangerZone {hv_id_d} pose=({hx_d:.2f},{hy_d:.2f}) min_d={d_d:.3f}")
            self.prev_hv_in_danger_zone = hv_in_danger

        # 3) 주기 로그 (너무 많으면 주행에 영향 → 0.5~1.0초 추천)
        if now_t - self.last_hv_log_t > self.hv_log_period:
            self.last_hv_log_t = now_t
            s = f"[HVPOSE] V{self.id_str} "
            if self.hv19_active:
                s += f"HV19=({self.hv19_x:.2f},{self.hv19_y:.2f}) "
            if self.hv20_active:
                s += f"HV20=({self.hv20_x:.2f},{self.hv20_y:.2f}) "
            print(s)

        # 마지막에 publish 호출
        self._control_vehicle(target_vel_req)


    def _control_vehicle(self, target_vel):
        # 속도 명령을 "가속/감속 모두" 램프로 제한
        if target_vel > self.current_cmd_vel:
            self.current_cmd_vel += ACCEL_STEP
            if self.current_cmd_vel > target_vel:
                self.current_cmd_vel = target_vel
        else:
            self.current_cmd_vel -= DECEL_STEP
            if self.current_cmd_vel < target_vel:
                self.current_cmd_vel = target_vel

        # -0.01 홀드를 허용해야 하므로 0으로 클램프하면 안 됨
        # 너무 큰 음수만 제한
        if self.current_cmd_vel < -0.05:
            self.current_cmd_vel = -0.05

        # 경로 추종 (look-ahead + PID-ish)
        min_dist = 1e9
        idx = 0
        path_len = len(self.path)

        for i in range(path_len):
            px, py = self.path[i]
            d = math.hypot(px - self.curr_x, py - self.curr_y)
            if d < min_dist:
                min_dist = d
                idx = i

        target_idx = idx
        for i in range(idx, path_len):
            if math.hypot(self.path[i][0] - self.curr_x, self.path[i][1] - self.curr_y) >= CTRL_PARAMS["look_ahead"]:
                target_idx = i
                break

        tx, ty = self.path[target_idx]
        desired_yaw = math.atan2(ty - self.curr_y, tx - self.curr_x)

        yaw_err = desired_yaw - self.curr_yaw
        while yaw_err > math.pi:
            yaw_err -= 2 * math.pi
        while yaw_err < -math.pi:
            yaw_err += 2 * math.pi

        dt = 0.1
        self.integral_error = max(-1.0, min(1.0, self.integral_error + yaw_err * dt))
        p = CTRL_PARAMS["kp"] * yaw_err
        i = CTRL_PARAMS["ki"] * self.integral_error
        d = CTRL_PARAMS["kd"] * (yaw_err - self.prev_error) / dt
        cte = min_dist * CTRL_PARAMS["k_cte"] * (-1.0 if yaw_err < 0 else 1.0)

        steer = max(-1.5, min(1.5, float(p + i + d + cte)))
        self.prev_error = yaw_err

        cmd = Accel()
        cmd.linear.x = float(self.current_cmd_vel)
        cmd.angular.z = float(steer)
        self.pub_cmd.publish(cmd)

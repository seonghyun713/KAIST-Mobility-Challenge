#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Accel, PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import json
import math
import time
import numpy as np
import os

class MergeCorridorController(Node):
    # Tuning Constants
    MERGE_ZONE = {'x_min': 3.50, 'x_max': 5.31, 'y_min': -2.66, 'y_max': 0.77}
    CUSTOM_ZONE = {'x_min': 4.0, 'x_max': 6.0, 'y_min': -1.35, 'y_max': 0.77}
    
    # Vehicle & Control
    CAR_LENGTH = 0.21
    MAX_SPEED_STRAIGHT = 2.0
    MAX_SPEED_CORNER = 1.05
    LANE_CHANGE_COOLDOWN = 0.8
    
    # Lookahead & PID
    MIN_LOOKAHEAD = 0.48
    MAX_LOOKAHEAD = 0.85
    PID_KP = 1.5
    PID_KI = 0.05
    CORNER_THRESHOLD = 0.15

    def __init__(self):
        super().__init__('Task_2_legend_ace_controller')
        self.get_logger().info("🏎️ Task2 Controller")

        # --- QoS & Communication ---
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        self.pose_sub = self.create_subscription(PoseStamped, '/CAV_01', self.pose_callback, qos)
        self.control_pub = self.create_publisher(Accel, '/CAV_01_accel', 10)
        
        self.hv_data = {}
        for i in range(19, 37):
            self.create_subscription(PoseStamped, f'/HV_{i}', lambda m, id=i: self.hv_callback(m, id), qos)

        # --- Path Loading ---
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PATH_DIR = os.path.join(BASE_DIR, "path")

        self.paths = {
            1: self.load_path(os.path.join(PATH_DIR, "lane1.json")),
            2: self.load_path(os.path.join(PATH_DIR, "lane2.json")),
            3: self.load_path(os.path.join(PATH_DIR, "lane3.json")),
        }

        # Convert paths to numpy arrays for fast access
        self.path_arrs = {k: np.column_stack((v['X'], v['Y'])) if v['X'] else np.empty((0, 2)) for k, v in self.paths.items()}

        # --- HV Configuration ---
        self.LANE_HV_MAP = {
            1: list(range(19, 23)),  # Fast Lane
            2: list(range(23, 31)),  # Slow Lane
            3: list(range(31, 37))   # Static Lane
        }

        # --- State Variables ---
        self.curr_lane = 2
        self.last_idx = None
        self.x = 0.0; self.y = 0.0; self.yaw = 0.0; self.spd = 0.0
        
        self.prev_x = 0.0; self.prev_y = 0.0; self.prev_time = time.time()
        self.spd_err_sum = 0.0
        self.prev_accel = 0.0
        self.smoothed_v = 0.0
        
        self.pose_cnt = 0
        self.is_ready = False
        self.last_lc_time = 0.0
        self.start_time = None

        self.create_timer(0.02, self.control_loop)

    def load_path(self, filename):
        if not os.path.exists(filename): return {'X': [], 'Y': []}
        try:
            with open(filename, 'r') as f:
                d = json.load(f)
                return {'X': d.get('X') or d.get('x', []), 'Y': d.get('Y') or d.get('y', [])}
        except: return {'X': [], 'Y': []}

    def set_path(self, lane_id):
        if self.path_arrs[lane_id].shape[0] == 0: return
        self.curr_lane = lane_id
        self.last_idx = None
        self.spd_err_sum = 0.0 # Reset Integral Term on LC
        self.last_lc_time = time.time()

    def hv_callback(self, msg, hv_id):
        """ Updates HV position and estimates velocity using a low-pass filter. """
        cx, cy = msg.pose.position.x, msg.pose.position.y
        now = time.time()
        fv = 0.0
        if hv_id in self.hv_data:
            prev = self.hv_data[hv_id]
            dt = now - prev['t']
            if dt > 0.0:
                v = math.hypot(cx - prev['x'], cy - prev['y']) / dt
                fv = 0.4 * prev['v'] + 0.6 * v if v < 10.0 else prev['v']
        self.hv_data[hv_id] = {'x': cx, 'y': cy, 'v': fv, 't': now}

    def pose_callback(self, msg):
        """ Updates Ego state (Position, Yaw, Speed). """
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.yaw = msg.pose.orientation.z
        
        now = time.time()
        dt = now - self.prev_time
        if dt > 0:
            dist = math.hypot(self.x - self.prev_x, self.y - self.prev_y)
            self.spd = dist / dt if dist > 0.005 else 0.0
        
        self.prev_x, self.prev_y, self.prev_time = self.x, self.y, now
        self.pose_cnt += 1
        
        if not self.is_ready and self.pose_cnt > 3:
            self.get_logger().info("✅ Controller Ready")
            self.is_ready = True
            if self.start_time is None: self.start_time = time.time()

    def get_nearest_index(self):
        arr = self.path_arrs[self.curr_lane]
        if len(arr) == 0: return 0
        if self.last_idx is None:
            self.last_idx = np.argmin(np.hypot(arr[:,0] - self.x, arr[:,1] - self.y))
            return self.last_idx
        
        # Local search around last index for efficiency
        s = self.last_idx
        indices = np.arange(s, s + 100) % len(arr)
        check = arr[indices]
        self.last_idx = indices[np.argmin(np.hypot(check[:,0] - self.x, check[:,1] - self.y))]
        return self.last_idx

    # --- Logic: Zone & Safety Check ---
    def is_in_merge_zone(self):
        return (self.MERGE_ZONE['x_min'] <= self.x <= self.MERGE_ZONE['x_max'] and 
                self.MERGE_ZONE['y_min'] <= self.y <= self.MERGE_ZONE['y_max'])
    
    def is_in_custom_zone(self):
        return (self.CUSTOM_ZONE['x_min'] <= self.x <= self.CUSTOM_ZONE['x_max'] and 
                self.CUSTOM_ZONE['y_min'] <= self.y <= self.CUSTOM_ZONE['y_max'])

    def check_side_safety(self, target_lane):
        """
        [Gatekeeper] Blocks LC if ANY car is physically overlapping.
        Zone: Rear 0.5m ~ Front 0.5m | Side 0.3m (Blind Spot)
        """
        ids = self.LANE_HV_MAP.get(target_lane, [])
        for hv_id in ids:
            if hv_id not in self.hv_data: continue
            d = self.hv_data[hv_id]
            
            # Transform World -> Ego-Local
            dx, dy = d['x'] - self.x, d['y'] - self.y
            lx = math.cos(-self.yaw) * dx - math.sin(-self.yaw) * dy
            ly = math.sin(-self.yaw) * dx + math.cos(-self.yaw) * dy
            
            if (-0.5 < lx < 0.5) and (abs(ly) < 0.30):
                return False # Blocked
        return True # Safe

    def get_lane_info(self, target_lane):
        """ Calculates nearest Front/Rear gaps & speeds. Includes special merge logic. """
        min_f, f_v = 999.0, 0.0
        min_r, r_v = 999.0, 0.0
        
        arr = self.path_arrs[target_lane]
        if len(arr) == 0: return (999.0, 0.0, 999.0, 0.0)

        # Path vector for dot product
        dists = np.hypot(arr[:,0] - self.x, arr[:,1] - self.y)
        idx = np.argmin(dists)
        n_idx = (idx + 5) % len(arr)
        vx, vy = arr[n_idx, 0] - arr[idx, 0], arr[n_idx, 1] - arr[idx, 1]

        ids = list(self.LANE_HV_MAP.get(target_lane, []))
        
        # Extended Check: If I am in Lane 1 Merge Zone, check Lane 2 as well
        if target_lane == 1 and self.is_in_merge_zone():
            ids.extend(self.LANE_HV_MAP[2])

        for hv_id in ids:
            if hv_id not in self.hv_data: continue
            d = self.hv_data[hv_id]
            
            dx, dy = d['x'] - self.x, d['y'] - self.y
            raw_dist = math.hypot(dx, dy)
            if raw_dist > 6.0: continue # Optimization

            dot = dx * vx + dy * vy
            gap = raw_dist - self.CAR_LENGTH

            # Front Car
            if dot > 0:
                if gap < min_f:
                    min_f = gap
                    f_v = d['v']
            # Rear Car
            else:
                # [Merge Protection] If Lane 2 car is dangerously close, spike the cost
                check_merge_rear = self.is_in_merge_zone() and (target_lane == 2)
                if check_merge_rear and gap < 0.5:
                    if gap < min_f: # Treat as a 'Wall' to deter entry
                        min_f = 0.01 
                        f_v = d['v']
                    continue

                # Ignore rear cars if I am significantly faster and safe
                am_i_faster = self.spd > (d['v'] - 0.2)
                if am_i_faster and gap > 0.5: continue

                if gap < min_r:
                    min_r = gap
                    r_v = d['v']

        return min_f, f_v, min_r, r_v

    # --- Cost Function ---
    def calculate_cost(self, lane_id):
        cost = 0.0
        
        # 1. Critical: Prohibit Lane 1 in Merge Zone
        if self.is_in_merge_zone() and lane_id == 1:
            return 999999.0
        
        # 2. Avoid Lane 3 (Static Obstacles)
        if lane_id == 3: cost += 3.0

        fg, fv, rg, rv = self.get_lane_info(lane_id)

        # 3. Safety: Time-To-Collision (TTC) Logic
        if rg < 1.2:
            app_spd = rv - self.spd
            if app_spd > 0:
                ttc = rg / app_spd
                cost += 100000.0 if ttc < 0.5 else (1.0/ttc) * 2000.0
            elif rg < 0.3: cost += 5000.0

        # 4. Zone-Specific Cost Logic
        # [Zone A] Custom/Merge Zone: Aggressive & Agile behavior
        if self.is_in_custom_zone():
            # Short detection range (1.0m) for complex scenarios
            if fg < 1.0:
                # (A) Static Obstacles/Vehicles
                if fv < 0.5:
                    if fg < 0.6: 
                        cost += (0.6 - fg) * 1000.0  # High cost for immediate avoidance
                    else: 
                        cost += 200.0                # Soft pressure
                
                # (B) Moving Vehicles
                else:
                    if fg < 0.5: 
                        cost += 1000.0 # Collision risk
                    
                    # Penalize speed difference (Aggressive overtaking logic)
                    diff = self.MAX_SPEED_STRAIGHT - fv
                    if diff > 0:
                        cost += (diff ** 2) * 200.0
                        # Incentive to leave current lane if blocked
                        if lane_id == self.curr_lane and diff > 0.8:
                            cost += 200.0 
                    
                    # Proximity penalty
                    cost += (1.0 / max(fg, 0.1)) * 20.0

            # (C) Empty Lane Bonus
            else:
                cost -= 30.0 

        # [Zone B] Normal Area: Conservative & Smooth behavior
        else:
            # Long detection range (5.0m)
            if fg < 5.0:
                if fg < 0.3: cost += 5000.0 # Safety limit violation
                
                # Linear speed matching cost (Smoother)
                diff = self.MAX_SPEED_STRAIGHT - fv
                if diff > 0: cost += diff * 100.0
                
                # Handling static objects (mainly for Lane 3)
                if lane_id == 3 and fv < 0.1:
                    cost += (2.0 - fg) * 1000.0 if fg < 2.0 else 0
                else:
                    cost += (1.0 / max(fg, 0.1)) * 40.0
        
        # 5. Stability: Hysteresis cost to prevent rapid lane switching
        if lane_id != self.curr_lane: cost += 5.0
        
        return cost

    def check_curvature(self):
        arr = self.path_arrs[self.curr_lane]
        if self.last_idx is None: return 0.0
        
        f_idx = (self.last_idx + 40) % len(arr)
        scan = f_idx
        ld = 0.5
        
        for _ in range(10):
            scan = (scan + 1) % len(arr)
            dx, dy = arr[scan, 0] - arr[f_idx, 0], arr[scan, 1] - arr[f_idx, 1]
            if math.hypot(dx, dy) > ld: break
            
        vx, vy = arr[(f_idx+1)%len(arr), 0] - arr[f_idx, 0], arr[(f_idx+1)%len(arr), 1] - arr[f_idx, 1]
        fyaw = math.atan2(vy, vx)
        
        dx, dy = arr[scan, 0] - arr[f_idx, 0], arr[scan, 1] - arr[f_idx, 1]
        ly = math.sin(-fyaw) * dx + math.cos(-fyaw) * dy
        return abs(2.0 * ly / (ld ** 2))

    # --- Yield Logic ---
    def should_yield(self):
        # Only yield if in merge zone, stuck in Lane 1, and Lane 2 is blocked
        if not self.is_in_merge_zone(): return False
        if self.curr_lane == 2: return False
        
        # If unsafe to change to Lane 2 -> Yield/Stop
        return not self.check_side_safety(2)

    # --- Main Decision ---
    def decision_logic(self, cur_k):
        if abs(cur_k) > self.CORNER_THRESHOLD: return
        if self.check_curvature() > self.CORNER_THRESHOLD: return

        costs = {}
        # Search relevant lanes only
        search = [self.curr_lane]
        if self.curr_lane == 1: search.append(2)
        elif self.curr_lane == 2: search.extend([1, 3])
        elif self.curr_lane == 3: search.append(2)

        for l in search: costs[l] = self.calculate_cost(l)

        best = min(costs, key=costs.get)
        curr_cost = costs.get(self.curr_lane, 99999.0)

        # Check Emergency
        is_emerg = (self.is_in_merge_zone() and self.curr_lane == 1)
        
        if best != self.curr_lane:
            do_change = False
            if is_emerg:
                do_change = True
            elif (time.time() - self.last_lc_time) > self.LANE_CHANGE_COOLDOWN:
                if curr_cost - costs[best] > 20.0: do_change = True

            # [Safety Gatekeeper] Final Physical Check
            if do_change:
                if self.check_side_safety(best):
                    self.get_logger().info(f"✅ LC Approved: {self.curr_lane}->{best}")
                    self.set_path(best)
                else:
                    # Dangerous overlap detected -> Stay in current lane
                    pass

    # --- Control Loop ---
    def control_loop(self):

        if not self.is_ready:
            self.control_pub.publish(Accel())
            return

        idx = self.get_nearest_index()
        arr = self.path_arrs[self.curr_lane]
        if len(arr) == 0:
            self.control_pub.publish(Accel())
            return
        # 1. Pure Pursuit Lookahead
        ld = np.clip(0.48 + 0.15 * self.spd, self.MIN_LOOKAHEAD, self.MAX_LOOKAHEAD)
        s_idx = idx
        for _ in range(50):
            s_idx = (s_idx + 1) % len(arr)
            if math.hypot(arr[s_idx, 0] - self.x, arr[s_idx, 1] - self.y) > ld: break
        
        tx, ty = arr[s_idx]
        dx, dy = tx - self.x, ty - self.y
        lx = math.cos(-self.yaw) * dx - math.sin(-self.yaw) * dy
        ly = math.sin(-self.yaw) * dx + math.cos(-self.yaw) * dy
        k = 2.0 * ly / (ld ** 2)

        # 2. Decision
        self.decision_logic(k)

        # 3. Speed Planning
        tv = self.MAX_SPEED_STRAIGHT
        is_turn = abs(k) > 0.40
        if is_turn:
            bf = min(1.0, (abs(k) - 0.40) * 2.5)
            tv = max(self.MAX_SPEED_CORNER, self.MAX_SPEED_STRAIGHT * (1.0 - bf * 0.55))

        fg, fv, rg, rv = self.get_lane_info(self.curr_lane)
        
        # Tailgating Boost (If rear is too close and faster)
        if rg < 1.0 and rv > self.spd: tv = self.MAX_SPEED_STRAIGHT + 0.3
        
        # ACC (Front Car Follow)
        if fg < 0.6:
            ratio = max(0.0, (fg - 0.15) / (0.6 - 0.15))
            tv = min(tv, fv * 0.9 + tv * 0.1 * ratio)
            
        # [Yield Override] Stop if stuck in Merge Zone
        if self.should_yield():
            tv = 0.0

        # 4. Actuation (PID + LPF)
        self.smoothed_v = 0.8 * self.smoothed_v + 0.2 * tv
        err = self.smoothed_v - self.spd
        
        # [Corrected] Integral Accumulation with Anti-Windup
        self.spd_err_sum += err
        self.spd_err_sum = np.clip(self.spd_err_sum, -1.0, 1.0) # Limit integral term
        
        acc_cmd = self.PID_KP * err + self.PID_KI * self.spd_err_sum
        
        # Soft Launch Logic
        if self.spd < 0.5 and self.smoothed_v > 0.5 and not is_turn:
            acc_cmd = max(acc_cmd, 1.5)
        elif self.spd < 0.1 and self.smoothed_v > 0.1 and acc_cmd < 0.6:
            acc_cmd = 0.8

        acc = 0.3 * float(acc_cmd) + 0.7 * self.prev_accel
        self.prev_accel = acc
        
        if self.spd < 0.1 and acc < 0.0: acc = 0.0
        acc = max(acc, -1.0)

        # Lateral Control (Stanley-like)
        nx, ny = arr[idx, 0] - self.x, arr[idx, 1] - self.y
        cte = math.sin(-self.yaw) * nx + math.cos(-self.yaw) * ny
        k_cte = 2.5 if (self.curr_lane == 2 and cte < -0.15) else 1.1
        steer = (k * 1.0) + (cte * k_cte)

        cmd = Accel()
        cmd.linear.x = acc
        cmd.angular.z = float(steer)
        self.control_pub.publish(cmd)

def main():
    rclpy.init()
    node = MergeCorridorController()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

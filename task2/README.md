# Task 2 — Mixed Traffic Control (HV + CAV)

This task focuses on vehicle control and overtaking logic in a mixed traffic
scenario where human-driven vehicles (HVs) and a CAV coexist.

---

## Files Overview

- Path Files (*.json)  
  Lane reference paths used for lane-based projection and reasoning.

- projection.py  
  Projects CAV and HV positions onto lane coordinates (s, d) using reference paths.

- decision_fsm.py  
  FSM-based decision logic for lane keeping and lane change based on surrounding HVs.

- controller.py  
  Low-level control module that generates acceleration and steering commands
  based on the decision output.

- visualization_proj.py  
  Utility for visualizing projection results and lane association.

---

## Execution Flow

Projection (projection.py)  
↓  
Decision FSM (decision_fsm.py)  
↓  
Controller (controller.py)

---

## How to Run

### Terminal 1 — Decision FSM

```bash
source /opt/ros/foxy/setup.bash
source ~/Mobility_Challenge_Simulator/install/setup.bash
export ROS_DOMAIN_ID=100

cd ~/Desktop/Mobility_Challenge_Simulator/src/central_control/
python3 decision_fsm.py


### Terminal 2 — Controller

```bash
source /opt/ros/foxy/setup.bash
source ~/Mobility_Challenge_Simulator/install/setup.bash
export ROS_DOMAIN_ID=100

cd ~/Desktop/Mobility_Challenge_Simulator/src/central_control/
python3 controller.py


# KAIST Mobility Challenge — CAV Control & Simulation

This repository contains our team’s implementation for the  
**KAIST Mobility Challenge**,  
a competition focused on **connected and autonomous vehicle (CAV) control**
in a multi-agent simulation environment.

The project is developed using **ROS 2** and targets robust vehicle control
under realistic traffic and interaction scenarios.

---

## Competition Overview
- **Event**: KAIST Mobility Challenge
- **Focus**: Cooperative and autonomous vehicle control
- **Platform**: ROS 2–based simulator
- **Scenario**: Multi-vehicle interaction in a shared environment

The objective is to design and implement control algorithms that enable
stable, safe, and efficient autonomous driving behavior in complex scenarios.

---

## Project Overview
Our team focuses on **trajectory tracking and motion control** for autonomous vehicles
within the provided simulation framework.

The system:
- Subscribes to vehicle state and waypoint information
- Computes control commands based on reference trajectories
- Publishes acceleration and steering commands in real time
- Is designed to be modular and easily extensible

---

## System Architecture
- **ROS 2 Nodes**
  - State subscription (pose, velocity)
  - Control computation
  - Command publishing
- **Control Logic**
  - Reference trajectory following
  - Smooth acceleration control
- **Simulation Interface**
  - Direct integration with the KAIST Mobility Challenge simulator

---

## Team & Contribution
This project is developed as a **team-based competition entry**.

---

## Notes
- This repository follows a structured Git workflow  
  (feature branches → dev → main).
- Build artifacts and logs are excluded via `.gitignore`.

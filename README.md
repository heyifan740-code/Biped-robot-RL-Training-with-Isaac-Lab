# 🦆 My biped robot RL Training with Isaac Lab

A reinforcement learning project for the OpenDuckMini bipedal robot, developed using **NVIDIA Isaac Lab** (built on Isaac Sim 5.1).

This project demonstrates an end-to-end workflow: importing a custom URDF biped, configuring the physics environment, and training a locomotion policy using **PPO (Proximal Policy Optimization)**. A key focus of this work was addressing common bipedal instability issues.

## 🎥 Simulation Demos

The following demos showcase the robot's performance after training with the custom curriculum.

| **Stable Locomotion (0.6 m/s)** | **Robustness / Push Recovery** |
| :---: | :---: |
| <img src="media/walk_demo.gif" width="100%"> | <img src="media/push_demo.gif" width="100%"> |
| *Smooth gait with minimal body tilt* | *Recovering from external disturbances* |

> **Note:** Full training videos and Sim2Real tests can be found in the `media/` directory.

## Key Features & Highlights

* **Sim2Real Ready Pipeline**:
  - Integrated the OpenDuckMini URDF into Isaac Sim 5.1.
  - Configured precise actuator PD gains (`stiffness` & `damping`) to match physical hardware characteristics.

* **Advanced Reward Shaping**:
  - Solved the local minima "skiing gait" problem by implementing a custom curriculum.
  - **Feet Air Time**: Incentivized stepping motions to prevent dragging feet.
  - **Joint Regularization**: Penalized excessive joint deviation to maintain a natural standing posture.
  - **Orientation Stability**: Heavily penalized base tilt to ensure upright locomotion.

* **Modern Tech Stack**:
  - Utilizing the latest **Isaac Lab (Main Branch)** architecture.
  - Training via **RSL-RL** library on NVIDIA RTX GPU.

## 📂 Project Structure

This repository contains the essential custom components extracted from the Isaac Lab environment:

```text
OpenDuck-IsaacLab-RL/
├── assets/
│   └── robots/my_biped/       # Converted USD assets and source URDF/meshes
├── config/
│   └── my_biped/              # Environment configurations (Reward functions, Observations)
│       └── agents/            # PPO algorithm hyperparameters
└── media/                     # Demo videos, GIFs, and plots

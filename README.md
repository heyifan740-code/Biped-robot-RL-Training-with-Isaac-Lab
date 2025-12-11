# 🦆 My biped robot RL Training with Isaac Lab

A reinforcement learning project for the OpenDuckMini bipedal robot, developed using **NVIDIA Isaac Lab** (built on Isaac Sim 5.1).

This project demonstrates an end-to-end workflow: importing a custom URDF biped, configuring the physics environment, and training a locomotion policy using **PPO (Proximal Policy Optimization)**. A key focus of this work was addressing common bipedal instability issues.

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
└── media/                     # Demo videos and 

```
## How to Run

To execute this training environment, integrate these files back into a standard Isaac Lab installation:

1. **Prerequisites**: Ensure NVIDIA Isaac Lab (v5.1+) is installed.

2. **Asset Integration**: Copy the contents of `assets/` and `config/` into your local Isaac Lab `source/` directory structure.

3. **Task Registration**: Register the new task in your local `isaaclab_tasks/__init__.py`:

   ```python
   gym.register(
       id="Isaac-Velocity-MyBiped-v0",
       entry_point="isaaclab.envs:ManagerBasedRLEnv",
       kwargs={
           "env_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.my_biped.flat_env_cfg:MyBipedFlatEnvCfg",
           "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.my_biped.agents.rsl_rl_ppo_cfg:MyBipedPPORunnerCfg",
       },
   )

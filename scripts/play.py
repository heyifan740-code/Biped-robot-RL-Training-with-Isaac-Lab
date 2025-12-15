# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""



import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


# ... 其他 imports ...
import torch

# [新增] 引入 Isaac Sim 的调试绘图工具
from isaacsim.util.debug_draw import _debug_draw

import gymnasium as gym
import os
import time
import torch
#
import pandas as pd
import numpy as np
#
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    
    # ================== 【插入在这里】 ==================
    print("🔓 解锁时长限制: 将最大回合时间设为 300秒")
    env_cfg.episode_length_s = 300.0  # 足够你录 7500 帧了
    # ====================================================
    
    
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # [新增] 初始化录制列表
    trajectory_log = []
    print("🔴 开始录制... 目标速度已锁死为 0.7 m/s")

    # 1. 获取机器人对象
    robot_entity = env.unwrapped.scene["robot"]
    
    # 2. 打印关节名称列表 (这就是 CSV 从左到右的列名！)
    print("\n" + "="*50)
    print("📢 仿真关节顺序 (CSV 列顺序):")
    print(robot_entity.joint_names)
    print("="*50 + "\n")
    
    # 临时暂停一下，让你看清楚再继续
    input("按回车键继续 >>> ")


    # [修复 1] 在进入循环前，必须先定义这些列表！
    log_pos = []
    log_vel = []
    log_torque = []

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()


        # [新增] 强制锁死速度指令 (Sim-to-Real 核心步骤)
        # 必须在 policy(obs) 之前执行，确保网络看到的是正确的指令
        # 注意：这里假设你的机器人名叫 "Robot"，指令名叫 "base_velocity"
        try:
            # 1. 构造目标速度 (Vx=0.7)
            target_vel = torch.tensor([0.7, 0.0, 0.0], device=env.unwrapped.device)
            
            # 2. 获取当前指令的 Tensor 引用
            # 注意："base_velocity" 必须和你 Config 里的名字一致
            cmd_tensor = env.unwrapped.command_manager.get_command("base_velocity")
            
            # 3. 原地覆盖 (In-place update)，这会直接改变环境中的指令
            cmd_tensor[:] = target_vel
            
        except Exception as e:
            # 避免刷屏报错，只在第一帧提示
            if timestep == 0:
                print(f"⚠️ 覆盖指令失败: {e}")



        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)


            # # [新增] 数据录制逻辑
            # # 获取第 0 个环境的机器人关节位置
            # # 注意：Isaac Lab 新版数据通常在 scene["Robot"] 里
            # try:
            #     # 尝试获取关节位置 (Joint Positions)
            #     # 这里的 "Robot" 必须和你 Config 里的 self.scene.robot 的名字一致
            #     current_joints = env.unwrapped.scene["robot"].data.joint_pos[0].cpu().numpy()
            #     trajectory_log.append(current_joints)
                
            #     # 录制 1500 帧 (约 30秒) 后自动保存
            #     if len(trajectory_log) == 1500:
            #         print("💾 数据已收集 1500 帧，正在保存为 CSV...")
            #         df = pd.DataFrame(trajectory_log)
            #         # 保存到 logs 目录下，方便查找
            #         save_path = os.path.join(log_dir, "walk_0.80.csv")
            #         df.to_csv(save_path, index=False, header=False)
            #         print(f"✅ 文件已保存至: {save_path}")
            #         print("💡 你现在可以停止脚本，或者继续观察。")
            # except KeyError:
            #     print("❌ 找不到 'Robot' 资产，请检查 env_cfg 里的机器人名字")


            # try:
            #     # 获取机器人句柄
            #     robot = env.unwrapped.scene["robot"]
                
            #     # 1. 抓取数据 (转为 CPU numpy)
            #     # 注意：数据都在 GPU 上，必须 .cpu().numpy()
            #     p = robot.data.joint_pos[0].cpu().numpy()
            #     v = robot.data.joint_vel[0].cpu().numpy()
            #     t = robot.data.applied_torque[0].cpu().numpy()
                
            #     # 2. 存入列表
            #     log_pos.append(p)
            #     log_vel.append(v)
            #     log_torque.append(t)
                
            #     # 3. 进度打印 & 保存
            #     curr_len = len(log_pos)
            #     if curr_len % 100 == 0:
            #         print(f"🎥 Recorded {curr_len} frames...")

            #     if curr_len == 1000: # 或者你想要的时长
            #         print("💾 正在保存全量数据 robot_data_0.2.npz ...")
                    
            #         # 保存为 .npz 文件
            #         save_path = os.path.abspath("robot_data_0.2.npz")
            #         np.savez(save_path, 
            #                  pos=np.array(log_pos), 
            #                  vel=np.array(log_vel), 
            #                  torques=np.array(log_torque),
            #                  names=np.array(robot.data.joint_names)) # 把关节名字也存进去
                             
            #         print(f"✅ 保存成功: {save_path}")
            #         break

            # except Exception as e:
            #     print(f"❌ 录制出错: {e}")

            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

            # # ... 在 while 循环内部 ...
    
            # # 1. 获取绘图接口实例 (放在循环里或者循环外都可以)
            # draw_interface = _debug_draw.acquire_debug_draw_interface()
            
            # # ... env.step(actions) ...
            # # ... obs 更新 ...

            # # [新增] --- 速度可视化逻辑 ---
            
            # # A. 获取机器人数据 (假设机器人名字叫 "robot"，请根据你的 Config 确认)
            # # env.unwrapped.scene 包含了仿真场景里的所有物体
            # robot = env.unwrapped.scene["robot"]
            
            # # B. 获取基座线速度 (Body Frame / 机器人自身坐标系)
            # # root_lin_vel_b 的形状是 [num_envs, 3] -> (vx, vy, vz)
            # # 我们只关心第 0 个环境
            # lin_vel_b = robot.data.root_lin_vel_b[0] 
            
            # vx = lin_vel_b[0].item()  # 前进速度
            # vy = lin_vel_b[1].item()  # 侧向速度
            # total_speed = torch.norm(lin_vel_b[:2]).item() # 水平总速度
            
            # # C. 获取机器人当前位置 (World Frame)
            # # 用来确定把字写在哪里
            # root_pos_w = robot.data.root_pos_w[0].cpu().numpy()
            
            # # D. 定义文字显示的位置 (在机器人头顶上方 0.5米处)
            # text_pos = [root_pos_w[0], root_pos_w[1], root_pos_w[2] + 0.6]
            
            # # E. 准备显示的字符串
            # # 格式: Vx: 前进速度 | Speed: 总速度
            # display_text = f"Vx: {vx:.2f} m/s\nSpeed: {total_speed:.2f} m/s"
            
            # # F. 绘制文字
            # # draw_text(位置xyz, 内容, 字体大小, 颜色RGBA)
            # # 颜色: [1, 1, 0, 1] 是黄色
            # draw_interface.clear_lines() # 清除上一帧的残留（虽然 draw_text 通常只有一帧寿命，但加上是个好习惯）
            # draw_interface.draw_text(text_pos, display_text, 20, [1.0, 1.0, 0.0, 1.0])
            
            # # ---------------------------
            # ... 在 env.step(actions) 之后 ...

            # 1. 获取机器人数据
            robot = env.unwrapped.scene["robot"]
            
            # 获取世界坐标系下的位置和速度
            # root_pos_w: [N, 3] -> (x, y, z)
            # root_lin_vel_w: [N, 3] -> (vx, vy, vz)
            root_pos = robot.data.root_pos_w[0].cpu().numpy()
            root_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
            
            # 2. 计算画线的起点和终点
            # 起点：稍微高出机器人一点点 (z + 0.5)，方便看
            start_point = root_pos.copy()
            start_point[2] += 0.5 
            
            # 终点：起点 + 速度向量
            # 这里的 1.0 是缩放比例，如果线太短看不清，可以改成 2.0
            end_point = start_point + (root_vel * 1.0) 
            
            # 3. 绘制线条
            try:
                # 获取绘图接口 (带容错处理)
                from isaacsim.util.debug_draw import _debug_draw
                draw_interface = _debug_draw.acquire_debug_draw_interface()
                
                # 清除旧线
                draw_interface.clear_lines()
                
                # 画一条黄色的粗线
                # draw_lines(起点列表, 终点列表, 颜色列表, 线宽列表)
                draw_interface.draw_lines(
                    [start_point.tolist()], 
                    [end_point.tolist()], 
                    [(1.0, 1.0, 0.0, 1.0)], # 黄色 RGBA
                    [5.0] # 线宽
                )
            except Exception as e:
                # 如果还是报错，就只在终端打印，不让脚本崩溃
                pass

            # 4. 同时在终端打印数值 (双重保险)
            speed = np.linalg.norm(root_vel[:2]) # 只看水平速度
            print(f"\r🚀 Velocity: {speed:.2f} m/s", end="")
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, MySceneCfg
from isaaclab_assets.robots.my_biped import MY_BIPED_CFG
from isaaclab.managers import RewardTermCfg as RewTerm, ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.envs import mdp
# [新增] 正确导入噪声配置类
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# --------------------------------------------------------
# 自定义观测配置类(customize observations)
# --------------------------------------------------------
@configclass
class MyObservationsCfg:
    """自定义观测配置：分离 Actor 和 Critic"""

    # 1. 定义 Policy (Actor) 的观测组：给实机部署用的 (Blind)
    @configclass
    class PolicyCfg(ObsGroup):
        """Actor 只看实机能获取的数据"""
        
        #  base_lin_vel (实机测不准)
        #  height_scan (实机没激光雷达)
        
        # 使用 Unoise，去掉 mdp.utils 前缀
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    # 2. 定义 Critic 的观测组(Privileged)
    @configclass
    class CriticCfg(ObsGroup):
        """Critic 依然可以看到所有真理，帮助训练"""
        
        # 修正：直接使用 Unoise
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = False 
            self.concatenate_terms = True

    # 3. 注册这两个组
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()





# --------------------------------------------------------
# 主环境配置类 env config class
# --------------------------------------------------------
@configclass
class MyBipedFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    
    # 挂载自定义的观测配置
    observations: MyObservationsCfg = MyObservationsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        
        # 1. 替换机器人
        self.scene.robot = MY_BIPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 2. 地形设为平地
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None 
        self.scene.height_scanner = None 
        
        # 3. 动作缩放
        self.actions.joint_pos.scale = 0.25 

        
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        
        
        self.rewards.track_lin_vel_xy_exp.weight = 2.0

        #self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5


        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.lin_vel_z_l2.weight = -2.0

        self.rewards.action_rate_l2.weight = -0.05
        # 关节加速度惩罚
        # 防止电机瞬间力矩过大（保护齿轮）
        #self.rewards.dof_acc_l2 = RewTerm(func=mdp.dof_acc_l2, weight=-2.5e-7)

        # func 改成了 mdp.joint_acc_l2
        #self.rewards.joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
        self.rewards.is_alive = RewTerm(func=mdp.is_alive, weight=0.3)

        # 恢复并加重限位惩罚
        if hasattr(self.rewards, "dof_pos_limits"):
            self.rewards.dof_pos_limits.weight = -5.0
        
        # 腾空奖励 大
        if hasattr(self.rewards, "feet_air_time"):
            self.rewards.feet_air_time.params["sensor_cfg"].body_names = [".*foot.*"]
            self.rewards.feet_air_time.weight = 2.0
            self.rewards.feet_air_time.params["threshold"] = 0.1
            
        # ========== 修复正则匹配与禁用不必要项 ==========
        self.scene.contact_forces.body_filter = [".*foot.*", "body_bottom"] 

        if hasattr(self.terminations, "base_contact"):
            self.terminations.base_contact.params["sensor_cfg"].body_names = ["body_bottom"]
            
        if hasattr(self.rewards, "undesired_contacts"):
            self.rewards.undesired_contacts = None

        if hasattr(self.events, "add_base_mass"):
            self.events.add_base_mass = None
            
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None
            
        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None
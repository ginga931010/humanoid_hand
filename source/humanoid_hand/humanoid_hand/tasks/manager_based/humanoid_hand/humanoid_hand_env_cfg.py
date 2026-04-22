# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg

from . import mdp as local_mdp 

##
# Scene Definition (定義場景)
##

@configclass
class HumanoidHandSceneCfg(InteractiveSceneCfg):
    """仿生人形手掌場景設定"""

    # 1. 地板與光照
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    
    # 2. 桌子
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.75),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            
            # 這裡就可以設定物理材質了！
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,  # 靜摩擦力 (很高，不易滑)
                dynamic_friction=1.0, # 動摩擦力
                restitution=0.0,      # 彈性 (不反彈)
            ),

        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0, 0.375)),
    )

    # 3. 機器人
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/ginga/humanoid_hand/source/humanoid_hand/humanoid_hand/assets/New_URDF_export_description/usd/movable_humanoid_hand.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0, # [建議] 降低這個值，對小物件比較穩定
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=16, # [建議] 增加迭代次數，提高小物件的計算精度
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={'Revolute_33': 0.0, 'Revolute_37': 0.0, 'Revolute_42': 0.0, 'Revolute_47': 0.0,
                        'Revolute_51': 0.5, 'Revolute_34': 0.0, 'Revolute_35': 0.0, 'Revolute_38': 0.0,
                        'Revolute_39': 0.0, 'Revolute_43': 0.0, 'Revolute_44': 0.0, 'Revolute_48': 0.0, 
                        'Revolute_49': 0.0, 'Revolute_52': 0.0, 'Revolute_36': 0.0, 'Revolute_40': 0.0, 
                        'Revolute_45': 0.0, 'Revolute_50': 0.0, 'Revolute_53': 0.0, 'Revolute_54': 0.0, 
                        'Revolute_55': 0.0},
            pos=(0.5, 0.25, 0.85), 
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "hand_actuator": IdealPDActuatorCfg(   # [修改] 改用 IdealPD，這會比較聽話
                joint_names_expr=[
                    "Revolute_33", "Revolute_34", 
                    "Revolute_37", "Revolute_38", 
                    "Revolute_42", "Revolute_43", 
                    "Revolute_47", "Revolute_48", 
                    "Revolute_51", "Revolute_52", "Revolute_53"
                ],
                # IdealPD 不需要 effort_limit_sim，它直接用力矩上限
                effort_limit=1.0,      # 給它超大的力，確認它到底能不能動
                velocity_limit=1.0,    
                stiffness=0.032,         # P gain: 對於手指(幾克重)，20已經很大了
                damping=0.0048,            # D gain: 設為 P 的 10% 左右來消抖
            ),
        },
    )

    # 4. 目標物體
    # [修正] 移除原本路徑中的 /Objects/ 中間層，直接放在環境根目錄下
    
    # A. 方塊
    object_cube: RigidObjectCfg = RigidObjectCfg(
        # 改為 Object_Cube (避免中間層)
        prim_path="{ENV_REGEX_NS}/Object_Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05), # 方塊大小 (5公分)
            collision_props=sim_utils.CollisionPropertiesCfg(),
            
            # 這裡就可以設定物理材質了！
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,  # 靜摩擦力 (很高，不易滑)
                dynamic_friction=1.0, # 動摩擦力
                restitution=0.0,      # 彈性 (不反彈)
            ),
            
            # 設定顏色 (紅色) 方便辨識
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            
            # 剛體屬性
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.78)),
    )

    # B. 球體
    object_sphere: RigidObjectCfg = RigidObjectCfg(
        # 改為 Object_Sphere
        prim_path="{ENV_REGEX_NS}/Object_Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.2, 0.78)),
    )

    # C. 圓柱體
    object_cylinder: RigidObjectCfg = RigidObjectCfg(
        # 改為 Object_Cylinder
        prim_path="{ENV_REGEX_NS}/Object_Cylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 1.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.2, 0.85)),
    )


##
# MDP Settings
##

@configclass
class ActionsCfg:
    """Action Spaces"""
    fingers_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        # [修正] 不要用 [".*"]，必須指定你真正想要控制的那 11 個關節
        joint_names=[
            "Revolute_33", "Revolute_34", 
            "Revolute_37", "Revolute_38", 
            "Revolute_42", "Revolute_43", 
            "Revolute_47", "Revolute_48", 
            "Revolute_51", "Revolute_52", "Revolute_53"
        ],
        scale=2.0,
        use_default_offset=False, 
    )

@configclass
class ObservationsCfg:
    """Observation Spaces"""
    
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=local_mdp.object_position_dynamic) 
        object_type = ObsTerm(func=local_mdp.object_type_one_hot)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event (Reset & Randomization)"""
    
    # [注意] asset_cfg 這裡使用 SceneEntityCfg("object_cube")
    # 這會去對應 HumanoidHandSceneCfg 裡面的變數名稱 "object_cube"
    # 所以即使我們改了 prim_path，這裡也不需要改，只要變數名沒變就好。
    
    friction_cube = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object_cube"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.7, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    friction_sphere = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object_sphere"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.7, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    friction_cylinder = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object_cylinder"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.7, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    select_object = EventTerm(
        func=local_mdp.reset_active_object,
        mode="reset",
        params={
            "object_names": ["object_cube", "object_sphere", "object_cylinder"],
        },
    )

    reset_hand = EventTerm(
        func=local_mdp.reset_hand_pose_conditional,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "offset_top_grasp_pos": (0.0, 0.06, 0.095),
            "offset_top_grasp_rot": (0.5, 0.8660254, 0.0, 0.0),
            "offset_side_grasp_pos": (-0.095, 0.035, -0.005),
            "offset_side_grasp_rot": (0.7071068, 0.0, 0.7071068, 0.0),
        },
    )

@configclass
class RewardsCfg:
    """Reward Terms"""
    
    # 1. 活著就有獎勵 (避免過早結束)
    alive = RewTerm(func=mdp.is_alive, weight=0.0)
    
    # 2. 懲罰動作劇烈程度 (Action Rate) - 讓動作平滑
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.25)
    
    # 3. [核心] 指尖靠近物體獎勵
    fingertip_distance = RewTerm(
        func=local_mdp.fingertip_object_distance,
        weight=20.0,
        params={
            "std": 0.05,  # 距離 1cm 時獎勵下降顯著
            "minimal_height": 0.0, # 隨時都計算
            # [重要] 請將這裡的 regex 改為你真正的指尖 Link 名稱
            # 如果不知道，先用 ".*" 雖然不準但能跑
            # 建議格式: ["distal_thumb", "distal_index", ...]
            "fingertip_cfg": SceneEntityCfg("robot", body_names=[
                "Component2_1", "Component5_1", "Component10_1","Component16_1"]), 
        },
    )
    
    # 4. [核心] 抬起獎勵 (Binary) - 這是最重要的目標
    # 假設桌子高度 0.75，物體本身高 0.05，所以 0.85 代表抬起約 5~10 公分
    object_lifted = RewTerm(
        func=local_mdp.object_is_lifted_by_type, # 改用新的函數
        weight=50.0,
        params={
            # 根據上面的計算填入數值
            "height_cube": 0.785,     # 方塊比較矮
            "height_sphere": 0.80,    # 球體中等
            "height_cylinder": 0.83,  # 圓柱本身就很高，閾值要設高一點，不然還沒離地就拿分了
        },
    )
    
    thumb_tip_tracking = RewTerm(
        func=local_mdp.fingertip_object_distance,
        weight=100.0,  # 給予高權重，強迫它重視大拇指的位置
        params={
            "std": 0.03,  # [高精度要求] 只有非常靠近(5cm內)才開始給高分
            "minimal_height": 0.0,
            
            # [關鍵] 這裡只填入大拇指尖的 Link 名稱
            # 請根據你的 URDF 確認名稱，這裡假設是包含 53 的 link (通常是拇指末端)
            # 或者是 "Link_53", "distal_thumb" 等
            "fingertip_cfg": SceneEntityCfg("robot", body_names=["Component20_1"]), 
        },
    )
    # 5. [輔助] 高度連續獎勵 - 引導它慢慢往上
    # object_height = RewTerm(
    #     func=local_mdp.object_height_continuous,
    #     weight=5.0,
    #     params={
    #         "initial_height": 0.78, # 物體在桌上的初始高度
    #         "max_height": 0.3,      # 最多算到抬高 30cm
    #     },
    # )

@configclass
class TerminationsCfg:
    """Termination Conditions"""
    
    # 1. 超時終止 (Time Out)
    # 當步數超過 episode_length_s / dt 時觸發
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 2. 物體掉落終止 (Object Dropped)
    # 桌子高度是 0.75m，如果物體掉到 0.5m 以下，判定為失敗
    object_dropped = DoneTerm(
        func=local_mdp.object_dropped, # 呼叫我們剛剛寫的函數
        params={"threshold": 0.5},     # 閾值設定
    )
##
# Environment Configuration
##

@configclass
class HumanoidHandEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Humanoid Hand RL Environment."""
    
    scene: HumanoidHandSceneCfg = HumanoidHandSceneCfg(num_envs=16, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    episode_length_s = 5.0
    decimation = 2 
    
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005, 
        render_interval=2, 
        # disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
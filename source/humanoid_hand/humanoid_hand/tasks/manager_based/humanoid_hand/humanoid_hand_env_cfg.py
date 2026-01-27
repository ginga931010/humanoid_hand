# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
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
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={".*": 0.0},
            pos=(0.5, 0.25, 0.75), 
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "hand_actuator": ImplicitActuatorCfg(
                joint_names_expr=[
                    "Revolute_33", "Revolute_34", 
                    "Revolute_37", "Revolute_38", 
                    "Revolute_42", "Revolute_43", 
                    "Revolute_47", "Revolute_48", 
                    "Revolute_51", "Revolute_52", "Revolute_53"
                ],
                effort_limit_sim=1000,
                velocity_limit_sim=10,    
                stiffness=2000,
                damping=200,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
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
            height=0.10,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 1.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.2, 0.80)),
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
        scale=1.0,
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
            "offset_top_grasp_pos": (0.0, 0.1, 0.05),
            "offset_top_grasp_rot": (0.7071, 0.7071, 0.0, 0.0),
            "offset_side_grasp_pos": (0.0, 0.05, 0.0),
            "offset_side_grasp_rot": (1.0, 0.0, 0.0, 0.0),
        },
    )

@configclass
class RewardsCfg:
    """Reward Terms"""
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

@configclass
class TerminationsCfg:
    """Termination Conditions"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

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
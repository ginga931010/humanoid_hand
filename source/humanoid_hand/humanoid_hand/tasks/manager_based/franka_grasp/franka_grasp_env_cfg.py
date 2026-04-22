# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.envs import mdp
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
# 引入自定義的模組
from . import mdp as local_mdp

@configclass
class FrankaGraspSceneCfg(InteractiveSceneCfg):
    """場景設定：Franka + 桌子 + 方塊 + 固定相機"""

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
            size=(1.5, 1.5, 0.8), # 桌高 0.8
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
        
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0, 0.4)), # 中心在 0.4, 頂部在 0.8
    )

    # 3. Franka 機器人
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            # 使用 Isaac Lab 內建的 Franka USD
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.1, 0.0, 0.8), # 放在桌子上
            rot=(1.0, 0.0, 0.0, 0.0),
            # 預設關節角度 (Ready Pose)
            joint_pos={
                "panda_joint1": 0.0, "panda_joint2": -0.785, "panda_joint3": 0.0,
                "panda_joint4": -2.356, "panda_joint5": 0.0, "panda_joint6": 1.571,
                "panda_joint7": 0.785, "panda_finger_joint.*": 0.04,
            },
        ),
        actuators={
            "panda_shoulder": IdealPDActuatorCfg(
                joint_names_expr=["panda_joint[1-7]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_fingers": IdealPDActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2000.0,
                damping=50.0,
            ),
        },
    )

    # 4. 目標物 (方塊)
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 1.0)),
    )

    # 5. [關鍵] 深度相機 (Fixed Camera)
    # 5. [修正] 深度相機 (Fixed Camera)
    # 5. 深度相機 (Fixed Camera)
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        debug_vis=True,
        # 1. Sensor 屬性
        update_period=0,
        height=84,
        width=84,
        data_types=["distance_to_image_plane"],
        
        # 2. Spawner 屬性 (物理鏡頭參數)
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
        ),
        
        # 3. [修正] 位置與旋轉 (使用 offset 代替 init_state)
        # [推薦設定] 斜前方 45 度角
        # 位置：x=1.2 (桌子前緣外), z=1.0 (比桌子高 0.2m)
        # 這樣看過去，能看到物體的側面，也能看到深度變化
        offset=CameraCfg.OffsetCfg(
            pos=(1.7, 0.0, 1.9),
            #rot=(-0.8733046, 0, -0.4871745, 0), # y 60 度俯視
            #rot=(0.5735764 ,0 , 0.819152, 0), # y 110 度俯視
            #rot=(0.707, 0.0, 0.707, 0.0), # y 90 度俯視
            #rot=(0, 0 ,0 ,1), #z180
            rot=(0.0, -0.3827, 0.0, 0.9239),
            #rot=(0.0, 0.3827, -0.3827, 0.9239), # 
            convention="world",
        ),
    )


@configclass
class ActionsCfg:
    # 使用關節位置控制 (相對值)
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )
    # 夾爪控制 (二值化：開/關)
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04}, 
        close_command_expr={"panda_finger_joint.*": 0.0},   # [修正] 把 ["0.0", "0.0"] 改成 0.0
    )
# observations for play franka.py
# @configclass
# class ObservationsCfg:
#     @configclass
#     class PolicyCfg(ObsGroup):
#         # [Actor 輸入]
#         # 1. 深度圖 (End-to-End)
#         concatenate_terms = False
#         depth_image = ObsTerm(
#             func=mdp.image, # <--- 改用這個自定義函數
#             params={
#                 "sensor_cfg": SceneEntityCfg("camera"),
#                 "data_type": "distance_to_image_plane",
#                 "normalize": True,
#             },
#         )
#         # 2. 本體感覺 (關節資訊)
#         joint_pos = ObsTerm(func=mdp.joint_pos_rel)
#         joint_vel = ObsTerm(func=mdp.joint_vel_rel)

#     @configclass
#     class CriticCfg(ObsGroup):
#         # [Critic 輸入] - 上帝視角
#         concatenate_terms = True
#         joint_pos = ObsTerm(func=mdp.joint_pos_rel)
#         joint_vel = ObsTerm(func=mdp.joint_vel_rel)
#         object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object_cube")})
#         object_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object_cube")})
        
#     policy: PolicyCfg = PolicyCfg()
#     critic: CriticCfg = CriticCfg()

#observations for training
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # [Actor 輸入]
        # 1. 深度圖 (End-to-End)
        concatenate_terms = True
        depth_image = ObsTerm(
            func=local_mdp.image_flattened, # <--- 改用這個自定義函數
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "distance_to_image_plane",
                "normalize": True,
            },
        )
        # 2. 本體感覺 (關節資訊)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

    @configclass
    class CriticCfg(ObsGroup):
        # [Critic 輸入] - 上帝視角
        concatenate_terms = True
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object_cube")})
        object_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object_cube")})
        
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    # 1. Reset Robot -> Pre-grasp Pose
    reset_robot = EventTerm(
        func=local_mdp.reset_robot_to_ready_pose,
        mode="reset",
        params={"robot_cfg": SceneEntityCfg("robot")},
    )
    # 2. Reset Object -> Under Hand
    reset_object = EventTerm(
        func=local_mdp.reset_object_under_hand,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object_cube"),
            "offset_range": (0.1, 0.1), # 隨機生成在 20cm x 20cm 範圍內
        },
    )

@configclass
class RewardsCfg:
    # 1. 抬起獎勵 (主目標)
    lift_success = RewTerm(
        func=local_mdp.object_is_lifted_v2, # 記得去 rewards.py 新增這個函數
        weight=0.0, 
        params={
            "object_cfg": SceneEntityCfg("object_cube"),
            "gripper_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
            "height_threshold": 0.85,
            "dist_threshold": 0.15, # 10公分內才算有效抬起
        },
    )
    # 2. 距離懲罰 (引導靠近)
    approach_object = RewTerm(
        func=local_mdp.gripper_distance_reward,
        weight=20.0,
        params={
            "target_cfg": SceneEntityCfg("object_cube"),
            "gripper_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
        },
    )
    # 3. 動作懲罰 (平滑)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)


    align_gripper = RewTerm(
        func=local_mdp.align_gripper_to_object,
        weight=5, # 權重不用太大，輔助用
        params={
            "target_cfg": SceneEntityCfg("object_cube"),
            "gripper_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
        },
    )

    # 4. [新增] 開合引導
    # 教它 "近了就要抓"
    grasp_incentive = RewTerm(
        func=local_mdp.grasp_shaping_reward,
        weight=0.5, # 這個權重可以跟距離獎勵 (1.0) 相當
        params={
            "target_cfg": SceneEntityCfg("object_cube"),
            "gripper_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
            "open_threshold": 0.12, # 6公分以內算 "近"，開始鼓勵閉合
            "cube_width": 0.05,
        },
    )
    #weight0.5 open_threshold0.13 change finger_pos > 0.0275
@configclass
class TerminationsCfg:
    # 超時
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 掉落 (如果低於桌子高度太久)
    # 這裡簡單判斷：如果物體掉到地上就結束
    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object_cube")},
    )

@configclass
class FrankaGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Franka Visual Grasping."""
    scene: FrankaGraspSceneCfg = FrankaGraspSceneCfg(num_envs=4, env_spacing=5.0) # 有相機渲染，環境數少一點以免顯存爆
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    episode_length_s = 5.0 # 短時間任務
    decimation = 2 # 控制頻率 50Hz (如果 sim dt=0.01)

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.01,
        render_interval=2,
        
    )
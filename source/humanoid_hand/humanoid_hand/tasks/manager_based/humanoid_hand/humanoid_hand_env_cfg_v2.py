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
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg

from . import mdp as local_mdp 


##
# Scene Definition (定義場景)
##

@configclass
class HumanoidHandSceneCfg(InteractiveSceneCfg):
    """仿生人形手掌場景設定 (v2 Multibody)"""

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
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=2,
                kinematic_enabled=True,
            
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,  
                dynamic_friction=1.0, 
                restitution=0.0,      
            ),
            
            
        ),
        
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0, 0.375)),
    )

    # 3. 機器人 (套用新版 URDF 結構)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            # [請確認] 記得替換為新版 USD 的正確路徑
            usd_path="/home/ginga/humanoid_hand/source/humanoid_hand/humanoid_hand/assets/urdf_v2_multibody_description/usd/movable_v2.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                 
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=32, 
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # [更新] 套用新的 11 個受驅動關節，初始角度設為 0.0
            # step 1
            # joint_pos={
            #     'Revolute_12': 1.4, 'Revolute_14': 0.2, 'Revolute_16': 1.4, 
            #     'Revolute_18': 0.2, 'Revolute_20': 1.4, 'Revolute_22': 0.2, 
            #     'Revolute_24': 1.4, 'Revolute_26': 0.2, 'Revolute_7': 0.1, 
            #     'Revolute_8': 0.0, 'Revolute_29': 0.0
            # },
            #step 2
            # joint_pos={
            #     'Revolute_12': 1, 'Revolute_14': 0.2, 'Revolute_16': 1, 
            #     'Revolute_18': 0.2, 'Revolute_20': 1, 'Revolute_22': 0.2, 
            #     'Revolute_24': 1, 'Revolute_26': 0.2, 'Revolute_7': 0.1, 
            #     'Revolute_8': 0.0, 'Revolute_29': 0.0
            # },
            # step 3 
            joint_pos={
                'Revolute_12': 0.5, 'Revolute_14': 0.2, 'Revolute_16': 0.5, 
                'Revolute_18': 0.2, 'Revolute_20': 0.5, 'Revolute_22': 0.2, 
                'Revolute_24': 0.5, 'Revolute_26': 0.2, 'Revolute_7': 0.1, 
                'Revolute_8': 0.0, 'Revolute_29': 0.0
            },
            # step 4 
            # joint_pos={
            #     'Revolute_12': 0.0, 'Revolute_14': 0.0, 'Revolute_16': 0.0, 
            #     'Revolute_18': 0.0, 'Revolute_20': 0.0, 'Revolute_22': 0.0, 
            #     'Revolute_24': 0.0, 'Revolute_26': 0.0, 'Revolute_7': 0.0, 
            #     'Revolute_8': 0.0, 'Revolute_29': 0.0
            # },
            pos=(0.5, 0.25, 0.5), 
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "hand_actuator": ImplicitActuatorCfg(   
                # [更新] 指定新的 11 個致動器關節
                joint_names_expr=[
                    "Revolute_12", "Revolute_14", "Revolute_16", "Revolute_18",
                    "Revolute_20", "Revolute_22", "Revolute_24", "Revolute_26",
                    "Revolute_7", "Revolute_8", "Revolute_29"
                ],
                # 既然用了隱式求解器，我們就可以放心地使用正常的物理參數了！
                # effort_limit=1.0,      # 恢復到 0.5 Nm (隱式求解器會安全地截斷)
                # velocity_limit=5.0,    # 限制最高轉速
                effort_limit_sim=4.0,
                velocity_limit_sim=10.0,
                # PD Gains 可以調回合理的數值，再也不怕爆炸了！
                stiffness=5,         # 放心給 2.0，手指會很有力且穩定
                damping=0.2,           # 給予 5% 的阻尼來消除震盪       
            ),
        },
    )

    # 4. 目標物體 (維持不變)
    object_cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,  
                dynamic_friction=1.0, 
                restitution=0.0,      
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=2,
                linear_damping=0.0,   # <--- 控制平移的阻力
                angular_damping=0.0,  # <--- 控制旋轉的阻力
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.78)),
    )

    object_sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=2,
                linear_damping=0.0,   # <--- 控制平移的阻力
                angular_damping=0.0,  # <--- 控制旋轉的阻力
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,  
                dynamic_friction=1.0, 
                restitution=0.0,      
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.2, 0.78)),
    )

    object_cylinder: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_Cylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.035,
            height=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.78)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=2,
                linear_damping=0.0,   # <--- 控制平移的阻力
                angular_damping=0.0,  # <--- 控制旋轉的阻力
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,  
                dynamic_friction=1.0, 
                restitution=0.0,      
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.2, 0.895)),
    )

##
# MDP Settings
##

@configclass
class ActionsCfg:
    """Action Spaces"""
    fingers_pos = local_mdp.EMAJointPositionActionCfg(
        asset_name="robot", 
        # [更新] 行動空間也對應這 11 個關節
        joint_names=[
            "Revolute_12", "Revolute_14", "Revolute_16", "Revolute_18",
            "Revolute_20", "Revolute_22", "Revolute_24", "Revolute_26",
            "Revolute_7", "Revolute_8", "Revolute_29"
        ],
        alpha=0.5,
        
        # 1. 關閉預設 offset (這樣它就不會去吃你 init_state 的 0.0 了)
        use_default_offset=False, 
        
        # 2. 分別給定每個關節的 Scale (行程的一半)
        scale={
            "Revolute_12": 0.8726, "Revolute_14": 0.8726, "Revolute_16": 0.8726,
            "Revolute_18": 0.8726, "Revolute_20": 0.8726, "Revolute_22": 0.8726,
            "Revolute_24": 0.8726, "Revolute_26": 0.8726,
            "Revolute_7": 0.6981,   # 限度 [-0.698, 0.698] -> scale=(0.698-(-0.698))/2
            "Revolute_8": 0.8726,   # 限度 [-0.698, 1.047] -> scale=(1.047-(-0.698))/2
            "Revolute_29": 0.523599      # 請依照你後來補上的極限，填入 (上限-下限)/2
        },
        
        # 3. [新增] 手動給定 Offset (上下限的中點)
        offset={
            "Revolute_12": 0.8726, "Revolute_14": 0.8726, "Revolute_16": 0.8726,
            "Revolute_18": 0.8726, "Revolute_20": 0.8726, "Revolute_22": 0.8726,
            "Revolute_24": 0.8726, "Revolute_26": 0.8726,
            "Revolute_7": 0.0,      # 對稱的，中點為 0
            "Revolute_8": 0.1745,   # 限度 [-0.698, 1.047] 的中點
            "Revolute_29": -0.523599      # 請依照你後來補上的極限，填入 (上限+下限)/2
        }
    
    )

@configclass
class ObservationsCfg:
    """Observation Spaces"""
    
    @configclass
    class PolicyCfg(ObsGroup):
        # 1. 本體感覺 (Proprioception)
        joint_pos = ObsTerm(
            func=local_mdp.active_joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    joint_names=[
                        "Revolute_12", "Revolute_14", "Revolute_16", "Revolute_18",
                        "Revolute_20", "Revolute_22", "Revolute_24", "Revolute_26",
                        "Revolute_7", "Revolute_8", "Revolute_29"
                    ]
                )
            }
        )
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # 2. 讓 Agent 記得自己上一幀下了什麼指令
        last_action = ObsTerm(func=mdp.last_action)
        
        # # 3. 關節位置誤差 (虛擬觸覺)
        # joint_pos_error = ObsTerm(
        #     func=local_mdp.joint_position_error,
        #     params={
        #         # 記得填入你實際的那 11 個 joint_names
        #         "asset_cfg": SceneEntityCfg(
        #             "robot", 
        #             joint_names=[
        #                 "Revolute_12", "Revolute_14", "Revolute_16", "Revolute_18",
        #                 "Revolute_20", "Revolute_22", "Revolute_24", "Revolute_26",
        #                 "Revolute_7", "Revolute_8", "Revolute_29"
        #             ]
        #         )
        #     }
        # )
        
        # 4. 外部環境資訊 (Exteroception)
        object_position = ObsTerm(
            func=local_mdp.object_position_dynamic,
            noise=UniformNoiseCfg(
                n_min=-0.005, # 注意：UniformNoise 通常使用範圍 (n_min, n_max)
                n_max=0.005 # 加入 5mm 的隨機雜訊
            )
        ) 
        object_type = ObsTerm(func=local_mdp.object_type_one_hot)

        tactile_proxy = ObsTerm(
            func=local_mdp.tactile_proxy_fusion, # 注意：請替換為您存放該函數的正確路徑
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    joint_names=[
                        "Revolute_12", "Revolute_14", "Revolute_16", "Revolute_18",
                        "Revolute_20", "Revolute_22", "Revolute_24", "Revolute_26",
                        "Revolute_7", "Revolute_8", "Revolute_29"
                    ],
                )
            }
        )
        
        
        def __post_init__(self):
            self.enable_corruption = True # Policy 需要雜訊，增加強健性 (Robustness)
            self.concatenate_terms = True

    # [關鍵新增] 定義 Critic 的觀察空間
    @configclass
    class CriticCfg(PolicyCfg):
        def __post_init__(self):
            # Critic 是上帝視角，不需要雜訊，這會讓 Value Function 估計得更準！
            self.enable_corruption = False 
            self.concatenate_terms = True
        
    

    # 將兩個群組實例化
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg() # 加上這一行，警告就會消失了

@configclass
class EventCfg:
    """Event (Reset & Randomization)"""
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
            "offset_top_grasp_pos": (0.0, 0.06, 0.11),
            "offset_top_grasp_rot": (0.497571, 0.8674232, 0.0, 0.0),
            "offset_side_grasp_pos": (-0.095, 0.035, 0.0),
            "offset_side_grasp_rot": (0.7071068, 0.0, 0.7071068, 0.0),
        },
    )

@configclass
class RewardsCfg:
    reach_surface = RewTerm(
        func=local_mdp.fingertip_surface_distance,
        weight=10.0,
        params={
            "fingertip_cfg": SceneEntityCfg(
                "robot", 
                body_names=[
                    "Component1_1",  # 食指指尖
                    "Component4_1",  # 中指指尖
                    "Component7_1",  # 無名指指尖
                    "Component10_1", # 小拇指指尖
                    "Component23_1"  # 大拇指指尖
                ]
            ),
            "alpha": 20.0,
            "approx_radius": 0.03 # 根據您的球體或圓柱體半徑微調
        }
    )

    # ==========================================
    # 組件二：空間對立與包覆對齊獎勵 (r_align)
    # 鼓勵指尖移動到物體背側形成力封閉
    # ==========================================
    opposition_alignment = RewTerm(
        func=local_mdp.finger_opposition_alignment,
        weight=5.0,
        params={
            "fingertip_cfg": SceneEntityCfg(
                "robot", 
                body_names=[
                    "Component1_1", "Component4_1", "Component7_1", "Component10_1", "Component23_1"
                ]
            ),
            # 請將 "palm_base_link" 替換為您 xacro 檔中實際的手掌/基座連桿名稱
            "palm_cfg": SceneEntityCfg("robot", body_names=["base_link"]) 
        }
    )

    # ==========================================
    # 組件三：虛擬力封閉獎勵 (r_force)
    # 在接觸後激勵持續施加向內的握力
    # ==========================================
    virtual_force = RewTerm(
        func=local_mdp.virtual_force_closure,
        weight=2.0,
        params={
            "fingertip_cfg": SceneEntityCfg(
                "robot", 
                body_names=[
                    "Component1_1", "Component4_1", "Component7_1", "Component10_1", "Component23_1"
                ]
            ),
            "joint_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    "Revolute_12", "Revolute_14", "Revolute_16", "Revolute_18",
                    "Revolute_20", "Revolute_22", "Revolute_24", "Revolute_26",
                    "Revolute_7", "Revolute_8", "Revolute_29"
                ],
            ),
            "contact_threshold": 0.04
        }
    )

    # ==========================================
    # 組件四：階段性提升獎勵 (r_lift)
    # ==========================================
    object_lifted = RewTerm(
        func=local_mdp.object_is_lifted_by_type, # 改用新的函數
        weight=50.0,
        params={
            # 根據上面的計算填入數值
            "height_cube": 0.8,     # 方塊比較矮
            "height_sphere": 0.79,    # 球體中等
            "height_cylinder": 0.875,  # 圓柱本身就很高，閾值要設高一點，不然還沒離地就拿分了
        },
    )

    # ==========================================
    # 組件五：可操作性懲罰 (r_penalty)
    # 懲罰過大的動作變化率以防止手指震顫
    # ==========================================
    action_rate = RewTerm(
        func=local_mdp.action_rate_penalty,
        weight=-0.05,
    )
    
@configclass
class TerminationsCfg:
    """Termination Conditions"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropped = DoneTerm(
        func=local_mdp.object_dropped,
        params={"threshold": 0.7},     
    )

##
# Environment Configuration
##

@configclass
class HumanoidHandEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Humanoid Hand RL Environment."""
    
    scene: HumanoidHandSceneCfg = HumanoidHandSceneCfg(num_envs=2, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    episode_length_s = 5.0
    decimation = 5
    
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005, 
        render_interval=5, 
        physx=sim_utils.PhysxCfg(
            # 修正後的參數名稱（注意加上 max_ 前綴）
            max_position_iteration_count=4,
            max_velocity_iteration_count=1,
            
            # 解決你之前警告的核心參數
            enable_external_forces_every_iteration=True,
            
            # 建議同時設定 solver 類型，TGS 通常比 PGS 更穩定
            solver_type=1, # 0: PGS, 1: TGS (Temporal Gauss-Seidel)
            
            # 針對接觸密集任務的緩衝區設定
            gpu_max_rigid_contact_count=2**21,
            gpu_max_rigid_patch_count=2**18,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import math
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_robot_to_ready_pose(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
):
    robot: Articulation = env.scene[robot_cfg.name]

    # [修正] 確保維度正確 (num_envs, num_dof)
    # Franka = 9 DOFs (7 arm + 2 gripper)
    # 這裡的 ready_joints 只有 9 個值，如果你的 Franka 資源有不同關節數要調整
    ready_joints = torch.tensor(
        [0.0, 0.4, 0.0, -2.4, 0.0, 2.7, 0.785, 0.04, 0.04], 
        device=env.device
    )
    
    # 擴展到 (num_envs, 9)
    target_joints = ready_joints.repeat(len(env_ids), 1)

    # 加一點隨機雜訊
    noise = (torch.rand_like(target_joints) - 0.5) * 0.05
    target_joints += noise
    
    # [關鍵] 夾爪 (最後兩個) 設為開啟 (0.04)
    target_joints[:, 7] = 0.04
    target_joints[:, 8] = 0.04

    # [關鍵] 設定位置，並且將速度歸零！
    target_vel = torch.zeros_like(target_joints)
    
    # 寫入模擬器
    robot.write_joint_state_to_sim(target_joints, target_vel, env_ids=env_ids)

def reset_object_under_hand(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    offset_range: tuple = (0.1, 0.1), # x, y 的隨機範圍
):
    """
    [配合預夾取]
    將物體隨機生成在「手掌正下方」的範圍內。
    這保證了每次開始 RL 時，物體都在相機視野內，且手在物體上方。
    """
    # 1. 取得機器人 End-Effector (手掌) 的當前位置
    # 這裡我們用一個簡單的預估值，因為上面的 reset_robot_to_ready_pose 已經將手固定在某個區域
    # Franka Ready Pose 的 End-Effector 大約在 (0.5, 0.0, 0.5) (相對於 World)
    # 我們直接將物體生成在 (0.5, 0.0) 附近的桌面上
    
    center_x, center_y = 0.5, 0.0
    
    # 2. 隨機生成物體位置
    obj_pos = torch.zeros((len(env_ids), 3), device=env.device)
    
    # X 軸範圍 (手掌下方前後)
    obj_pos[:, 0] = center_x + (torch.rand(len(env_ids), device=env.device) - 0.5) * offset_range[0] * 2
    # Y 軸範圍 (手掌下方左右)
    obj_pos[:, 1] = center_y + (torch.rand(len(env_ids), device=env.device) - 0.5) * offset_range[1] * 2
    # Z 軸 (桌面高度 + 物體半高) -> 假設桌高 0.0 (相對), 物體半高 0.025
    # 如果桌子有厚度，請加上桌子高度
    obj_pos[:, 2] = table_height = 0.8 
    obj_pos[:, 2] = table_height + 0.025 + 0.005

    # 3. 隨機旋轉 (繞 Z 軸)
    obj_rot = torch.zeros((len(env_ids), 4), device=env.device)
    obj_rot[:, 0] = 1.0 # w
    # (簡單起見先不轉，或者你可以加 random yaw)

    # 4. 寫入模擬器
    object_tensor: RigidObject = env.scene[object_cfg.name]
    # 注意：這裡假設 root_state 是 Global 的
    # 如果物體是 spawn 在 env 下，需要加上 env_origins
    root_state = object_tensor.data.default_root_state[env_ids].clone()
    
    # 加上環境原點 (這是最重要的一步，不然所有物體都在世界中心)
    env_origins = env.scene.env_origins[env_ids]
    root_state[:, :3] = obj_pos + env_origins
    
    object_tensor.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
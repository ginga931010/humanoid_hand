# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import TYPE_CHECKING, List

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject, Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_active_object(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    object_names: List[str],
    asset_cfg: SceneEntityCfg = None,
):
    """
    隨機選擇一個物體放到桌上，其他的移到地底。
    [修正] 加入 env_origins 補償，確保物體分佈在各個環境的桌子上。
    """
    # 1. 初始化 ID 紀錄
    if not hasattr(env, "object_type_id"):
        env.object_type_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # 2. 為需要 reset 的環境隨機生成物體 ID
    num_choices = len(object_names)
    random_ids = torch.randint(0, num_choices, (len(env_ids),), device=env.device)
    
    # 更新環境變數
    env.object_type_id[env_ids] = random_ids

    # [關鍵修正] 取得環境的原點座標 (Env Origins)
    # env_origins: (num_envs, 3) -> 每個環境在世界中的 (x, y, z) 偏移量
    env_origins = env.scene.env_origins[env_ids]

    # 3. 執行移動邏輯
    for i, name in enumerate(object_names):
        # 從 scene 取得物件
        obj = env.scene[name]
        
        # 取得預設狀態 (這通常是 Local 座標，或是未包含 env_spacing 的座標)
        default_root_state = obj.data.default_root_state[env_ids].clone()
        
        # 判斷選中狀態
        is_selected = (random_ids == i)
        
        # 準備新的位置
        new_pos = default_root_state[:, :3]
        new_rot = default_root_state[:, 3:7]
        
        # A. 如果是被選中的 -> 放在桌上
        if is_selected.any():
            # [關鍵修正] 加上環境原點偏移量！
            # 這樣 Local (0.5, 0, 0) + Origin (10, 0, 0) = World (10.5, 0, 0)
            new_pos[is_selected] += env_origins[is_selected]

            # 加入隨機噪聲
            pos_noise = (torch.rand((is_selected.sum(), 3), device=env.device) - 0.5) * 0.04
            pos_noise[:, 2] = 0.0 # Z 軸不加噪聲
            new_pos[is_selected] += pos_noise
        
        # B. 如果沒被選中 -> 移到地底
        is_not_selected = ~is_selected
        if is_not_selected.any():
            # 移到地底不需要加 env_origins，反正看不見
            new_pos[is_not_selected, 2] = -10.0
            
        # 4. 寫入模擬器
        obj.write_root_pose_to_sim(torch.cat([new_pos, new_rot], dim=-1), env_ids=env_ids)


def reset_hand_pose_conditional(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    offset_top_grasp_pos: tuple,
    offset_top_grasp_rot: tuple,
    offset_side_grasp_pos: tuple,
    offset_side_grasp_rot: tuple,
):
    """
    根據當前選中的物體類型，將手臂(手掌)瞬移到對應的 Pre-grasp Pose。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    
    if not hasattr(env, "object_type_id"):
        return 

    active_ids = env.object_type_id[env_ids]
    
    # 參數準備
    obj_names = ["object_cube", "object_sphere", "object_cylinder"]
    target_pos = torch.zeros((len(env_ids), 3), device=env.device)
    target_rot = torch.zeros((len(env_ids), 4), device=env.device)
    
    off_top_p = torch.tensor(offset_top_grasp_pos, device=env.device)
    off_top_r = torch.tensor(offset_top_grasp_rot, device=env.device)
    off_side_p = torch.tensor(offset_side_grasp_pos, device=env.device)
    off_side_r = torch.tensor(offset_side_grasp_rot, device=env.device)

    # 填入目標位置
    for i, name in enumerate(obj_names):
        mask = (active_ids == i)
        if not mask.any():
            continue
            
        # [注意] 這裡是讀取 root_pos_w (World Position)
        # 如果上面的 reset_active_object 成功把物體移到正確的世界座標
        # 這裡讀到的就會是正確的座標，手就會跟著去正確的地方
        obj_pos_w = env.scene[name].data.root_pos_w[env_ids]
        
        target_pos[mask] = obj_pos_w[mask]
        
        if i == 2: # Cylinder
            target_pos[mask] += off_side_p
            target_rot[mask] = off_side_r
        else: # Cube/Sphere
            target_pos[mask] += off_top_p
            target_rot[mask] = off_top_r

    # 加入隨機噪聲
    pos_noise = (torch.rand_like(target_pos) - 0.5) * 0.01
    target_pos += pos_noise
    
    # 寫入模擬器
    robot.write_root_pose_to_sim(torch.cat([target_pos, target_rot], dim=-1), env_ids=env_ids)
    
    # 重置關節
    default_joints = robot.data.default_joint_pos[env_ids].clone()
    robot.write_joint_state_to_sim(default_joints, torch.zeros_like(default_joints), env_ids=env_ids)
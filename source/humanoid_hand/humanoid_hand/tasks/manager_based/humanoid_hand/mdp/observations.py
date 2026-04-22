# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_position_dynamic(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    回傳「當前桌面上那個物體」的世界座標。
    修正：移除了 unused 的 sensor_cfg 參數，解決 ValueError
    """
    # 初始化
    if not hasattr(env, "object_type_id"):
        env.object_type_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    obj_names = ["object_cube", "object_sphere", "object_cylinder"]
    target_pos = torch.zeros((env.num_envs, 3), device=env.device)
    
    # 根據 ID 填入對應物體的座標
    for i, name in enumerate(obj_names):
        env_indices = (env.object_type_id == i).nonzero(as_tuple=True)[0]
        if len(env_indices) > 0:
            obj = env.scene[name]
            target_pos[env_indices] = obj.data.root_pos_w[env_indices]

    return target_pos

def object_type_one_hot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    回傳物體類型的 One-hot 向量。
    修正：移除了 unused 的 sensor_cfg 參數，解決 ValueError
    """
    if not hasattr(env, "object_type_id"):
        env.object_type_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        
    num_types = 3 
    return torch.nn.functional.one_hot(env.object_type_id, num_classes=num_types).float()

def joint_position_error(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    [虛擬觸覺] 計算關節位置誤差 (Target Position - Current Position)
    當手指被物體擋住時，這個誤差值會變大，形同觸覺感測。
    """
    # 取得機器人 asset
    asset = env.scene[asset_cfg.name]

    # 取得「目標位置」與「當前位置」
    # joint_pos_target 是經過 action scale 與 offset 轉換後，實際送到 PD 控制器的目標角度
    target_pos = asset.data.joint_pos_target[:, asset_cfg.joint_ids]
    current_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # 計算並回傳誤差
    error = target_pos - current_pos
    return error

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def tactile_proxy_fusion(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    將 joint_pos_error 與 joint_vel 進行特徵交叉與級聯，作為虛擬觸覺訊號。
    """
    asset = env.scene[asset_cfg.name]
    
    # 獲取當前關節位置與速度 (維度將由 asset_cfg.joint_ids 決定，預期為 11)
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    
    # 使用底層實際的關節目標位置 (joint_pos_target) 來計算誤差，這比使用 action 更準確
    target_pos = asset.data.joint_pos_target[:, asset_cfg.joint_ids]
    
    # 1. 計算關節位置誤差
    joint_pos_error = target_pos - joint_pos
    
    # 2. 特徵交叉 (Feature Crossing): error * exp(-|vel|)
    velocity_penalty = torch.exp(-5.0 * torch.abs(joint_vel))
    contact_signal = joint_pos_error * velocity_penalty
    
    # 3. 級聯 (Concatenation): 回傳維度為 11 + 11 + 11 = 33
    return torch.cat([joint_pos_error, joint_vel, contact_signal], dim=-1)


def active_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    [Sim-to-Real 關鍵] 僅讀取主動關節 (有裝設馬達的關節) 的實際角度。
    這能確保 RL Agent 的輸入維度與實機感測器 (Encoder) 讀數完全一致，
    避免神經網路過度依賴現實中無法觀測的被動連桿數據。
    """
    # 取得機器人 asset
    asset = env.scene[asset_cfg.name]
    
    # 透過 asset_cfg.joint_ids，只提取我們指定的 11 個馬達關節的數值
    # 維度: [num_envs, 11]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]
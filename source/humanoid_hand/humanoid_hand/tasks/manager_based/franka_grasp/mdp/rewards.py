# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_is_lifted(env: "ManagerBasedRLEnv", height_threshold: float, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    如果物體高度超過閾值，回傳 1.0。
    """
    obj = env.scene[object_cfg.name]
    # Z 軸高度
    is_lifted = obj.data.root_pos_w[:, 2] > height_threshold
    return is_lifted.float()

def gripper_distance_reward(env: "ManagerBasedRLEnv", target_cfg: SceneEntityCfg, gripper_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    計算夾爪中心到物體的距離懲罰 (引導手靠近物體)。
    """
    # 取得物體位置
    obj_pos = env.scene[target_cfg.name].data.root_pos_w
    
    # 取得夾爪位置 (End Effector)
    # 這裡假設 gripper_cfg 指向的是夾爪的 frame (body 0)
    hand_pos = env.scene[gripper_cfg.name].data.body_pos_w[:, gripper_cfg.body_ids[0]]
    
    dists = torch.norm(hand_pos - obj_pos, dim=-1)
    # 取所有指尖的平均距離
    mean_dist = torch.mean(dists, dim=-1)
    
    sigma = 0.3
    reward = 1.0 - torch.tanh(mean_dist / sigma)
    if mean_dist < 0.18:
        reward += 1 * (1.0 - mean_dist / 0.2) # 額外獎勵，距離越近越多
    return reward

from isaaclab.utils.math import quat_apply

def align_gripper_to_object(
    env, 
    target_cfg: SceneEntityCfg, 
    gripper_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    [修正版] 指向獎勵
    優化：解決近距離向量翻轉問題，並將獎勵範圍修正為 0~1
    """
    # 1. 取得位置
    obj_pos = env.scene[target_cfg.name].data.root_pos_w
    # 取得 Hand Link 位置 (通常是 panda_hand, 手腕處)
    hand_pos = env.scene[gripper_cfg.name].data.body_pos_w[:, gripper_cfg.body_ids[0]]
    hand_quat = env.scene[gripper_cfg.name].data.body_quat_w[:, gripper_cfg.body_ids[0]]

    # 2. 計算向量與距離
    target_vec = obj_pos - hand_pos
    dist = torch.norm(target_vec, dim=-1)
    
    # 歸一化方向向量
    target_dir = target_vec / (dist.unsqueeze(-1) + 1e-6)

    # 3. 計算夾爪指向 (Franka Z-axis)
    z_axis = torch.zeros_like(target_dir)
    z_axis[:, 2] = 1.0
    # [修正] 使用 isaaclab 標準函數 quat_apply
    gripper_dir = quat_apply(hand_quat, z_axis)

    # 4. 計算 Dot Product
    dot_prod = torch.sum(target_dir * gripper_dir, dim=-1)

    # 5. [關鍵修正] 獎勵整形 (Reward Shaping)
    
    # A. 範圍轉換 (-1~1 -> 0~1)
    # 使用 (dot + 1) / 2 雖然是線性的，但 0.5 * (dot + 1) 比較溫和
    # 或者用更激勵的指數函數: reward = dot^2 (但要注意方向)
    # 這裡推薦使用 Power 函數，讓只有 "很準" 的時候才給高分，稍微歪掉就給低分
    # 例如：((dot + 1) / 2) ^ 2
    # 背對(dot=-1) -> 0.0
    # 垂直(dot= 0) -> 0.25
    # 對準(dot= 1) -> 1.0
    align_reward = torch.pow((dot_prod + 1.0) / 2.0, 2)

    # B. [防翻轉機制] Singularity Protection
    # 當手掌中心距離物體非常近 (例如 < 2cm) 時，向量方向會變得不可靠 (或翻轉)
    # 這時候我們假設 "既然這麼近了，角度應該是對的"，直接給最大獎勵
    # 這樣可以避免機器人成功夾取時反而被扣分
    close_mask = dist < 0.02
    align_reward[close_mask] = 1.0
    
    return align_reward
def grasp_shaping_reward(
    env, 
    target_cfg: SceneEntityCfg, 
    gripper_cfg: SceneEntityCfg,
    open_threshold: float = 0.08,  # [修正] 設定大於方塊半徑 (方塊寬5cm, 半徑約4cm-7cm包含對角線)
    cube_width: float = 0.05       # 定義方塊寬度
) -> torch.Tensor:
    """
    修正後的夾取引導獎勵：防止 Reward Hacking。
    """
    # 1. 計算距離
    obj_pos = env.scene[target_cfg.name].data.root_pos_w
    hand_pos = env.scene[gripper_cfg.name].data.body_pos_w[:, gripper_cfg.body_ids[0]]
    dist = torch.norm(obj_pos - hand_pos, dim=-1)

    # 2. 取得手指狀態 (單指位置)
    # Franka: 0.04 (Open, Total 8cm), 0.0 (Closed)
    robot = env.scene[gripper_cfg.name]
    finger_pos = robot.data.joint_pos[:, -2:].mean(dim=-1)

    rewards = torch.zeros_like(dist)
    
    # --- 階段一：靠近階段 (Approach Phase) ---
    # 條件：距離 > 0.08 (稍微大於方塊，避免還沒對準就想夾)
    # 策略：這時候手必須是開的！如果手是關的，給予懲罰。
    # 為什麼用懲罰？ -> 避免它為了拿 "張開獎勵" 而故意不靠近。
    # 現在它在遠處張開手只能拿到 0 分，唯一能拿正分的方法是 "靠近"。
    
    is_far = dist > open_threshold
    # 如果在遠處，且手指小於 0.035 (代表沒全開)，給予負分
    # 0.035 * 2 = 7cm > 5cm (方塊)，確保能容納方塊
    rewards[is_far] = torch.where(
        finger_pos[is_far] < 0.035, 
        0.0, # 懲罰
        0.0   # 沒事
    )

    # --- 階段二：夾取階段 (Grasp Phase) ---
    # 條件：距離 < 0.08
    # 策略：鼓勵手指位置接近 "方塊寬度"。
    # 方塊寬 0.05 -> 單指應在 0.025 左右。
    # 如果 finger_pos < 0.03 (開始閉合) 且 > 0.01 (沒夾空)，給獎勵。
    
    is_near = ~is_far
    
    # 判斷是否正在嘗試夾取 (手指位置小於 0.035)
    is_closing = (finger_pos < 0.035) & (finger_pos > 0.025)
    
    # [關鍵] 只有在 "近" 且 "嘗試夾" 的時候給正分
    rewards[is_near] = torch.where(
        is_closing[is_near],
        1.0, # 給予引導獎勵
        0.0   
    )
    
    # [進階] 如果真的夾到了 (0.02 < finger_pos < 0.03)，給更高分
    # 這代表它夾住了方塊 (物理引擎擋住了手指)
    is_grasping_obj = (finger_pos > 0.0275) & (finger_pos < 0.0325)
    rewards[is_near & is_grasping_obj] += 0.5

    return rewards


def object_is_lifted_v2(
    env, 
    object_cfg: SceneEntityCfg, 
    gripper_cfg: SceneEntityCfg,
    height_threshold: float = 0.85, # 桌子0.8 + 5cm
    dist_threshold: float = 0.12    # [關鍵] 必須在手心附近
) -> torch.Tensor:
    
    # 1. 檢查高度
    obj_pos = env.scene[object_cfg.name].data.root_pos_w
    is_high_enough = obj_pos[:, 2] > height_threshold
    
    # 2. [新增] 檢查距離 (防止撞飛騙分)
    hand_pos = env.scene[gripper_cfg.name].data.body_pos_w[:, gripper_cfg.body_ids[0]]
    dist = torch.norm(obj_pos - hand_pos, dim=-1)
    is_close_to_hand = dist < dist_threshold
    
    # 只有 "夠高" 且 "在手裡" 才算分
    return (is_high_enough & is_close_to_hand).float()
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def _get_active_object_pose(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """
    [Helper] 取得「當前桌面上那個物體」的位置與旋轉 (World Frame)。
    回傳: (position, rotation)
    """
    if not hasattr(env, "object_type_id"):
        env.object_type_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # 預設三個物體名稱 (需與 Scene 設定一致)
    obj_names = ["object_cube", "object_sphere", "object_cylinder"]
    
    # 準備容器
    target_pos = torch.zeros((env.num_envs, 3), device=env.device)
    target_rot = torch.zeros((env.num_envs, 4), device=env.device)
    
    # 根據 ID 填入對應物體的 Pose
    for i, name in enumerate(obj_names):
        env_indices = (env.object_type_id == i).nonzero(as_tuple=True)[0]
        if len(env_indices) > 0:
            obj = env.scene[name]
            target_pos[env_indices] = obj.data.root_pos_w[env_indices]
            target_rot[env_indices] = obj.data.root_quat_w[env_indices]

    return target_pos, target_rot


def fingertip_object_distance(
    env: "ManagerBasedRLEnv", 
    std: float, 
    minimal_height: float, 
    fingertip_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    計算指尖到物體的平均距離，並轉化為獎勵 (距離越近獎勵越高)。
    
    Args:
        std: 獎勵函數的寬度 (Sensitivity)，越小代表要求越精確。
        minimal_height: 只有當手掌高於此高度時才開始計算獎勵 (避免手在桌下亂抓)。
        fingertip_cfg: 定義哪些 link 是指尖。
    """
    # 1. 取得指尖位置 (World Frame)
    # body_pos_w 形狀: (num_envs, num_fingertips, 3)
    fingertip_pos = env.scene[fingertip_cfg.name].data.body_pos_w[:, fingertip_cfg.body_ids]
    
    # 2. 取得當前目標物位置
    object_pos, _ = _get_active_object_pose(env)
    
    # 3. 計算距離
    # 讓 object_pos 擴展維度以配合指尖數量: (num_envs, 1, 3)
    target_pos_expanded = object_pos.unsqueeze(1)
    
    # 計算每個指尖到物體中心的歐式距離
    # shape: (num_envs, num_fingertips)
    dists = torch.norm(fingertip_pos - target_pos_expanded, dim=-1)
    
    # 取所有指尖的平均距離
    mean_dist = torch.mean(dists, dim=-1)
    
    # 4. 計算獎勵 (使用 Tanh Kernel)
    # 距離為 0 時獎勵為 1，距離無限大時獎勵為 0
    reward = 1.0 - torch.tanh(mean_dist / std)
    
    # 5. 過濾條件：如果手太低 (還沒準備好)，就不給獎勵，避免它學會趴在地上
    robot = env.scene["robot"]
    # 假設 root frame 的 Z 軸代表手掌高度
    hand_height = robot.data.root_pos_w[:, 2]
    reward = torch.where(hand_height > minimal_height, reward, torch.zeros_like(reward))
    
    return reward


def object_is_lifted_by_type(
    env: "ManagerBasedRLEnv", 
    height_cube: float,
    height_sphere: float,
    height_cylinder: float,
) -> torch.Tensor:
    """
    [Binary Reward] 根據物體類型，分別判斷是否達到指定的抬起高度。
    
    Args:
        height_cube: 方塊的判定高度閾值 (World Z)
        height_sphere: 球體的判定高度閾值 (World Z)
        height_cylinder: 圓柱體的判定高度閾值 (World Z)
    """
    # 1. 取得物體位置
    # (假設 _get_active_object_pose 已經在同一個檔案中定義過)
    object_pos, _ = _get_active_object_pose(env)
    
    # 2. 確保 object_type_id 存在
    if not hasattr(env, "object_type_id"):
        env.object_type_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        
    # 3. 建立動態閾值 Tensor
    # 先建立一個跟 env 數量一樣大的 Tensor
    thresholds = torch.zeros(env.num_envs, device=env.device)
    
    # 根據 ID 填入對應的閾值
    # ID 0: Cube
    thresholds[env.object_type_id == 0] = height_cube
    # ID 1: Sphere
    thresholds[env.object_type_id == 1] = height_sphere
    # ID 2: Cylinder
    thresholds[env.object_type_id == 2] = height_cylinder
    
    # 4. 判斷高度 (Z軸) 是否大於對應的閾值
    is_lifted = (object_pos[:, 2] > thresholds)
    
    return is_lifted.float()

def object_height_continuous(
    env: "ManagerBasedRLEnv", 
    initial_height: float,
    max_height: float
) -> torch.Tensor:
    """
    [Continuous Reward] 物體越高，獎勵越多 (引導它往上抬)。
    """
    object_pos, _ = _get_active_object_pose(env)
    
    # 計算相對於桌面的高度增量
    height_diff = object_pos[:, 2] - initial_height
    
    # 限制在 [0, max_height] 區間，避免它把東西拋飛到天花板來刷分
    height_reward = torch.clamp(height_diff, min=0.0, max=max_height)
    
    return height_reward









import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
# 假設您的檔案中有此 helper function 
# from.helpers import _get_active_object_pose 

# ==========================================
# 組件一：表面感知之接近獎勵 (r_reach)
# ==========================================
def fingertip_surface_distance(
    env: ManagerBasedRLEnv, 
    fingertip_cfg: SceneEntityCfg, 
    alpha: float = 20.0,
    approx_radius: float = 0.03
) -> torch.Tensor:
    """
    計算指尖至物體表面的指數衰減距離獎勵。
    """
    # 獲取動態目標物位置
    object_pose, _ = _get_active_object_pose(env)
    object_pos = object_pose[:, :3] 
    
    # 獲取指尖位置
    fingertip_asset = env.scene[fingertip_cfg.name]
    fingertip_pos = fingertip_asset.data.body_pos_w[:, fingertip_cfg.body_ids, :3]
    
    # 計算指尖到物體中心的距離
    dist_to_center = torch.norm(fingertip_pos - object_pos.unsqueeze(1), dim=-1)
    
    # 近似計算到表面的距離 (扣除物體半徑，並限制最小值為 0 避免穿透獎勵)
    surface_dist = torch.clamp(dist_to_center - approx_radius, min=0.0)
    
    # 使用指數衰減 (Exponential Decay) 取代 Tanh
    reward = torch.exp(-alpha * surface_dist).mean(dim=1)
    return reward

# ==========================================
# 組件二：空間對立與包覆對齊獎勵 (r_align)
# ==========================================
# ==========================================
# 組件二：空間對立與包覆對齊獎勵 (r_align) (修正版)
# ==========================================
def finger_opposition_alignment(
    env: ManagerBasedRLEnv,
    fingertip_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    鼓勵指尖移動到物體相對於手掌的「背面」，形成力封閉包覆的餘弦獎勵。
    """
    object_pose, _= _get_active_object_pose(env)
    object_pos = object_pose[:, :3] # [num_envs, 3]
    
    palm_asset = env.scene[palm_cfg.name]
    # 【修正點】：加入  提取單一連桿，確保維度為 [num_envs, 3] 而非 [num_envs, 1, 3]
    palm_pos = palm_asset.data.body_pos_w[:, palm_cfg.body_ids, :3].squeeze(1) # [num_envs, 3] 
    
    fingertip_asset = env.scene[fingertip_cfg.name]
    fingertip_pos = fingertip_asset.data.body_pos_w[:, fingertip_cfg.body_ids, :3] # [num_envs, 5, 3]
    
    # 計算手掌指向物體的向量 (v_palm)
    v_palm_to_obj = object_pos - palm_pos # [num_envs, 3]
    v_palm_to_obj_norm = v_palm_to_obj / (torch.norm(v_palm_to_obj, dim=-1, keepdim=True) + 1e-6)
    
    # 計算物體指向各指尖的向量 (v_tip)
    v_obj_to_tip = fingertip_pos - object_pos.unsqueeze(1) # [num_envs, 5, 3]
    v_obj_to_tip_norm = v_obj_to_tip / (torch.norm(v_obj_to_tip, dim=-1, keepdim=True) + 1e-6)
    
    # 計算內積 (Cosine Similarity)
    cos_theta = torch.sum(v_palm_to_obj_norm.unsqueeze(1) * v_obj_to_tip_norm, dim=-1) # [num_envs, 5]
    
    # 只有當餘弦值為正 (代表指尖在物體背側) 時才給予獎勵，並沿著手指維度加總
    reward = torch.clamp(cos_theta, min=0.0).sum(dim=1) # [num_envs]
    return reward


# ==========================================
# 組件三：虛擬力封閉獎勵 (r_force) (修正版)
# ==========================================
def virtual_force_closure(
    env: ManagerBasedRLEnv,
    fingertip_cfg: SceneEntityCfg,
    joint_cfg: SceneEntityCfg,
    contact_threshold: float = 0.04
) -> torch.Tensor:
    """
    僅當指尖距離物體極近時，獎勵正向的關節位置誤差，激勵持續施加向內的握力。
    """
    object_pose, _ = _get_active_object_pose(env)
    object_pos = object_pose[:, :3]
    
    fingertip_asset = env.scene[fingertip_cfg.name]
    fingertip_pos = fingertip_asset.data.body_pos_w[:, fingertip_cfg.body_ids, :3]
    
    # 判斷指尖群的平均距離是否小於接觸閾值 (物體半徑 + 容差)
    mean_dist = torch.norm(fingertip_pos - object_pos.unsqueeze(1), dim=-1).mean(dim=1)
    is_close = (mean_dist < contact_threshold).float() # [num_envs]
    
    robot = env.scene[joint_cfg.name]
    joint_pos = robot.data.joint_pos[:, joint_cfg.joint_ids]
    
    # 【修正點】：使用底層實際的目標關節位置，取代歸一化在 [-1, 1] 的 action，以確保誤差計算的物理意義
    target_pos = robot.data.joint_pos_target[:, joint_cfg.joint_ids]
    
    # 計算絕對誤差並取平均
    pos_error = torch.abs(target_pos - joint_pos).mean(dim=1) # [num_envs]
    
    # 只有在 is_close = 1 時，誤差才會轉化為獎勵
    return is_close * pos_error # [num_envs]

# ==========================================
# 組件四：階段性提升獎勵 (r_lift)
# ==========================================
def phased_lifting_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.2,
    initial_height: float = 0.05,
    dense_weight: float = 1.0,
    bonus_weight: float = 50.0
) -> torch.Tensor:
    """
    結合密集高度追蹤與巨額成功紅利的階段性提升獎勵。
    """
    object_pose, _ = _get_active_object_pose(env)
    z_current = object_pose[:, 2]
    
    # 計算 0 到 1 的密集獎勵 (依據高度進度)
    progress = (z_current - initial_height) / (target_height - initial_height)
    dense_reward = torch.clamp(progress, min=0.0, max=1.0) * dense_weight
    
    # 達到目標高度時給予一次性紅利
    success_bonus = (z_current >= target_height).float() * bonus_weight
    
    return dense_reward + success_bonus

# ==========================================
# 組件五：可操作性懲罰 (r_penalty)
# ==========================================
def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    懲罰連續時間步驟間過大的動作變化率，確保物理合理性並防止震顫 。
    """
    # 取當前 action 與前一次 action
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    
    # 平方和懲罰
    penalty = torch.sum(torch.square(action - prev_action), dim=1)
    return penalty
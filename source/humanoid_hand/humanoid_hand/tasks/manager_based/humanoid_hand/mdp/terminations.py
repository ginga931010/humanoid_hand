# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_dropped(env: "ManagerBasedRLEnv", threshold: float) -> torch.Tensor:
    """
    [Termination] 當「當前選中的物體」高度低於閾值，或水平範圍離開桌面時，視為掉落並終止。
    
    Args:
        threshold (float): 高度閾值 (World Z)。桌子高度是 0.75，建議設為 0.4~0.5。
    """
    # 1. 確保有 ID 紀錄
    if not hasattr(env, "object_type_id"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    obj_names = ["object_cube", "object_sphere", "object_cylinder"]
    
    # 2. 建立 Tensor 儲存每個環境中 Active Object 的 XYZ 座標
    # 高度預設為 10.0 (安全值)，XY 預設為桌子中心 (0.5, 0.0)
    active_obj_z = torch.ones(env.num_envs, device=env.device) * 10.0
    active_obj_x = torch.ones(env.num_envs, device=env.device) * 0.5
    active_obj_y = torch.zeros(env.num_envs, device=env.device)
    
    for i, name in enumerate(obj_names):
        env_indices = (env.object_type_id == i).nonzero(as_tuple=True)[0]
        
        if len(env_indices) > 0:
            obj = env.scene[name]
            # 一次抓取 XYZ 座標 (shape: [num_env_indices, 3])
            pos = obj.data.root_pos_w[env_indices] - env.scene.env_origins[env_indices]
            
            active_obj_x[env_indices] = pos[:, 0]
            active_obj_y[env_indices] = pos[:, 1]
            active_obj_z[env_indices] = pos[:, 2]

    # 3. 條件 A: 判斷是否低於高度閾值
    is_below_height = active_obj_z < threshold
    
    # 4. 條件 B: 判斷是否離開桌面範圍
    # 桌子中心: (0.5, 0.0), 尺寸: (0.8, 0.8)
    # 理論邊界: X (0.1 ~ 0.9), Y (-0.4 ~ 0.4)
    # 我們向外加一點點緩衝區 (buffer=0.05)，避免物體剛好掛在邊緣就被判定失敗
    buffer = 0.05
    min_x, max_x = 0.1 - buffer, 0.9 + buffer
    min_y, max_y = -0.4 - buffer, 0.4 + buffer
    
    is_off_table_x = (active_obj_x < min_x) | (active_obj_x > max_x)
    is_off_table_y = (active_obj_y < min_y) | (active_obj_y > max_y)
    is_off_table = is_off_table_x | is_off_table_y

    # 5. 結合條件：只要「太低」或是「離開桌面」，都判定為掉落 (終止)
    return is_below_height | is_off_table
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
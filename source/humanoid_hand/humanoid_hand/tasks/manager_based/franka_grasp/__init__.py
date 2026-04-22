# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from .franka_grasp_env_cfg import FrankaGraspEnvCfg
from .agents.rsl_rl_ppo_cfg import FrankaGraspPPORunnerCfg
from . import agents
env_cfg = FrankaGraspEnvCfg()
gym.register(
    id="Play-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # [關鍵修正] 使用 "cfg" 作為 key，並傳入實例化物件
        # ManagerBasedRLEnv.__init__(cfg=...) 需要這個參數
        "cfg": env_cfg,
        
        # 這個參數是用來給 train.py 讀取的，Environment 本身不會用到，但傳進去無妨
        "rsl_rl_cfg_entry_point": FrankaGraspPPORunnerCfg,
    },
)
# 註冊環境
gym.register(
    id="Franka-Grasp-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaGraspEnvCfg,
        "rsl_rl_cfg_entry_point": FrankaGraspPPORunnerCfg,
    },
)
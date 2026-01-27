# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import gymnasium as gym

# from . import agents

# ##
# # Register Gym environments.
# ##


# gym.register(
#     id="Humanoid-Hand-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.humanoid_hand_env_cfg:HumanoidHandEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
#     },
# )
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

# 引入你的 Config Class
from .humanoid_hand_env_cfg import HumanoidHandEnvCfg

# 如果你有用 RSL_RL，也引入它的 Config (但不要放在 gym.register 的 kwargs 裡傳給 env)
from .agents.rsl_rl_ppo_cfg import PPORunnerCfg

gym.register(
    id="Humanoid-Hand-v0",  # 請確保這裡的 ID 跟你在 play_hand.py 呼叫的一模一樣
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # [關鍵修正 1] Key 必須是 "cfg"
        # [關鍵修正 2] Value 必須是實例化後的物件 (加上括號)
        "cfg": HumanoidHandEnvCfg(),
        
        # 如果你需要把 RSL_RL 的設定綁定在這個 Env ID 上，
        # 通常不建議放在 kwargs 傳給 Env (因為 Env 看不懂)，
        # 而是由 train.py 透過 gym.spec(env_id).kwargs.get("rsl_rl_cfg_entry_point") 去抓。
        # 但如果你的 Env 沒報錯說 "unexpected keyword argument"，留著也行。
        # 為了保險起見，建議先只留 cfg。
    },
)
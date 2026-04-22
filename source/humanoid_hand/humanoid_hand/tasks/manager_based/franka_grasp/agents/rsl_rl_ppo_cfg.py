# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class FrankaGraspPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32 # 增加步數，因為視覺訓練需要更長的序列回饋
    max_iterations = 2000
    save_interval = 500
    experiment_name = "franka_visual_grasp"
    empirical_normalization = True # [重要] 開啟歸一化，這對 CNN 和 MLP 混合輸入很重要

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        
        # RSL_RL 會自動偵測：
        # Policy input 包含 image -> 使用 NatureCNN backbone
        # Critic input 是 vector -> 使用 MLP
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01, # 保持一定的探索
        num_learning_epochs=2,
        num_mini_batches=4, # 增加 batch 切分，減少顯存佔用 (因為有圖片)
        learning_rate=1e-5, # 降低學習率，視覺訓練較不穩定
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
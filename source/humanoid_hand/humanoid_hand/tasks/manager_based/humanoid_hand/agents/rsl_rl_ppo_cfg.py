# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 500
    save_interval = 250
    experiment_name = "cartpole_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class HumanoidHandPPORunnerCfg1(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 1500
    save_interval = 250
    experiment_name = "humanoid_hand"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class HumanoidHandPPORunnerCfg2(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24  # 每個環境採樣步數 (24 * 0.01s * 2 = 0.48s 的軌跡)
    max_iterations = 500   # 總訓練迭代次數
    save_interval = 250     # 每 100 次迭代存檔一次

    # 1. 實驗監控
    experiment_name = "humanoid_hand_grasp"
    run_name = ""
    logger = "tensorboard" # 或 "wandb"
    
    # 2. 策略網路結構 (Policy Network)
    # 這裡的設計是關鍵
    policy = RslRlPpoActorCriticCfg(
        # [初始探索噪聲] 設大一點 (1.0)，讓它一開始動作大一點，才不會卡在原地
        init_noise_std=1.0, 
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # [網路層數] [512, 256, 128] 是抓取任務的黃金比例
        # 為什麼？因為抓取涉及摩擦力，是非線性的，需要一定深度來擬合
        # 但也不要太深 (e.g. 1024, 1024)，那樣訓練太慢且難收斂
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        
        # [激活函數] 推薦 ELU，比 ReLU 平滑，對連續控制(馬達)比較友善
        activation="elu", 
    )

    # 3. PPO 演算法參數
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        
        # [熵係數 Entropy] 這是最關鍵的參數！
        # 0.01 ~ 0.005: 高探索。強迫 Agent 多嘗試奇怪動作 (適合你的大拇指雨刷)
        # < 0.001: 低探索。Agent 會很快趨於保守 (容易學不到轉大拇指)
        entropy_coef=0.01, 
        
        num_learning_epochs=5,
        
        # [Batch Size] 4096 envs * 24 steps / 4 mini_batches = 24576
        # 確保這個數字夠大，梯度估計才準
        num_mini_batches=4,
        
        # [學習率] 3e-4 是標準值
        # 如果發現震盪太厲害，可以降到 1e-4
        learning_rate=3e-4, 
        schedule="adaptive", # 自動調整學習率
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class HumanoidHandV2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 25  # 每個環境採樣步數 (50 * 0.01s * 2 = 1s 的軌跡)
    max_iterations = 200  # 總訓練迭代次數
    save_interval = 100     # 每 100 次迭代存檔一次

    # 1. 實驗監控
    experiment_name = "humanoid_hand_grasp_v2"
    run_name = ""
    logger = "tensorboard" # 或 "wandb"
    clip_actions = 1.0
    # 2. 策略網路結構 (Policy Network)
    # 這裡的設計是關鍵
    policy = RslRlPpoActorCriticCfg(
        # [初始探索噪聲] 設大一點 (1.0)，讓它一開始動作大一點，才不會卡在原地
        init_noise_std=1.0, 
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        
        # [激活函數] 推薦 ELU，比 ReLU 平滑，對連續控制(馬達)比較友善
        activation="elu", 
        
    )

    # 3. PPO 演算法參數
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        
        # [熵係數 Entropy] 這是最關鍵的參數！
        # 0.01 ~ 0.005: 高探索。強迫 Agent 多嘗試奇怪動作 (適合你的大拇指雨刷)
        # < 0.001: 低探索。Agent 會很快趨於保守 (容易學不到轉大拇指)
        entropy_coef=0.01, 
        
        num_learning_epochs=5,
        
        # [Batch Size] 4096 envs * 24 steps / 4 mini_batches = 24576
        # 確保這個數字夠大，梯度估計才準
        num_mini_batches=4,
        
        # [學習率] 3e-4 是標準值
        # 如果發現震盪太厲害，可以降到 1e-4
        learning_rate=1e-4, 
        schedule="adaptive", # 自動調整學習率
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        
    )
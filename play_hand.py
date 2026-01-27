import argparse
from isaaclab.app import AppLauncher

# 1. 啟動 Isaac Sim
parser = argparse.ArgumentParser(description="Test Bionic Hand Env")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 導入必要的庫
import gymnasium as gym
import torch
import source.humanoid_hand.humanoid_hand.tasks.manager_based.humanoid_hand  # 確保這行路徑對應你的資料夾名稱

def main():
    # 建立環境
    env = gym.make("Humanoid-Hand-v0", render_mode="rgb_array")
    
    print("[INFO] Environment created.")
    
    # 取得環境資訊
    # 注意：Isaac Lab 的 action_space 形狀通常是 (num_envs, action_dim)
    action_space_shape = env.unwrapped.action_space.shape
    print(f"[INFO] Action space shape: {action_space_shape}")
    
    # [關鍵修正] 
    # 如果 shape 是 (16, 21)，我們取最後一個維度 [-1] 作為動作數量
    # 如果 shape 是 (21,)，取 [0] 也是 21，所以用 [-1] 最安全
    action_dim = action_space_shape[-1]
    num_envs = env.unwrapped.num_envs
    
    print(f"[INFO] Number of Envs: {num_envs}")
    print(f"[INFO] Action Dimension: {action_dim}")
    
    # Reset
    obs, _ = env.reset()
    
    # 模擬迴圈
    for i in range(100000):
        # [關鍵修正] 使用正確的 action_dim
        actions = torch.rand((num_envs, action_dim), device=env.unwrapped.device) * 2 - 1
        
        # Step
        obs, rew, terminated, truncated, info = env.step(actions)
        
        if i % 2000 == 0:
            print(f"[INFO] Resetting environment at step {i}")
            env.reset()

    env.close()

if __name__ == "__main__":
    main()
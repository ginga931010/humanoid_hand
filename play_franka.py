import argparse
import numpy as np
import os
import cv2
from isaaclab.app import AppLauncher

# 1. 啟動 Isaac Sim
parser = argparse.ArgumentParser(description="Test Franka Grasp Depth Env")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 導入必要的庫
import gymnasium as gym
import torch

# [重要] 這裡的 import 路徑必須指向你的 franka_grasp 資料夾
# 根據你的 play_hand.py，你的資料夾結構似乎比較深，請確保這行能找到 __init__.py
# 如果報錯 ModuleNotFoundError，請檢查資料夾名稱是否正確
try:
    import source.humanoid_hand.humanoid_hand.tasks.manager_based.franka_grasp
except ImportError:
    # 備用路徑 (如果你是在 source/humanoid_hand 下執行)
    import humanoid_hand.tasks.manager_based.franka_grasp

def main():
    # 建立環境 (使用我們註冊的新 ID)
    # render_mode="rgb_array" 讓 Isaac Sim 在背景渲染相機，這對視覺訓練很重要
    env = gym.make("Play-Franka", render_mode="rgb_array")
    
    print("[INFO] Environment created.")
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG
    from isaaclab.utils.math import quat_rotate
    from isaaclab.managers import SceneEntityCfg
    # --- 機器人檢查 ---
    robot = env.unwrapped.scene["robot"]
    # 取得關節限制
    # Franka 有 7 個手臂關節 + 2 個夾爪關節 = 9 個 DOF
    dof_limits = robot.data.default_joint_pos_limits
    
    print(f"\n[DEBUG] Robot Info Check:")
    print(f"Joint Names: {robot.data.joint_names}")
    # 檢查有沒有抓到關節限制
    if dof_limits is not None:
        print(f"Lower Limits (First 5): {dof_limits[0, :5, 0]}") 
        print(f"Upper Limits (First 5): {dof_limits[0, :5, 1]}")
    
    # --- Action Space 檢查 ---
    action_space_shape = env.unwrapped.action_space.shape
    print(f"\n[INFO] Action space shape: {action_space_shape}")
    
    # 取得動作維度 (預期是 9: 7 arm + 2 gripper)
    action_dim = action_space_shape[-1]
    num_envs = env.unwrapped.num_envs
    
    print(f"[INFO] Number of Envs: {num_envs}")
    print(f"[INFO] Action Dimension: {action_dim}")
    
    # --- Reset & Observation 檢查 ---
    print("\n[INFO] Resetting environment...")
    obs, _ = env.reset()
    
    # [關鍵] 檢查 Policy 的觀測值，確認是否有收到深度圖
    print("\n[DEBUG] Observation Keys & Shapes:")
    if isinstance(obs, dict) and "policy" in obs:
        for key, value in obs["policy"].items():
            print(f"  Policy - {key}: {value.shape}")
            # 驗證深度圖形狀，應該是 (num_envs, 1, 84, 84) 或類似
            if key == "depth_image":
                print(f"    -> Camera Depth Image detected! Resolution: {value.shape[-2]}x{value.shape[-1]}")
    else:
        print("  [Warning] Observation structure is not a dict or missing 'policy' key.")

    # --- 模擬迴圈 ---
    print("\n[INFO] Starting simulation loop...")
    # [新增] 初始化除錯用的座標軸標記
    # FRAME_MARKER_CFG 會畫出紅綠藍三色軸
    # [修正] 初始化 Visual Marker
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    
    # [關鍵修正] 必須指定 USD 路徑，否則會報 Path.__init__ 錯誤
    frame_marker_cfg.prim_path = "/World/Visuals/HandFrame"
    
    # 設定縮放大小
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    
    # 現在這裡不會報錯了
    hand_marker = VisualizationMarkers(frame_marker_cfg)

    # 取得機器人和夾爪的索引 (為了在迴圈裡拿數據)
    # 注意：需使用 env.unwrapped 才能存取底層 scene
    robot_entity = env.unwrapped.scene["robot"]
    # 找出 "panda_hand" 的 body index
    hand_body_ids = robot_entity.find_bodies("panda_hand")[0]
    for i in range(10000):
        # 隨機動作測試
        actions = torch.rand((num_envs, action_dim), device=env.unwrapped.device) * 2 - 1
        
        # Step
        obs, rew, terminated, truncated, info = env.step(actions)
        if i % 2 == 0: # 不需要每一步都畫，省點效能
            # 1. 取得手掌的當前位置與旋轉
            # shape: (num_envs, 3) 和 (num_envs, 4)
            hand_pos = robot_entity.data.body_pos_w[:, hand_body_ids].squeeze(1) # shape: (num_envs, 3)
            hand_quat = robot_entity.data.body_quat_w[:, hand_body_ids].squeeze(1) # shape: (num_envs, 4)
            
            # 2. 視覺化座標軸
            # 這會在手掌的位置畫出它的 X, Y, Z 軸
            hand_marker.visualize(translations=hand_pos, orientations=hand_quat)
        if i % 500 == 0:
            # 1. 取出第一張圖 (env_0)
            # shape: (84, 84, 1)
            depth_tensor = obs["policy"]["depth_image"][0, :, :, 0]
            
            # 2. [關鍵] 印出原始數值的統計資料！
            # 這是唯一能告訴我們相機到底有沒有壞掉的方法
            d_min = depth_tensor.min().item()
            d_max = depth_tensor.max().item()
            d_mean = depth_tensor.mean().item()
            
            print(f"\n[DEBUG Step {i}] Raw Depth Stat -> Min: {d_min:.4f}, Max: {d_max:.4f}, Mean: {d_mean:.4f}")
            
            # 3. 判斷情況
            if d_min == float('inf') or d_mean > 1000:
                print("  [Error] 相機看到無限遠 (Inf)！請檢查相機位置與旋轉 (rot)。")
            elif d_max == 0:
                print("  [Error] 畫面全黑 (0)！相機可能在物體內部或被遮擋。")
            else:
                # 4. 動態歸一化 (Dynamic Normalization)
                # 自動把當前畫面中最亮跟最暗的地方拉開到 0-255
                depth_img = depth_tensor.cpu().numpy()
                
                # 避免分母為 0
                denom = (d_max - d_min) if (d_max - d_min) > 1e-5 else 1.0
                
                # 拉伸對比度：(數值 - 最小值) / 範圍 * 255
                depth_vis = (depth_img - d_min) / denom * 255.0
                
                # 轉成 uint8 並存檔
                depth_vis = np.clip(depth_vis, 0, 255).astype(np.uint8)
                
                # 使用 colormap 讓深度圖更容易看 (變成彩色的熱力圖)
                # 如果沒有 cv2.applyColorMap 也可以只存灰階
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                
                save_path = f"debug_depth_{i:04d}.png"
                cv2.imwrite(save_path, depth_color)
                print(f"  [Success] Saved normalized depth image to {save_path}")
        # 每 100 步印一次平均獎勵，確認 Reward Function 有在運作
        if i % 100 == 0:
            print(f"[Step {i:04d}] Mean Reward: {rew.mean().item():.4f}")
            
            # 檢查是否有環境因為掉落而終止
            num_done = terminated.sum().item()
            if num_done > 0:
                print(f"  -> {num_done} environments terminated (e.g. object dropped).")

        if i % 2000 == 0:
            print(f"[INFO] Force Resetting all environments at step {i}")
            env.reset()

    env.close()

if __name__ == "__main__":
    main()
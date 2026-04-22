import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
def image_flattened(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    data_type: str = "distance_to_image_plane",
    normalize: bool = True
) -> torch.Tensor:
    """
    獲取相機圖像並將其壓扁成 1D 向量，以便餵給 MLP Policy。
    """
    # 1. 呼叫 Isaac Lab 原生的 image 函數取得 (N, H, W, C)
    # 注意：這裡直接重用 mdp.image 的邏輯，或者手動抓 sensor data
    sensor = env.scene.sensors[sensor_cfg.name]
    data = sensor.data.output[data_type]
    
    # 2. 處理歸一化 (深度圖 0~10m -> 0~1)
    if normalize:
        if "distance" in data_type or "depth" in data_type:
            # 假設有效深度範圍是 2.0 公尺 (超過都切掉)
            data = torch.clamp(data, 0.0, 2.0)
            data = data / 2.0
    
    # 3. [關鍵] 壓扁 (Flatten)
    # 輸入: (num_envs, H, W, 1) -> 輸出: (num_envs, H*W*1)
    batch_size = data.shape[0]
    flat_data = data.view(batch_size, -1)
    
    return flat_data
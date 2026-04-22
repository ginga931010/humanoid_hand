import numpy as np
from scipy.spatial.transform import Rotation

def get_lookat_quat(cam_pos, target_pos):
    """
    計算從 cam_pos 看向 target_pos 的四元數 (w, x, y, z)
    假設相機座標系定義 (Convention) 為:
    - Forward: -Z (這是大多數 3D 軟體如 Blender/OpenGL 的相機預設)
    - Up: +Y
    
    但在 Isaac Lab 的 'world' convention 中，我們通常需要將其轉到底層的 World Frame。
    這裡我們計算的是將相機的 -Z 軸對準目標向量的旋轉。
    """
    cam_pos = np.array(cam_pos)
    target_pos = np.array(target_pos)
    
    # 1. 計算方向向量 (Forward Vector)
    # 我們希望相機的 -Z 軸指向目標
    forward = target_pos - cam_pos
    forward = forward / np.linalg.norm(forward)
    
    # 2. 定義 Up Vector (世界座標的上方)
    up = np.array([0.0, 0.0, 1.0])
    
    # 3. 建立旋轉矩陣 (LookAt Matrix)
    # Z-axis: 指向相機後方 (-forward)
    z_axis = -forward
    # X-axis: Right (up cross z)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    # Y-axis: Up (z cross x)
    y_axis = np.cross(z_axis, x_axis)
    
    # 旋轉矩陣 R (Column-major)
    R = np.eye(3)
    R[:, 0] = x_axis
    R[:, 1] = y_axis
    R[:, 2] = z_axis
    
    # 4. 轉成 Quaternion (Scipy 輸出是 x, y, z, w)
    r = Rotation.from_matrix(R)
    quat = r.as_quat() 
    
    # 轉成 Isaac Sim 格式 (w, x, y, z)
    isaac_quat = [quat[3], quat[0], quat[1], quat[2]]
    
    return isaac_quat

# --- 設定你的座標 ---
camera_position = [1.1, 0.0, 1.1]  # 你想要放置相機的位置 (桌子斜前方)
target_position = [0.5, 0.0, 0.8]  # 你想要看的位置 (桌子中心表面)

q = get_lookat_quat(camera_position, target_position)

print("-" * 30)
print(f"Camera Pos: {camera_position}")
print(f"Target Pos: {target_position}")
print("-" * 30)
print(f"Calculated Quaternion (w, x, y, z):")
print(f"rot=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f})")
print("-" * 30)
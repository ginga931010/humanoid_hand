from scipy.spatial.transform import Rotation as R

# 1. 定義第一個旋轉：Z 軸轉 180 度 (面向後方)
r_z = R.from_euler('z', 180, degrees=True)

# 2. 定義第二個旋轉：Y 軸轉 45 度 (往下看)
# 注意：這裡假設你是要在「轉身後」的基礎上轉 Y，所以是 Intrinsic (內旋)
r_y = R.from_euler('y', 45, degrees=True) 
# 如果你覺得方向反了（變成往上看），把 45 改成 -45 即可

# 3. 組合旋轉：矩陣乘法順序
# 在 Scipy 中，如果是 Intrinsic (小寫 xyz)，順序是先左後右的疊加
# 但最直觀的是直接用乘法：r_final = r_z * r_y 代表先做 Z，再做 Y (Local Frame)
r_final = r_z * r_y

# 4. 取得四元數 (Isaac Sim 需要 w, x, y, z 順序)
# Scipy 預設回傳 (x, y, z, w)，所以需要調整順序！
xyzw = r_final.as_quat()
wxyz = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])

print(f"你的四元數是: {wxyz}")
# 預期輸出: (0.0, -0.38268343, 0.0, 0.92387953)
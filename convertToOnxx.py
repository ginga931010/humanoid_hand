import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic

# ==========================================
# 1. 參數設定 (請務必替換為你訓練時的真實設定)
# ==========================================
MODEL_PATH = "/home/ginga/humanoid_hand/logs/rsl_rl/humanoid_hand_grasp_v2/06-01-01-01_INS1.0/model_796.pt"  # 替換成你的 rsl_rl .pt 權重檔路徑
ONNX_OUTPUT_PATH = "/home/ginga/humanoid_hand/humanoid_hand_actor.onnx"

# 這些維度必須與你 Isaac Sim 環境中的設定完全一致
# 例如：目標座標(3) + 關節位置(11) + 關節速度(11) = 25
NUM_OBS = 72      
NUM_ACTIONS = 11

# 必須與訓練時的 yaml 或 config 設定檔一致 (rsl_rl 預設通常是這組或 [512, 256, 128])
ACTOR_HIDDEN_DIMS = [512, 256, 128] 

# ==========================================
# 2. 載入 rsl_rl 的模型權重
# ==========================================
# 初始化 ActorCritic 結構 (為了能順利載入 state_dict)
policy = ActorCritic(
    num_actor_obs=NUM_OBS,
    num_critic_obs=NUM_OBS,
    num_actions=NUM_ACTIONS,
    actor_hidden_dims=ACTOR_HIDDEN_DIMS,
    critic_hidden_dims=ACTOR_HIDDEN_DIMS, # Critic 的設定不影響匯出，隨意填即可
    activation='elu', # rsl_rl 預設通常是 elu，請確認你的訓練設定
)

# 載入 .pt 檔案 (讀取 state_dict)
print(f"Loading checkpoint from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
policy.load_state_dict(checkpoint['model_state_dict'])

# 切換到推論模式
policy.eval()

# ==========================================
# 3. 萃取 Actor 網路並匯出 ONNX
# ==========================================
# 我們寫一個乾淨的 Wrapper，只保留 Actor 的前向傳播
class ActorInference(nn.Module):
    def __init__(self, actor_network):
        super().__init__()
        self.actor = actor_network

    def forward(self, obs):
        # 確保輸出不包含其他訓練用的附加資訊
        return self.actor(obs)

# 提取 rsl_rl policy 內的 actor 模組
actor_engine = ActorInference(policy.actor)

# 建立一個 Dummy Input (假資料) 給 TensorRT 追蹤計算圖
# 維度為 [batch_size, num_obs]
dummy_input = torch.randn(1, NUM_OBS, dtype=torch.float32)

print(f"Exporting Actor network to {ONNX_OUTPUT_PATH}...")
torch.onnx.export(
    actor_engine,
    dummy_input,
    ONNX_OUTPUT_PATH,
    export_params=True,
    opset_version=14,               # 建議使用 14 或 11，相容性較好
    do_constant_folding=True,       # 執行常數折疊優化
    input_names=['obs'],            # 為輸入節點命名
    output_names=['action'],        # 為輸出節點命名
    dynamic_axes={                  # 允許未來推論時改變 batch_size
        'obs': {0: 'batch_size'}, 
        'action': {0: 'batch_size'}
    }
)

print("ONNX export complete! You can now deploy it to the Jetson Brain Docker.")
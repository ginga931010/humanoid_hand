import torch
# [修正 1] 匯入控制邏輯類別 (從 joint_actions.py)
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

# [修正 2] 匯入設定檔類別 (直接從 mdp 模組拿最安全)
from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.utils import configclass

class EMAJointPositionAction(JointPositionAction):
    """自定義：帶有 EMA 平滑濾波的關節位置控制器"""
    cfg: "EMAJointPositionActionCfg"
    
    def __init__(self, cfg: "EMAJointPositionActionCfg", env):
        super().__init__(cfg, env)
        # 初始化儲存前一次的平滑目標值
        self._smoothed_targets = torch.zeros_like(self.processed_actions)

    def reset(self, env_ids: slice | torch.Tensor | None = None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        
        # 回合重置時，將平滑基準對齊當前馬達的實際目標角度，避免重置瞬間暴衝
        self._smoothed_targets[env_ids] = self._asset.data.joint_pos_target[env_ids][:, self._joint_ids]

    def process_actions(self, actions: torch.Tensor):
        # 1. 執行原版的計算 (這會自動套用你寫好的 scale 和 offset 字典)
        # 執行後 self.processed_actions 會變成 RL 想要達到的「原始目標角度」
        super().process_actions(actions)
        
        # 2. 套用 EMA 低通濾波
        alpha = self.cfg.alpha
        self._smoothed_targets = alpha * self.processed_actions + (1.0 - alpha) * self._smoothed_targets
        
        # 3. 覆寫回去，讓環境去執行平滑後的指令
        self.processed_actions[:] = self._smoothed_targets


@configclass
class EMAJointPositionActionCfg(JointPositionActionCfg):
    """EMA 關節控制器的設定檔"""
    class_type: type = EMAJointPositionAction
    alpha: float = 0.2  # [新增] 平滑係數，範圍 (0, 1]。1.0 代表無平滑，越小越平滑
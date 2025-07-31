from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class AsymmetricRewardConfig:
    # Loss region shaping
    loss_clip_r: float = -1.0            # clip downside around -1R
    loss_penalty_scale: float = 1.25     # base penalty slope up to 1R
    tail_loss_penalty_scale: float = 0.75  # extra penalty beyond 1R

    # Gain region shaping
    gain_knee_r: float = 1.5             # knee point where convex amplification starts
    tail_gain_scale: float = 2.0         # slope beyond knee

    # Drawdown and MAE penalties (per step)
    dd_penalty_scale: float = 0.25       # penalize increase in drawdown
    mae_penalty_scale: float = 0.05      # penalize large adverse excursions in R
    mae_floor_r: float = 0.5             # threshold before MAE penalty triggers

    # Position sizing/holding regularizers
    size_penalty_scale: float = 0.02     # discourage oversized positions (proportional)
    time_penalty_scale: float = 0.005    # discourage excessive holding

    # Utility-based loss aversion (prospect-theory-like)
    loss_aversion_lambda: float = 1.8    # > 1 increases pain of losses
    alpha_gain: float = 0.88             # concavity for gains (not used when convex tail engaged)
    beta_loss: float = 0.88              # convexity for losses

    # Hard caps
    min_reward: float = -5.0
    max_reward: float = 10.0


class AsymmetricReward:
    """
    Risk-aware asymmetric reward with:
      - Clipped downside and heavier penalties beyond -1R
      - Convex upside amplification beyond knee (e.g., 1.5R) to encourage tail winners
      - Drawdown and MAE-based shaping to favor small frequent losses and rare large wins
      - Size and time penalties to discourage overtrading/overfitting
      - Optional utility shaping (prospect theory)
    Expected step_info keys:
      - r: per-trade or per-step R multiple (pnl / nominal_risk)
      - dd_delta: change in drawdown this step (>= 0 when drawdown worsens)
      - mae_r: current trade MAE in R
      - size_ratio: position nominal risk / allowed nominal risk
      - time_in_trade: bars in current trade
    """

    def __init__(self, cfg: Optional[AsymmetricRewardConfig] = None):
        self.cfg = cfg or AsymmetricRewardConfig()

    def __call__(self, env, step_info: Dict[str, Any]) -> float:
        r = float(step_info.get("r", 0.0))
        dd_delta = max(0.0, float(step_info.get("dd_delta", 0.0)))
        mae_r = max(0.0, float(step_info.get("mae_r", 0.0)))
        size_ratio = max(0.0, float(step_info.get("size_ratio", 0.0)))
        time_in_trade = max(0.0, float(step_info.get("time_in_trade", 0.0)))

        reward = 0.0

        # Base utility component (piecewise and prospect-theory-like)
        if r < 0:
            # Loss utility with loss aversion
            loss_mag = abs(r)
            base_pen = self.cfg.loss_penalty_scale * min(loss_mag, abs(self.cfg.loss_clip_r))
            tail_pen = self.cfg.tail_loss_penalty_scale * max(0.0, loss_mag - abs(self.cfg.loss_clip_r))
            util = -(self.cfg.loss_aversion_lambda * (base_pen + tail_pen) ** self.cfg.beta_loss)
            reward += util
        else:
            # Gains: mild up to knee, then convex amplification
            base_gain = min(self.cfg.gain_knee_r, r)
            tail_gain = max(0.0, r - self.cfg.gain_knee_r) * self.cfg.tail_gain_scale
            util = (base_gain ** self.cfg.alpha_gain) + tail_gain
            reward += util

        # Drawdown penalty
        reward -= self.cfg.dd_penalty_scale * dd_delta

        # MAE penalty (encourage cutting quickly)
        if mae_r > self.cfg.mae_floor_r:
            reward -= self.cfg.mae_penalty_scale * (mae_r - self.cfg.mae_floor_r)

        # Size penalty (discourage oversizing)
        reward -= self.cfg.size_penalty_scale * max(0.0, size_ratio - 1.0)

        # Time penalty (discourage stalling)
        reward -= self.cfg.time_penalty_scale * time_in_trade

        # Clamp reward
        reward = float(max(self.cfg.min_reward, min(self.cfg.max_reward, reward)))
        return reward


def asymmetric_reward(env, step_info: Dict[str, Any]) -> float:
    """
    Convenience function to use default config.
    """
    return AsymmetricReward()(env, step_info)
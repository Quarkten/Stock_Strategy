import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from src.data.lob import LOB
from src.data.synthetic_lob import generate_synthetic_lob

# Gymnasium is listed in requirements; import lazily to avoid hard fail if missing at import time
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None
    spaces = None


@dataclass
class CostModel:
    slippage_ticks: float = 0.0
    fee_per_share: float = 0.0
    spread_widen_prob: float = 0.0
    spread_widen_ticks: float = 0.0
    latency_bars: int = 0
    latency_dist: str = "fixed"
    latency_shape: float = 2.0
    latency_scale: float = 1.0
    seed: int = 42
    data_augmentation: bool = False

    def get_latency(self, rng: random.Random) -> int:
        if self.latency_dist == "fixed":
            return self.latency_bars
        elif self.latency_dist == "gamma":
            return int(rng.gammavariate(self.latency_shape, self.latency_scale))
        else:
            return 0

    def apply_slippage(self, price: float, side: str, rng: random.Random) -> float:
        slip = self.slippage_ticks
        if self.data_augmentation:
            slip = rng.uniform(0.5, 1.5) * self.slippage_ticks

        if rng.random() < self.spread_widen_prob:
            slip += self.spread_widen_ticks
        if side.upper() == "BUY" or side.upper() == "LONG":
            return price + slip
        return price - slip

@dataclass
class SafetyLimits:
    max_position_size_shares: float
    max_leverage: float
    per_trade_max_loss_r: float
    daily_max_loss_pct: float


class TradingEnv(gym.Env if gym else object):
    """
    Gym-like environment that wraps a bar-by-bar trading simulation aligned with existing strategy constraints.
    Objectives:
      - Observations include engineered features, risk metrics, regime flags, and position context
      - Actions include entry/exit/hold (discrete) and continuous adjustments for size, stop-loss/take-profit multipliers, and time-stop
      - Reward is provided externally (configurable), but a default asymmetric shaping is available in reward module
      - Safety layer overrides actions violating risk constraints (no naked violations)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        strategy,
        reward_fn,
        config: Dict[str, Any],
        seed: int = 42,
        lookback_window: int = 20,
    ):
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required for TradingEnv")

        super().__init__()
        self.strategy = strategy
        self.config = config or {}
        self.seed_val = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Data must include engineered features; caller can use src/utils/features.add_indicators
        if data is None or data.empty:
            raise ValueError("TradingEnv requires non-empty data DataFrame")
        self.df = data.sort_index()
        self.ptr = 0

        # LOB
        self.lobs = generate_synthetic_lob(self.df)
        self.lob = LOB()

        # Cost Model
        cost_config = self.config.get("backtest_costs", {})
        self.cost = CostModel(
            slippage_ticks=float(cost_config.get("slippage_ticks", 0.01)),
            fee_per_share=float(cost_config.get("fee_per_share", 0.005)),
            spread_widen_prob=float(cost_config.get("spread_widen_prob", 0.05)),
            spread_widen_ticks=float(cost_config.get("spread_widen_ticks", 0.02)),
            data_augmentation=cost_config.get("data_augmentation", False),
            seed=seed,
        )

        # Risk/safety limits
        capital = float(self.config.get("capital", 100000.0))
        risk_per_trade = float(self.config.get("risk_per_trade", 0.005))
        daily_max_loss = float(self.config.get("daily_max_loss", 0.015))
        self.safety = SafetyLimits(
            max_position_size_shares=float(self.config.get("env", {}).get("max_position_size_shares", 1e6)),
            max_leverage=float(self.config.get("env", {}).get("max_leverage", 2.0)),
            per_trade_max_loss_r=float(self.config.get("env", {}).get("per_trade_max_loss_r", 1.2)),
            daily_max_loss_pct=daily_max_loss,
        )
        self.capital = capital
        self.nominal_risk = self.capital * risk_per_trade

        # Position state
        self.position: Optional[Dict[str, Any]] = None
        self.daily_pnl = 0.0
        self.episode_pnl = 0.0
        self.trades: List[Dict[str, Any]] = []

        # Observation construction
        self.feature_cols = self._infer_feature_columns(self.df)
        self.lookback_window = lookback_window
        # Observation = features + regime one-hot + position context (6) + risk context (3) + LOB (40)
        self.n_lob_levels = 10
        n_features = len(self.feature_cols) + 3 + 6 + 3 + self.n_lob_levels * 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.lookback_window, n_features), dtype=np.float32)

        # Actions flattened to a single Box for SB3 compatibility:
        # [disc_onehot(4), size_mult(0..1), stop_mult(0.5..2.5), tp_mult(0.5..3.0), time_stop(0..1)]
        low = np.array([0, 0, 0, 0, 0.0, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1.0, 2.5, 3.0, 1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Reward function callable: reward = reward_fn(env, step_info)
        self.reward_fn = reward_fn

        # Time-stop bounds
        self.max_time_stop_bars = int(self.config.get("env", {}).get("max_time_stop_bars", 48))

        # Done criteria
        self.max_bars = int(self.config.get("env", {}).get("max_bars_per_episode", len(self.df)))

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = self.seed_val
        self.seed_val = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.seed(seed)
        self.ptr = 0
        self.position = None
        self.daily_pnl = 0.0
        self.episode_pnl = 0.0
        self.trades.clear()
        self._update_lob()
        obs = self._build_observation()
        info = {}
        return obs, info

    def step(self, action):
        # Decode flattened action
        action = np.array(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 9:
            raise ValueError("Action must be length 9: 4 one-hot + 5 continuous")
        disc_oh = action[:4]
        disc = int(np.argmax(disc_oh))
        size_mult, stop_mult, tp_mult, tstop_scaled = float(action[4]), float(action[5]), float(action[6]), float(action[7])
        tstop_bars = int(round(tstop_scaled * self.max_time_stop_bars))

        # Safety override will be applied if necessary
        reward = 0.0
        terminated = False
        truncated = False

        # Retrieve current bar
        if self.ptr >= len(self.df):
            return self._build_observation(), 0.0, True, False, {}

        bar = self.df.iloc[self.ptr]
        close = float(bar["close"])
        atr = float(bar.get("atr", np.nan))
        if math.isnan(atr) or atr <= 0:
            # advance pointer until ATR exists
            self.ptr += 1
            return self._build_observation(), 0.0, False, False, {"skip": "no_atr"}

        # Execute action
        info: Dict[str, Any] = {}

        # Apply latency
        latency = self.cost.get_latency(self.rng)
        if self.ptr + latency >= len(self.df):
            return self._build_observation(), 0.0, True, False, {"skip": "latency"}

        bar = self.df.iloc[self.ptr + latency]
        close = float(bar["close"])
        atr = float(bar.get("atr", np.nan))

        if disc == 1 or disc == 2:
            # Open new position if flat
            if self.position is None:
                side = "LONG" if disc == 1 else "SHORT"

                # Match order against LOB
                trades = self.lob.match_order("BUY" if side == "LONG" else "SELL", size_mult * self.nominal_risk)
                if not trades:
                    return self._build_observation(), 0.0, False, False, {"skip": "no_liquidity"}

                entry_price = np.average([t[0] for t in trades], weights=[t[1] for t in trades])
                shares = sum(t[1] for t in trades)

                stop_dist = stop_mult * atr
                tp_dist = tp_mult * atr
                stop = entry_price - stop_dist if side == "LONG" else entry_price + stop_dist
                tp = entry_price + tp_dist if side == "LONG" else entry_price - tp_dist
                risk_per_share = abs(entry_price - stop)

                if risk_per_share > 0:
                    # --- Safety Layer ---
                    # 1. Clip stop loss to per-trade max loss
                    max_stop_dist = self.safety.per_trade_max_loss_r * atr
                    if stop_dist > max_stop_dist:
                        stop_dist = max_stop_dist
                        stop = entry_price - stop_dist if side == "LONG" else entry_price + stop_dist
                        risk_per_share = abs(entry_price - stop)

                    # 2. Calculate position size
                    shares = max(0.0, (self.nominal_risk / risk_per_share) * size_mult)

                    # 3. Clip shares by max position size and leverage
                    shares = min(shares, self.safety.max_position_size_shares)
                    max_shares_by_leverage = (self.capital * self.safety.max_leverage) / entry_price
                    shares = min(shares, max_shares_by_leverage)

                    if shares > 0:
                        entry_fee = shares * self.cost.fee_per_share
                        self.position = {
                            "side": side,
                            "entry": entry_price,
                            "stop": stop,
                            "tp": tp,
                            "shares": shares,
                            "open_idx": self.ptr,
                            "tstop_bars": tstop_bars,
                            "atr": atr,
                            "initial_risk": risk_per_share,
                            "breakeven_r": float(getattr(self.strategy, "breakeven_r", 1.0)),
                            "entry_fee": entry_fee,
                        }
                        info["opened"] = side
        elif disc == 3:
            # Close if position
            if self.position is not None:
                trades = self.lob.match_order("SELL" if self.position["side"] == "LONG" else "BUY", self.position["shares"])
                if not trades:
                    return self._build_observation(), 0.0, False, False, {"skip": "no_liquidity"}

                exit_price = np.average([t[0] for t in trades], weights=[t[1] for t in trades])
                pnl = self._close_position(exit_price)
                reward += self._compute_reward(pnl, exit_price)
                info["closed"] = True

        # Update position management on this bar
        if self.position is not None:
            # Move to breakeven after +1R
            pos = self.position
            if pos["side"] == "LONG":
                move = close - pos["entry"]
                if move >= pos["initial_risk"] and pos["stop"] < pos["entry"]:
                    pos["stop"] = pos["entry"]
            else:
                move = pos["entry"] - close
                if move >= pos["initial_risk"] and pos["stop"] > pos["entry"]:
                    pos["stop"] = pos["entry"]

            # Time-stop check
            if pos["tstop_bars"] > 0 and (self.ptr - pos["open_idx"]) >= pos["tstop_bars"]:
                pnl = self._close_position(close)
                reward += self._compute_reward(pnl, close)
                info["time_stop"] = True

            # Stop/TP intrabar check (simplified: use close)
            if self.position is not None:
                pos = self.position
                hit = False
                if pos["side"] == "LONG":
                    if close <= pos["stop"]:
                        exit_price = self.cost.apply_slippage(pos["stop"], "SELL", self.rng)
                        pnl = self._close_position(exit_price)
                        reward += self._compute_reward(pnl, exit_price)
                        info["stopped"] = True
                        hit = True
                    elif close >= pos["tp"]:
                        exit_price = self.cost.apply_slippage(pos["tp"], "SELL", self.rng)
                        pnl = self._close_position(exit_price)
                        reward += self._compute_reward(pnl, exit_price)
                        info["took_profit"] = True
                        hit = True
                else:
                    if close >= pos["stop"]:
                        exit_price = self.cost.apply_slippage(pos["stop"], "BUY", self.rng)
                        pnl = self._close_position(exit_price)
                        reward += self._compute_reward(pnl, exit_price)
                        info["stopped"] = True
                        hit = True
                    elif close <= pos["tp"]:
                        exit_price = self.cost.apply_slippage(pos["tp"], "BUY", self.rng)
                        pnl = self._close_position(exit_price)
                        reward += self._compute_reward(pnl, exit_price)
                        info["took_profit"] = True
                        hit = True
                # Daily loss cap ends episode
                if self.daily_pnl <= -(self.safety.daily_max_loss_pct * self.capital):
                    terminated = True
                    info["daily_halt"] = True

        # Advance pointer and update LOB
        self.ptr += 1
        if self.ptr >= self.max_bars or self.ptr >= len(self.df):
            terminated = True
        else:
            self._update_lob()

        obs = self._build_observation()
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ----------------------------
    # Helpers
    # ----------------------------
    def _infer_feature_columns(self, df: pd.DataFrame) -> List[str]:
        candidates = [
            "close", "atr", "ema20", "ema50", "macd_hist", "bb_upper", "bb_lower", "vwap", "vol_ratio",
        ]
        return [c for c in candidates if c in df.columns]

    def _regime_one_hot(self, row: pd.Series) -> np.ndarray:
        vr = float(row.get("vol_ratio", 0.0))
        if vr >= 0.01:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)  # HIGH_VOL
        if vr >= 0.005:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)  # MED_VOL
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)      # LOW_VOL

    def _position_context(self) -> np.ndarray:
        if self.position is None:
            return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        pos = self.position
        # is_long, stop_dist_R, tp_dist_R, time_in_trade, entry_close_dist, shares_norm
        is_long = 1.0 if pos["side"] == "LONG" else 0.0
        stop_r = abs(pos["entry"] - pos["stop"]) / max(1e-9, pos["atr"])
        tp_r = abs(pos["tp"] - pos["entry"]) / max(1e-9, pos["atr"])
        time_in = (self.ptr - pos["open_idx"])
        shares_norm = np.tanh(pos["shares"] / 10000.0)
        entry_close_dist = 0.0
        return np.array([is_long, stop_r, tp_r, float(time_in), entry_close_dist, shares_norm], dtype=np.float32)

    def _risk_context(self) -> np.ndarray:
        # daily pnl in R, episode pnl in R, recent streak placeholder
        daily_r = self.daily_pnl / max(1e-9, self.nominal_risk)
        episode_r = self.episode_pnl / max(1e-9, self.nominal_risk)
        streak = 0.0
        return np.array([daily_r, episode_r, streak], dtype=np.float32)

    def _build_observation(self) -> np.ndarray:
        obs_list = []
        for i in range(self.lookback_window):
            idx = max(0, self.ptr - self.lookback_window + 1 + i)
            row = self.df.iloc[idx]

            feats = np.array([float(row[c]) for c in self.feature_cols], dtype=np.float32)
            regime = self._regime_one_hot(row)

            # For simplicity, we use the current position and risk context for all past observations
            pos_ctx = self._position_context()
            risk_ctx = self._risk_context()

            # LOB features
            # For simplicity, we use the current LOB for all past observations
            lob_snapshot = self.lob.get_snapshot(self.n_lob_levels)
            bids = lob_snapshot['bids']
            asks = lob_snapshot['asks']
            lob_features = np.zeros(self.n_lob_levels * 4)
            for j in range(self.n_lob_levels):
                if j < len(bids):
                    lob_features[j*4] = bids[j][0]
                    lob_features[j*4+1] = bids[j][1]
                if j < len(asks):
                    lob_features[j*4+2] = asks[j][0]
                    lob_features[j*4+3] = asks[j][1]

            obs = np.concatenate([feats, regime, pos_ctx, risk_ctx, lob_features], axis=0).astype(np.float32)
            obs_list.append(obs)

        return np.array(obs_list, dtype=np.float32)

    def _update_lob(self):
        self.lob = LOB()
        lob_snapshot = self.lobs[self.ptr]
        for price, size in lob_snapshot['bids']:
            self.lob.add_order('BID', price, size)
        for price, size in lob_snapshot['asks']:
            self.lob.add_order('ASK', price, size)

    def _close_position(self, exit_px: float) -> float:
        assert self.position is not None
        pos = self.position
        pnl = (exit_px - pos["entry"]) * pos["shares"] if pos["side"] == "LONG" else (pos["entry"] - exit_px) * pos["shares"]

        # Apply fees
        entry_fee = pos.get("entry_fee", 0.0)
        exit_fee = pos["shares"] * self.cost.fee_per_share
        pnl -= (entry_fee + exit_fee)

        self.daily_pnl += pnl
        self.episode_pnl += pnl
        trade = dict(pos)
        trade["exit"] = exit_px
        trade["pnl"] = pnl
        self.trades.append(trade)
        self.position = None
        return pnl

    def _compute_reward(self, pnl: float, price_ref: float) -> float:
        """
        Delegate to external reward function. Falls back to simple asymmetric utility if not provided.
        """
        if self.reward_fn is not None:
            return float(self.reward_fn(self, {"pnl": pnl, "price": price_ref}))
        # default: penalize losses more than reward wins; convex tail for > 1.5R
        r = pnl / max(1e-9, self.nominal_risk)
        if r < 0:
            return float(-min(1.0, abs(r)) * 1.25 - max(0.0, abs(r) - 1.0) * 0.75)
        gain = min(1.5, r) + max(0.0, r - 1.5) * 2.0
        return float(gain)

    def render(self):
        pass

    def close(self):
        pass
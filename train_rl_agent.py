import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.strategies.intraday_strategy import IntradayStrategy
from src.utils.data_fetcher import DataFetcher
from src.utils.features import add_indicators
from src.agents.envs.trading_env import TradingEnv
from src.agents.rewards.asymmetric_reward import AsymmetricReward, AsymmetricRewardConfig

# SB3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


@dataclass
class RLConfig:
    agent: str = "ppo"  # ppo | sac
    timesteps: int = 10000
    seed: int = 42
    # PPO
    ppo_lr: float = 3e-4
    ppo_batch_size: int = 2048
    # SAC
    sac_lr: float = 3e-4
    sac_batch_size: int = 256


def load_merged_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    import yaml

    with open(config_path, "r") as f:
        base = yaml.safe_load(f)
    # Optionally merge config/rl.yaml if present
    rl_over = {}
    if os.path.exists("config/rl.yaml"):
        with open("config/rl.yaml", "r") as f:
            rl_over = yaml.safe_load(f) or {}
    # Shallow merge: rl overrides base keys
    merged = base.copy()
    merged.update(rl_over or {})
    return merged


def build_dataset(data_fetcher: DataFetcher, symbol: str, timeframe: str, start: Optional[str], end: Optional[str], cfg: Dict[str, Any]) -> pd.DataFrame:
    df = data_fetcher.fetch_data(symbol, _normalize_timeframe(timeframe), start=start, end=end)
    if df is None or df.empty:
        raise RuntimeError("No data loaded for training")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # default to NY tz
        tz = cfg.get("timezone", "America/New_York")
        df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
        df = df.set_index("timestamp").sort_index()
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        tz = cfg.get("timezone", "America/New_York")
        df = df.tz_convert(tz)
        df = df.sort_index()

    # Feature engineering
    df = add_indicators(df, cfg)
    # Drop NaNs at beginning
    df = df.dropna().copy()
    return df


def make_env(df: pd.DataFrame, strategy: IntradayStrategy, cfg: Dict[str, Any], reward_cfg: Optional[Dict[str, Any]] = None, seed: int = 42):
    # Reward
    arc = AsymmetricRewardConfig(**(reward_cfg or {})) if reward_cfg else AsymmetricRewardConfig()
    reward_fn = AsymmetricReward(arc)
    def _env_fn():
        env = TradingEnv(data=df, strategy=strategy, reward_fn=reward_fn, config=cfg, seed=seed)
        return Monitor(env)
    return _env_fn


def train(agent: str, env_fn, rl_cfg: RLConfig, model_path: str, eval_only: bool = False):
    if agent.lower() == "ppo":
        model = PPO("MlpPolicy", DummyVecEnv([env_fn]), learning_rate=rl_cfg.ppo_lr, verbose=1, seed=rl_cfg.seed, batch_size=rl_cfg.ppo_batch_size)
    elif agent.lower() == "sac":
        model = SAC("MlpPolicy", DummyVecEnv([env_fn]), learning_rate=rl_cfg.sac_lr, verbose=1, seed=rl_cfg.seed, batch_size=rl_cfg.sac_batch_size)
    else:
        raise ValueError("Unsupported agent, choose from: ppo, sac")

    if not eval_only:
        model.learn(total_timesteps=int(rl_cfg.timesteps))
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
    else:
        if os.path.exists(model_path + ".zip"):
            model = model.load(model_path, env=DummyVecEnv([env_fn]))
        # quick eval rollout
        env = DummyVecEnv([env_fn])
        obs = env.reset()
        ep_rewards = []
        ep_ret = 0.0
        for _ in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_ret += float(reward.mean())
            if done.any():
                ep_rewards.append(ep_ret)
                ep_ret = 0.0
                obs = env.reset()
        print(json.dumps({"episodes": len(ep_rewards), "mean_ep_reward": float(np.mean(ep_rewards) if ep_rewards else 0.0)}))


def _normalize_timeframe(tf: str) -> str:
    mapping = {"5m": "5min", "10m": "10min", "15m": "15min", "60m": "1hour", "1h": "1hour", "1d": "1day", "1day": "1day"}
    return mapping.get(tf, tf)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", type=str, default="ppo", help="ppo|sac")
    p.add_argument("--symbol", type=str, default="SPY")
    p.add_argument("--timeframe", type=str, default="5min")
    p.add_argument("--start", type=str, required=False)
    p.add_argument("--end", type=str, required=False)
    p.add_argument("--timesteps", type=int, default=10000)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--model-path", type=str, default="models/rl_model")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_merged_config()
    rl_cfg = RLConfig(agent=args.agent, timesteps=int(args.timesteps), seed=int(cfg.get("seed", 42)))
    # Strategy and data
    strategy = IntradayStrategy(cfg)
    data_fetcher = DataFetcher(cfg)
    df = build_dataset(data_fetcher, args.symbol or cfg.get("target_symbol", "SPY"), args.timeframe or cfg.get("timeframes", {}).get("short_term", ["5min"])[0], args.start, args.end, cfg)
    # Env
    env_fn = make_env(df, strategy, cfg, reward_cfg=cfg.get("reward", {}), seed=rl_cfg.seed)
    # Train/Eval
    train(args.agent, env_fn, rl_cfg, args.model_path, eval_only=bool(args.eval_only))


if __name__ == "__main__":
    main()
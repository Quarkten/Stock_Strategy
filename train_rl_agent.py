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
from sb3_contrib import TQC
from src.agents.replay.sb3_per_buffer import SB3PrioritizedReplayBuffer
from src.agents.policies.transformer_policy import TransformerPolicy, TransformerFeaturesExtractor
import wandb
from wandb.integration.sb3 import WandbCallback

@dataclass
class RLConfig:
    agent: str = "ppo"  # ppo | sac | tqc
    policy: str = "mlp" # mlp | transformer
    timesteps: int = 10000
    seed: int = 42
    # PPO
    ppo_lr: float = 3e-4
    ppo_batch_size: int = 2048
    # SAC
    sac_lr: float = 3e-4
    sac_batch_size: int = 256
    # TQC
    tqc_lr: float = 7.3e-4
    tqc_batch_size: int = 256


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


def train(agent: str, env_fn, rl_cfg: RLConfig, model_path: str, eval_only: bool = False, wandb_project: Optional[str] = None):
    run = None
    if wandb_project:
        run = wandb.init(
            project=wandb_project,
            config=rl_cfg,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard logs
            monitor_gym=True,       # auto-upload videos of agents
            save_code=True,         # optional
        )

    vec_env = DummyVecEnv([env_fn])

    if agent.lower() == "ppo":
        model = PPO("MlpPolicy", vec_env, learning_rate=rl_cfg.ppo_lr, verbose=1, seed=rl_cfg.seed, batch_size=rl_cfg.ppo_batch_size, tensorboard_log=f"runs/{run.id}" if run else None)
    elif agent.lower() == "sac":
        model = SAC("MlpPolicy", vec_env, learning_rate=rl_cfg.sac_lr, verbose=1, seed=rl_cfg.seed, batch_size=rl_cfg.sac_batch_size, tensorboard_log=f"runs/{run.id}" if run else None)
    elif agent.lower() == "tqc":
        policy_kwargs = {}
        if rl_cfg.policy == "transformer":
            policy_kwargs["features_extractor_class"] = TransformerFeaturesExtractor
            policy_kwargs["features_extractor_kwargs"] = dict(features_dim=256)

        replay_buffer_kwargs = {"stratified_config": cfg.get("replay")}
        model = TQC(
            TransformerPolicy if rl_cfg.policy == "transformer" else "MlpPolicy",
            vec_env,
            learning_rate=rl_cfg.tqc_lr,
            verbose=1,
            seed=rl_cfg.seed,
            batch_size=rl_cfg.tqc_batch_size,
            replay_buffer_class=SB3PrioritizedReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"runs/{run.id}" if run else None,
        )
    else:
        raise ValueError("Unsupported agent, choose from: ppo, sac, tqc")

    if not eval_only:
        callback = WandbCallback(model_save_path=f"models/{run.id}" if run else model_path) if run else None
        model.learn(total_timesteps=int(rl_cfg.timesteps), callback=callback)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        if run:
            run.finish()
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


def walk_forward_train(
    agent: str,
    df: pd.DataFrame,
    strategy: IntradayStrategy,
    cfg: Dict[str, Any],
    rl_cfg: RLConfig,
    train_days: int,
    val_days: int,
    step_days: int,
    wandb_project: Optional[str] = None,
    eval_log_path: Optional[str] = None,
):
    unique_dates = df.index.normalize().unique()
    start_idx = 0
    end_idx = train_days
    step_idx = 0
    all_trades = []

    while end_idx < len(unique_dates):
        train_start_date = unique_dates[start_idx]
        train_end_date = unique_dates[end_idx - 1]
        val_start_date = unique_dates[end_idx]
        val_end_date = unique_dates[min(end_idx + val_days - 1, len(unique_dates) - 1)]

        print(f"--- Walk-forward step {step_idx} ---")
        print(f"Training from {train_start_date.date()} to {train_end_date.date()}")
        print(f"Validating from {val_start_date.date()} to {val_end_date.date()}")

        train_df = df.loc[train_start_date:train_end_date]
        val_df = df.loc[val_start_date:val_end_date]

        model_path = f"models/{rl_cfg.agent}_{step_idx}"

        # Train
        train_env_fn = make_env(train_df, strategy, cfg, reward_cfg=cfg.get("reward", {}), seed=rl_cfg.seed)
        train(agent, train_env_fn, rl_cfg, model_path, eval_only=False, wandb_project=wandb_project)

        # Evaluate
        eval_env = DummyVecEnv([make_env(val_df, strategy, cfg, reward_cfg=cfg.get("reward", {}), seed=rl_cfg.seed)])
        model = TQC.load(model_path, env=eval_env) if agent == 'tqc' else SAC.load(model_path, env=eval_env) if agent == 'sac' else PPO.load(model_path, env=eval_env)

        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = eval_env.step(action)

        trades = eval_env.envs[0].unwrapped.trades
        all_trades.extend(trades)

        start_idx += step_days
        end_idx += step_days
        step_idx += 1

    if eval_log_path and all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(eval_log_path, index=False)
        print(f"Saved evaluation trades to {eval_log_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", type=str, default="ppo", help="ppo|sac|tqc")
    p.add_argument("--policy", type=str, default="mlp", help="mlp|transformer")
    p.add_argument("--symbol", type=str, default="SPY")
    p.add_argument("--timeframe", type=str, default="5min")
    p.add_argument("--start", type=str, required=False)
    p.add_argument("--end", type=str, required=False)
    p.add_argument("--timesteps", type=int, default=10000)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--model-path", type=str, default="models/rl_model")
    p.add_argument("--wandb-project", type=str, help="Wandb project name")
    p.add_argument("--walk-forward", action="store_true", help="Enable walk-forward training")
    p.add_argument("--train-days", type=int, default=252, help="Number of days for training window")
    p.add_argument("--val-days", type=int, default=63, help="Number of days for validation window")
    p.add_argument("--step-days", type=int, default=63, help="Number of days to step forward")
    p.add_argument("--eval-log-path", type=str, default="data/tqc_trades.csv", help="Path to save evaluation trade log")
    p.add_argument("--reward-params", type=str, help="JSON string of reward parameters to override")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_merged_config()

    if args.reward_params:
        import json
        reward_params = json.loads(args.reward_params)
        cfg["reward"].update(reward_params)

    rl_cfg = RLConfig(
        agent=args.agent,
        policy=args.policy,
        timesteps=int(args.timesteps),
        seed=int(cfg.get("seed", 42)),
        tqc_lr=float(cfg.get("training", {}).get("tqc", {}).get("learning_rate", 7.3e-4)),
        tqc_batch_size=int(cfg.get("training", {}).get("tqc", {}).get("batch_size", 256)),
    )
    # Strategy and data
    strategy = IntradayStrategy(cfg)
    data_fetcher = DataFetcher(cfg)
    df = build_dataset(data_fetcher, args.symbol or cfg.get("target_symbol", "SPY"), args.timeframe or cfg.get("timeframes", {}).get("short_term", ["5min"])[0], args.start, args.end, cfg)

    if args.walk_forward:
        walk_forward_train(
            agent=args.agent,
            df=df,
            strategy=strategy,
            cfg=cfg,
            rl_cfg=rl_cfg,
            train_days=args.train_days,
            val_days=args.val_days,
            step_days=args.step_days,
            wandb_project=args.wandb_project,
            eval_log_path=args.eval_log_path,
        )
    else:
        # Env
        env_fn = make_env(df, strategy, cfg, reward_cfg=cfg.get("reward", {}), seed=rl_cfg.seed)
        # Train/Eval
        train(args.agent, env_fn, rl_cfg, args.model_path, eval_only=bool(args.eval_only), wandb_project=args.wandb_project)


if __name__ == "__main__":
    main()
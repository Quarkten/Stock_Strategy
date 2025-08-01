import argparse
from pettingzoo.test import parallel_api_test
from sb3_contrib import TQC
import supersuit as ss
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.envs.marl_env import MultiAgentTradingEnv
from src.agents.envs.trading_env import TradingEnv
from src.strategies.intraday_strategy import IntradayStrategy
from src.utils.data_fetcher import DataFetcher
from train_rl_agent import build_dataset, load_merged_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=10000)
    args = parser.parse_args()

    cfg = load_merged_config()

    # Create the single-agent trading environment
    strategy = IntradayStrategy(cfg)
    data_fetcher = DataFetcher(cfg)
    df = build_dataset(data_fetcher, cfg.get("target_symbol", "SPY"), "5min", None, None, cfg)
    trading_env = TradingEnv(df, strategy, None, cfg)

    # Create the multi-agent environment
    marl_env = MultiAgentTradingEnv(trading_env, n_agents=args.num_agents)

    # Wrap the environment for stable-baselines3
    env = ss.pettingzoo_env_to_vec_env_v1(marl_env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    # Create and train the TQC agent
    model = TQC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)

    print("MARL training complete.")

if __name__ == "__main__":
    main()

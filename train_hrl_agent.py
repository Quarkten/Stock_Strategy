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
from src.agents.envs.hrl_env import HRLEnv
from src.agents.envs.trading_env import TradingEnv
from src.agents.rewards.asymmetric_reward import AsymmetricReward, AsymmetricRewardConfig
from src.agents.hrl_manager import Manager
from src.agents.hrl_worker import Worker

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def main():
    # ... (parsing args and loading configs)

    # Create the TradingEnv
    trading_env = TradingEnv(...)

    # Create the HRLEnv
    hrl_env = HRLEnv(trading_env, manager_config, worker_config)

    # Create the Manager and Worker agents
    manager = Manager("MlpPolicy", hrl_env.manager_observation_space, ...)
    worker = Worker("MlpPolicy", hrl_env.worker_observation_space, ...)

    # HRL Training Loop
    for episode in range(num_episodes):
        manager_obs, _ = hrl_env.reset()
        done = False
        while not done:
            # Manager chooses a goal
            manager_action, _ = manager.predict(manager_obs)

            # Worker tries to achieve the goal
            worker_obs, worker_reward, worker_done, _ = hrl_env.step(manager_action)

            # Train the worker
            worker.learn(...)

            # Manager gets a reward at the end of the trade
            manager_reward = hrl_env.get_manager_reward()

            # Train the manager
            manager.learn(...)

            manager_obs = hrl_env.get_manager_obs()
            done = hrl_env.is_done()

if __name__ == "__main__":
    main()

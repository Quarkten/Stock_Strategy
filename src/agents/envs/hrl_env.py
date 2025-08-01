import numpy as np
from gymnasium import spaces
from .trading_env import TradingEnv

class HRLEnv:
    """
    A Hierarchical Reinforcement Learning (HRL) environment that wraps the TradingEnv.
    It consists of a high-level Manager and a low-level Worker.
    """
    def __init__(self, trading_env: TradingEnv, manager_config, worker_config):
        self.trading_env = trading_env
        self.manager_config = manager_config
        self.worker_config = worker_config

        # Manager observation and action space
        self.manager_observation_space = self._create_manager_observation_space()
        self.manager_action_space = spaces.Discrete(7)  # 0: neutral, 1-3: long (1R, 2R, 3R), 4-6: short (1R, 2R, 3R)

        # Worker observation and action space
        self.worker_observation_space = self._create_worker_observation_space()
        self.worker_action_space = self.trading_env.action_space

    def _create_manager_observation_space(self):
        # Condensed market state + portfolio value
        trading_obs_dim = self.trading_env.observation_space.shape[0]
        return spaces.Box(low=-np.inf, high=np.inf, shape=(trading_obs_dim + 1,), dtype=np.float32)

    def _create_worker_observation_space(self):
        # Full market state + goal from manager
        trading_obs_dim = self.trading_env.observation_space.shape[0]
        return spaces.Box(low=-np.inf, high=np.inf, shape=(trading_obs_dim + 1,), dtype=np.float32)

    def reset(self):
        obs, info = self.trading_env.reset()
        manager_obs = self._get_manager_obs(obs)
        return manager_obs, info

    def _get_manager_obs(self, trading_obs):
        portfolio_value = self.trading_env.capital + self.trading_env.episode_pnl
        return np.append(trading_obs, portfolio_value)

    def _get_worker_obs(self, trading_obs, goal):
        return np.append(trading_obs, goal)

    def step(self, manager_action):
        """
        A full HRL step, from manager action to the end of a trade.
        """
        goal = self._get_goal_from_manager_action(manager_action)

        # Worker executes the trade
        done = False
        worker_reward = 0
        while not done:
            worker_obs = self._get_worker_obs(self.trading_env._build_observation(), goal)
            # Here we would get the worker's action from the worker agent
            # For now, we'll just use a placeholder
            worker_action = self.worker_action_space.sample()

            obs, reward, terminated, truncated, info = self.trading_env.step(worker_action)
            done = terminated or truncated
            worker_reward += self._calculate_worker_reward(reward, goal)

        # Manager gets a reward at the end of the trade
        manager_reward = self.trading_env.trades[-1]['pnl'] if self.trading_env.trades else 0

        manager_obs = self._get_manager_obs(obs)

        return manager_obs, manager_reward, done, info

    def _get_goal_from_manager_action(self, action):
        if action == 0: # Neutral
            return 0
        elif action <= 3: # Long
            return action
        else: # Short
            return -(action - 3)

    def _calculate_worker_reward(self, pnl, goal):
        # Reward the worker for getting closer to the goal R-multiple
        current_r = pnl / self.trading_env.nominal_risk if self.trading_env.nominal_risk > 0 else 0
        return -abs(current_r - goal)

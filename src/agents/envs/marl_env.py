import functools
from copy import deepcopy

import numpy as np
from gymnasium import spaces

from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import from_parallel

from .trading_env import TradingEnv

class MultiAgentTradingEnv(ParallelEnv):
    """
    A multi-agent trading environment that wraps the TradingEnv.
    """
    metadata = {"render_modes": ["human"], "name": "marl_trading_v0"}

    def __init__(self, trading_env: TradingEnv, n_agents: int = 2):
        self.trading_env = trading_env
        self.n_agents = n_agents
        self.agents = [f"agent_{i}" for i in range(n_agents)]

        # PettingZoo API
        self.observation_spaces = {agent: self.trading_env.observation_space for agent in self.agents}
        self.action_spaces = {agent: self.trading_env.action_space for agent in self.agents}

    def reset(self, seed=None, options=None):
        obs, info = self.trading_env.reset(seed=seed, options=options)
        observations = {agent: obs for agent in self.agents}
        infos = {agent: info for agent in self.agents}
        return observations, infos

    def step(self, actions):
        # The LOB is shared. The actions of all agents will affect the same LOB.
        # This is a simplified model where we process the actions sequentially.
        # A more realistic model would handle simultaneous actions.

        all_obs, all_rewards, all_terminated, all_truncated, all_infos = {}, {}, {}, {}, {}

        for agent, action in actions.items():
            obs, reward, terminated, truncated, info = self.trading_env.step(action)
            all_obs[agent] = obs
            all_rewards[agent] = reward
            all_terminated[agent] = terminated
            all_truncated[agent] = truncated
            all_infos[agent] = info

            if terminated or truncated:
                self.agents.remove(agent)

        return all_obs, all_rewards, all_terminated, all_truncated, all_infos

    def render(self):
        return self.trading_env.render()

    def close(self):
        self.trading_env.close()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.trading_env.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.trading_env.action_space

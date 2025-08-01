import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from typing import Optional, Any, Dict, List
from gymnasium import spaces

from .prioritized_buffer import PrioritizedReplayBuffer, StratifiedBinsConfig

class SB3PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        stratified_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage)

        cfg = StratifiedBinsConfig(**stratified_config) if stratified_config else StratifiedBinsConfig()
        self.per_buffer = PrioritizedReplayBuffer(cfg)
        self.last_batch_indices = None

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]]) -> None:

        # sb3 passes obs with shape (n_envs, obs_dim), we are using n_envs=1
        obs = obs[0]
        next_obs = next_obs[0]

        transition = {
            "obs": obs,
            "next_obs": next_obs,
            "action": action,
            "reward": float(reward),
            "done": bool(done),
            "info": infos[0],
        }
        self.per_buffer.add(transition)

    def sample(self, batch_size: int, env: Optional["VecNormalize"] = None) -> ReplayBufferSamples:
        transitions, indices, is_weights = self.per_buffer.sample(batch_size)

        self.last_batch_indices = indices

        # Convert list of transitions to ReplayBufferSamples
        obs = np.array([t["obs"] for t in transitions])
        next_obs = np.array([t["next_obs"] for t in transitions])
        actions = np.array([t["action"] for t in transitions])
        rewards = np.array([t["reward"] for t in transitions])
        dones = np.array([t["done"] for t in transitions])

        return ReplayBufferSamples(
            observations=self.to_torch(obs),
            actions=self.to_torch(actions),
            next_observations=self.to_torch(next_obs),
            dones=self.to_torch(dones),
            rewards=self.to_torch(rewards),
        )

    def update_priorities(self, priorities: np.ndarray) -> None:
        if self.last_batch_indices is not None:
            self.per_buffer.update_priorities(self.last_batch_indices, priorities)

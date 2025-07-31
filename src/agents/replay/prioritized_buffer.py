from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StratifiedBinsConfig:
    # R buckets define emphasis on tails
    r_edges: Tuple[float, ...] = (-math.inf, -1.0, 0.0, 1.0, 2.0, math.inf)
    # Volatility regimes, map externally computed regime flags to indices {LOW, MED, HIGH}
    regime_labels: Tuple[str, ...] = ("LOW_VOL", "MED_VOL", "HIGH_VOL")
    # Sampling temperature: higher alpha increases preference to rare bins
    alpha: float = 0.6
    # PER-like beta to correct bias during training (if used)
    beta: float = 0.4
    # Minimum sampling prob per bin
    eps: float = 1e-6
    # Max buffer size
    capacity: int = 200_000


class PrioritizedReplayBuffer:
    """
    Stratified prioritized replay buffer across regime classes and R-buckets.

    Each transition must include:
      obs, action, reward, next_obs, done, info
    where info contains:
      - 'r': R multiple for the transition or terminal trade
      - 'regime': one of {"LOW_VOL", "MED_VOL", "HIGH_VOL"} (or mapped)
      - optional 'priority': initial priority

    Sampling:
      - Maintain counts and priorities for (regime_idx, r_bucket_idx) bins
      - Compute sampling weights W_ij ~ (count_ij + eps)^(-alpha) normalized over all non-empty bins
      - Within-bin sampling by normalized priorities; fallback to uniform if missing
      - Returns (batch transitions, indices, is_weights) for optional PER correction
    """

    def __init__(self, cfg: Optional[StratifiedBinsConfig] = None, seed: int = 42):
        self.cfg = cfg or StratifiedBinsConfig()
        self.capacity = int(self.cfg.capacity)
        self.rng = np.random.default_rng(seed)

        # Storage
        self.storage: List[Dict[str, Any]] = []
        self.priorities: List[float] = []

        # Bin index -> list of indices in storage
        self.regime_to_idx = {label: i for i, label in enumerate(self.cfg.regime_labels)}
        self.bins: Dict[Tuple[int, int], List[int]] = {}

    # ------------- Public API -------------

    def add(self, transition: Dict[str, Any]) -> None:
        """
        Add a transition with required fields:
          transition = {
            "obs": ...,
            "action": ...,
            "reward": float,
            "next_obs": ...,
            "done": bool,
            "info": {"r": float, "regime": str, "priority": Optional[float]}
          }
        """
        if len(self.storage) >= self.capacity:
            # FIFO eviction: pop oldest, also remove from bins
            self._evict_oldest()

        r_val = float(transition.get("info", {}).get("r", 0.0))
        regime = str(transition.get("info", {}).get("regime", "LOW_VOL"))
        reg_idx = self._regime_idx(regime)
        r_bucket = self._r_bucket_idx(r_val)

        idx = len(self.storage)
        self.storage.append(transition)

        prio = float(transition.get("info", {}).get("priority", 1.0))
        self.priorities.append(max(1e-6, prio))

        key = (reg_idx, r_bucket)
        self.bins.setdefault(key, []).append(idx)

    def sample(self, batch_size: int) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """
        Returns transitions, indices, importance-sampling weights
        """
        if not self.storage:
            return [], np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        # Compute bin weights
        keys = list(self.bins.keys())
        counts = np.array([len(self.bins[k]) for k in keys], dtype=np.float32)
        # Prefer bins with fewer samples (emphasize tails) via inverse power
        probs_bins = (counts + self.cfg.eps) ** (-self.cfg.alpha)
        probs_bins = probs_bins / probs_bins.sum()

        # Draw bins for each sample
        chosen_keys = self.rng.choice(len(keys), size=batch_size, replace=True, p=probs_bins)
        batch_indices: List[int] = []

        # Within each chosen bin, sample by priorities
        for k_idx in chosen_keys:
            key = keys[k_idx]
            idxs = self.bins.get(key, [])
            if not idxs:
                # fallback: uniform over full buffer
                batch_indices.append(int(self.rng.integers(low=0, high=len(self.storage))))
                continue
            # compute normalized priorities for selected indices
            prios = np.array([self.priorities[i] for i in idxs], dtype=np.float32)
            p = prios / prios.sum() if prios.sum() > 0 else np.ones_like(prios) / len(prios)
            pick_local = int(self.rng.choice(len(idxs), p=p))
            batch_indices.append(int(idxs[pick_local]))

        # Importance weights (optional, PER correction)
        # Approximate prob: P(i) = sum_over_bins P(bin) * P(i | bin)
        # We recompute P(i|bin) as priority/sum for the bin where the sample was drawn.
        is_weights = []
        for bi, k_idx in zip(batch_indices, chosen_keys):
            key = keys[int(k_idx)]
            idxs = self.bins.get(key, [bi])
            prios = np.array([self.priorities[i] for i in idxs], dtype=np.float32)
            p_i_bin = float(self.priorities[bi] / max(1e-12, prios.sum()))
            p_i = float(probs_bins[int(k_idx)] * p_i_bin)
            w = (len(self.storage) * p_i) ** (-self.cfg.beta) if p_i > 0 else 1.0
            is_weights.append(w)

        # Normalize IS weights
        is_arr = np.array(is_weights, dtype=np.float32)
        is_arr /= (is_arr.max() + 1e-9)

        transitions = [self.storage[i] for i in batch_indices]
        return transitions, np.array(batch_indices, dtype=np.int64), is_arr

    def update_priorities(self, indices: np.ndarray, new_prios: np.ndarray) -> None:
        for i, p in zip(indices, new_prios):
            self.priorities[int(i)] = float(max(1e-6, p))

    def __len__(self) -> int:
        return len(self.storage)

    # ------------- Internals -------------

    def _evict_oldest(self):
        # remove index 0 and fix indices in bins
        if not self.storage:
            return
        self.storage.pop(0)
        self.priorities.pop(0)
        # Rebuild bins by decrementing indices and dropping -1
        new_bins: Dict[Tuple[int, int], List[int]] = {}
        for key, idxs in self.bins.items():
            remapped = []
            for idx in idxs:
                ni = idx - 1
                if ni >= 0:
                    remapped.append(ni)
            if remapped:
                new_bins[key] = remapped
        self.bins = new_bins

    def _regime_idx(self, regime: str) -> int:
        return self.regime_to_idx.get(regime, 0)

    def _r_bucket_idx(self, r: float) -> int:
        edges = self.cfg.r_edges
        # find bin such that edges[i] <= r < edges[i+1]
        for i in range(len(edges) - 1):
            if r >= edges[i] and r < edges[i + 1]:
                return i
        return len(edges) - 2
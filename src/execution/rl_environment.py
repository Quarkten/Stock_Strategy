
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.agents.reward_functions import calculate_asymmetric_reward
import pandas as pd
from typing import Optional, Dict, Any

from src.execution.backtester import Backtester, Position
from src.strategies.intraday_strategy import IntradayStrategy
from src.utils.data_fetcher import DataFetcher

class TradingEnv(gym.Env):
    """
    A custom OpenAI Gym environment for training a reinforcement learning agent
    to trade stocks.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config: Dict[str, Any], symbol: str, timeframe: str, start: str, end: str):
        super(TradingEnv, self).__init__()

        self.config = config
        self.symbol = symbol
        self.timeframe = timeframe
        self.start = start
        self.end = end

        self.strategy = IntradayStrategy(self.config)
        self.data_fetcher = DataFetcher(self.config)
        self.backtester = Backtester(
            strategy=self.strategy,
            data_fetcher=self.data_fetcher,
            config=self.config,
            symbol=self.symbol,
            timeframe=self.timeframe,
            start=self.start,
            end=self.end,
        )

        # Define action and observation spaces
        self._define_spaces()

        # Load and preprocess data
        self.df = self.backtester._load_data()
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.trade_history = []

    def _define_spaces(self):
        """Defines the action and observation spaces for the environment."""
        # Action space: A single continuous box for all actions.
        # 0: trade_action (0-1: hold, 1-2: buy, 2-3: sell)
        # 1: position_sizing (0.1 to 1.0)
        # 2: stop_loss (0.001 to 0.1)
        # 3: take_profit (0.001 to 0.2)
        self.action_space = spaces.Box(
            low=np.array([0, 0.1, 0.001, 0.001], dtype=np.float32),
            high=np.array([3, 1.0, 0.1, 0.2], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )

        # Observations
        # A single flat box for the observation space.
        # Concatenates market data, position context, and risk metrics.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> tuple:
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.backtester.trades = []
        self.backtester.capital = self.config.get('capital', 100000)
        self.trade_history = []
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: np.ndarray) -> tuple:
        """
        Executes one time step within the environment.

        Args:
            action (np.ndarray): The action to take.

        Returns:
            tuple: A tuple containing the next observation, the reward,
                   a boolean indicating whether the episode is terminated,
                   a boolean indicating whether the episode is truncated,
                   and a dictionary with additional information.
        """
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False  # We don't have a truncation condition for now

        # Execute the action and get the PnL for the step
        pnl = self._execute_action(action)

        # Calculate the reward
        reward = self._calculate_reward(pnl)

        if pnl != 0:
            self.trade_history.append(pnl)

        # Get the next observation
        obs = self._get_observation()
        info = {}

        return obs, reward, terminated, truncated, info

    





    

    

    

    

    def _get_observation(self) -> np.ndarray:
        """Constructs the observation for the current time step."""
        bar = self.df.iloc[self.current_step]
        market_data = np.array([
            bar['open'],
            bar['high'],
            bar['low'],
            bar['close'],
            bar['volume'],
            bar['ema20'],
            bar['atr'],
            bar['bb_upper'],
            bar['bb_lower'],
            bar['macd_hist'],
        ], dtype=np.float32)

        has_position = 1.0 if self.backtester.open_position else 0.0
        position_side = 0.0
        unrealized_pnl = 0.0
        if self.backtester.open_position:
            position_side = 1.0 if self.backtester.open_position.side == "LONG" else 2.0
            unrealized_pnl = self.backtester.open_position.pnl or 0.0

        position_context = np.array([has_position, position_side, unrealized_pnl], dtype=np.float32)

        risk_metrics = np.array([
            self.backtester.daily_pnl,
            self.backtester.consecutive_losses
        ], dtype=np.float32)

        return np.concatenate([market_data, position_context, risk_metrics])

    def _get_observation(self) -> np.ndarray:
        """Constructs the observation for the current time step."""
        bar = self.df.iloc[self.current_step]
        market_data = np.array([
            bar['open'],
            bar['high'],
            bar['low'],
            bar['close'],
            bar['volume'],
            bar['ema20'],
            bar['atr'],
            bar['bb_upper'],
            bar['bb_lower'],
            bar['macd_hist'],
        ], dtype=np.float32)

        has_position = 1.0 if self.backtester.open_position else 0.0
        position_side = 0.0
        unrealized_pnl = 0.0
        if self.backtester.open_position:
            position_side = 1.0 if self.backtester.open_position.side == "LONG" else 2.0
            unrealized_pnl = self.backtester.open_position.pnl or 0.0

        position_context = np.array([has_position, position_side, unrealized_pnl], dtype=np.float32)

        risk_metrics = np.array([
            self.backtester.daily_pnl,
            self.backtester.consecutive_losses
        ], dtype=np.float32)

        return np.concatenate([market_data, position_context, risk_metrics])

    def _get_observation(self) -> np.ndarray:
        """Constructs the observation for the current time step."""
        bar = self.df.iloc[self.current_step]
        market_data = np.array([
            bar['open'],
            bar['high'],
            bar['low'],
            bar['close'],
            bar['volume'],
            bar['ema20'],
            bar['atr'],
            bar['bb_upper'],
            bar['bb_lower'],
            bar['macd_hist'],
        ], dtype=np.float32)

        has_position = 1.0 if self.backtester.open_position else 0.0
        position_side = 0.0
        unrealized_pnl = 0.0
        if self.backtester.open_position:
            position_side = 1.0 if self.backtester.open_position.side == "LONG" else 2.0
            unrealized_pnl = self.backtester.open_position.pnl or 0.0

        position_context = np.array([has_position, position_side, unrealized_pnl], dtype=np.float32)

        risk_metrics = np.array([
            self.backtester.daily_pnl,
            self.backtester.consecutive_losses
        ], dtype=np.float32)

        return np.concatenate([market_data, position_context, risk_metrics])

    def _get_observation(self) -> np.ndarray:
        """Constructs the observation for the current time step."""
        bar = self.df.iloc[self.current_step]
        market_data = np.array([
            bar['open'],
            bar['high'],
            bar['low'],
            bar['close'],
            bar['volume'],
            bar['ema20'],
            bar['atr'],
            bar['bb_upper'],
            bar['bb_lower'],
            bar['macd_hist'],
        ], dtype=np.float32)

        has_position = 1.0 if self.backtester.open_position else 0.0
        position_side = 0.0
        unrealized_pnl = 0.0
        if self.backtester.open_position:
            position_side = 1.0 if self.backtester.open_position.side == "LONG" else 2.0
            unrealized_pnl = self.backtester.open_position.pnl or 0.0

        position_context = np.array([has_position, position_side, unrealized_pnl], dtype=np.float32)

        risk_metrics = np.array([
            self.backtester.daily_pnl,
            self.backtester.consecutive_losses
        ], dtype=np.float32)

        return np.concatenate([market_data, position_context, risk_metrics])

    def _get_observation(self) -> np.ndarray:
        """Constructs the observation for the current time step."""
        bar = self.df.iloc[self.current_step]
        market_data = np.array([
            bar['open'],
            bar['high'],
            bar['low'],
            bar['close'],
            bar['volume'],
            bar['ema20'],
            bar['atr'],
            bar['bb_upper'],
            bar['bb_lower'],
            bar['macd_hist'],
        ], dtype=np.float32)

        has_position = 1.0 if self.backtester.open_position else 0.0
        position_side = 0.0
        unrealized_pnl = 0.0
        if self.backtester.open_position:
            position_side = 1.0 if self.backtester.open_position.side == "LONG" else 2.0
            unrealized_pnl = self.backtester.open_position.pnl or 0.0

        position_context = np.array([has_position, position_side, unrealized_pnl], dtype=np.float32)

        risk_metrics = np.array([
            self.backtester.daily_pnl,
            self.backtester.consecutive_losses
        ], dtype=np.float32)

        return np.concatenate([market_data, position_context, risk_metrics])

    def _execute_action(self, action: np.ndarray) -> float:
        """
        Executes the given action in the backtester and returns the PnL.
        """
        trade_action_raw, position_sizing, stop_loss, take_profit = action

        # Decode the trade action
        if 0 <= trade_action_raw < 1:
            trade_action = 0  # Hold
        elif 1 <= trade_action_raw < 2:
            trade_action = 1  # Buy
        else:
            trade_action = 2  # Sell

        pnl = 0.0

        # Check for exits first
        if self.backtester.open_position:
            pnl = self.backtester.check_for_exit(self.df.iloc[self.current_step])
            if pnl != 0.0:
                return pnl

        # Execute new trade actions
        if trade_action == 1:  # Buy
            self._execute_trade("LONG", position_sizing, stop_loss, take_profit)
        elif trade_action == 2:  # Sell
            self._execute_trade("SHORT", position_sizing, stop_loss, take_profit)

        return pnl

    def _execute_trade(self, side: str, position_sizing: float, stop_loss_pct: float, take_profit_pct: float):
        """Executes a trade based on the given side and action."""
        if self.backtester.open_position:
            return  # Can't open a new position if one is already open

        bar = self.df.iloc[self.current_step]
        entry_price = bar['close']

        if side == "LONG":
            stop_price = entry_price * (1 - stop_loss_pct)
            target_price = entry_price * (1 + take_profit_pct)
        else:  # SHORT
            stop_price = entry_price * (1 + stop_loss_pct)
            target_price = entry_price * (1 - take_profit_pct)

        # Calculate position size
        risk_per_share = abs(entry_price - stop_price)
        risk_amount = self.backtester.capital * self.backtester.risk_per_trade_pct * position_sizing
        size = int(np.floor(risk_amount / risk_per_share)) if risk_per_share > 0 else 0

        if size > 0:
            self.backtester.open_position = Position(
                side=side,
                entry_time=bar.name,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                size=size,
                setup_name="rl_agent",
            )

    def _calculate_reward(self, pnl: float) -> float:
        """
        Calculates the reward for the current step.
        """
        # This is a placeholder for the is_tail_win logic
        is_tail_win = self._is_tail_win(pnl)

        return calculate_asymmetric_reward(
            pnl,
            self.backtester.daily_pnl,
            is_tail_win
        )

    def _is_tail_win(self, pnl: float) -> bool:
        """
        Determines if a given PnL value is a "tail win".
        """
        if pnl <= 0 or not self.trade_history:
            return False

        wins = [p for p in self.trade_history if p > 0]
        if not wins:
            return False

        win_percentile = np.percentile(wins, 80)
        return pnl >= win_percentile

    def render(self, mode='human', close=False):
        """Renders the environment to the screen."""
        pass

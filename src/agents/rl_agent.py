import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import talib
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from typing import Dict, Any, Tuple


class ForexTradingEnv(gym.Env):
    """Custom trading environment for RL agent"""
    
    def __init__(self, data, initial_balance=10000, lookback_window=20):
        super(ForexTradingEnv, self).__init__()
        
        # Preprocess data
        self.data = self._preprocess_data(data.reset_index(drop=True))
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price data + portfolio info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback_window * 5 + 3,), dtype=np.float32
        )
        
        self.reset()
    
    def _preprocess_data(self, df):
        """Add technical indicators and normalize data"""
        # Ensure required columns exist
        if '4. close' not in df.columns:
            df['4. close'] = df['close']
        if '5. volume' not in df.columns:
            df['5. volume'] = 0
            
        # Add technical indicators
        df['rsi'] = talib.RSI(df['4. close'], timeperiod=14)
        df['sma_20'] = talib.SMA(df['4. close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['4. close'], timeperiod=50)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['4. close'])
        
        # Normalize values
        for col in ['4. close', '5. volume', 'rsi', 'sma_20', 'sma_50', 'macd', 'macd_signal']:
            if col in df.columns:
                df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-10)
                
        return df.fillna(0)
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation with technical indicators"""
        # Get price data for lookback window
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        obs_data = []
        for i in range(start_idx, end_idx):
            if i < len(self.data):
                row = self.data.iloc[i]
                obs_data.extend([
                    row.get('4. close', 0),
                    row.get('5. volume', 0),
                    row.get('rsi', 0),
                    row.get('sma_20', 0),
                    row.get('macd', 0)
                ])
            else:
                obs_data.extend([0]*5)
        
        # Pad if necessary
        while len(obs_data) < self.lookback_window * 5:
            obs_data.extend([0]*5)
        
        # Add portfolio information
        current_price = self.data.iloc[min(self.current_step, len(self.data)-1)].get('4. close', 0)
        portfolio_data = [
            self.balance / self.initial_balance,
            self.shares_held,
            current_price
        ]
        
        return np.array(obs_data + portfolio_data, dtype=np.float32)
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
        
        current_price = self.data.iloc[self.current_step].get('4. close', 0)
        prev_net_worth = self.net_worth
        
        # Execute action
        if action == 1:  # Buy
            if self.balance >= current_price:
                shares_to_buy = self.balance // current_price
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * current_price
                self.trades.append(('BUY', shares_to_buy, current_price))
        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.trades.append(('SELL', self.shares_held, current_price))
                self.shares_held = 0
        
        # Calculate net worth
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Calculate reward with clipping to avoid extreme values
        reward = np.clip(
            (self.net_worth - prev_net_worth) / self.initial_balance,
            -1.0, 1.0
        )
        
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, False, {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held
        }

class RLTrader:
    """Reinforcement Learning Trader using PPO"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}  # Store trained models by symbol
        self.models_dir = config.get('models_dir', 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
    def train_model(self, symbol, data, episodes=100):
        """Train the RL model on historical data"""
        print(f"Training RL model for {symbol}...")
        
        try:
            # Create and validate environment
            env = ForexTradingEnv(data)
            check_env(env)  # Validate the environment
            
            # Wrap environment for vectorized training
            env = DummyVecEnv([lambda: env])
            
            # Create PPO model
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                tensorboard_log=f"{self.models_dir}/logs/{symbol}"
            )
            
            # Train the model
            model.learn(total_timesteps=episodes*1000)
            
            # Save the model
            model_path = f"{self.models_dir}/{symbol}_ppo"
            model.save(model_path)
            self.models[symbol] = model
            
            print(f"Training completed for {symbol}")
            return model
            
        except Exception as e:
            print(f"Error training RL model for {symbol}: {e}")
            return self._create_mock_model()
    
    def _create_mock_model(self):
        """Create a simple mock model for fallback"""
        env = ForexTradingEnv(pd.DataFrame({'close': [1]}))
        model = PPO(
            "MlpPolicy",
            DummyVecEnv([lambda: env]),
            verbose=0
        )
        return model
    
    def predict_action(self, symbol, current_data):
        """Predict trading action for current market data"""
        try:
            model = self.models.get(symbol)
            if model is None:
                # Try to load saved model
                model_path = f"{self.models_dir}/{symbol}_ppo"
                if os.path.exists(model_path + ".zip"):
                    model = PPO.load(model_path)
                    self.models[symbol] = model
                else:
                    return 0  # Hold
            
            # Create temporary environment
            env = ForexTradingEnv(current_data)
            obs = env.reset()
            
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            return int(action)
            
        except Exception as e:
            print(f"Error predicting action for {symbol}: {e}")
            return 0  # Default to hold
    
    def get_trading_signal(self, symbol, data):
        """Get trading signal (simplified interface)"""
        action = self.predict_action(symbol, data)
        return {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(action, 'HOLD')
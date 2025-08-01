import pandas as pd
import numpy as np
import h5py
from src.utils.data_fetcher import DataFetcher
from src.utils.features import add_indicators
from train_rl_agent import load_merged_config

def prepare_offline_dataset(trade_log_path: str, ohlcv_data: pd.DataFrame, output_path: str):
    """
    Prepares an offline RL dataset from a trade log and OHLCV data.

    Args:
        trade_log_path: Path to the trade log CSV file.
        ohlcv_data: A pandas DataFrame with OHLCV data.
        output_path: Path to save the processed dataset.
    """
    trade_log = pd.read_csv(trade_log_path)

    observations = []
    actions = []
    rewards = []
    terminals = []

    for _, trade in trade_log.iterrows():
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time'])

        trade_data = ohlcv_data.loc[entry_time:exit_time]

        for i in range(len(trade_data) - 1):
            obs = trade_data.iloc[i].values
            next_obs = trade_data.iloc[i+1].values

            # For simplicity, we'll use a placeholder for actions and rewards
            action = np.zeros(9) # Placeholder
            reward = 0 # Placeholder

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminals.append(0)

        # Add the terminal transition
        if len(trade_data) > 0:
            observations.append(trade_data.iloc[-1].values)
            actions.append(np.zeros(9)) # Placeholder
            rewards.append(trade['pnl'])
            terminals.append(1)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('observations', data=np.array(observations, dtype=np.float32))
        f.create_dataset('actions', data=np.array(actions, dtype=np.float32))
        f.create_dataset('rewards', data=np.array(rewards, dtype=np.float32))
        f.create_dataset('terminals', data=np.array(terminals, dtype=np.float32))

    print(f"Offline dataset saved to {output_path}")

if __name__ == '__main__':
    cfg = load_merged_config()
    data_fetcher = DataFetcher(cfg)
    df = add_indicators(data_fetcher.fetch_data(cfg.get("target_symbol", "SPY"), "5min", None, None, cfg), cfg)
    prepare_offline_dataset('data/baseline_trades.csv', df, 'data/offline_dataset.h5')

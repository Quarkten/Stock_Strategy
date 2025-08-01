import numpy as np
import pandas as pd

def generate_synthetic_lob(ohlcv_data: pd.DataFrame, n_levels: int = 10, base_spread: float = 0.01):
    """
    Generates synthetic limit order book (LOB) data from OHLCV data.

    Args:
        ohlcv_data: A pandas DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
        n_levels: The number of levels to generate for each side of the LOB.
        base_spread: The base spread between the best bid and ask.

    Returns:
        A list of LOB snapshots, where each snapshot is a dictionary with 'bids' and 'asks'.
    """
    lobs = []
    for _, row in ohlcv_data.iterrows():
        mid_price = (row['high'] + row['low']) / 2
        volatility = (row['high'] - row['low']) / mid_price
        spread = base_spread + volatility

        lob = {'bids': [], 'asks': []}

        # Generate bids
        for i in range(n_levels):
            price = mid_price - (spread / 2) * (1 + i * 0.1)
            size = row['volume'] * np.random.uniform(0.1, 0.5) * (1 - i * 0.05)
            lob['bids'].append((price, size))

        # Generate asks
        for i in range(n_levels):
            price = mid_price + (spread / 2) * (1 + i * 0.1)
            size = row['volume'] * np.random.uniform(0.1, 0.5) * (1 - i * 0.05)
            lob['asks'].append((price, size))

        lobs.append(lob)

    return lobs

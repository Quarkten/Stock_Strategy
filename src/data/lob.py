from collections import defaultdict
from typing import List, Tuple

class LOB:
    """
    A simple limit order book (LOB) implementation.
    """
    def __init__(self):
        self.bids = defaultdict(float)
        self.asks = defaultdict(float)

    def add_order(self, side: str, price: float, size: float):
        """
        Add a new order to the book.
        """
        if side.upper() == 'BID':
            self.bids[price] += size
        elif side.upper() == 'ASK':
            self.asks[price] += size

    def remove_order(self, side: str, price: float, size: float):
        """
        Remove an existing order from the book.
        """
        if side.upper() == 'BID':
            if price in self.bids:
                self.bids[price] -= size
                if self.bids[price] <= 0:
                    del self.bids[price]
        elif side.upper() == 'ASK':
            if price in self.asks:
                self.asks[price] -= size
                if self.asks[price] <= 0:
                    del self.asks[price]

    def match_order(self, side: str, size: float, impact_factor: float = 1e-5) -> List[Tuple[float, float]]:
        """
        Match an incoming market order against the book and apply market impact.
        """
        trades = []
        if side.upper() == 'BUY':
            while size > 0 and self.asks:
                best_ask = min(self.asks.keys())
                trade_size = min(size, self.asks[best_ask])
                trades.append((best_ask, trade_size))
                self.remove_order('ASK', best_ask, trade_size)
                size -= trade_size
                self._apply_market_impact('BUY', trade_size, impact_factor)
        elif side.upper() == 'SELL':
            while size > 0 and self.bids:
                best_bid = max(self.bids.keys())
                trade_size = min(size, self.bids[best_bid])
                trades.append((best_bid, trade_size))
                self.remove_order('BID', best_bid, trade_size)
                size -= trade_size
                self._apply_market_impact('SELL', trade_size, impact_factor)
        return trades

    def _apply_market_impact(self, side: str, trade_size: float, impact_factor: float):
        """
        Apply market impact to the LOB after a trade.
        """
        if side.upper() == 'BUY':
            # Shift ask prices up
            new_asks = defaultdict(float)
            for price, size in self.asks.items():
                new_price = price + trade_size * impact_factor
                new_asks[new_price] += size
            self.asks = new_asks
        elif side.upper() == 'SELL':
            # Shift bid prices down
            new_bids = defaultdict(float)
            for price, size in self.bids.items():
                new_price = price - trade_size * impact_factor
                new_bids[new_price] += size
            self.bids = new_bids

    def get_snapshot(self, n_levels: int = 10) -> dict:
        """
        Return a snapshot of the current state of the LOB.
        """
        bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
        asks = sorted(self.asks.items(), key=lambda x: x[0])
        return {'bids': bids[:n_levels], 'asks': asks[:n_levels]}

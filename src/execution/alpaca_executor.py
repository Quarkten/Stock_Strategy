import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

class AlpacaExecutor:
    def __init__(self, config):
        self.api = tradeapi.REST(
            key_id=config['alpaca_api_key'],
            secret_key=config['alpaca_secret_key'],
            base_url=config['alpaca_base_url'],
            api_version='v2'
        )

    def execute_trade(self, symbol, action, size):
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=size,
                side=action.lower(),
                type='market',
                time_in_force='gtc'
            )
            print(f"Successfully submitted {action} order for {size} shares of {symbol}.")
            return order
        except APIError as e:
            print(f"Error executing trade: {e}")
            return None

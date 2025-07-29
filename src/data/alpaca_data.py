import os
import logging
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class AlpacaData:
    def __init__(self, config):
        self.config = config
        load_dotenv() # Load environment variables from .env file

        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets') # Default to paper trading

        print(f"DEBUG: ALPACA_API_KEY in AlpacaData: {self.api_key}")
        print(f"DEBUG: ALPACA_SECRET_KEY in AlpacaData: {self.secret_key}")
        print(f"DEBUG: ALPACA_BASE_URL in AlpacaData: {self.base_url}")

        if not self.api_key or not self.secret_key:
            print("Warning: Alpaca API keys not found in environment variables. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
            # Fallback to hardcoded keys for testing if necessary, but warn the user
            # self.api_key = 'YOUR_ALPACA_API_KEY'
            # self.secret_key = 'YOUR_ALPACA_SECRET_KEY'
            # self.base_url = 'https://paper-api.alpaca.markets'

        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url=self.base_url
        )

    def get_historical_data(self, symbol, interval='1D', period='1y'):
        """
        Fetch historical data from Alpaca.
        Intervals: 1Min, 5Min, 15Min, 1H, 1D
        Period is not directly used by Alpaca's get_bars, instead we use start/end dates.
        """
        try:
            print(f"Fetching data for {symbol} with interval {interval} from Alpaca...")

            # Map intervals to Alpaca intervals
            alpaca_interval_map = {
                '1min': '1Min',
                '5min': '5Min',
                '15min': '15Min',
                '30min': '30Min', # Alpaca supports 30Min
                '60min': '1H',
                '1h': '1H',
                'daily': '1D',
                '1d': '1D',
                'weekly': '1D', # Alpaca doesn't have direct weekly bars, need to aggregate daily
                '1w': '1D',
                'monthly': '1D', # Alpaca doesn't have direct monthly bars, need to aggregate daily
                '1M': '1D',
            }
            alpaca_interval = alpaca_interval_map.get(interval, '1D')

            # Calculate start and end dates based on period
            end_date = datetime.now()
            if period == '1mo':
                start_date = end_date - timedelta(days=30)
            elif period == '1y':
                start_date = end_date - timedelta(days=365)
            elif period == '5y':
                start_date = end_date - timedelta(days=5 * 365)
            elif period == '10y':
                start_date = end_date - timedelta(days=10 * 365)
            elif period == '20y':
                start_date = end_date - timedelta(days=20 * 365)
            elif period == '60d':
                start_date = end_date - timedelta(days=60)
            else: # Default to 1 year
                start_date = end_date - timedelta(days=365)

            # Alpaca's get_bars returns a BarSet object
            # Ensure correct date format for Alpaca API
            # For daily intervals, Alpaca expects YYYY-MM-DD
            # For intraday intervals, it expects RFC3339 (ISO 8601) format
            if alpaca_interval == '1D':
                start_param = start_date.strftime("%Y-%m-%d")
                end_param = end_date.strftime("%Y-%m-%d")
            else:
                start_param = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                end_param = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

            bars = self.api.get_bars(
                symbol,
                alpaca_interval,
                start=start_param,
                end=end_param,
                limit=10000, # Max limit for historical data
                feed='iex'  # Use IEX feed for free data
            ).df

            if bars is None or bars.empty:
                print(f"No data returned for {symbol} from Alpaca")
                return pd.DataFrame()

            # Rename columns to match standard format
            bars.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }, inplace=True)

            # Ensure datetime index
            bars.index = pd.to_datetime(bars.index)

            print(f"Successfully fetched {len(bars)} data points for {symbol} from Alpaca")
            return bars

        except APIError as e:
            logging.error(f"Alpaca API error fetching historical data for {symbol}: {e}")
            return None
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol} from Alpaca: {e}")
            return None

    def get_current_price(self, symbol):
        """Get real-time price from Alpaca"""
        try:
            print(f"Fetching current price for {symbol} from Alpaca...")
            bar = self.api.get_latest_bar(symbol)
            if bar:
                df = pd.DataFrame([{
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                }], index=[datetime.now()])
                print(f"Successfully fetched current price for {symbol} from Alpaca: {bar.close}")
                return df
            else:
                print(f"No current price data found for {symbol} from Alpaca.")
                return None
        except APIError as e:
            logging.error(f"Alpaca API error getting current price for {symbol}: {e}")
            return None
        except Exception as e:
            logging.error(f"Error getting current price for {symbol} from Alpaca: {e}")
            return None

    def get_chart_data_for_llm(self, symbol, timeframes=['daily', 'weekly', '4h', '1h']):
        """Fetch specific chart data for LLM analysis from Alpaca"""
        chart_data = {}
        for tf in timeframes:
            # Alpaca doesn't have '4h' or 'weekly' directly, use '1H' or '1D' and let LLM interpret
            alpaca_tf = '1H' if tf == '4h' else ('1D' if tf == 'weekly' else tf)
            data = self.get_historical_data(symbol, alpaca_tf, '1y') # Fetch 1 year for context
            if data is not None and not data.empty:
                # Take a recent subset for the LLM prompt, e.g., last 5 candles
                chart_data[tf] = data.tail(5).to_string()
            else:
                chart_data[tf] = f"No data available for {tf}"
        return chart_data

    def get_multi_timeframe_data(self, symbol):
        """Fetch data for all configured timeframes from Alpaca"""
        data = {}
        # Assuming config['timeframes'] contains keys like 'short_term' and 'trend'
        # and their values are lists of intervals like ['1min', '5min', '15min']
        all_timeframes = self.config['timeframes'].get('short_term', []) + [self.config['timeframes'].get('trend')]
        for tf in all_timeframes:
            if tf: # Ensure tf is not None
                data[tf] = self.get_historical_data(symbol, tf)
        return data

    def get_news(self, symbol=None, start_date=None, end_date=None, limit=10):
        """
        Fetch news articles from Alpaca.
        Can filter by symbol, date range, and limit the number of articles.
        """
        try:
            print(f"Fetching news from Alpaca for symbol {symbol or 'all'}...")
            news = self.api.get_news(
                symbol=symbol,
                start=start_date if start_date else None,
                end=end_date if end_date else None,
                limit=limit
            )
            
            if not news:
                print(f"No news found for {symbol or 'all'} from Alpaca.")
                return []

            news_list = []
            for article in news:
                news_list.append({
                    'headline': article.headline,
                    'summary': article.summary,
                    'created_at': article.created_at.isoformat(),
                    'symbols': article.symbols,
                    'url': article.url
                })
            print(f"Successfully fetched {len(news_list)} news articles from Alpaca.")
            return news_list

        except APIError as e:
            logging.error(f"Alpaca API error fetching news: {e}")
            return []
        except Exception as e:
            logging.error(f"Error fetching news from Alpaca: {e}")
            return []

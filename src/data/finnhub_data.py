import os
import logging
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import finnhub

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class FinnhubData:
    def __init__(self, config):
        self.config = config
        load_dotenv() # Load environment variables from .env file

        self.api_key = os.getenv('FINNHUB_API_KEY')

        if not self.api_key:
            print("Warning: Finnhub API key not found in environment variables. Please set FINNHUB_API_KEY.")

        # Initialize Finnhub client (uncomment and adjust if you have a specific client library)
        self.client = finnhub.Client(api_key=self.api_key)

    def get_historical_data(self, symbol, interval='1D', period='1y'):
        """
        Fetch historical data from Finnhub.
        Finnhub's historical data is typically for candles (OHLCV).
        Intervals: 1, 5, 15, 30, 60, D, W, M (minutes, daily, weekly, monthly)
        """
        try:
            print(f"Fetching historical data for {symbol} with interval {interval} from Finnhub...")

            # Map intervals to Finnhub intervals
            finnhub_interval_map = {
                '1min': '1',
                '5min': '5',
                '15min': '15',
                '30min': '30',
                '1h': '60',
                '1D': 'D',
                'daily': 'D',
                '1w': 'W',
                'weekly': 'W',
                '1M': 'M',
                'monthly': 'M',
            }
            finnhub_interval = finnhub_interval_map.get(interval, 'D')

            # Calculate start and end timestamps
            end_timestamp = int(datetime.now().timestamp())
            if period == '1mo':
                start_timestamp = int((datetime.now() - timedelta(days=30)).timestamp())
            elif period == '1y':
                start_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
            elif period == '5y':
                start_timestamp = int((datetime.now() - timedelta(days=5 * 365)).timestamp())
            elif period == '10y':
                start_timestamp = int((datetime.now() - timedelta(days=10 * 365)).timestamp())
            elif period == '20y':
                start_timestamp = int((datetime.now() - timedelta(days=20 * 365)).timestamp())
            elif period == '60d':
                start_timestamp = int((datetime.now() - timedelta(days=60)).timestamp())
            else: # Default to 1 year
                start_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())

            res = self.client.stock_candles(symbol, finnhub_interval, start_timestamp, end_timestamp)

            bars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

            if res and res['s'] == 'ok':
                data = {
                    'time': [datetime.fromtimestamp(t) for t in res['t']],
                    'open': res['o'],
                    'high': res['h'],
                    'low': res['l'],
                    'close': res['c'],
                    'volume': res['v']
                }
                bars = pd.DataFrame(data).set_index('time')
                bars.index = pd.to_datetime(bars.index)

            if bars.empty:
                print(f"No data returned for {symbol} from Finnhub")
                return pd.DataFrame()

            print(f"Successfully fetched {len(bars)} data points for {symbol} from Finnhub")
            return bars

        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol} from Finnhub: {e}")
            return None

    def get_current_price(self, symbol):
        """
        Get real-time price from Finnhub.
        """
        try:
            print(f"Fetching current price for {symbol} from Finnhub...")
            quote = self.client.quote(symbol)
            if quote and 'c' in quote:
                current_price = quote['c']
                df = pd.DataFrame([{
                    'open': quote.get('o'),
                    'high': quote.get('h'),
                    'low': quote.get('l'),
                    'close': current_price,
                    'volume': quote.get('v')
                }], index=[datetime.now()])
                print(f"Successfully fetched current price for {symbol} from Finnhub: {current_price}")
                return df
            else:
                print(f"No current price data found for {symbol} from Finnhub.")
                return None
        except Exception as e:
            logging.error(f"Error getting current price for {symbol} from Finnhub: {e}")
            return None

    def get_chart_data_for_llm(self, symbol, timeframes=['daily', 'weekly', '4h', '1h']):
        """Fetch specific chart data for LLM analysis from Finnhub (placeholder)"""
        chart_data = {}
        for tf in timeframes:
            data = self.get_historical_data(symbol, tf, '1y') # Fetch 1 year for context
            if data is not None and not data.empty:
                chart_data[tf] = data.tail(5).to_string()
            else:
                chart_data[tf] = f"No data available for {tf}"
        return chart_data

    def get_multi_timeframe_data(self, symbol):
        """Fetch data for all configured timeframes from Finnhub (placeholder)"""
        data = {}
        all_timeframes = self.config['timeframes'].get('short_term', []) + [self.config['timeframes'].get('trend')]
        for tf in all_timeframes:
            if tf: # Ensure tf is not None
                data[tf] = self.get_historical_data(symbol, tf)
        return data

    def get_news(self, symbol=None, start_date=None, end_date=None, limit=10):
        """
        Fetch news articles from Finnhub.
        """
        try:
            print(f"Fetching news from Finnhub for symbol {symbol or 'all'}...")
            # Placeholder for fetching news
            # news = self.client.company_news(symbol, _from=start_date.strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'))

            # For now, return an empty list as a placeholder
            news_list = []

            # Example of how you might process the results from Finnhub's company_news
            # if news:
            #     for article in news:
            #         news_list.append({
            #             'headline': article.get('headline'),
            #             'summary': article.get('summary'),
            #             'created_at': datetime.fromtimestamp(article.get('datetime')).isoformat(),
            #             'symbols': article.get('related').split(',') if article.get('related') else [],
            #             'url': article.get('url')
            #         })

            print(f"Successfully fetched {len(news_list)} news articles from Finnhub.")
            return news_list

        except Exception as e:
            logging.error(f"Error fetching news from Finnhub: {e}")
            return []

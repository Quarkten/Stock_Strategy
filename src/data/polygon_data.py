import os
import logging
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
# Assuming a Polygon API client library is installed, e.g., polygon-api-client
from polygon import RESTClient

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class PolygonData:
    def __init__(self, config):
        self.config = config
        load_dotenv() # Load environment variables from .env file

        # Prefer env var; fallback to config key if provided
        self.api_key = os.getenv('POLYGON_API_KEY') or self.config.get('polygon_api_key')

        if not self.api_key:
            logging.error("Polygon API key missing. Set POLYGON_API_KEY in env or polygon_api_key in config/config.yaml.")
            # Defer client init; methods will raise clearer error
            self.client = None
            return

        # Initialize Polygon client
        try:
            self.client = RESTClient(self.api_key)
        except Exception as e:
            logging.error(f"Failed to initialize Polygon RESTClient: {e}")
            self.client = None

    def get_historical_data(self, symbol, interval='1D', start=None, end=None, period='1y'):
        """
        Fetch historical data from Polygon.
        Intervals: 1Min, 5Min, 15Min, 1H, 1D
        """
        try:
            if self.client is None:
                raise RuntimeError("Polygon REST client is not initialized due to missing/invalid API key.")
            print(f"Fetching data for {symbol} with interval {interval} from Polygon...")

            # Map intervals to Polygon intervals (adjust based on actual Polygon API client)
            # This is a placeholder and needs to be adjusted based on the actual Polygon API client's requirements
            multiplier = 1
            timespan = 'day'
            if interval == '1min':
                timespan = 'minute'
            elif interval == '5min':
                multiplier = 5
                timespan = 'minute'
            elif interval == '15min':
                multiplier = 15
                timespan = 'minute'
            elif interval == '1h':
                timespan = 'hour'
            elif interval == '1D':
                timespan = 'day'
            else:
                logging.warning(f"Unsupported interval for Polygon: {interval}. Defaulting to 1D.")
                timespan = 'day'

            # Determine date range: prefer explicit start/end if provided; otherwise derive from period
            if start is not None and end is not None:
                # Accept common formats: 'YYYYMMDD', 'YYYY-MM-DD', datetime
                def _parse_date(d):
                    if isinstance(d, datetime):
                        return d
                    s = str(d)
                    try:
                        if len(s) == 8 and s.isdigit():
                            return datetime.strptime(s, "%Y%m%d")
                        return datetime.fromisoformat(s)
                    except Exception:
                        # Fallback: treat as already acceptable string and let API handle; but keep as datetime if possible
                        return datetime.fromisoformat(s)
                start_date = _parse_date(start)
                end_date = _parse_date(end)
            else:
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

            # Polygon REST API expects ISO date strings for from_/to
            from_date_str = start_date.strftime('%Y-%m-%d')
            to_date_str = end_date.strftime('%Y-%m-%d')

            # Placeholder for fetching data using the Polygon client
            # Replace with actual API call, e.g.:
            # aggs = self.client.get_aggs(
            #     ticker=symbol,
            #     multiplier=multiplier,
            #     timespan=timespan,
            #     from_=from_date_str,
            #     to=to_date_str,
            #     limit=50000 # Adjust limit as needed
            # )

            aggs = self.client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_date_str,
                to=to_date_str,
                limit=50000 # Adjust limit as needed
            )

            data = []
            for a in aggs:
                data.append({
                    'time': datetime.fromtimestamp(a.timestamp / 1000), # Convert ms to datetime
                    'open': a.open,
                    'high': a.high,
                    'low': a.low,
                    'close': a.close,
                    'volume': a.volume
                })
            bars = pd.DataFrame(data).set_index('time')
            bars.index = pd.to_datetime(bars.index)


            if bars.empty:
                print(f"No data returned for {symbol} from Polygon")
                return pd.DataFrame()

            print(f"Successfully fetched {len(bars)} data points for {symbol} from Polygon")
            return bars

        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol} from Polygon: {e}")
            return None

    def get_current_price(self, symbol):
        """Get real-time price from Polygon (placeholder)"""
        logging.warning("get_current_price not implemented for PolygonData yet.")
        return None

    def get_chart_data_for_llm(self, symbol, timeframes=['daily', 'weekly', '4h', '1h']):
        """Fetch specific chart data for LLM analysis from Polygon (placeholder)"""
        chart_data = {}
        for tf in timeframes:
            data = self.get_historical_data(symbol, tf, '1y') # Fetch 1 year for context
            if data is not None and not data.empty:
                chart_data[tf] = data.tail(5).to_string()
            else:
                chart_data[tf] = f"No data available for {tf}"
        return chart_data

    def get_multi_timeframe_data(self, symbol):
        """Fetch data for all configured timeframes from Polygon (placeholder)"""
        data = {}
        # Assuming config['timeframes'] contains keys like 'short_term' and 'trend'
        # and their values are lists of intervals like ['1min', '5min', '15min']
        all_timeframes = self.config['timeframes'].get('short_term', []) + [self.config['timeframes'].get('trend')]
        for tf in all_timeframes:
            if tf: # Ensure tf is not None
                data[tf] = self.get_historical_data(symbol, tf)
        return data

    def get_news(self, symbol=None, start_date=None, end_date=None, limit=10):
        """Fetch news articles from Polygon (placeholder)"""
        logging.warning("get_news not implemented for PolygonData yet.")
        return []

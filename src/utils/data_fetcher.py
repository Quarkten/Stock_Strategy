import requests
import os
import json
from datetime import datetime, timedelta
import logging
import pandas as pd

from src.data.alpaca_data import AlpacaData
from src.data.finnhub_data import FinnhubData
from src.data.polygon_data import PolygonData

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, config):
        self.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_KEY") or config.get('alpha_vantage_api_key')
        self.fred_api_key = os.getenv("FRED_API_KEY") or config.get('fred_api_key')
        self.config = config

    def _choose_provider(self, explicit: str | None = None):
        """
        Select a data provider class instance based on explicit selection or available API keys.
        Order: explicit -> Alpaca -> Finnhub -> Polygon.
        """
        if explicit:
            name = explicit.lower()
            if name == "alpaca":
                return "alpaca", AlpacaData(self.config)
            if name == "finnhub":
                return "finnhub", FinnhubData(self.config)
            if name == "polygon":
                return "polygon", PolygonData(self.config)

        # Auto-select based on environment keys presence
        if os.getenv('ALPACA_API_KEY') and os.getenv('ALPACA_SECRET_KEY'):
            return "alpaca", AlpacaData(self.config)
        if os.getenv('FINNHUB_API_KEY'):
            return "finnhub", FinnhubData(self.config)
        if os.getenv('POLYGON_API_KEY'):
            return "polygon", PolygonData(self.config)

        # Default to Alpaca (may return empty if keys missing, but keeps behavior predictable)
        return "alpaca", AlpacaData(self.config)

    def _map_timeframe(self, timeframe: str, provider: str) -> str:
        """
        Normalize common timeframe aliases to provider-specific intervals.
        Accepts inputs like: '1day','1d','daily','1hour','1h','60min','1min','5min','15min','30min'.
        """
        tf = timeframe.lower().strip()

        # Canonicalize
        alias_map = {
            "1day": "1d", "daily": "1d", "1d": "1d", "day": "1d",
            "1hour": "1h", "1hr": "1h", "hour": "1h", "60min": "1h", "1h": "1h",
            "1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min"
        }
        canonical = alias_map.get(tf, tf)

        provider = provider.lower()
        if provider == "alpaca":
            # AlpacaData.get_historical_data map accepts '1D','1H','1Min','5Min','15Min','30Min'
            mapping = {
                "1d": "1D",
                "1h": "1H",
                "1min": "1min",
                "5min": "5min",
                "15min": "15min",
                "30min": "30min",
            }
            return mapping.get(canonical, "1D")
        if provider == "finnhub":
            # FinnhubData.get_historical_data expects e.g. '1D','1h','1min','5min','15min','30min'
            # Internally it maps to its own codes; we pass a readable alias.
            mapping = {
                "1d": "1D",
                "1h": "1h",
                "1min": "1min",
                "5min": "5min",
                "15min": "15min",
                "30min": "30min",
            }
            return mapping.get(canonical, "1D")
        if provider == "polygon":
            # PolygonData.get_historical_data expects '1D','1h','1min','5min','15min','30min'
            mapping = {
                "1d": "1D",
                "1h": "1h",
                "1min": "1min",
                "5min": "5min",
                "15min": "15min",
                "30min": "30min",
            }
            return mapping.get(canonical, "1D")

        return "1D"

    def fetch_data(self, symbol: str, timeframe: str, limit: int | None = None, provider: str | None = None, period: str | None = None, **kwargs):
        """
        Minimal OHLCV fetch wrapper used by run.py.
        - Logs inputs
        - Auto-selects provider (or uses explicit)
        - Maps timeframe aliases to provider-specific intervals
        - Calls provider.get_historical_data
        - Trims to last 'limit' rows if provided
        Returns: pandas DataFrame indexed by datetime with columns [open, high, low, close, volume]
        """
        prov_name, prov = self._choose_provider(provider)
        interval = self._map_timeframe(timeframe, prov_name)
        logger.info(f"DataFetcher.fetch_data: symbol={symbol}, timeframe={timeframe} -> interval={interval}, limit={limit}, provider={prov_name}")
        # If polygon API key is missing/invalid, skip polygon as primary to avoid repeated failing calls
        if prov_name == "polygon":
            poly_key = os.getenv('POLYGON_API_KEY') or self.config.get('polygon_api_key')
            if not poly_key:
                logger.warning("Polygon API key missing/invalid; skipping Polygon as primary and using Alpaca/Finnhub fallbacks.")
                prov_name, prov = "alpaca", AlpacaData(self.config)
                interval = self._map_timeframe(timeframe, prov_name)
        # Short-circuit: if polygon is selected but API key invalid, avoid repeated attempts; rely on fallbacks
        if prov_name == "polygon" and not (os.getenv('POLYGON_API_KEY') or self.config.get('polygon_api_key')):
            logger.warning("Polygon API key missing or invalid; skipping Polygon and using fallbacks.")
            # Force first provider to Alpaca, then Finnhub
            prov_name, prov = "alpaca", AlpacaData(self.config)

        # Choose a period if not provided. Rough heuristic based on limit.
        # The provider classes typically accept 'period' or start/end; here we use period when available.
        if not period and limit:
            if interval in ("1min", "5min", "15min", "30min"):
                period = "60d"   # intraday: ~2 months
            elif interval in ("1h", "1H"):
                period = "1y"
            else:
                period = "5y"    # daily or larger

        # Providers have signature get_historical_data(symbol, interval='1D', start=None, end=None, period=None)
        # Normalize start/end into datetime objects for providers that require datetime
        def _parse_dt(val):
            if val is None:
                return None
            if isinstance(val, pd.Timestamp):
                return val.to_pydatetime()
            if isinstance(val, datetime):
                return val
            s = str(val)
            try:
                if len(s) == 8 and s.isdigit():
                    return datetime.strptime(s, "%Y%m%d")
                return datetime.fromisoformat(s)
            except Exception:
                # Fallback: try pandas parser
                try:
                    return pd.to_datetime(s).to_pydatetime()
                except Exception:
                    return None

        if "start" in kwargs or "end" in kwargs:
            parsed_start = _parse_dt(kwargs.get("start"))
            parsed_end = _parse_dt(kwargs.get("end"))
            if parsed_start is not None:
                kwargs["start"] = parsed_start
            if parsed_end is not None:
                kwargs["end"] = parsed_end

        # Attempt with chosen provider; on failure, try fallbacks Alpaca -> Finnhub (excluding the chosen one)
        providers_order = [(prov_name, prov)]
        # Build fallbacks
        fallback_map = {
            "alpaca": AlpacaData(self.config),
            "finnhub": FinnhubData(self.config),
            "polygon": PolygonData(self.config),
        }
        for name, instance in list(fallback_map.items()):
            if name != prov_name:
                providers_order.append((name, instance))
        df = pd.DataFrame()
        last_err = None
        for name, instance in providers_order:
            try:
                if hasattr(instance, "get_historical_data"):
                    df = instance.get_historical_data(symbol, interval=interval, period=period, **kwargs)
                else:
                    logger.error(f"Provider '{name}' lacks get_historical_data()")
                    df = pd.DataFrame()
                if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                    prov_name = name  # Note: report actual working provider
                    break
            except Exception as e:
                last_err = e
                logger.error(f"Error in provider '{name}' get_historical_data: {e}")
                df = pd.DataFrame()
        if df is None:
            df = pd.DataFrame()
        if df.empty and last_err:
            # Keep last error logged; returning empty is fine for caller
            pass

        if df is None:
            return pd.DataFrame()
        if not isinstance(df, pd.DataFrame):
            logger.error("Provider returned a non-DataFrame result")
            return pd.DataFrame()

        # Normalize columns if present
        cols_lower = [c.lower() for c in df.columns]
        rename_map = {}
        for want in ["open", "high", "low", "close", "volume"]:
            if want not in cols_lower:
                # try common variants (already lowercased above)
                variants = {"open": ["o", "open"], "high": ["h", "high"], "low": ["l", "low"], "close": ["c", "close"], "volume": ["v", "volume", "vol"]}
                for var in variants[want]:
                    if var in cols_lower:
                        rename_map[df.columns[cols_lower.index(var)]] = want
                        break
        if rename_map:
            df = df.rename(columns=rename_map)

        # Ensure index is datetime if 'time' column exists
        if 'time' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            try:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
            except Exception:
                pass

        # Trim to limit if requested
        if limit is not None and limit > 0:
            df = df.tail(limit)

        return df

    def fetch_alpha_vantage_news(self, symbols=None, topics=None, time_from=None, time_to=None):
        if not self.alpha_vantage_api_key:
            logger.warning("ALPHA_VANTAGE_KEY not set. Skipping Alpha Vantage news fetch.")
            return ""

        base_url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.alpha_vantage_api_key,
            "sort": "RELEVANCE",
            "limit": 50
        }
        if symbols:
            params["symbols"] = ",".join(symbols)
        if topics:
            params["topics"] = ",".join(topics)
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            articles = [
                f"Title: {item['title']}\nSummary: {item['summary']}"
                for item in data.get('feed', [])
            ]
            return data.get('feed', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return ""

    def fetch_fred_data(self, series_id):
        if not self.fred_api_key:
            logger.warning("FRED_API_KEY not set. Skipping FRED data fetch.")
            return ""

        base_url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            observations = data.get('observations', [])
            if observations:
                latest = observations[0]
                return f"{series_id}: Date={latest['date']}, Value={latest['value']}"
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching FRED data: {e}")
            return ""

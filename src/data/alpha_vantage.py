import os
from alpha_vantage.timeseries import TimeSeries
from src.data.database import DatabaseManager
from datetime import datetime

class AlphaVantageData:
    def __init__(self, config):
        self.config = config
        self.api_key = os.getenv('ALPHA_VANTAGE_KEY')
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.db = DatabaseManager(config['db_path'])
        
    def get_historical_data(self, symbol, interval='5min', period='1mo'):
        """Fetch historical data with caching"""
        # Check cache first
        cached = self.db.get_cached_data(symbol, interval)
        if cached:
            return cached
            
        data, _ = self.ts.get_intraday(symbol=symbol, interval=interval, outputsize='full')
        self.db.cache_data(symbol, interval, data)
        return data
        
    def get_current_price(self, symbol):
        """Get real-time price"""
        data, _ = self.ts.get_quote_endpoint(symbol)
        price = float(data['05. price'])
        self.db.log_price(symbol, price, datetime.now())
        return price
        
    def get_multi_timeframe_data(self, symbol):
        """Fetch data for all configured timeframes"""
        data = {}
        for tf in self.config['timeframes']['short_term'] + [self.config['timeframes']['trend']]:
            data[tf] = self.get_historical_data(symbol, tf)
        return data
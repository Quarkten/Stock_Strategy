import os
import logging
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import time
from alpha_vantage.timeseries import TimeSeries
from src.data.database import DatabaseManager
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class AlphaVantageRateLimitError(Exception):
    """Custom exception for Alpha Vantage API rate limit errors."""
    pass

class AlphaVantageData:
    def __init__(self, config):
        self.config = config
        # Look for environment variable named 'ALPHA_VANTAGE_KEY'
        self.api_key = os.getenv('ALPHA_VANTAGE_KEY')
        
        if not self.api_key:
            # Fallback to direct assignment if environment variable not found
            load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv('ALPHA_VANTAGE_KEY')
        if not self.api_key:
            self.api_key = 'R4AC25T3PGQITZ5T'  # Fallback to hardcoded key if not set
            print("Warning: Using hardcoded API key. Consider setting ALPHA_VANTAGE_KEY environment variable.")
        
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.db = DatabaseManager(config['db_path'])
        self.rate_limit_delay = 12  # Alpha Vantage free tier: 5 calls per minute
        self.max_retries = 3
        self.initial_retry_delay = 5 # seconds
        
    def get_historical_data(self, symbol, interval='daily', period='1mo'):
        """Fetch historical data with proper error handling and retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                print(f"Fetching data for {symbol} with interval {interval} (Attempt {attempt + 1}/{self.max_retries})...")
                
                # Map intervals to Alpha Vantage functions
                if interval in ['1min', '5min', '15min', '30min', '60min', '1h']:
                    # Map '1h' to '60min' for Alpha Vantage
                    av_interval = '60min' if interval == '1h' else interval
                    output_size = 'full' if period and period != '1mo' else 'compact' # Use 'full' for backtesting periods
                    data, meta_data = self.ts.get_intraday(
                        symbol=symbol, 
                        interval=av_interval, 
                        outputsize=output_size
                    )
                elif interval in ['daily', '1d']:
                    data, meta_data = self.ts.get_daily(symbol=symbol, outputsize='compact')
                elif interval in ['weekly', '1w']:
                    data, meta_data = self.ts.get_weekly(symbol=symbol)
                elif interval in ['monthly', '1M']:
                    data, meta_data = self.ts.get_monthly(symbol=symbol)
                else:
                    # Default to daily
                    data, meta_data = self.ts.get_daily(symbol=symbol, outputsize='compact')
                
                # Check for rate limit message in meta_data or data (Alpha Vantage sometimes returns errors in data)
                if isinstance(meta_data, dict) and "Note" in meta_data and "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day." in meta_data["Note"]:
                    raise AlphaVantageRateLimitError("Alpha Vantage API rate limit reached.")
                if isinstance(data, dict) and "Error Message" in data and "Invalid API call" in data["Error Message"]:
                    # This might indicate other errors, but often includes rate limits
                    raise AlphaVantageRateLimitError("Alpha Vantage API error, possibly rate limit.")

                # Rename columns to a more standard format
                data.rename(columns={
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close',
                    '5. volume': 'volume'
                }, inplace=True)
                
                if data.empty:
                    print(f"No data returned for {symbol}")
                    return self._get_mock_price_data(symbol)
                
                # Store in database
                self.db.store_price_data(symbol, interval, data)
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                print(f"Successfully fetched {len(data)} data points for {symbol}")
                return data
                
            except AlphaVantageRateLimitError as e:
                logging.error(f"Alpha Vantage API rate limit reached: {e}")
                if attempt < self.max_retries - 1:
                    retry_delay = self.initial_retry_delay * (2 ** attempt)
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise e # Re-raise the specific error after max retries
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                print("Using mock data...")
                return self._get_mock_price_data(symbol)
        return self._get_mock_price_data(symbol) # Should not be reached if retries are handled correctly
    
    def get_current_price(self, symbol):
        """Get real-time price and return as DataFrame with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                print(f"Fetching current price for {symbol} (Attempt {attempt + 1}/{self.max_retries})...")
                data, _ = self.ts.get_quote_endpoint(symbol)
                
                # Check for rate limit message in data
                if isinstance(data, dict) and "Error Message" in data and "Invalid API call" in data["Error Message"]:
                    raise AlphaVantageRateLimitError("Alpha Vantage API error, possibly rate limit.")

                price = float(data.iloc[0]['05. price'])
                # Return as a DataFrame with appropriate columns for consistency
                df = pd.DataFrame([{
                    'open': float(data.iloc[0]['02. open']), 
                    'high': float(data.iloc[0]['03. high']), 
                    'low': float(data.iloc[0]['04. low']), 
                    'close': price, 
                    'volume': float(data.iloc[0]['06. volume'])
                }], index=[datetime.now()])
                self.db.log_price(symbol, 'realtime', df) # Log with a 'realtime' interval
                return df
            except AlphaVantageRateLimitError as e:
                logging.error(f"Alpha Vantage API rate limit reached: {e}")
                if attempt < self.max_retries - 1:
                    retry_delay = self.initial_retry_delay * (2 ** attempt)
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise e # Re-raise the specific error after max retries
            except Exception as e:
                print(f"Error getting current price for {symbol}: {e}")
                return self._get_mock_price_data(symbol)
        return self._get_mock_price_data(symbol) # Should not be reached if retries are handled correctly

    def _get_mock_price_data(self, symbol):
        """Generates mock price data for a given symbol."""
        base_prices = {
            'AAPL': 150.0,
            'GOOG': 2500.0,
            'TSLA': 200.0,
            'MSFT': 300.0,
            'AMZN': 3000.0,
            'META': 250.0,
            'NVDA': 400.0,
            'SPY': 450.0 # Added SPY for consistency
        }
        mock_price = base_prices.get(symbol, 100.0)
        mock_data = pd.DataFrame({
            'open': [mock_price],
            'high': [mock_price + 1],
            'low': [mock_price - 1],
            'close': [mock_price],
            'volume': [100000]
        }, index=[datetime.now()])
        return mock_data

    def get_chart_data_for_llm(self, symbol, timeframes=['daily', 'weekly', '4h', '1h']):
        """Fetch specific chart data for LLM analysis"""
        chart_data = {}
        for tf in timeframes:
            # Alpha Vantage doesn't have '4h' directly, so we'll use '60min' and let the LLM interpret
            av_tf = '60min' if tf == '4h' else tf
            data = self.get_historical_data(symbol, av_tf, 'full') # Fetch full data for context
            if data is not None and not data.empty:
                # Take a recent subset for the LLM prompt, e.g., last 5 candles
                chart_data[tf] = data.tail(5).to_string()
            else:
                chart_data[tf] = f"No data available for {tf}"
        return chart_data
        
    def get_multi_timeframe_data(self, symbol):
        """Fetch data for all configured timeframes"""
        data = {}
        for tf in self.config['timeframes']['short_term'] + [self.config['timeframes']['trend']]:
            data[tf] = self.get_historical_data(symbol, tf)
        return data
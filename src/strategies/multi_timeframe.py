import pandas as pd
import numpy as np

class MultiTimeframeTrader:
    def __init__(self, config, db):
        self.config = config
        self.db = db
        self.params = config['strategy']['params']
        
        # Initialize strategy parameters
        self.primary_timeframe = self.params.get('primary_timeframe', '1h')
        self.secondary_timeframe = self.params.get('secondary_timeframe', '15m')
        self.ema_short_period = self.params.get('ema_short_period', 20)
        self.ema_long_period = self.params.get('ema_long_period', 50)
        self.atr_period = self.params.get('atr_period', 14)
        self.atr_multiplier = self.params.get('atr_multiplier', 1.5)
        self.breakout_period = self.params.get('breakout_period', 5)
        self.risk_per_trade = self.params.get('risk_per_trade', 0.01)  # 1% risk per trade

    def _calculate_ema(self, series, period):
        """Calculate Exponential Moving Average (EMA)"""
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_atr(self, df, period):
        """Calculate Average True Range (ATR)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        # True Range is the maximum of the three values
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the moving average of TR
        return tr.rolling(window=period).mean()

    def get_multi_timeframe_data(self, symbol):
        """Fetch data for all timeframes - compatible with existing project structure"""
        timeframes = [self.primary_timeframe, self.secondary_timeframe]
        data = {}
        
        for tf in timeframes:
            # Use the database method that matches your project structure
            if hasattr(self.db, 'get_price_data'):
                data[tf] = self.db.get_price_data(symbol, tf)
            elif hasattr(self.db, 'get_ohlcv'):
                data[tf] = self.db.get_ohlcv(symbol, tf, limit=max(self.ema_long_period, 100))
            else:
                # Fallback method
                data[tf] = self.db.get_historical_data(symbol, tf)
        
        return data

    def generate_signal(self, symbol, account_balance=None, historical_data=None):
        """
        Generate trading signals based on multi-timeframe analysis
        
        Args:
            symbol (str): The trading symbol.
            account_balance (float, optional): Current account balance for position sizing. Defaults to None.
            historical_data (dict, optional): Pre-loaded historical data for different timeframes.
                                             Expected format: {'1h': df_1h, '15m': df_15m}.
                                             If None, data will be fetched using get_multi_timeframe_data.
        Returns:
            tuple: (signal, pattern, timeframe, size, trend_dir, trend_str)
        """
        try:
            if historical_data:
                timeframe_data = historical_data
            else:
                # Fetch multi-timeframe data using the compatible method
                timeframe_data = self.get_multi_timeframe_data(symbol)
            
            primary_data = timeframe_data.get(self.primary_timeframe)
            secondary_data = timeframe_data.get(self.secondary_timeframe)
            
            # Check if we have data
            if primary_data is None or secondary_data is None:
                return "HOLD", None, None, 0, "neutral", 0.0
            
            # Convert to DataFrame if needed
            if not isinstance(primary_data, pd.DataFrame):
                primary_data = pd.DataFrame(primary_data)
            if not isinstance(secondary_data, pd.DataFrame):
                secondary_data = pd.DataFrame(secondary_data)
            
            # Check if we have sufficient data
            if (len(primary_data) < self.ema_long_period or 
                len(secondary_data) < self.atr_period + self.breakout_period):
                return "HOLD", None, None, 0, "neutral", 0.0
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close']
            for df in [primary_data, secondary_data]:
                for col in required_columns:
                    if col not in df.columns:
                        # Try alternative column names
                        alt_names = {
                            'open': ['Open', 'OPEN'],
                            'high': ['High', 'HIGH'],
                            'low': ['Low', 'LOW'],
                            'close': ['Close', 'CLOSE']
                        }
                        found = False
                        for alt_name in alt_names.get(col, []):
                            if alt_name in df.columns:
                                df[col] = df[alt_name]
                                found = True
                                break
                        if not found:
                            return "HOLD", None, None, 0, "neutral", 0.0
            
            # Calculate EMAs for primary trend direction
            primary_data.loc[:, 'ema_short'] = self._calculate_ema(
                primary_data['close'], 
                self.ema_short_period
            )
            primary_data.loc[:, 'ema_long'] = self._calculate_ema(
                primary_data['close'], 
                self.ema_long_period
            )
            
            # Determine primary trend direction and strength
            latest_primary = primary_data.iloc[-1]
            ema_diff = latest_primary['ema_short'] - latest_primary['ema_long']
            
            if ema_diff > 0:
                trend_dir = "up"
                trend_str = abs(ema_diff) / latest_primary['close']
            elif ema_diff < 0:
                trend_dir = "down"
                trend_str = abs(ema_diff) / latest_primary['close']
            else:
                trend_dir = "neutral"
                trend_str = 0.0
            
            # Calculate ATR for secondary timeframe
            secondary_data.loc[:, 'atr'] = self._calculate_atr(secondary_data, self.atr_period)
            
            # Get latest secondary candle
            latest_secondary = secondary_data.iloc[-1]
            
            # Calculate breakout levels
            prev_candles = secondary_data.iloc[-(self.breakout_period + 1):-1]
            resistance = prev_candles['high'].max()
            support = prev_candles['low'].min()
            
            # Initialize default values
            signal = "HOLD"
            pattern = None
            timeframe = self.secondary_timeframe
            size = 0
            
            # Check for breakout patterns aligned with primary trend
            if trend_dir == "up":
                # Bullish breakout pattern
                if latest_secondary['close'] > resistance:
                    pattern = f"Bullish Breakout ({self.breakout_period}-period)"
                    signal = "BUY"
                    
                    # Calculate position size if account balance is available
                    if account_balance and latest_secondary['atr'] > 0:
                        stop_loss_price = latest_secondary['close'] - (latest_secondary['atr'] * self.atr_multiplier)
                        risk_per_share = latest_secondary['close'] - stop_loss_price
                        if risk_per_share > 0:
                            dollar_risk = account_balance * self.risk_per_trade
                            size = max(0, dollar_risk / risk_per_share)
            
            elif trend_dir == "down":
                # Bearish breakout pattern
                if latest_secondary['close'] < support:
                    pattern = f"Bearish Breakout ({self.breakout_period}-period)"
                    signal = "SELL"
                    
                    # Calculate position size if account balance is available
                    if account_balance and latest_secondary['atr'] > 0:
                        stop_loss_price = latest_secondary['close'] + (latest_secondary['atr'] * self.atr_multiplier)
                        risk_per_share = stop_loss_price - latest_secondary['close']
                        if risk_per_share > 0:
                            dollar_risk = account_balance * self.risk_per_trade
                            size = max(0, dollar_risk / risk_per_share)
            
            return signal, pattern, timeframe, size, trend_dir, trend_str
            
        except Exception as e:
            # Log the error if you have logging set up
            print(f"Error in generate_signal: {e}")
            return "HOLD", None, None, 0, "neutral", 0.0

    def get_signal_strength(self, symbol):
        """Get signal strength for compatibility with other strategies"""
        signal, pattern, timeframe, size, trend_dir, trend_str = self.generate_signal(symbol)
        
        if signal == "BUY":
            return min(trend_str * 100, 100)  # Cap at 100
        elif signal == "SELL":
            return max(-trend_str * 100, -100)  # Cap at -100
        else:
            return 0.0
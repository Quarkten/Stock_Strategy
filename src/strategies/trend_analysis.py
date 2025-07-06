import talib
import numpy as np

class TrendAnalyzer:
    @staticmethod
    def determine_trend(df, fast_period=20, slow_period=50, adx_period=14):
        df['ema_fast'] = talib.EMA(df['Close'], timeperiod=fast_period)
        df['ema_slow'] = talib.EMA(df['Close'], timeperiod=slow_period)
        df['adx'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=adx_period)
        
        current_fast = df['ema_fast'].iloc[-1]
        current_slow = df['ema_slow'].iloc[-1]
        current_adx = df['adx'].iloc[-1]
        
        if current_fast > current_slow:
            direction = "bullish"
        elif current_fast < current_slow:
            direction = "bearish"
        else:
            direction = "neutral"
        
        strength = min(100, max(0, current_adx * 1.5))
        return direction, strength, current_fast, current_slow, current_adx
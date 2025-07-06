import talib
import numpy as np
from scipy.signal import find_peaks

class PatternDetector:
    @staticmethod
    def detect_candlestick_patterns(df):
        patterns = {}
        patterns['hammer'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        patterns['inverted_hammer'] = talib.CDLINVERTEDHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        # ... all other candlestick patterns
        return patterns
    
    @staticmethod
    def detect_chart_patterns(df):
        patterns = []
        # Double Top/Bottom detection
        if PatternDetector.is_double_top(df):
            patterns.append('double_top')
        if PatternDetector.is_double_bottom(df):
            patterns.append('double_bottom')
        # ... other chart patterns
        return patterns
    
    @staticmethod
    def is_double_top(df, window=20, tolerance=0.02):
        # ... double top detection logic
    
    @staticmethod
    def is_double_bottom(df, window=20, tolerance=0.02):
        # ... double bottom detection logic
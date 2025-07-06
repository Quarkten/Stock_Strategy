import numpy as np
from src.strategies.pattern_detection import PatternDetector
from src.strategies.trend_analysis import TrendAnalyzer
from src.utils.risk_management import calculate_position_size

class TradingAgent:
    def __init__(self, config, db):
        self.config = config
        self.db = db
        self.pattern_detector = PatternDetector()
        self.trend_analyzer = TrendAnalyzer()
        
    def analyze_market(self, symbol):
        """Full market analysis pipeline"""
        # Get multi-timeframe data
        data = self.get_multi_timeframe_data(symbol)
        
        # Trend analysis
        trend_direction, trend_strength = self.analyze_trend(data)
        
        # Pattern detection
        patterns = self.detect_patterns(data)
        
        # Generate signals
        signals = self.generate_signals(patterns, trend_direction)
        
        return signals
    
    def get_multi_timeframe_data(self, symbol):
        """Fetch data for all timeframes"""
        timeframes = self.config['timeframes']['short_term'] + [self.config['timeframes']['trend']]
        return {tf: self.db.get_price_data(symbol, tf) for tf in timeframes}
    
    def analyze_trend(self, data):
        """Analyze trend on higher timeframe"""
        trend_tf = self.config['timeframes']['trend']
        return self.trend_analyzer.determine_trend(data[trend_tf])
    
    def detect_patterns(self, data):
        """Detect patterns across timeframes"""
        patterns = {}
        for tf in self.config['timeframes']['short_term']:
            patterns[tf] = self.pattern_detector.detect_all(data[tf])
        return patterns
    
    def generate_signals(self, patterns, trend_direction):
        """Generate trading signals based on patterns and trend"""
        signals = []
        for tf, tf_patterns in patterns.items():
            for pattern in tf_patterns:
                if self.is_valid_s
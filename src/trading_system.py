import os
import time
import pytz
from datetime import datetime
from src.data.forex_factory import ForexFactoryScraper
from src.data.twitter_sentiment import TwitterSentimentAnalyzer
from src.data.alpha_vantage import AlphaVantageData
from src.data.database import DatabaseManager
from src.agents.rl_agent import RLTrader
from src.strategies.multi_timeframe import MultiTimeframeTrader

class TradingSystem:
    def __init__(self, config):
        self.config = config
        self.db = DatabaseManager(config['db_path'])
        self.forex_scraper = ForexFactoryScraper(config)
        self.twitter_analyzer = TwitterSentimentAnalyzer(config)
        self.data_fetcher = AlphaVantageData(config)
        self.rl_trader = RLTrader(config)
        self.mt_trader = MultiTimeframeTrader(config, self.db)
        self.trained_models = {}
        
    def run_daily_setup(self):
        print("Running daily setup...")
        self.forex_scraper.scrape_calendar()
        self.twitter_analyzer.capture_tweets()
        top_symbols = self.db.get_top_stocks(self.config['num_top_stocks'])
        print(f"Top stocks for today: {top_symbols}")
        
        for symbol, _ in top_symbols:
            print(f"Training RL model for {symbol}...")
            data = self.data_fetcher.get_historical_data(symbol, '1d', '5y')
            self.trained_models[symbol] = self.rl_trader.train_model(symbol, data)
        print("Daily setup complete.")
    
    def trading_loop(self):
        print("Starting trading loop...")
        top_symbols = self.db.get_top_stocks(self.config['num_top_stocks'])
        
        while True:
            now = datetime.now(pytz.timezone(self.config['timezone']))
            market_close = now.replace(hour=15, minute=45, second=0, microsecond=0)
            
            if now > market_close:
                print("Market closed. Stopping trading loop.")
                break
                
            for symbol, _ in top_symbols:
                try:
                    signal, pattern, timeframe, size, trend_dir, trend_str = (
                        self.mt_trader.generate_signal(symbol)
                    
                    if signal != "HOLD":
                        self.execute_trade(symbol, signal, pattern, timeframe, size, trend_dir, trend_str)
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
            
            time.sleep(300)  # 5 minutes
    
    def execute_trade(self, symbol, action, pattern, timeframe, size, trend_dir, trend_str):
        price = self.data_fetcher.get_current_price(symbol)
        print(f"Executing {action.upper()} for {symbol} at {price:.2f}")
        print(f"Pattern: {pattern} | Timeframe: {timeframe}")
        print(f"Trend: {trend_dir} ({trend_str:.1f}%) | Size: {size} shares")
        
        # In real implementation, connect to broker API here
        self.db.log_trade(action, symbol, price, size, 'MultiTimeframe', pattern, 0, timeframe)
        
        # Visualize the pattern
        from src.utils.pattern_visualization import visualize_multi_timeframe
        visualize_multi_timeframe(symbol, timeframe, pattern)
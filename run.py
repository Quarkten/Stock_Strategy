import yaml
import logging
import argparse
from datetime import datetime
import pandas as pd
from typing import Optional

from src.strategies.intraday_strategy import IntradayStrategy
from scraper import ForexCalendarScraper, ScrapingConfig
from src.utils.data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingApplication:
    def __init__(self, config_path='config/config.yaml'):
        """Initializes the trading application."""
        self.config = self.load_config(config_path)
        self.strategy = IntradayStrategy(self.config)
        self.data_fetcher = DataFetcher(self.config)
        self.scraper = ForexCalendarScraper(ScrapingConfig())
        self.daily_loss = 0
        self.trades = []

    def load_config(self, path):
        """Loads the configuration from a YAML file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def run_premarket_preparation(self):
        """
        Executes the pre-market routine as defined in the trading plan.
        This includes reviewing news, economic data, and determining a daily bias.
        """
        logging.info("Starting pre-market preparation...")
        # Fetch news and economic data
        calendar_data, news_data = self.scraper.get_news()
        if calendar_data:
            logging.info(f"Found {len(calendar_data)} calendar events.")
        if news_data:
            logging.info(f"Found news across {len(news_data)} sections.")
        
        # Fetch historical data for bias evaluation
        spy_daily_data = self.data_fetcher.fetch_data('SPY', '1day', limit=100)
        spy_1h_data = self.data_fetcher.fetch_data('SPY', '1hour', limit=100)
        
        # Evaluate daily bias
        daily_bias = self.strategy.evaluate_daily_bias_tjr_style(spy_daily_data, None, None, spy_1h_data)
        logging.info(f"Preliminary Daily Bias: {daily_bias}")

    def run_trading_session(self):
        """
        Runs the main trading session, checking for trades and enforcing risk rules.
        """
        logging.info("Starting trading session...")
        # This is a simplified loop. In a real application, this would be event-driven.
        # For now, we'll simulate a few trade checks.
        for i in range(5):
            self.check_for_trades()
            if self.daily_loss >= self.config['daily_max_loss'] * self.config['capital']:
                logging.warning("Daily max loss reached. Stopping trading for the day.")
                break

    def check_for_trades(self):
        """
        Checks for new trade opportunities and validates them against risk rules.
        """
        # This is a placeholder for the trading logic
        logging.info("Checking for new trade opportunities...")
        # In a real scenario, you would fetch live data and evaluate signals.
        # Here we will just simulate a trade for demonstration purposes.
        entry_price = 450.0
        stop_loss_price = 449.0
        
        if self.strategy.check_risk_before_trade(entry_price, stop_loss_price):
            self.simulate_trade(entry_price, stop_loss_price)
        else:
            logging.warning("Trade aborted due to risk management rules.")

    def simulate_trade(self, entry_price, stop_loss_price):
        """
        Simulates a trade execution and updates the daily P&L.
        """
        # Simulate a trade and update daily loss
        # This is a placeholder
        position_size = self.strategy.calculate_position_size(entry_price, stop_loss_price)
        trade_outcome = -1 * abs(entry_price - stop_loss_price) * position_size # Simulate a losing trade
        
        self.strategy.update_pnl(trade_outcome)
        self.daily_loss = -self.strategy.daily_pnl # Keep track of total loss for the day
        
        self.trades.append({
            'entry_time': datetime.now(),
            'outcome': trade_outcome,
            'reason': 'Simulated trade'
        })
        logging.info(f"Simulated trade executed. Outcome: {trade_outcome}. Total daily loss: {self.daily_loss}")

    def run_post_market_review(self):
        """
        Executes the post-market routine for logging and performance analysis.
        """
        logging.info("Starting post-market review...")
        if not self.trades:
            logging.info("No trades were made today.")
            return

        # Create a DataFrame from the trades
        trades_df = pd.DataFrame(self.trades)
        
        # Log all trades
        logging.info("--- Daily Trades Log ---")
        for trade in self.trades:
            logging.info(f"  - Time: {trade['entry_time']}, Outcome: {trade['outcome']}, Reason: {trade['reason']}")

        # Analyze performance
        total_pnl = trades_df['outcome'].sum()
        win_rate = len(trades_df[trades_df['outcome'] > 0]) / len(trades_df) if len(trades_df) > 0 else 0
        
        logging.info("--- Performance Summary ---")
        logging.info(f"Total P&L: {total_pnl}")
        logging.info(f"Win Rate: {win_rate:.2%}")
        logging.info(f"Total Trades: {len(self.trades)}")

        # Save trades to a CSV file
        log_file = f"data/trade_log_{datetime.now().strftime('%Y%m%d')}.csv"
        trades_df.to_csv(log_file, index=False)
        logging.info(f"Trade log saved to {log_file}")

    def run(self):
        self.run_premarket_preparation()
        self.run_trading_session()
        self.run_post_market_review()

def parse_args():
    parser = argparse.ArgumentParser(description="Trading/backtest runner")
    parser.add_argument("--start", type=str, help="Backtest start date YYYYMMDD")
    parser.add_argument("--end", type=str, help="Backtest end date YYYYMMDD")
    parser.add_argument("--symbol", type=str, help="Symbol, e.g., SPY")
    parser.add_argument("--timeframe", type=str, help="Timeframe, e.g., 5m, 15m, 1h, 1day")
    parser.add_argument("--rth-only", action="store_true", help="Filter to Regular Trading Hours")
    parser.add_argument("--capital", type=float, help="Starting capital")
    parser.add_argument("--daily-max-loss-pct", type=float, help="Daily max loss percent (e.g. 0.015)")
    parser.add_argument("--risk-per-trade-pct", type=float, help="Risk per trade percent (e.g. 0.005)")
    parser.add_argument("--csv-out", type=str, help="Path to master CSV for trades")
    return parser.parse_args()

def backtest_main(args):
    # Lazy import to avoid hard dependency in demo mode
    from src.execution.backtester import Backtester

    app = TradingApplication()
    cfg = app.config.copy()

    # Merge CLI overrides
    symbol = args.symbol or cfg.get("target_symbol", "SPY")
    timeframe = args.timeframe or cfg.get("strategy", {}).get("params", {}).get("secondary_timeframe", cfg.get("timeframes", {}).get("short_term", ["5min"])[0])
    capital = args.capital if args.capital is not None else cfg.get("capital", 100000)
    daily_max_loss_pct = args.daily_max_loss_pct if args.daily_max_loss_pct is not None else cfg.get("daily_max_loss", 0.015)
    risk_per_trade_pct = args.risk_per_trade_pct if args.risk_per_trade_pct is not None else cfg.get("risk_per_trade", 0.005)
    csv_out = args.csv_out or "data/trades_master.csv"
    rth_only = bool(args.rth_only)

    # Update strategy finance params for backtest
    cfg["capital"] = capital
    cfg["daily_max_loss"] = daily_max_loss_pct
    cfg["risk_per_trade"] = risk_per_trade_pct

    strategy = IntradayStrategy(cfg)
    data_fetcher = DataFetcher(cfg)

    bt = Backtester(strategy=strategy,
                    data_fetcher=data_fetcher,
                    config=cfg,
                    symbol=symbol,
                    timeframe=timeframe,
                    start=args.start,
                    end=args.end,
                    rth_only=rth_only,
                    csv_out=csv_out)
    bt.run()

if __name__ == "__main__":
    args = parse_args()
    if args.start and args.end:
        backtest_main(args)
    else:
        app = TradingApplication()
        app.run()
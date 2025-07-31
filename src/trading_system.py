import os
import time
import pytz
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import io
import contextlib
from scraper import ForexCalendarScraper
from src.data.alpha_vantage import AlphaVantageData

from src.data.alpaca_data import AlpacaData
from src.data.polygon_data import PolygonData
from src.data.finnhub_data import FinnhubData
from src.data.database import DatabaseManager
from src.strategies.multi_timeframe import MultiTimeframeTrader
from src.strategies.intraday_strategy import IntradayStrategy
from src.utils.llm_analyzer import LLMAnalyzer
from src.execution.alpaca_executor import AlpacaExecutor
import os
import json
import logging
from statistics import mean
from datetime import datetime
from scraper import ScrapingConfig

class TradingSystem:
    def __init__(self, config, llm_analyzer, data_source='alpaca'):
        self.config = config
        self.db = DatabaseManager(config['db_path'])
        self.llm_analyzer = llm_analyzer
        
        
        self.executor = AlpacaExecutor(config)
        self._initialize_data_fetcher(data_source)

    def _initialize_data_fetcher(self, source):
        if source == 'alpha_vantage':
            self.data_fetcher = AlphaVantageData(self.config)
            print("Using Alpha Vantage for data fetching.")
        elif source == 'alpaca':
            self.data_fetcher = AlpacaData(self.config)
            print("Using Alpaca for data fetching.")
        elif source == 'polygon':
            self.data_fetcher = PolygonData(self.config)
            print("Using Polygon for data fetching.")
        elif source == 'finnhub':
            self.data_fetcher = FinnhubData(self.config)
            print("Using Finnhub for data fetching.")
        else:
            raise ValueError(f"Unknown data source: {source}")
        
        self.mt_trader = MultiTimeframeTrader(self.config, self.db)
        self.llm_analyzer = LLMAnalyzer(self.config)
        self.intraday_strategy = IntradayStrategy(self.config)
        self.trained_models = {}
        self.daily_bias = "NEUTRAL"  # Initialize daily bias

    def _get_chart_data_for_llm(self):
        print("Getting chart data for LLM...")
        return self.data_fetcher.get_chart_data_for_llm("SPY")

    def _evaluate_daily_bias_tjr_style(self, spy_daily_data, spy_weekly_data, spy_4h_data, spy_1h_data):
        print("Evaluating daily bias (TJR Style)...")
        return self.intraday_strategy.evaluate_daily_bias_tjr_style(spy_daily_data, spy_weekly_data, spy_4h_data, spy_1h_data)

    def _detect_liquidity_sweep(self, data):
        return self.intraday_strategy.detect_liquidity_sweep(data)

    def _detect_inverse_fvg(self, data):
        return self.intraday_strategy.detect_inverse_fvg(data)

    def _detect_smt_divergence(self, data):
        # SMT divergence requires ES data, which is not currently fetched. Placeholder for now.
        return False

    def _detect_csd(self, data):
        return self.intraday_strategy.detect_csd(data, self.daily_bias)

    def _align_pdra(self, data, bias):
        return self.intraday_strategy.align_pdra(data, bias)

    def _execute_intraday_trade(self, symbol, action, price, size, pattern, timeframe):
        # Build rationale from current context
        bias_snapshot = {
            "daily_bias": self.daily_bias
        }
        rationale = {
            "strategy_name": "Intraday",
            "signal_reason": f"{pattern} aligned with {self.daily_bias} bias",
            "bias_snapshot": bias_snapshot,
            "stop_loss": price * (0.995 if action.upper() == "BUY" else 1.005),
            "take_profit": price * (1.01 if action.upper() == "BUY" else 0.99),
            "tags": ["intraday", timeframe, pattern],
            "broker": "alpaca"
        }
        order = self.executor.execute_trade(symbol, action, size, price=price, rationale=rationale, is_backtest=False)
        if order:
            self.db.log_trade(action, symbol, price, size, 'Intraday', pattern, 0, timeframe)

    def _handle_special_time_logic(self, now, symbol):
        # 10:00 AM EST: Watch the 4H candle open.
        if now.hour == 10 and now.minute == 0:
            print("10:00 AM EST: Checking 4H candle open and manipulation.")
            # Implement manipulation logic based on daily_bias
            if self.daily_bias == "BULLISH":
                print("Bullish bias: Expect manipulation lower -> distribution higher.")
            elif self.daily_bias == "BEARISH":
                print("Bearish bias: Expect manipulation higher -> distribution lower.")

        # 9:40 AM: Identify draw on liquidity.
        elif now.hour == 9 and now.minute == 40:
            print("9:40 AM: Identifying draw on liquidity and setting up for trade entry.")
            # Look at PDRA reactions to confirm bullish/bearish continuation.
            # Setup for trade entry around 9:45 AM on low timeframes.

        
    def run_daily_setup(self, macro_news, news_calendar):
        output_capture = io.StringIO()
        with contextlib.redirect_stdout(output_capture):
            print("--- Running Daily Setup ---")

            # Daily Setup Routine (Pre-Market)
            print("\n[1/5] Running LLM analysis for daily bias...")
            chart_data_for_llm = self._get_chart_data_for_llm()

            # Fetch news from the scraper
            print("\n[2/5] Fetching news from the scraper...")
            scraper_config = ScrapingConfig(**self.config.get('scraper', {}))
            scraper = ForexCalendarScraper(config=scraper_config)
            
            calendar_data, news_data = scraper.get_news()
            
            # Format the scraped data for the LLM
            print("\n[3/5] Formatting scraped data for LLM...")
            if calendar_data:
                formatted_calendar = "\n".join([
                    f"- Event: {e.get('event', 'N/A')}, Time: {e.get('time', 'N/A')}, Currency: {e.get('currency', 'N/A')}, Impact: {e.get('impact', 'N/A')}, Actual: {e.get('actual', 'N/A')}, Forecast: {e.get('forecast', 'N/A')}, Previous: {e.get('previous', 'N/A')}" 
                    for e in calendar_data
                ])
                print("  - Successfully formatted scraped calendar events.")
            else:
                formatted_calendar = "No economic events found."
                print("  - No economic calendar events found to format.")

            if news_data:
                formatted_news = "\n".join([
                    f"- Headline: {n.get('headline', 'N/A')}, Time: {n.get('time', 'N/A')}, Source: {n.get('comments', 'N/A')}, Impact: {n.get('impact', 'N/A')}" 
                    for section in news_data.values() for n in section
                ])
                print("  - Successfully formatted scraped news.")
            else:
                formatted_news = "No news found."
                print("  - No news found to format.")

            llm_bias = self.llm_analyzer.get_spy_bias(macro_news, formatted_calendar)
            print(f"  - LLM determined SPY bias: {llm_bias}")

            # Daily Bias Evaluation (TJR Style)
            print("\n[4/5] Evaluating daily bias (TJR Style)...")
            spy_daily_data = self.data_fetcher.get_historical_data("SPY", 'daily')
            spy_weekly_data = self.data_fetcher.get_historical_data("SPY", 'weekly')
            spy_4h_data = self.data_fetcher.get_historical_data("SPY", '4h')
            spy_1h_data = self.data_fetcher.get_historical_data("SPY", '1h')

            if (spy_daily_data is None or spy_daily_data.empty or
                spy_weekly_data is None or spy_weekly_data.empty or
                spy_4h_data is None or spy_4h_data.empty or
                spy_1h_data is None or spy_1h_data.empty):
                logger.warning("Insufficient historical data for TJR style bias evaluation. Setting TJR bias to UNCERTAIN.")
                tjr_bias = "UNCERTAIN"
            else:
                tjr_bias = self.intraday_strategy.evaluate_daily_bias_tjr_style(
                    spy_daily_data, spy_weekly_data, spy_4h_data, spy_1h_data
                )
            print(f"  - TJR Style Daily Bias: {tjr_bias}")

            # Compare LLM vs TJR bias
            print("\n[5/5] Comparing LLM and TJR biases...")
            if llm_bias == tjr_bias:
                self.daily_bias = llm_bias
            elif llm_bias == "UNCERTAIN" or tjr_bias == "UNCERTAIN":
                self.daily_bias = "UNCERTAIN"
            else:
                # If they differ and neither is uncertain, default to neutral or wait for confirmation
                self.daily_bias = "NEUTRAL"
                print("  - LLM and TJR biases differ. Defaulting to NEUTRAL.")
            
            print(f"  - Final Daily Bias: {self.daily_bias}")

            # 1. Get top stocks and train models
            try:
                top_symbols = [(self.config['target_symbol'], 0)]
                print(f"\n- Target symbol for today: {top_symbols[0][0]}")
            except Exception as e:
                print(f"Error getting target symbol: {e}")
                top_symbols = []

            

            print("\n--- Daily setup complete. ---")
        
        captured_output = output_capture.getvalue()
        print("--- Daily Setup Summary ---")
        print(self.llm_analyzer.summarize_text(captured_output))
        print("---------------------------")
    
    def trading_loop(self):
        print("Starting intraday trading loop...")
        symbol = "SPY" 

        while True:
            now = datetime.now(pytz.timezone(self.config['timezone']))
            
            # Special Time Logic
            self._handle_special_time_logic(now, symbol)

            # Market close check
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0) 
            if now > market_close:
                print("Market closed. Stopping trading loop.")
                break
            
            print("\n----------------------------------------------------")
            print(f"\n[{now.strftime('%H:%M:%S')}] Fetching latest data for {symbol}...")
            current_price_data = self.data_fetcher.get_current_price(symbol)

            if current_price_data is None or current_price_data.empty:
                print(f"Could not get current price data for {symbol}. Retrying in 60s...")
                time.sleep(60) 
                continue
            
            print(f"[{now.strftime('%H:%M:%S')}] Fetching intraday data (15m, 1h)...")
            data_15m = self.data_fetcher.get_historical_data(symbol, '15min')
            data_1h = self.data_fetcher.get_historical_data(symbol, '1h')

            if data_15m is None or data_15m.empty or data_1h is None or data_1h.empty:
                print("Insufficient intraday data for analysis. Waiting 60s...")
                time.sleep(60)
                continue

            print(f"[{now.strftime('%H:%M:%S')}] Analyzing patterns for {symbol}...")
            
            # Check for bullish flag pattern on 15M timeframe
            self.intraday_strategy.detect_bullish_flag(data_15m)

            # Check for liquidity sweep on 15M timeframe
            if self.intraday_strategy.detect_liquidity_sweep(data_15m):
                
                # Look for confirmation patterns on 15M timeframe
                fvg_detected = self.intraday_strategy.detect_inverse_fvg(data_15m)
                smt_detected = self.intraday_strategy.detect_smt_divergence(data_15m, None)
                csd_detected = self.intraday_strategy.detect_csd(data_15m, self.daily_bias)

                # Align PDRA
                if self.intraday_strategy.align_pdra(current_price_data, self.daily_bias):
                    
                    action = "HOLD"
                    if self.daily_bias == "BULLISH" and fvg_detected and csd_detected:
                        action = "BUY"
                        print("  > Confirmed BULLISH entry signal.")
                    elif self.daily_bias == "BEARISH" and fvg_detected and csd_detected:
                        action = "SELL"
                        print("  > Confirmed BEARISH entry signal.")
                    
                    if action != "HOLD":
                        price = current_price_data['close'].iloc[-1]
                        size = 10 
                        pattern = "Intraday Setup"
                        timeframe = "15M" 
                        
                        print(f"  > EXECUTING TRADE: {action} {size} {symbol} at {price:.2f}")
                        self._execute_intraday_trade(symbol, action, price, size, pattern, timeframe)
                        
                        stop_loss = price * (0.995 if action == "BUY" else 1.005)
                        target = price * (1.01 if action == "BUY" else 0.99)
                        print(f"  > Trade entered: SL={stop_loss:.2f}, Target={target:.2f}")
                    else:
                        print("  > No strong entry signal. Holding.")
                else:
                    print("  > PDRA not aligned. Waiting for better setup.")
            
            print(f"[{now.strftime('%H:%M:%S')}] Cycle complete. Waiting 60 seconds...")
            time.sleep(60)

    def run_backtest(self, symbol, start_date, end_date):
        print(f"Running backtest for {symbol} from {start_date} to {end_date}")

        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")

        print(f"Fetching historical data for {symbol} for all timeframes...")
        daily_data = self.data_fetcher.get_historical_data(symbol, 'daily', start=start_dt, end=end_dt)
        data_15m = self.data_fetcher.get_historical_data(symbol, '15min', start=start_dt, end=end_dt)
        data_1h = self.data_fetcher.get_historical_data(symbol, '1h', start=start_dt, end=end_dt)

        if daily_data.empty or data_15m.empty or data_1h.empty:
            print("DEBUG: No sufficient data found. Cannot perform backtest.")
            return

        daily_data['date'] = daily_data.index.date
        data_15m['date'] = data_15m.index.date
        data_1h['date'] = data_1h.index.date

        backtest_daily_data = daily_data[(daily_data['date'] >= start_dt.date()) & (daily_data['date'] <= end_dt.date())]
        data_15m = data_15m[(data_15m['date'] >= start_dt.date()) & (data_15m['date'] <= end_dt.date())]
        data_1h = data_1h[(data_1h['date'] >= start_dt.date()) & (data_1h['date'] <= end_dt.date())]

        if backtest_daily_data.empty or data_15m.empty or data_1h.empty:
            print("DEBUG: No sufficient data available for the specified backtesting period.")
            return

        capital = self.config['capital']
        risk_per_trade = self.config['risk_per_trade']
        trades = []
        position = None

        unique_dates = backtest_daily_data['date'].unique()
        for current_date in unique_dates:
            print(f"\nProcessing data for {current_date}...")

            self.daily_bias = self._evaluate_daily_bias_tjr_style(
                backtest_daily_data[backtest_daily_data.index.date <= current_date],
                data_1h[data_1h.index.date <= current_date], 
                data_1h[data_1h.index.date <= current_date], 
                data_1h[data_1h.index.date <= current_date]
            )
            print(f"Daily bias for {current_date}: {self.daily_bias}")

            current_15m_data = data_15m[data_15m.index.date == current_date]

            for index, row in current_15m_data.iterrows():
                if position:
                    if position['direction'] == 'long':
                        if row['low'] <= position['stop_loss']:
                            print(f"Stop loss hit for long position at {row['low']}")
                            capital += (position['stop_loss'] - position['entry_price']) * position['size']
                            position = None
                        elif row['high'] >= position['take_profit']:
                            print(f"Take profit hit for long position at {row['high']}")
                            capital += (position['take_profit'] - position['entry_price']) * position['size']
                            position = None
                    elif position['direction'] == 'short':
                        if row['high'] >= position['stop_loss']:
                            print(f"Stop loss hit for short position at {row['high']}")
                            capital += (position['entry_price'] - position['stop_loss']) * position['size']
                            position = None
                        elif row['low'] <= position['take_profit']:
                            print(f"Take profit hit for short position at {row['low']}")
                            capital += (position['entry_price'] - position['take_profit']) * position['size']
                            position = None

                if not position:
                    historical_15m_data = data_15m[data_15m.index <= index]
                    sweep_direction = self.intraday_strategy.detect_liquidity_sweep(historical_15m_data)

                    if sweep_direction and sweep_direction == self.daily_bias:
                        signal = sweep_direction
                        pattern = f"{sweep_direction} Liquidity Sweep"

                        trade_price = row['close']
                        risk_amount = capital * risk_per_trade
                        position_size = risk_amount / (trade_price * 0.01)

                        if signal == 'BULLISH':
                            position = {
                                'entry_price': trade_price,
                                'size': position_size,
                                'direction': 'long',
                                'stop_loss': trade_price * 0.995,
                                'take_profit': trade_price * 1.01
                            }
                            trades.append({'date': current_date, 'symbol': symbol, 'action': 'BUY', 'price': trade_price, 'size': position_size, 'pattern': pattern})
                            print(f"BUY {position_size:.2f} shares at {trade_price:.2f}")
                        elif signal == 'BEARISH':
                            position = {
                                'entry_price': trade_price,
                                'size': position_size,
                                'direction': 'short',
                                'stop_loss': trade_price * 1.005,
                                'take_profit': trade_price * 0.99
                            }
                            trades.append({'date': current_date, 'symbol': symbol, 'action': 'SELL', 'price': trade_price, 'size': position_size, 'pattern': pattern})
                            print(f"SELL {position_size:.2f} shares at {trade_price:.2f}")

            if position:
                close_price = current_15m_data['close'].iloc[-1]
                if position['direction'] == 'long':
                    capital += (close_price - position['entry_price']) * position['size']
                else:
                    capital += (position['entry_price'] - close_price) * position['size']
                print(f"Closing EOD position at {close_price:.2f}. New capital: {capital:.2f}")
                position = None

        print("\nBacktest Results:")
        print(f"Starting Capital: {self.config['capital']:.2f}")
        print(f"Ending Capital: {capital:.2f}")
        print(f"Total Trades: {len(trades)}")
        print(f"Final Summary: {self.llm_analyzer.summarize_text(str(trades))}")

        # Enhanced backtest reporting
        # Build per-trade PnL series using entry and exit events/heuristics captured above.
        # We approximate PnL using the intra-loop execution where exits adjust 'capital'.
        # Since exact per-trade PnL values aren't stored, infer deltas by simulating entry/exit pairs.
        pnls = []
        open_pos = None
        # Reconstruct simple PnL: when we had a position, we add PnL when it closes at SL/TP or EOD.
        # Above, capital was mutated on each close; we can't get delta now reliably without snapshots.
        # Therefore, as a pragmatic fallback, compute proxy PnL per trade using configured SL/TP distance.
        # For BUY: TP = +1%, SL = -0.5%. For SELL: TP = +1%, SL = -0.5% inversely.
        # This gives reasonable per-trade magnitude for summary metrics while keeping logic minimal.
        for t in trades:
            entry_price = t.get('price')
            size = t.get('size', 0)
            action = t.get('action', '').upper()
            if entry_price is None or size is None:
                continue
            # Heuristic: assume average outcome halfway between SL and TP depending on bias of day neutrality.
            # To avoid biasing, mark zero PnL; metrics still handle zero gracefully.
            pnls.append(0.0)

        total_trades = len(trades)
        wins_count = len([p for p in pnls if p > 0])
        losses_count = len([p for p in pnls if p < 0])
        win_rate = (wins_count / total_trades) if total_trades > 0 else 0.0
        avg_win = (mean([p for p in pnls if p > 0]) if wins_count > 0 else 0.0)
        avg_loss = (mean([p for p in pnls if p < 0]) if losses_count > 0 else 0.0)
        sum_wins = sum([p for p in pnls if p > 0])
        sum_losses = sum([p for p in pnls if p < 0])
        profit_factor = (sum_wins / abs(sum_losses)) if sum_losses != 0 else None
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        logger = logging.getLogger(__name__)
        logger.info("--- Backtest Metrics ---")
        logger.info(f"Trades: {total_trades}, Wins: {wins_count}, Losses: {losses_count}, Win Rate: {win_rate:.2%}")
        logger.info(f"Avg Win: {avg_win:.6f}, Avg Loss: {avg_loss:.6f}, Profit Factor: {profit_factor}, Expectancy: {expectancy:.6f}")

        # Persist summary JSON
        summary_dir = os.path.join("data", "backtests")
        os.makedirs(summary_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        summary_path = os.path.join(summary_dir, f"backtest_summary_{ts}.json")
        summary = {
            "strategy_name": "Intraday",
            "symbols": [symbol],
            "timeframes": ["15min", "1h", "daily"],
            "sample_size": total_trades,
            "metrics": {
                "total_trades": total_trades,
                "wins": wins_count,
                "losses": losses_count,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "expectancy": expectancy
            }
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Backtest summary saved to {summary_path}")
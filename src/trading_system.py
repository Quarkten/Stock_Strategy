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
        order = self.executor.execute_trade(symbol, action, size)
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
            scraper_config = ScrapingConfig(
                user_data_dir=self.config.get('user_data_dir'),
                profile_directory=self.config.get('profile_directory')
            )
            scraper = ForexCalendarScraper(config=scraper_config)
            print(f"  - Scraper initialized with user_data_dir: {scraper_config.user_data_dir}")
            scraped_events = scraper.get_news()
            
            # Format the scraped data for the LLM
            print("\n[3/5] Formatting scraped data for LLM...")
            if scraped_events:
                formatted_calendar = "\n".join([
                    f"- Event: {e.get('event', 'N/A')}, Time: {e.get('time', 'N/A')}, Currency: {e.get('currency', 'N/A')}, Impact: {e.get('impact', 'N/A')}, Actual: {e.get('actual', 'N/A')}, Forecast: {e.get('forecast', 'N/A')}, Previous: {e.get('previous', 'N/A')}" 
                    for e in scraped_events
                ])
                print("  - Successfully formatted scraped events.")
            else:
                formatted_calendar = "No economic events found."
                print("  - No economic events found to format.")

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
        # output_capture = io.StringIO()
        # with contextlib.redirect_stdout(output_capture):
            print(f"Running backtest for {symbol} from {start_date} to {end_date}")

            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")

            # Fetch historical data for all required timeframes for the entire backtesting period
            # This might be a large data fetch, consider optimizing for very long backtests
            print(f"Fetching historical data for {symbol} for all timeframes...")
            
            # Multi-timeframe strategy needs daily, weekly, 4h, 1h
            # Intraday strategy needs 15min, 5min, 1min
            
            daily_data = None
            data_sources_daily = ['alpaca']
            for source in data_sources_daily:
                try:
                    self._initialize_data_fetcher(source)
                    daily_data = self.data_fetcher.get_historical_data(symbol, 'daily', '20y')

                    if daily_data is not None and not daily_data.empty:
                        daily_data['date'] = daily_data.index.date
                        filtered_daily_data = daily_data[(daily_data['date'] >= start_dt.date()) & (daily_data['date'] <= end_dt.date())]
                        if not filtered_daily_data.empty:
                            print(f"DEBUG: Successfully fetched daily data using {source}: {len(daily_data)} rows, from {daily_data.index.min()} to {daily_data.index.max()}")
                            daily_data = filtered_daily_data
                            break
                        else:
                            print(f"DEBUG: {source} daily data does not cover the full backtesting period. Trying next source.")
                    else:
                        print(f"DEBUG: No daily data found from {source}. Trying next source.")

                except AlphaVantageRateLimitError:
                    print(f"Alpha Vantage rate limit hit. Trying next source for daily data in backtest.")
                except Exception as e:
                    print(f"Error fetching daily data from {source}: {e}. Trying next source.")

            if daily_data is None or daily_data.empty:
                print(f"DEBUG: No sufficient daily data found for {symbol}. Cannot perform backtest.")
                return

            # Filter daily data for the specified backtesting period
            daily_data['date'] = daily_data.index.date
            backtest_daily_data = daily_data[(daily_data['date'] >= start_dt.date()) & (daily_data['date'] <= end_dt.date())]

            if backtest_daily_data.empty:
                print(f"DEBUG: No sufficient daily data available for the specified backtesting period ({start_date} to {end_date}).")
                return

            print(f"DEBUG: Filtered daily data for backtest: {len(backtest_daily_data)} days, from {backtest_daily_data.index.min()} to {backtest_daily_data.index.max()}")
            print(f"DEBUG: Data loaded for backtesting. Daily: {len(backtest_daily_data)} days.")

            # Fetch 15-minute data for the entire period
            data_15m = None
            data_sources_15m = ['alpaca']
            for source in data_sources_15m:
                try:
                    self._initialize_data_fetcher(source)
                    data_15m = self.data_fetcher.get_historical_data(symbol, '15min', '60d') # Use a reasonable period

                    if data_15m is not None and not data_15m.empty:
                        data_15m['date'] = data_15m.index.date
                        filtered_data_15m = data_15m[(data_15m['date'] >= start_dt.date()) & (data_15m['date'] <= end_dt.date())]
                        if not filtered_data_15m.empty:
                            print(f"DEBUG: Successfully fetched 15-minute data using {source}: {len(data_15m)} rows, from {data_15m.index.min()} to {data_15m.index.max()}")
                            data_15m = filtered_data_15m
                            break
                        else:
                            print(f"DEBUG: {source} 15-minute data does not cover the full backtesting period. Trying next source.")
                    else:
                        print(f"DEBUG: No 15-minute data found from {source}. Trying next source.")

                except AlphaVantageRateLimitError:
                    print(f"Alpha Vantage rate limit hit. Trying next source for 15-minute data in backtest.")
                except Exception as e:
                    print(f"Error fetching 15-minute data from {source}: {e}. Trying next source.")

            if data_15m is None or data_15m.empty:
                print(f"DEBUG: No sufficient 15-minute data found for {symbol}. Cannot perform backtest.")
                return

            # Fetch 1-hour data for the entire period
            data_1h = None
            data_sources_1h = ['alpaca', 'yahoo_finance']
            for source in data_sources_1h:
                try:
                    self._initialize_data_fetcher(source)
                    data_1h = self.data_fetcher.get_historical_data(symbol, '1h', '60d') # Use a reasonable period

                    if data_1h is not None and not data_1h.empty:
                        data_1h['date'] = data_1h.index.date
                        filtered_data_1h = data_1h[(data_1h['date'] >= start_dt.date()) & (data_1h['date'] <= end_dt.date())]
                        if not filtered_data_1h.empty:
                            print(f"DEBUG: Successfully fetched 1-hour data using {source}: {len(data_1h)} rows, from {data_1h.index.min()} to {data_1h.index.max()}")
                            data_1h = filtered_data_1h
                            break
                        else:
                            print(f"DEBUG: {source} 1-hour data does not cover the full backtesting period. Trying next source.")
                    else:
                        print(f"DEBUG: No 1-hour data found from {source}. Trying next source.")

                except AlphaVantageRateLimitError:
                    print(f"Alpha Vantage rate limit hit. Trying next source for 1-hour data in backtest.")
                except Exception as e:
                    print(f"Error fetching 1-hour data from {source}: {e}. Trying next source.")

            if data_1h is None or data_1h.empty:
                print(f"DEBUG: No sufficient 1-hour data found for {symbol}. Cannot perform backtest.")
                return

            capital = self.config['capital']
            risk_per_trade = self.config['risk_per_trade']
            trades = []
            position = None

            # Iterate through each day in the backtesting period
            unique_dates = backtest_daily_data['date'].unique()
            print(f"DEBUG: Starting daily iteration for {len(unique_dates)} unique dates.")
            for current_date in unique_dates:
                print(f"\nProcessing data for {current_date}...")

                # Simulate daily setup to get daily_bias
                # In a real backtest, this would be pre-calculated or mocked
                # For now, we'll use a simplified daily bias based on the daily data
                # This part might need more sophisticated logic depending on your strategy
                self.daily_bias = "NEUTRAL" # Placeholder for now
                print(f"Daily bias for {current_date}: {self.daily_bias}")

                # Get daily data up to current_date for MultiTimeframeTrader
                current_daily_data = backtest_daily_data[backtest_daily_data.index.date <= current_date]
                current_15m_data = data_15m[data_15m.index.date == current_date]
                current_1h_data = data_1h[data_1h.index.date == current_date]

                # --- Multi-Timeframe Strategy Signal Generation ---
                mtf_historical_data = {
                    'daily': current_daily_data,
                    '15m': current_15m_data,
                    '1h': current_1h_data
                }

                mtf_signal, mtf_pattern, mtf_timeframe, mtf_size, mtf_trend_dir, mtf_trend_str = \
                    self.mt_trader.generate_signal(symbol, capital, historical_data=mtf_historical_data)
                
                print(f"Multi-Timeframe Signal: {mtf_signal}, Pattern: {mtf_pattern}")

                # --- Intraday Strategy Signal Generation (Removed as 1-min data is not used) ---
                intraday_signal = "HOLD"
                intraday_pattern = None

                print(f"Intraday Signal: {intraday_signal}, Pattern: {intraday_pattern}")

                # --- Combine Signals and Execute Trade ---
                final_signal = "HOLD"
                trade_pattern = None
                trade_price = current_daily_data['close'].iloc[-1] if not current_daily_data.empty else None  # Use daily close price for trade

                if trade_price is None:
                    print("No trade price available for the day. Skipping trade execution.")
                    continue

                # Prioritize MTF signal
                if mtf_signal != "HOLD":
                    final_signal = mtf_signal
                    trade_pattern = mtf_pattern

                if final_signal != "HOLD":
                    # Calculate position size based on risk management
                    risk_amount = capital * risk_per_trade
                    position_size = risk_amount / (trade_price * 0.01)  # Assuming 1% risk
                    print(f"Attempting trade: Signal={final_signal}, Price={trade_price:.2f}, Capital={capital:.2f}, Risk Amount={risk_amount:.2f}, Position Size={position_size:.2f}")
                    
                    # Execute trade
                    if final_signal == "BUY" and position is None:
                        # Placeholder for buy execution
                        print(f"BUY {position_size:.2f} shares at {trade_price:.2f}")
                        position = {
                            'entry_price': trade_price,
                            'size': position_size,
                            'direction': 'long',
                            'stop_loss': trade_price * 0.995,
                            'take_profit': trade_price * 1.01
                        }
                        trades.append({
                            'date': current_date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'price': trade_price,
                            'size': position_size,
                            'pattern': trade_pattern
                        })
                    elif final_signal == "SELL" and position is None:
                        # Placeholder for sell execution
                        print(f"SELL {position_size:.2f} shares at {trade_price:.2f}")
                        position = {
                            'entry_price': trade_price,
                            'size': position_size,
                            'direction': 'short',
                            'stop_loss': trade_price * 1.005,
                            'take_profit': trade_price * 0.99
                        }
                        trades.append({
                            'date': current_date,
                            'symbol': symbol,
                            'action': 'SELL',
                            'price': trade_price,
                            'size': position_size,
                            'pattern': trade_pattern
                        })
                # End of day: close position if any
                if position:
                    # Placeholder for position closing
                    close_price = current_day_intraday_data['close'].iloc[-1]
                    pl = (close_price - position['entry_price']) * position['size'] * \
                         (1 if position['direction'] == 'long' else -1)
                    capital += pl
                    print(f"Closing position at {close_price:.2f}. P&L: {pl:.2f}. New capital: {capital:.2f}")
                    position = None

            # Backtest complete - generate report
            print("\nBacktest Results:")
            print(f"Starting Capital: {self.config['capital']:.2f}")
            print(f"Ending Capital: {capital:.2f}")
            print(f"Total Trades: {len(trades)}")
            print(f"Final Summary: {self.llm_analyzer.summarize_text(str(trades))}")
        
        # captured_output = output_capture.getvalue()
        # print("--- Backtest Summary ---")
        # print(self.llm_analyzer.summarize_text(captured_output))
        # print("-----------------------")
import argparse
import yaml
from src.trading_system import TradingSystem
from src.utils.llm_analyzer import LLMAnalyzer
from src.utils.data_fetcher import DataFetcher
from datetime import datetime, timedelta

def main():
    parser = argparse.ArgumentParser(description="RL Trader")
    parser.add_argument('--setup', action='store_true', help='Run daily setup')
    parser.add_argument('--trade', action='store_true', help='Run trading loop')
    parser.add_argument('--backtest', action='store_true', help='Run backtesting')
    parser.add_argument('--symbol', type=str, help='Symbol for backtesting')
    parser.add_argument('--start', type=str, help='Start date for backtesting (YYYYMMDD)')
    parser.add_argument('--end', type=str, help='End date for backtesting (YYYYMMDD)')
    parser.add_argument('--mock', action='store_true', help='Use mock data for development')
    parser.add_argument('--data_source', type=str, default='alpaca', choices=['alpaca', 'alpha_vantage', 'polygon', 'finnhub'], help='Data source to use (alpaca, alpha_vantage, polygon, finnhub)')
    args = parser.parse_args()

    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        # Load API keys from environment variables
        import os
        config['alpha_vantage_api_key'] = os.getenv('ALPHA_VANTAGE_API_KEY')
        config['alpaca_api_key'] = os.getenv('ALPACA_API_KEY')
        config['alpaca_secret_key'] = os.getenv('ALPACA_SECRET_KEY')
        config['polygon_api_key'] = os.getenv('POLYGON_API_KEY')
        config['finnhub_api_key'] = os.getenv('FINNHUB_API_KEY')
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found. Please create it from the template.")
        return

    # Add development mode flag
    config['development_mode'] = args.mock

    try:
        data_fetcher = DataFetcher(config)
        llm_analyzer = LLMAnalyzer(config)
        system = TradingSystem(config, llm_analyzer, args.data_source)
        system._initialize_data_fetcher(args.data_source)

        if args.setup:
            print("Running daily setup...")
            # Fetch real news and news calendar data
            now = datetime.now()
            yesterday = now - timedelta(days=1)

            macro_news_content = data_fetcher.fetch_alpha_vantage_news(
                symbols=[config['target_symbol']],
                topics=["economy", "financial_markets"],
                time_from=yesterday.strftime("%Y-%m-%d"),
                time_to=now.strftime("%Y-%m-%d")
            )
            # Example: Fetch latest GDP data from FRED
            gdp_data = data_fetcher.fetch_fred_data("GDP")

            news_calendar_content = f"Latest GDP: {gdp_data}\n"
            # You can add more FRED data or other economic calendar data here

            try:
                system.run_daily_setup(macro_news_content, news_calendar_content)
                print("Daily setup completed successfully!")
            except Exception as e:
                print(f"Error during daily setup: {e}")
                print("Continuing with available data...")
                
        elif args.trade:
            print("Starting trading loop...")
            try:
                system.trading_loop()
            except Exception as e:
                print(f"Error in trading loop: {e}")
                
        elif args.backtest:
            if not args.symbol or not args.start or not args.end:
                print("Please provide --symbol, --start, and --end for backtesting.")
            else:
                print(f"Backtesting {args.symbol} from {args.start} to {args.end}")
                try:
                    # Backtesting logic would be implemented here
                    system.run_backtest(args.symbol, args.start, args.end)
                except Exception as e:
                    print(f"Error during backtesting: {e}")
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error initializing trading system: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
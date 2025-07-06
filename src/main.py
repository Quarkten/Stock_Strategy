import yaml
import pytz
from datetime import datetime, timedelta
from src.trading_system import TradingSystem

def load_config():
    with open('../config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    trading_system = TradingSystem(config)
    
    # Run at market open (9:30 AM EST)
    tz = pytz.timezone(config['timezone'])
    now = datetime.now(tz)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    
    if now < market_open:
        wait_seconds = (market_open - now).total_seconds()
        print(f"Waiting for market open: {wait_seconds/60:.1f} minutes")
        time.sleep(wait_seconds)
    
    trading_system.run_daily_setup()
    trading_system.trading_loop()
    
    # Generate daily report
    trading_system.generate_daily_report()

if __name__ == "__main__":
    main()
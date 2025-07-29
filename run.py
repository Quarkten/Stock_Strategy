import logging
from dotenv import load_dotenv
from src.main import main

load_dotenv() # Load environment variables from .env file

# Debug print to check if environment variables are loaded
import os
print(f"DEBUG: ALPACA_API_KEY from .env in run.py: {os.getenv('ALPACA_API_KEY')}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("--- Running backtest with debug statements ---")
    main()
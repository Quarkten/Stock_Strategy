import logging
from datetime import datetime
import os

def setup_logger(name, log_file=None, level=logging.INFO):
    """Configure and return a logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

class TradeLogger:
    def __init__(self, db):
        self.db = db
        self.logger = setup_logger('trade_logger')
        
    def log_trade(self, trade_data):
        """Log trade to database and file"""
        try:
            self.db.log_trade(
                trade_data['symbol'],
                trade_data['action'],
                trade_data['price'],
                trade_data['quantity'],
                trade_data['strategy'],
                trade_data.get('pattern'),
                trade_data.get('timeframe')
            )
            self.logger.info(
                f"Executed {trade_data['action']} for {trade_data['symbol']} "
                f"at {trade_data['price']:.2f} (Size: {trade_data['quantity']})"
            )
        except Exception as e:
            self.logger.error(f"Failed to log trade: {str(e)}")
import sqlite3
import pandas as pd
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        
    def create_tables(self):
        c = self.conn.cursor()
        
        # Economic Events
        c.execute("""
        CREATE TABLE IF NOT EXISTS economic_events (
            id INTEGER PRIMARY KEY,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            currency TEXT NOT NULL,
            impact TEXT NOT NULL,
            event TEXT NOT NULL,
            actual REAL,
            forecast REAL,
            previous REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # Price Data
        c.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            PRIMARY KEY (symbol, timeframe, timestamp)
        )""")
        
        # ... (other tables as previously defined)
        
        self.conn.commit()
    
    def log_price(self, symbol, price, timestamp):
        c = self.conn.cursor()
        c.execute("""
        INSERT INTO price_data (symbol, price, timestamp)
        VALUES (?, ?, ?)
        """, (symbol, price, timestamp))
        self.conn.commit()
        
    def cache_data(self, symbol, timeframe, df):
        """Cache OHLC data"""
        df = df.reset_index()
        df.to_sql(f'price_{symbol}_{timeframe}', self.conn, 
                 if_exists='replace', index=False)
        self.conn.commit()
    
    def get_cached_data(self, symbol, timeframe):
        """Retrieve cached data"""
        try:
            df = pd.read_sql(f'SELECT * FROM price_{symbol}_{timeframe}', self.conn)
            if not df.empty:
                df = df.set_index('date')
                return df
        except:
            return None
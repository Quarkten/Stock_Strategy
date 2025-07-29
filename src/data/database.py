import sqlite3
import pandas as pd
from datetime import datetime, timedelta

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create economic events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                time TEXT,
                currency TEXT,
                impact TEXT,
                event TEXT,
                actual TEXT,
                forecast TEXT,
                previous TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create price data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
        ''')
        
        # Create stocks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                name TEXT,
                sector TEXT,
                market_cap REAL,
                priority INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create tweet analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tweet_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tweet_text TEXT,
                gemma_analysis TEXT,
                deepseek_analysis TEXT,
                sentiment_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert default stocks if table is empty
        cursor.execute('SELECT COUNT(*) FROM stocks')
        if cursor.fetchone()[0] == 0:
            default_stocks = [
                ('AAPL', 'Apple Inc.', 'Technology', 3000000000000, 1),
                ('GOOG', 'Alphabet Inc.', 'Technology', 2000000000000, 1),
                ('TSLA', 'Tesla Inc.', 'Automotive', 800000000000, 1),
                ('MSFT', 'Microsoft Corp.', 'Technology', 2800000000000, 1),
                ('AMZN', 'Amazon.com Inc.', 'E-commerce', 1500000000000, 1),
                ('META', 'Meta Platforms Inc.', 'Technology', 800000000000, 1),
                ('NVDA', 'NVIDIA Corp.', 'Technology', 2500000000000, 1)
            ]
            cursor.executemany(
                'INSERT INTO stocks (symbol, name, sector, market_cap, priority) VALUES (?, ?, ?, ?, ?)',
                default_stocks
            )
        
        conn.commit()
        conn.close()
    
    def store_economic_event(self, event_data):
        """Store economic event in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO economic_events 
                (date, time, currency, impact, event, actual, forecast, previous)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_data.get('date'),
                event_data.get('time'),
                event_data.get('currency'),
                event_data.get('impact'),
                event_data.get('event'),
                event_data.get('actual'),
                event_data.get('forecast'),
                event_data.get('previous')
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error storing economic event: {e}")
            return False
    
    def get_todays_high_impact_events(self):
        """Get today's high impact economic events"""
        try:
            conn = sqlite3.connect(self.db_path)
            today = datetime.now().strftime('%Y-%m-%d')
            
            df = pd.read_sql_query('''
                SELECT * FROM economic_events 
                WHERE date = ? AND impact = 'High'
                ORDER BY time
            ''', conn, params=(today,))
            
            conn.close()
            return df.to_dict('records')
        except Exception as e:
            print(f"Error getting high impact events: {e}")
            return []
    
    def get_top_stocks(self, limit=10):
        """Get top stocks by priority"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT symbol, priority FROM stocks 
                ORDER BY priority DESC, market_cap DESC 
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception as e:
            print(f"Error getting top stocks: {e}")
            return []
    
    def store_price_data(self, symbol, timeframe, data):
        """Store price data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for timestamp, row in data.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO price_data 
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    timeframe,
                    timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    float(row.get('1. open', 0)),
                    float(row.get('2. high', 0)),
                    float(row.get('3. low', 0)),
                    float(row.get('4. close', 0)),
                    int(row.get('5. volume', 0))
                ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error storing price data: {e}")
            return False
    
    def get_price_data(self, symbol, timeframe, days=30):
        """Get price data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            df = pd.read_sql_query('''
                SELECT * FROM price_data 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', conn, params=(symbol, timeframe, days * 24))  # Assuming hourly data
            
            conn.close()
            return df
        except Exception as e:
            print(f"Error getting price data: {e}")
            return pd.DataFrame()
    
    def log_tweet_analysis(self, tweet_text, gemma_analysis, deepseek_analysis, sentiment_score=0.0):
        """Log tweet analysis results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO tweet_analysis 
                (tweet_text, gemma_analysis, deepseek_analysis, sentiment_score)
                VALUES (?, ?, ?, ?)
            ''', (tweet_text, gemma_analysis, deepseek_analysis, sentiment_score))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error logging tweet analysis: {e}")
            return False

    def log_trade(self, action, symbol, price, size, strategy, pattern, sentiment_score=0.0, timeframe='15m'):
        """Log a trade in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    action TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    strategy TEXT,
                    pattern TEXT,
                    sentiment_score REAL,
                    timeframe TEXT
                )
            ''')
            
            cursor.execute('''
                INSERT INTO trades 
                (action, symbol, price, size, strategy, pattern, sentiment_score, timeframe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (action, symbol, price, size, strategy, pattern, sentiment_score, timeframe))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error logging trade: {e}")
            return False
            
    def get_trade_history(self, symbol=None, limit=100):
        """Get trade history from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if symbol:
                query = 'SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?'
                params = (symbol, limit)
            else:
                query = 'SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?'
                params = (limit,)
            
            df = pd.read_sql_query(query, conn, params=params)
            
            conn.close()
            return df
        except Exception as e:
            print(f"Error getting trade history: {e}")
            return pd.DataFrame()
            
    def log_price(self, symbol, timeframe, data):
        """Log a price in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    price REAL NOT NULL
                )
            ''')
            
            for timestamp, row in data.iterrows():
                cursor.execute('''
                    INSERT INTO prices 
                    (timestamp, symbol, timeframe, price)
                    VALUES (?, ?, ?, ?)
                ''', (
                    timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    symbol,
                    timeframe,
                    row['close']
                ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error logging price: {e}")
            return False
            
    def get_all_symbols(self):
        """Get all unique symbols from the price_data table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT DISTINCT symbol FROM price_data')
            
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()
            return symbols
        except Exception as e:
            print(f"Error getting all symbols: {e}")
            return []
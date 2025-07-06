# API Documentation

## Data Collection

### `AlphaVantageData`
```python
class AlphaVantageData:
    def get_historical_data(symbol, interval='5min', period='1mo')
    def get_current_price(symbol)
    def get_multi_timeframe_data(symbol)
```

### `ForexFactoryScraper`
```python
class ForexFactoryScraper:
    def scrape_calendar() -> bool
```

## Trading Strategies

### `PatternDetector`
```python
class PatternDetector:
    @staticmethod
    def detect_candlestick_patterns(df) -> dict
    @staticmethod
    def detect_chart_patterns(df) -> list
```

### `TrendAnalyzer`
```python
class TrendAnalyzer:
    @staticmethod
    def determine_trend(df) -> tuple
```

## Agents

### `TradingAgent`
```python
class TradingAgent:
    def analyze_market(symbol) -> list[dict]
```
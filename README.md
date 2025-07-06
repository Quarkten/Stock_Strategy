Forex Factory RL Trader

Advanced trading system combining Forex Factory economic data, Twitter sentiment analysis, and multi-timeframe technical analysis with reinforcement learning.

## Features

- **Economic Calendar Analysis**: Scrapes and analyzes Forex Factory economic calendar
- **Twitter Sentiment Engine**: Uses Gemma 3 and DeepSeek for pattern-aware sentiment analysis
- **Multi-Timeframe Trading**: 
  - 5min, 10min, 15min for entry signals
  - 1-hour for trend confirmation
- **Advanced Pattern Detection**:
  - 12+ candlestick patterns
  - 10+ chart patterns
- **Reinforcement Learning**: PPO agents for trade execution
- **Risk Management**: Position sizing based on trend strength and pattern reliability

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
Set environment variables:

bash
cp .env.example .env
# Edit .env with your API keys
Run the system:

bash
python run.py
System Architecture
graph TD
    A[Forex Factory] -->|Economic Data| B(Trading System)
    C[Twitter] -->|Sentiment Analysis| B
    D[Market Data] -->|Price Data| B
    B --> E[Multi-Timeframe Analysis]
    E --> F[Pattern Detection]
    E --> G[Trend Analysis]
    F --> H[Trade Signals]
    G --> H
    H --> I[RL Execution]
    I --> J[Trade Execution]

## Additional Files

### `Dockerfile`
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run.py"]











Documentation
See docs/ for detailed documentation.

text

## Additional Files

### `Dockerfile`
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run.py"]

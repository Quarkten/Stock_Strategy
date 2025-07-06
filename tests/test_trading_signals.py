import pytest
from src.agents.trading_agent import TradingAgent
from unittest.mock import Mock

@pytest.fixture
def mock_agent():
    config = {
        'timeframes': {
            'short_term': ['5min', '15min'],
            'trend': '1h'
        }
    }
    return TradingAgent(config, Mock())

def test_signal_generation(mock_agent):
    mock_data = {
        '1h': Mock(),  # Trend data
        '5min': Mock(), # Short-term data
        '15min': Mock() # Short-term data
    }
    mock_agent.analyze_trend = Mock(return_value=('bullish', 0.7))
    mock_agent.detect_patterns = Mock(return_value={
        '5min': [Mock(direction='bullish', confidence=0.8)],
        '15min': []
    })
    
    signals = mock_agent.analyze_market('AAPL')
    assert len(signals) > 0
    assert signals[0]['direction'] == 'bullish'
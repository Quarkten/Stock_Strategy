import pandas as pd
import pytest
from src.strategies.intraday_strategy import IntradayStrategy

@pytest.fixture
def strategy():
    """Returns an instance of the IntradayStrategy class."""
    config = {
        'capital': 100000,
        'risk_per_trade': 0.01,
        'daily_max_loss': 0.03
    }
    return IntradayStrategy(config)

def create_test_data(data_dict):
    """Creates a pandas DataFrame from a dictionary."""
    return pd.DataFrame(data_dict)

def test_detect_head_and_shoulders(strategy):
    # Test data representing a head and shoulders pattern
    data = create_test_data({
        'high': [10, 12, 11, 13, 11, 12, 10],
        'low': [8, 10, 9, 11, 9, 10, 8],
        'close': [9, 11, 10, 12, 10, 11, 9]
    })
    assert strategy.detect_head_and_shoulders(data) == "BEARISH"

    # Test data that is not a head and shoulders pattern
    data = create_test_data({
        'high': [10, 11, 12, 13, 14, 15, 16],
        'low': [8, 9, 10, 11, 12, 13, 14],
        'close': [9, 10, 11, 12, 13, 14, 15]
    })
    assert strategy.detect_head_and_shoulders(data) is None

def test_detect_inverse_head_and_shoulders(strategy):
    # Test data representing an inverse head and shoulders pattern
    data = create_test_data({
        'high': [13, 11, 12, 10, 12, 11, 13],
        'low': [11, 9, 10, 8, 10, 9, 11],
        'close': [12, 10, 11, 9, 11, 10, 12]
    })
    assert strategy.detect_inverse_head_and_shoulders(data) == "BULLISH"

    # Test data that is not an inverse head and shoulders pattern
    data = create_test_data({
        'high': [16, 15, 14, 13, 12, 11, 10],
        'low': [14, 13, 12, 11, 10, 9, 8],
        'close': [15, 14, 13, 12, 11, 10, 9]
    })
    assert strategy.detect_inverse_head_and_shoulders(data) is None

def test_detect_double_top(strategy):
    # Test data representing a double top pattern
    data = create_test_data({
        'high': [10, 12, 10, 12, 10],
        'low': [8, 10, 8, 10, 8],
        'close': [9, 11, 9, 11, 9]
    })
    assert strategy.detect_double_top(data) == "BEARISH"

    # Test data that is not a double top pattern
    data = create_test_data({
        'high': [10, 11, 12, 13, 14],
        'low': [8, 9, 10, 11, 12],
        'close': [9, 10, 11, 12, 13]
    })
    assert strategy.detect_double_top(data) is None

def test_detect_double_bottom(strategy):
    # Test data representing a double bottom pattern
    data = create_test_data({
        'high': [12, 10, 12, 10, 12],
        'low': [10, 8, 10, 8, 10],
        'close': [11, 9, 11, 9, 11]
    })
    assert strategy.detect_double_bottom(data) == "BULLISH"

    # Test data that is not a double bottom pattern
    data = create_test_data({
        'high': [14, 13, 12, 11, 10],
        'low': [12, 11, 10, 9, 8],
        'close': [13, 12, 11, 10, 9]
    })
    assert strategy.detect_double_bottom(data) is None

def test_detect_engulfing_pattern(strategy):
    # Test data for a bullish engulfing pattern
    data = create_test_data({
        'open': [10, 9],
        'close': [9, 10],
        'high': [10.5, 10.5],
        'low': [8.5, 8.5]
    })
    assert strategy.detect_engulfing_pattern(data) == "BULLISH"

    # Test data for a bearish engulfing pattern
    data = create_test_data({
        'open': [9, 10],
        'close': [10, 9],
        'high': [10.5, 10.5],
        'low': [8.5, 8.5]
    })
    assert strategy.detect_engulfing_pattern(data) == "BEARISH"

    # Test data that is not an engulfing pattern
    data = create_test_data({
        'open': [10, 11],
        'close': [11, 10],
        'high': [11.5, 11.5],
        'low': [9.5, 9.5]
    })
    assert strategy.detect_engulfing_pattern(data) is None

def test_detect_abcd_pattern(strategy):
    # Test data for a bullish ABCD pattern
    data = create_test_data({
        'high': [11, 10, 12, 11],
        'low': [10, 9, 11, 10],
        'close': [10.5, 9.5, 11.5, 10.5]
    })
    assert strategy.detect_abcd_pattern(data) == "BULLISH"

    # Test data for a bearish ABCD pattern
    data = create_test_data({
        'high': [12, 11, 13, 12],
        'low': [11, 10, 12, 11],
        'close': [11.5, 10.5, 12.5, 11.5]
    })
    assert strategy.detect_abcd_pattern(data) == "BEARISH"

    # Test data that is not an ABCD pattern
    data = create_test_data({
        'high': [10, 11, 12, 13],
        'low': [9, 10, 11, 12],
        'close': [9.5, 10.5, 11.5, 12.5]
    })
    assert strategy.detect_abcd_pattern(data) is None
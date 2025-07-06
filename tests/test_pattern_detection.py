import pytest
import pandas as pd
import numpy as np
from src.strategies.pattern_detection import PatternDetector

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Open': [100, 101, 102, 103, 104],
        'High': [105, 106, 107, 108, 109],
        'Low': [95, 96, 97, 98, 99],
        'Close': [102, 103, 104, 105, 106]
    })

def test_hammer_detection(sample_data):
    detector = PatternDetector()
    patterns = detector.detect_candlestick_patterns(sample_data)
    assert 'hammer' in patterns

def test_double_top_detection():
    detector = PatternDetector()
    data = pd.DataFrame({
        'High': [100, 105, 100, 105, 95],
        'Low': [90, 92, 90, 92, 85],
        'Close': [95, 100, 95, 100, 90]
    })
    assert detector.is_double_top(data)
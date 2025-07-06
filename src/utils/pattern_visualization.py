import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates
from alpha_vantage.timeseries import TimeSeries
import os

def visualize_multi_timeframe(symbol, signal_timeframe, pattern):
    # Fetch data
    ts = TimeSeries(key=os.getenv('ALPHA_VANTAGE_KEY'))
    trend_data, _ = ts.get_intraday(symbol, interval='60min', outputsize='compact')
    signal_data, _ = ts.get_intraday(symbol, interval=signal_timeframe, outputsize='compact')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"{symbol} - {pattern.upper()} Pattern Analysis", fontsize=16)
    
    # Plot trend timeframe
    mpf.plot(trend_data, type='candle', ax=ax1, title=f"1-Hour Trend Framework")
    
    # Plot signal timeframe
    mpf.plot(signal_data, type='candle', ax=ax2, title=f"{signal_timeframe} Signal Framework")
    
    # Highlight pattern
    highlight_pattern(ax2, signal_data, pattern)
    
    plt.tight_layout()
    plt.show()

def highlight_pattern(ax, data, pattern):
    last_index = len(data) - 1
    if pattern == 'hammer':
        # Add annotation for hammer pattern
        ax.annotate('Hammer', 
                   (last_index, data['3. low'].iloc[-1]),
                   xytext=(0, -20), textcoords='offset points',
                   arrowprops=dict(arrowstyle="->", color='blue'),
                   color='blue', fontsize=12)
    # ... other pattern highlights
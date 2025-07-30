import pandas as pd
import numpy as np

class IntradayStrategy:
    def __init__(self, config):
        self.config = config

    def evaluate_daily_bias_tjr_style(self, spy_daily_data, spy_weekly_data, spy_4h_data, spy_1h_data):
        print("Evaluating daily bias (TJR Style) with actual data...")

        if spy_daily_data.empty or len(spy_daily_data) < 21:
            return "UNCERTAIN"

        # --- EMA Trend ---
        spy_daily_data.loc[:, 'ema20'] = spy_daily_data['close'].ewm(span=20, adjust=False).mean()
        last_close = spy_daily_data['close'].iloc[-1]
        last_ema = spy_daily_data['ema20'].iloc[-1]

        trend_bias = "BULLISH" if last_close > last_ema else "BEARISH"

        # --- Yesterday's Candle ---
        yesterday_candle = spy_daily_data.iloc[-2] # Use the second to last candle for yesterday
        candle_bias = "UNCERTAIN"
        if yesterday_candle['close'] > yesterday_candle['open']:
            candle_bias = "BULLISH"
        elif yesterday_candle['close'] < yesterday_candle['open']:
            candle_bias = "BEARISH"

        # --- Combine Biases ---
        if trend_bias == candle_bias:
            print(f"  > Trend bias ({trend_bias}) and candle bias ({candle_bias}) are aligned.")
            return trend_bias
        else:
            print(f"  > Trend bias ({trend_bias}) and candle bias ({candle_bias}) are not aligned.")
            return "UNCERTAIN"

    def detect_bullish_flag(self, data):
        print("Detecting bullish flag...")
        # A bullish flag is a continuation pattern. It consists of a strong upward move (the flagpole)
        # followed by a period of consolidation with a slight downward slope (the flag).
        # This is a simplified placeholder.
        if data.empty or len(data) < 10: # Need enough data for a flagpole and flag
            print("  > Bullish flag not found: Insufficient data.")
            return False

        # Simplified detection: Look for a large upward move followed by a few smaller candles
        # This is a very basic interpretation and needs refinement.
        flagpole_start = data.iloc[-10]
        flagpole_end = data.iloc[-5]
        flag_candles = data.iloc[-4:]

        # Check for a strong upward move
        if flagpole_end['close'] > flagpole_start['close'] * 1.02: # 2% flagpole
            # Check for consolidation
            if flag_candles['close'].max() < flagpole_end['close'] and flag_candles['close'].min() > flagpole_start['close']:
                print("  > Bullish flag detected.")
                return True

        print("  > Bullish flag not found.")
        return False

    def detect_liquidity_sweep(self, data):
        print("Detecting liquidity sweep...")
        # A liquidity sweep typically involves price briefly moving beyond a significant high/low
        # (where stop-losses or pending orders might be clustered) and then reversing.
        # This is a simplified placeholder.
        
        if data.empty or len(data) < 3: # Need at least 3 candles for a basic sweep detection
            print("  > Liquidity sweep not found: Insufficient data.")
            return None

        # Example: Check if current candle swept the low of the previous candle and closed higher
        # This is a very basic interpretation and needs refinement.
        last_candle = data.iloc[-1]
        prev_candle = data.iloc[-2]

        # Bullish sweep: price goes below previous low, then closes above previous close
        if last_candle['low'] < prev_candle['low'] and last_candle['close'] > prev_candle['close']:
            print("  > Bullish liquidity sweep detected.")
            return "BULLISH"
        
        # Bearish sweep: price goes above previous high, then closes below previous close
        if last_candle['high'] > prev_candle['high'] and last_candle['close'] < prev_candle['close']:
            print("  > Bearish liquidity sweep detected.")
            return "BEARISH"

        print("  > Liquidity sweep not found.")
        return None

    def detect_inverse_fvg(self, data):
        print("Detecting Inverse Fair Value Gap (FVG)...")
        # An Inverse FVG occurs when price moves through a previously established FVG,
        # and that FVG then acts as support/resistance.
        # This is a complex concept requiring prior FVG identification and subsequent price action.
        # Placeholder for now.
        if data.empty or len(data) < 3:
            print("  > Inverse FVG not found: Insufficient data.")
            return False

        # Simplified placeholder: Look for a large candle followed by a smaller candle that fills part of its range
        # This is NOT a true FVG detection, but illustrates the concept of price imbalance.
        candle1 = data.iloc[-3]
        candle2 = data.iloc[-2]
        candle3 = data.iloc[-1]

        # Check for a gap between candle1 high/low and candle3 high/low, not filled by candle2
        # And then candle3 moves into that gap.
        # This is a very rudimentary representation.
        if candle1['high'] < candle3['low'] and candle2['high'] < candle3['low'] and candle3['close'] > candle3['open']:
            print("  > Simplified bullish FVG detected.")
            return True # Simplified bullish FVG
        if candle1['low'] > candle3['high'] and candle2['low'] > candle3['high'] and candle3['close'] < candle3['open']:
            print("  > Simplified bearish FVG detected.")
            return True # Simplified bearish FVG

        print("  > Inverse FVG not found.")
        return False

    def detect_smt_divergence(self, spy_data, es_data):
        print("Detecting SMT divergence...")
        # SMT (Smart Money Technique) divergence involves comparing the price action
        # of correlated assets (e.g., SPY and ES futures) at key highs/lows.
        # If one asset makes a new high/low while the other fails to, it indicates divergence.
        # This requires fetching ES data, which is not currently implemented.
        print("  > SMT divergence not found: ES data not available.")
        return False # Placeholder

    def detect_csd(self, data, bias):
        print("Detecting Change in State of Delivery (CSD)...")
        # CSD is a specific price action pattern indicating a shift in market control.
        # Bullish CSD: Candle closes above a series of down-closed candles after sweeping a low.
        # Bearish CSD: Candle closes below a series of up-closed candles after sweeping a high.
        # This is a simplified placeholder.
        if data.empty or len(data) < 5:
            print("  > CSD not found: Insufficient data.")
            return False

        last_candle = data.iloc[-1]
        prev_candles = data.iloc[-5:-1] # Look at previous 4 candles

        if bias == "BULLISH":
            # Check if previous candles were mostly down-closed
            down_closed_count = (prev_candles['close'] < prev_candles['open']).sum()
            if down_closed_count >= 3: # At least 3 out of 4 were down-closed
                # Check if last candle closed above the high of the down-closed series
                if last_candle['close'] > prev_candles['high'].max():
                    print("  > Bullish CSD detected.")
                    return True
        elif bias == "BEARISH":
            # Check if previous candles were mostly up-closed
            up_closed_count = (prev_candles['close'] > prev_candles['open']).sum()
            if up_closed_count >= 3: # At least 3 out of 4 were up-closed
                # Check if last candle closed below the low of the up-closed series
                if last_candle['close'] < prev_candles['low'].min():
                    print("  > Bearish CSD detected.")
                    return True
        print("  > CSD not found.")
        return False

    def align_pdra(self, data, bias):
        print("Aligning PDRA...")
        # PDRA (Premium/Discount Repricing Array) involves identifying specific price levels
        # (e.g., order blocks, mitigation blocks, fair value gaps) that act as support/resistance.
        # Alignment means price is in a discount zone for bullish bias or premium zone for bearish bias.
        # This is a highly subjective and complex concept in ICT trading.
        # Placeholder for now.
        
        # Simplified: If bullish, assume price is in a discount if it's near a recent low.
        if bias == "BULLISH":
            if not data.empty and len(data) > 5:
                recent_low = data['low'].min()
                if data['close'].iloc[-1] < recent_low * 1.005: # Within 0.5% of recent low
                    print("  > PDRA aligned for bullish bias.")
                    return True
        # Simplified: If bearish, assume price is in a premium if it's near a recent high.
        elif bias == "BEARISH":
            if not data.empty and len(data) > 5:
                recent_high = data['high'].max()
                if data['close'].iloc[-1] > recent_high * 0.995: # Within 0.5% of recent high
                    print("  > PDRA aligned for bearish bias.")
                    return True
        print("  > PDRA not aligned.")
        return False
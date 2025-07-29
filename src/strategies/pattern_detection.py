import talib
import numpy as np
from scipy.signal import find_peaks
import pandas as pd

class PatternDetector:
    @staticmethod
    def detect_candlestick_patterns(df):
        """
        Detect all candlestick patterns using TA-Lib and custom logic
        Returns a dictionary with pattern names and their values (100 = bullish, -100 = bearish, 0 = no pattern)
        """
        # Initialize pattern dictionary
        patterns = {}
        
        # Single candle patterns
        patterns['hammer'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        patterns['inverted_hammer'] = talib.CDLINVERTEDHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        patterns['hanging_man'] = talib.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Close'])
        patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        patterns['doji'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        patterns['long_legged_doji'] = PatternDetector.detect_long_legged_doji(df)
        patterns['spinning_top'] = PatternDetector.detect_spinning_top(df)
        patterns['marubozu_bullish'] = talib.CDLMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])
        patterns['marubozu_bearish'] = -patterns['marubozu_bullish']  # Bearish is inverse of bullish
        
        # Multi-candle patterns
        patterns['engulfing'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
        patterns['harami'] = talib.CDLHARAMI(df['Open'], df['High'], df['Low'], df['Close'])
        patterns['piercing'] = talib.CDLPIERCING(df['Open'], df['High'], df['Low'], df['Close'])
        patterns['dark_cloud_cover'] = talib.CDLDARKCLOUDCOVER(df['Open'], df['High'], df['Low'], df['Close'])
        patterns['morning_star'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'], penetration=0)
        patterns['evening_star'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'], penetration=0)
        patterns['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(df['Open'], df['High'], df['Low'], df['Close'])
        patterns['three_black_crows'] = talib.CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Close'])
        patterns['tweezer_top'] = PatternDetector.detect_tweezer_top_bottom(df, 'top')
        patterns['tweezer_bottom'] = PatternDetector.detect_tweezer_top_bottom(df, 'bottom')
        
        return patterns
    
    @staticmethod
    def detect_spinning_top(df, body_threshold=0.3, wick_threshold=0.3):
        """Detect spinning top pattern (small body, long upper and lower wicks)"""
        results = np.zeros(len(df))
        for i in range(len(df)):
            body_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
            total_range = df['High'].iloc[i] - df['Low'].iloc[i]
            
            if total_range == 0:
                continue
                
            upper_wick = df['High'].iloc[i] - max(df['Open'].iloc[i], df['Close'].iloc[i])
            lower_wick = min(df['Open'].iloc[i], df['Close'].iloc[i]) - df['Low'].iloc[i]
            
            # Conditions for spinning top
            if (body_size / total_range < body_threshold and
                upper_wick / total_range > wick_threshold and
                lower_wick / total_range > wick_threshold):
                results[i] = 100 if df['Close'].iloc[i] > df['Open'].iloc[i] else -100
                
        return results
    
    @staticmethod
    def detect_long_legged_doji(df, body_threshold=0.05, wick_threshold=0.4):
        """Detect long-legged doji pattern (very small body, long upper and lower wicks)"""
        results = np.zeros(len(df))
        for i in range(len(df)):
            body_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
            total_range = df['High'].iloc[i] - df['Low'].iloc[i]
            
            if total_range == 0:
                continue
                
            upper_wick = df['High'].iloc[i] - max(df['Open'].iloc[i], df['Close'].iloc[i])
            lower_wick = min(df['Open'].iloc[i], df['Close'].iloc[i]) - df['Low'].iloc[i]
            
            # Conditions for long-legged doji
            if (body_size / total_range < body_threshold and
                upper_wick / total_range > wick_threshold and
                lower_wick / total_range > wick_threshold):
                results[i] = 100  # Neutral pattern but often reversal signal
                
        return results
    
    @staticmethod
    def detect_tweezer_top_bottom(df, pattern_type='top', tolerance=0.005):
        """Detect tweezer top/bottom pattern (two candles with matching highs/lows)"""
        results = np.zeros(len(df))
        for i in range(1, len(df)):
            if pattern_type == 'top':
                # Two consecutive candles with similar highs
                high1 = df['High'].iloc[i-1]
                high2 = df['High'].iloc[i]
                if abs(high1 - high2) / high1 <= tolerance:
                    results[i] = -100  # Bearish reversal
            else:  # bottom
                # Two consecutive candles with similar lows
                low1 = df['Low'].iloc[i-1]
                low2 = df['Low'].iloc[i]
                if abs(low1 - low2) / low1 <= tolerance:
                    results[i] = 100  # Bullish reversal
        return results
    
    @staticmethod
    def detect_chart_patterns(df, window=100, tolerance=0.02, min_peaks=3):
        """
        Detect complex chart patterns
        Returns a list of detected patterns for each data point
        """
        patterns = [set() for _ in range(len(df))]
        highs = df['High'].values
        lows = df['Low'].values
        
        # Find peaks and valleys
        peaks, _ = find_peaks(highs, distance=window//10, prominence=(tolerance * np.mean(highs)))
        valleys, _ = find_peaks(-lows, distance=window//10, prominence=(tolerance * np.mean(lows)))
        
        for i in range(window, len(df)):
            segment_highs = highs[i-window:i]
            segment_lows = lows[i-window:i]
            
            # Find peaks and valleys in current segment
            seg_peaks = [p for p in peaks if i-window <= p < i]
            seg_valleys = [v for v in valleys if i-window <= v < i]
            
            if len(seg_peaks) < min_peaks or len(seg_valleys) < min_peaks:
                continue
                
            # 1. Head and Shoulders (bearish) / Inverse Head and Shoulders (bullish)
            if PatternDetector.is_head_and_shoulders(segment_highs, seg_peaks, segment_lows, seg_valleys, tolerance):
                patterns[i].add('head_and_shoulders')
            if PatternDetector.is_inverse_head_and_shoulders(segment_lows, seg_valleys, segment_highs, seg_peaks, tolerance):
                patterns[i].add('inverse_head_and_shoulders')
                
            # 2. Double Top/Bottom
            if PatternDetector.is_double_top(segment_highs, seg_peaks, segment_lows, tolerance):
                patterns[i].add('double_top')
            if PatternDetector.is_double_bottom(segment_lows, seg_valleys, segment_highs, tolerance):
                patterns[i].add('double_bottom')
                
            # 3. Triple Top/Bottom
            if PatternDetector.is_triple_top(segment_highs, seg_peaks, segment_lows, tolerance):
                patterns[i].add('triple_top')
            if PatternDetector.is_triple_bottom(segment_lows, seg_valleys, segment_highs, tolerance):
                patterns[i].add('triple_bottom')
                
            # 4. Rounding Bottom (saucer bottom)
            if PatternDetector.is_rounding_bottom(segment_lows, tolerance):
                patterns[i].add('rounding_bottom')
                
            # 5. Triangle Patterns
            triangle_type = PatternDetector.is_triangle(segment_highs, segment_lows, seg_peaks, seg_valleys, tolerance)
            if triangle_type:
                patterns[i].add(triangle_type)
                
            # 6. Channel Patterns
            channel_type = PatternDetector.is_channel(segment_highs, segment_lows, seg_peaks, seg_valleys, tolerance)
            if channel_type:
                patterns[i].add(channel_type)
                
            # 7. Wedge Patterns
            wedge_type = PatternDetector.is_wedge(segment_highs, segment_lows, seg_peaks, seg_valleys, tolerance)
            if wedge_type:
                patterns[i].add(wedge_type)
                
            # 8. Flag/Pennant
            if PatternDetector.is_flag(segment_highs, segment_lows, tolerance):
                patterns[i].add('flag')
            if PatternDetector.is_pennant(segment_highs, segment_lows, seg_peaks, seg_valleys, tolerance):
                patterns[i].add('pennant')
                
            # 9. Rectangle
            if PatternDetector.is_rectangle(segment_highs, segment_lows, seg_peaks, seg_valleys, tolerance):
                patterns[i].add('rectangle')
                
        return patterns
    
    @staticmethod
    def is_head_and_shoulders(highs, peaks, lows, valleys, tolerance):
        """Detect head and shoulders pattern (bearish reversal)"""
        if len(peaks) < 3:
            return False
            
        # Identify potential left shoulder, head, right shoulder
        for i in range(2, len(peaks)):
            left_shoulder = peaks[i-2]
            head = peaks[i-1]
            right_shoulder = peaks[i]
            
            # Head should be higher than shoulders
            if (highs[head] > highs[left_shoulder] and 
                highs[head] > highs[right_shoulder]):
                # Shoulders should be similar in height
                if abs(highs[left_shoulder] - highs[right_shoulder]) / highs[left_shoulder] <= tolerance:
                    # Find neckline (lowest point between shoulders)
                    neckline_start = min(left_shoulder, head)
                    neckline_end = max(head, right_shoulder)
                    neckline = min(lows[neckline_start:neckline_end+1])
                    
                    # Confirm breakdown below neckline
                    current_low = lows[-1]
                    if current_low < neckline:
                        return True
        return False
    
    @staticmethod
    def is_inverse_head_and_shoulders(lows, valleys, highs, peaks, tolerance):
        """Detect inverse head and shoulders pattern (bullish reversal)"""
        if len(valleys) < 3:
            return False
            
        for i in range(2, len(valleys)):
            left_shoulder = valleys[i-2]
            head = valleys[i-1]
            right_shoulder = valleys[i]
            
            # Head should be lower than shoulders
            if (lows[head] < lows[left_shoulder] and 
                lows[head] < lows[right_shoulder]):
                # Shoulders should be similar in depth
                if abs(lows[left_shoulder] - lows[right_shoulder]) / lows[left_shoulder] <= tolerance:
                    # Find neckline (highest point between shoulders)
                    neckline_start = min(left_shoulder, head)
                    neckline_end = max(head, right_shoulder)
                    neckline = max(highs[neckline_start:neckline_end+1])
                    
                    # Confirm breakout above neckline
                    current_high = highs[-1]
                    if current_high > neckline:
                        return True
        return False
    
    @staticmethod
    def is_double_top(highs, peaks, lows, tolerance):
        """Detect double top pattern (bearish reversal)"""
        if len(peaks) < 2:
            return False
            
        last_peak = peaks[-1]
        prev_peak = peaks[-2]
        
        # Peaks should be similar in height
        if abs(highs[last_peak] - highs[prev_peak]) / highs[prev_peak] <= tolerance:
            # Find trough between peaks
            trough = min(lows[prev_peak:last_peak+1])
            # Confirm breakdown below trough
            if lows[-1] < trough:
                return True
        return False
    
    @staticmethod
    def is_double_bottom(lows, valleys, highs, tolerance):
        """Detect double bottom pattern (bullish reversal)"""
        if len(valleys) < 2:
            return False
            
        last_valley = valleys[-1]
        prev_valley = valleys[-2]
        
        # Valleys should be similar in depth
        if abs(lows[last_valley] - lows[prev_valley]) / lows[prev_valley] <= tolerance:
            # Find peak between valleys
            peak = max(highs[prev_valley:last_valley+1])
            # Confirm breakout above peak
            if highs[-1] > peak:
                return True
        return False
    
    @staticmethod
    def is_triple_top(highs, peaks, lows, tolerance):
        """Detect triple top pattern (bearish reversal)"""
        if len(peaks) < 3:
            return False
            
        p1 = peaks[-3]
        p2 = peaks[-2]
        p3 = peaks[-1]
        
        # All three peaks should be similar in height
        if (abs(highs[p1] - highs[p2]) / highs[p1] <= tolerance and
            abs(highs[p2] - highs[p3]) / highs[p2] <= tolerance):
            # Find highest trough between peaks
            trough1 = min(lows[p1:p2+1])
            trough2 = min(lows[p2:p3+1])
            confirmation_level = max(trough1, trough2)
            # Confirm breakdown below confirmation level
            if lows[-1] < confirmation_level:
                return True
        return False
    
    @staticmethod
    def is_triple_bottom(lows, valleys, highs, tolerance):
        """Detect triple bottom pattern (bullish reversal)"""
        if len(valleys) < 3:
            return False
            
        v1 = valleys[-3]
        v2 = valleys[-2]
        v3 = valleys[-1]
        
        # All three valleys should be similar in depth
        if (abs(lows[v1] - lows[v2]) / lows[v1] <= tolerance and
            abs(lows[v2] - lows[v3]) / lows[v2] <= tolerance):
            # Find lowest peak between valleys
            peak1 = max(highs[v1:v2+1])
            peak2 = max(highs[v2:v3+1])
            confirmation_level = min(peak1, peak2)
            # Confirm breakout above confirmation level
            if highs[-1] > confirmation_level:
                return True
        return False
    
    @staticmethod
    def is_rounding_bottom(lows, tolerance, min_duration=20):
        """Detect rounding bottom pattern (bullish reversal)"""
        if len(lows) < min_duration:
            return False
            
        # Fit a quadratic curve to the lows
        x = np.arange(len(lows))
        coeffs = np.polyfit(x, lows, 2)
        fitted_curve = np.polyval(coeffs, x)
        
        # Calculate R-squared to measure fit quality
        residuals = lows - fitted_curve
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((lows - np.mean(lows))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Check if curve is concave up (U-shaped)
        if coeffs[0] > 0 and r_squared > 0.7:
            # Check if recent lows are rising
            if lows[-1] > lows[-min_duration//2]:
                return True
        return False
    
    @staticmethod
    def is_triangle(highs, lows, peaks, valleys, tolerance):
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        if len(peaks) < 2 or len(valleys) < 2:
            return None
            
        # Get recent peaks and valleys
        recent_peaks = peaks[-2:]
        recent_valleys = valleys[-2:]
        
        # Calculate slopes
        high_slope = (highs[recent_peaks[1]] - highs[recent_peaks[0]]) / (recent_peaks[1] - recent_peaks[0])
        low_slope = (lows[recent_valleys[1]] - lows[recent_valleys[0]]) / (recent_valleys[1] - recent_valleys[0])
        
        # Ascending triangle (flat top, rising bottom)
        if abs(high_slope) < tolerance and low_slope > tolerance:
            return 'ascending_triangle'
        
        # Descending triangle (falling top, flat bottom)
        if high_slope < -tolerance and abs(low_slope) < tolerance:
            return 'descending_triangle'
        
        # Symmetrical triangle (converging trendlines)
        if abs(high_slope + low_slope) < tolerance and high_slope < 0 and low_slope > 0:
            return 'symmetrical_triangle'
        
        return None
    
    @staticmethod
    def is_channel(highs, lows, peaks, valleys, tolerance):
        """Detect channel patterns (ascending, descending, horizontal)"""
        if len(peaks) < 3 or len(valleys) < 3:
            return None
            
        # Get multiple peaks/valleys for better slope calculation
        selected_peaks = peaks[-3:]
        selected_valleys = valleys[-3:]
        
        # Calculate slopes using linear regression
        high_slope, _ = np.polyfit(selected_peaks, highs[selected_peaks], 1)
        low_slope, _ = np.polyfit(selected_valleys, lows[selected_valleys], 1)
        
        # Parallel slopes indicate a channel
        if abs(high_slope - low_slope) < tolerance:
            # Ascending channel
            if high_slope > tolerance:
                return 'ascending_channel'
            # Descending channel
            elif high_slope < -tolerance:
                return 'descending_channel'
            # Horizontal channel
            else:
                return 'horizontal_channel'
        
        return None
    
    @staticmethod
    def is_wedge(highs, lows, peaks, valleys, tolerance):
        """Detect wedge patterns (rising or falling)"""
        if len(peaks) < 2 or len(valleys) < 2:
            return None
            
        # Get recent peaks and valleys
        recent_peaks = peaks[-2:]
        recent_valleys = valleys[-2:]
        
        # Calculate slopes
        high_slope = (highs[recent_peaks[1]] - highs[recent_peaks[0]]) / (recent_peaks[1] - recent_peaks[0])
        low_slope = (lows[recent_valleys[1]] - lows[recent_valleys[0]]) / (recent_valleys[1] - recent_valleys[0])
        
        # Both trendlines moving in same direction
        if abs(high_slope - low_slope) < tolerance:
            return None
            
        # Rising wedge (both trendlines up but converging)
        if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
            return 'rising_wedge'
        
        # Falling wedge (both trendlines down but converging)
        if high_slope < 0 and low_slope < 0 and high_slope < low_slope:
            return 'falling_wedge'
        
        return None
    
    @staticmethod
    def is_flag(highs, lows, tolerance, min_duration=10):
        """Detect flag pattern (continuation)"""
        if len(highs) < min_duration + 5:
            return False
            
        # Look for sharp price movement (flagpole)
        pole_start = len(highs) - min_duration - 5
        pole_end = pole_start + 5
        pole_height = abs(highs[pole_end] - lows[pole_start])
        
        if pole_height == 0:
            return False
            
        # Look for consolidation (flag) with parallel trendlines
        flag_highs = highs[pole_end:]
        flag_lows = lows[pole_end:]
        
        high_slope, _ = np.polyfit(np.arange(len(flag_highs)), flag_highs, 1)
        low_slope, _ = np.polyfit(np.arange(len(flag_lows)), flag_lows, 1)
        
        # Should have parallel trendlines
        if abs(high_slope - low_slope) > tolerance:
            return False
            
        # Flag should retrace less than 50% of pole
        flag_retrace = abs(flag_highs[-1] - flag_lows[-1])
        if flag_retrace / pole_height > 0.5:
            return False
            
        return True
    
    @staticmethod
    def is_pennant(highs, lows, peaks, valleys, tolerance):
        """Detect pennant pattern (continuation)"""
        if len(peaks) < 2 or len(valleys) < 2:
            return False
            
        # Pennant is a small symmetrical triangle after a sharp move
        if PatternDetector.is_triangle(highs, lows, peaks, valleys, tolerance) == 'symmetrical_triangle':
            # Should be preceded by a sharp move
            triangle_height = max(highs[-5:]) - min(lows[-5:])
            prior_move = abs(highs[-10] - lows[-10])
            
            if prior_move > 3 * triangle_height:
                return True
        return False
    
    @staticmethod
    
    def is_rectangle(highs, lows, peaks, valleys, tolerance):
        """Detect rectangle pattern (consolidation)"""
        if len(peaks) < 3 or len(valleys) < 3:
            return False

        # Get the most recent 3 highs and 3 lows
        recent_highs = highs[peaks[-3:]]
        recent_lows = lows[valleys[-3:]]

        # Highs should be within tolerance of each other
        high_range = np.max(recent_highs) - np.min(recent_highs)
        high_mean = np.mean(recent_highs)
        if high_mean == 0 or (high_range / high_mean) > tolerance:
            return False

        # Lows should be within tolerance of each other
        low_range = np.max(recent_lows) - np.min(recent_lows)
        low_mean = np.mean(recent_lows)
        if low_mean == 0 or (low_range / low_mean) > tolerance:
            return False

        # Rectangle should be wide enough (not a flat line)
        if (high_mean - low_mean) / high_mean < tolerance * 2:
            return False

        # Confirm price is still within the rectangle
        if not (np.min(recent_lows) <= highs[-1] <= np.max(recent_highs)):
            return False
        if not (np.min(recent_lows) <= lows[-1] <= np.max(recent_highs)):
            return False

        return True
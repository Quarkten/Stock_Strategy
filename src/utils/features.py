import pandas as pd
import numpy as np


def add_indicators(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """
    Compute standard indicators used by IntradayStrategy.evaluate_entry and the backtester.
    Indicators:
      - ATR (configurable period)
      - EMA20, EMA50
      - MACD histogram (12,26,9)
      - Bollinger Bands (20,2)
      - Session VWAP (resets each day)
      - vol_ratio = ATR/close
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    atr_period = 14
    if cfg:
        atr_period = int(
            (cfg.get("strategy", {}) or {}).get("params", {}).get("atr_period", 14)
        )

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(window=atr_period, min_periods=1).mean()

    # EMAs
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()

    # MACD hist
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = macd - signal

    # Bollinger Bands
    sma20 = close.rolling(window=20, min_periods=1).mean()
    std20 = close.rolling(window=20, min_periods=1).std(ddof=0)
    out["bb_upper"] = sma20 + 2 * std20
    out["bb_lower"] = sma20 - 2 * std20

    # Session VWAP
    out["vwap"] = session_vwap(out)

    # Volatility regime proxy
    out["vol_ratio"] = (out["atr"] / out["close"]).clip(lower=0.0)

    return out


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Per-day VWAP using close*volume / volume. Resets each session.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    vwap_vals = []
    cur_num = 0.0
    cur_den = 0.0
    cur_day = None
    for idx, row in df.iterrows():
        d = idx.date()
        if d != cur_day:
            cur_day = d
            cur_num = 0.0
            cur_den = 0.0
        price = float(row["close"])
        vol = float(row.get("volume", 1.0))
        cur_num += price * vol
        cur_den += vol if vol > 0 else 1.0
        vwap_vals.append(cur_num / max(1e-9, cur_den))
    return pd.Series(vwap_vals, index=df.index)


def encode_time_of_day_minutes(df: pd.DataFrame) -> pd.Series:
    """
    Returns minutes from session open (09:30 ET). If tz-aware index is not ET,
    this function assumes the index has already been converted by caller.
    """
    from datetime import time

    if df is None or df.empty:
        return pd.Series(dtype=float)

    open_t = time(9, 30)
    minutes = []
    for ts in df.index:
        m = (ts.hour - open_t.hour) * 60 + (ts.minute - open_t.minute)
        minutes.append(float(m))
    return pd.Series(minutes, index=df.index)
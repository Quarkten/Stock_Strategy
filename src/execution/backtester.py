import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


@dataclass
class BacktesterConfig:
    symbol: str
    timeframe: str
    start: str
    end: str
    rth_only: bool = True
    csv_out: str = "data/trades_master.csv"
    tz: str = "America/New_York"
    # Indicator params (defaults; may be overridden by strategy/config)
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    n_stop_atr: float = 1.0
    n_tp_atr: float = 1.8
    breakeven_r: float = 1.0
    loss_streak_pause: int = 3
    fill_policy: str = "next_open"  # "next_open" or "bar_close"


@dataclass
class Position:
    side: str  # "LONG" or "SHORT"
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    size: int
    setup_name: str
    reasons: List[str] = field(default_factory=list)
    signal_scores: Dict[str, float] = field(default_factory=dict)
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    r_multiple: Optional[float] = None

    def initial_risk(self) -> float:
        return abs(self.entry_price - self.stop_price)


class Backtester:
    def __init__(
        self,
        strategy,
        data_fetcher,
        config: dict,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        rth_only: bool = True,
        csv_out: str = "data/trades_master.csv",
    ):
        self.strategy = strategy
        self.data_fetcher = data_fetcher
        self.config = config
        self.params = BacktesterConfig(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            rth_only=rth_only,
            csv_out=csv_out,
        )
        # Load strategy-related numeric params if present
        strat_params = (config or {}).get("strategy", {}).get("params", {})
        self.params.atr_period = int(strat_params.get("atr_period", self.params.atr_period))
        self.params.n_stop_atr = float(config.get("atr_multiplier", config.get("strategy", {}).get("params", {}).get("atr_multiplier", self.params.n_stop_atr)))
        # bb defaults
        self.params.bb_period = int(config.get("backtest", {}).get("bb_period", self.params.bb_period))
        self.params.bb_std = float(config.get("backtest", {}).get("bb_std", self.params.bb_std))
        # risk control defaults
        self.params.breakeven_r = float(config.get("backtest", {}).get("breakeven_r", self.params.breakeven_r))
        self.params.loss_streak_pause = int(config.get("backtest", {}).get("loss_streak_pause", self.params.loss_streak_pause))
        self.params.fill_policy = str(config.get("backtest", {}).get("fill_policy", self.params.fill_policy))

        # Finance params mirrored from strategy/config for convenience
        self.capital = float(config.get("capital", 100000))
        self.daily_max_loss_pct = float(config.get("daily_max_loss", 0.015))
        self.risk_per_trade_pct = float(config.get("risk_per_trade", 0.005))

        self.trades: List[Dict[str, Any]] = []
        self.tz = ZoneInfo(self.params.tz)

    # ------------- Indicators -------------
    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def _bbands(close: pd.Series, period: int, std_mult: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ma = close.rolling(window=period, min_periods=period).mean()
        sd = close.rolling(window=period, min_periods=period).std()
        upper = ma + std_mult * sd
        lower = ma - std_mult * sd
        return ma, upper, lower

    @staticmethod
    def _vwap(df: pd.DataFrame) -> pd.Series:
        pv = df["close"] * df.get("volume", 0)
        cum_pv = pv.cumsum()
        cum_vol = df.get("volume", pd.Series(0, index=df.index)).cumsum().replace(0, np.nan)
        return cum_pv / cum_vol

    @staticmethod
    def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd - macd_signal

    def _ensure_tz(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        if df.index.tz is None:
            df.index = df.index.tz_localize(self.tz)
        else:
            df.index = df.index.tz_convert(self.tz)
        return df

    def _filter_rth(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        local = df.copy()
        # Keep weekdays 0..4 and times between 09:30-16:00
        mask_weekday = local.index.weekday <= 4
        times = local.index.time
        mask_time = (pd.Series(times, index=local.index) >= datetime.strptime("09:30:00", "%H:%M:%S").time()) & (
            pd.Series(times, index=local.index) <= datetime.strptime("16:00:00", "%H:%M:%S").time()
        )
        return local[mask_weekday & mask_time]

    def _preprocess_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.empty:
            return df
        df["ema20"] = self._ema(df["close"], span=20)
        df["atr"] = self._atr(df, period=self.params.atr_period)
        bb_mid, bb_u, bb_l = self._bbands(df["close"], period=self.params.bb_period, std_mult=self.params.bb_std)
        df["bb_mid"], df["bb_upper"], df["bb_lower"] = bb_mid, bb_u, bb_l
        df["vwap"] = self._vwap(df)
        df["macd_hist"] = self._macd_hist(df["close"])
        # Warmup drop
        warmup = max(20, self.params.atr_period, self.params.bb_period)
        return df.iloc[warmup:].dropna()

    def _load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data: {self.params.symbol} {self.params.timeframe} {self.params.start}..{self.params.end}")
        # Try providers with explicit start/end if supported; our wrapper passes kwargs through
        df = self.data_fetcher.fetch_data(
            self.params.symbol,
            self.params.timeframe,
            provider=self.config.get("data_source"),
            start=self.params.start,
            end=self.params.end,
        )
        if df is None or df.empty:
            logger.warning("Data fetch returned empty DataFrame.")
            return pd.DataFrame()
        df = self._ensure_tz(df)
        # Local slice by dates (in tz)
        try:
            start_ts = pd.Timestamp(self.params.start).tz_localize(self.tz) if pd.Timestamp(self.params.start).tz is None else pd.Timestamp(self.params.start).tz_convert(self.tz)
            end_ts = pd.Timestamp(self.params.end).tz_localize(self.tz) if pd.Timestamp(self.params.end).tz is None else pd.Timestamp(self.params.end).tz_convert(self.tz)
            df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        except Exception:
            # Fallback naive filter by date-only string
            df = df[(df.index.strftime("%Y%m%d") >= self.params.start) & (df.index.strftime("%Y%m%d") <= self.params.end)]
        if self.params.rth_only:
            df = self._filter_rth(df)
        return self._preprocess_indicators(df)

    # ------------- Main run -------------
    def run(self):
        df = self._load_data()
        if df.empty:
            logger.error("No data to backtest.")
            return

        # Day-wise grouping by local date
        day_index = df.index.tz_convert(self.tz) if df.index.tz is not None else df.index
        df = df.copy()
        df["__mkt_date__"] = day_index.date

        all_days = sorted(df["__mkt_date__"].unique())
        logger.info(f"Backtest days: {len(all_days)}")

        master_rows: List[Dict[str, Any]] = []

        for day in all_days:
            day_slice = df[df["__mkt_date__"] == day].copy()
            if day_slice.empty:
                continue

            # Compute bias once per day using prior 1d/1h if available via DataFetcher
            try:
                prior_daily = self.data_fetcher.fetch_data(self.params.symbol, "1day", limit=100, provider=self.config.get("data_source"))
                prior_1h = self.data_fetcher.fetch_data(self.params.symbol, "1h", limit=200, provider=self.config.get("data_source"))
            except Exception:
                prior_daily, prior_1h = pd.DataFrame(), pd.DataFrame()

            bias_result = self.strategy.evaluate_daily_bias_tjr_style(prior_daily, None, None, prior_1h)
            # Map to weight if legacy string
            if isinstance(bias_result, str):
                bias_label = bias_result
                bias_weight = 1.0 if bias_label in ("BULLISH", "BEARISH") else 0.5
                bias_reasons = [f"legacy_bias:{bias_label}"]
            elif isinstance(bias_result, dict):
                bias_label = bias_result.get("bias_label", "UNCERTAIN")
                bias_weight = float(bias_result.get("bias_weight", 0.5))
                bias_reasons = bias_result.get("reasons", [])
            else:
                bias_label, bias_weight, bias_reasons = "UNCERTAIN", 0.5, []

            day_pnl = 0.0
            consecutive_losses = 0
            trading_halted = False
            open_position: Optional[Position] = None

            for i in range(len(day_slice)):
                bar_time = day_slice.index[i]
                bar = day_slice.iloc[i]

                # Manage open position exits by stop/target or EOD
                if open_position is not None:
                    hit_stop = (bar.low <= open_position.stop_price) if open_position.side == "LONG" else (bar.high >= open_position.stop_price)
                    hit_target = (bar.high >= open_position.target_price) if open_position.side == "LONG" else (bar.low <= open_position.target_price)

                    # Breakeven move
                    init_r = open_position.initial_risk()
                    if init_r > 0:
                        if open_position.side == "LONG":
                            unreal = bar.high - open_position.entry_price
                        else:
                            unreal = open_position.entry_price - bar.low
                        if unreal >= self.params.breakeven_r * init_r:
                            # Move stop to entry if improves risk
                            if open_position.side == "LONG" and open_position.stop_price < open_position.entry_price:
                                open_position.stop_price = open_position.entry_price
                            if open_position.side == "SHORT" and open_position.stop_price > open_position.entry_price:
                                open_position.stop_price = open_position.entry_price

                    exit_now = False
                    exit_price = None
                    if hit_stop:
                        exit_now = True
                        # Fill policy
                        exit_price = open_position.stop_price if self.params.fill_policy == "bar_close" else float(day_slice.iloc[i + 1].open) if i + 1 < len(day_slice) else float(bar.close)
                    elif hit_target:
                        exit_now = True
                        exit_price = open_position.target_price if self.params.fill_policy == "bar_close" else float(day_slice.iloc[i + 1].open) if i + 1 < len(day_slice) else float(bar.close)

                    # Exit on EOD at bar close
                    is_last_bar = i == len(day_slice) - 1
                    if not exit_now and is_last_bar:
                        exit_now = True
                        exit_price = float(bar.close)

                    if exit_now:
                        open_position.exit_time = bar_time if is_last_bar or self.params.fill_policy == "bar_close" else (day_slice.index[i + 1] if i + 1 < len(day_slice) else bar_time)
                        open_position.exit_price = float(exit_price)
                        # PnL
                        if open_position.side == "LONG":
                            pnl = (open_position.exit_price - open_position.entry_price) * open_position.size
                        else:
                            pnl = (open_position.entry_price - open_position.exit_price) * open_position.size
                        r_mult = pnl / (open_position.initial_risk() * open_position.size) if open_position.initial_risk() > 0 else np.nan
                        open_position.pnl = pnl
                        open_position.r_multiple = r_mult
                        day_pnl += pnl
                        self.trades.append(self._trade_row_from_position(open_position, self.params.symbol, bias_weight))

                        if pnl < 0:
                            consecutive_losses += 1
                            if consecutive_losses >= self.params.loss_streak_pause:
                                trading_halted = True
                        else:
                            consecutive_losses = 0

                        open_position = None

                # Daily risk halt check (pre-trade gate)
                if trading_halted:
                    continue
                if day_pnl <= -(self.daily_max_loss_pct * self.capital):
                    trading_halted = True
                    continue

                # If flat, look for entries
                if open_position is None and not trading_halted:
                    state = {
                        "bias_label": bias_label,
                        "bias_weight": bias_weight,
                        "atr": float(bar.atr) if not np.isnan(bar.atr) else None,
                        "bb_upper": float(bar.bb_upper) if not np.isnan(bar.bb_upper) else None,
                        "bb_lower": float(bar.bb_lower) if not np.isnan(bar.bb_lower) else None,
                        "vwap": float(bar.vwap) if not np.isnan(bar.vwap) else None,
                        "ema20": float(bar.ema20) if not np.isnan(bar.ema20) else None,
                        "macd_hist": float(bar.macd_hist) if not np.isnan(bar.macd_hist) else None,
                        "config": {
                            "n_stop_atr": self.params.n_stop_atr,
                            "n_tp_atr": self.params.n_tp_atr,
                        },
                    }
                    # Strategy evaluate_entry expected to consume df context. Provide slice up to i for context.
                    context_df = day_slice.iloc[: i + 1]
                    signal = getattr(self.strategy, "evaluate_entry", None)
                    proposed = signal(context_df, state) if callable(signal) else None

                    if proposed:
                        entry_price = float(proposed.get("entry_price", float(bar.close)))
                        stop_price = float(proposed.get("stop_price", entry_price - (proposed.get("atr", bar.atr) or 1.0)))
                        target_price = float(proposed.get("target_price", entry_price + (proposed.get("atr", bar.atr) or 1.8)))
                        size_mult = float(proposed.get("size_multiplier", bias_weight))
                        # Sizing
                        per_share_risk = abs(entry_price - stop_price)
                        risk_amount = self.capital * self.risk_per_trade_pct * max(0.1, min(1.0, size_mult))
                        size = int(np.floor(risk_amount / per_share_risk)) if per_share_risk > 0 else 0
                        if size <= 0:
                            continue

                        # Strategy risk gate
                        if not self.strategy.check_risk_before_trade(entry_price, stop_price):
                            continue

                        # Place position, fill at next bar open if configured
                        if self.params.fill_policy == "next_open" and i + 1 < len(day_slice):
                            fill_price = float(day_slice.iloc[i + 1].open)
                            fill_time = day_slice.index[i + 1]
                        else:
                            fill_price = entry_price
                            fill_time = bar_time

                        open_position = Position(
                            side="LONG" if proposed.get("side", "LONG").upper() == "LONG" else "SHORT",
                            entry_time=fill_time,
                            entry_price=fill_price,
                            stop_price=stop_price,
                            target_price=target_price,
                            size=size,
                            setup_name=proposed.get("setup_name", "unknown"),
                            reasons=proposed.get("reasons", []),
                            signal_scores=proposed.get("signal_scores", {}),
                        )

            # End for bars
            # If any position remains, it was force-closed above at EOD

        # After all days
        self._finalize(csv_out=self.params.csv_out)

    def _trade_row_from_position(self, pos: Position, symbol: str, bias_weight: float) -> Dict[str, Any]:
        return {
            "date": pos.entry_time.date() if isinstance(pos.entry_time, pd.Timestamp) else None,
            "time": pos.entry_time.isoformat() if isinstance(pos.entry_time, pd.Timestamp) else None,
            "symbol": symbol,
            "setup": pos.setup_name,
            "side": pos.side,
            "entry": pos.entry_price,
            "stop": pos.stop_price,
            "target": pos.target_price,
            "exit": pos.exit_price,
            "pnl": pos.pnl,
            "r_multiple": pos.r_multiple,
            "size": pos.size,
            "bias_weight": bias_weight,
            "reasons": ";".join(pos.reasons or []),
            "momentum_score": pos.signal_scores.get("momentum", np.nan) if pos.signal_scores else np.nan,
            "mean_rev_score": pos.signal_scores.get("mean_rev", np.nan) if pos.signal_scores else np.nan,
            "vol_cap": pos.signal_scores.get("vol_cap", np.nan) if pos.signal_scores else np.nan,
            "pullback_score": pos.signal_scores.get("pullback", np.nan) if pos.signal_scores else np.nan,
            "exit_time": pos.exit_time.isoformat() if isinstance(pos.exit_time, pd.Timestamp) else None,
        }

    def _finalize(self, csv_out: Optional[str] = None):
        if not self.trades:
            logger.info("No trades generated in backtest.")
            return
        trades_df = pd.DataFrame(self.trades)
        # Summary
        total_pnl = float(trades_df["pnl"].sum())
        wins = trades_df[trades_df["pnl"] > 0]
        win_rate = float(len(wins)) / float(len(trades_df)) if len(trades_df) > 0 else 0.0
        avg_r = float(trades_df["r_multiple"].mean()) if "r_multiple" in trades_df.columns else np.nan
        logger.info("--- Backtest Summary ---")
        logger.info(f"Trades: {len(trades_df)} | WinRate: {win_rate:.2%} | Total PnL: {total_pnl:.2f} | Avg R: {avg_r:.2f}")

        # Save
        try:
            # Append or create
            if csv_out:
                mode = "a" if pd.io.common.file_exists(csv_out) else "w"
                header = not pd.io.common.file_exists(csv_out)
                trades_df.to_csv(csv_out, index=False, mode=mode, header=header)
                logger.info(f"Master trades CSV written to {csv_out}")
        except Exception as e:
            logger.error(f"Error writing master CSV: {e}")
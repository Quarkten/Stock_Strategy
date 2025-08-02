import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
import random
import os
from tqdm import tqdm

# This backtester is designed to match run.py expectations and integrate with IntradayStrategy.evaluate_entry
# It supports:
# - Data loading via DataFetcher (already in project)
# - Feature engineering (minimal inline; can be moved to src/utils/features.py later)
# - Day-by-day iteration with RTH filter
# - ATR-based stops/targets with breakeven, trailing placeholder
# - Simple slippage/fees/latency model
# - Daily max loss halt
# - Reproducibility via seed
# - CSV logging of trades
#
# Key interfaces:
#   bt = Backtester(strategy, data_fetcher, config, symbol, timeframe, start, end, rth_only, csv_out)
#   bt.run()
#
# Assumptions:
# - DataFetcher exposes fetch_data(symbol, timeframe, start=None, end=None, limit=None) returning DataFrame with columns:
#   ['timestamp','open','high','low','close','volume'] or index as datetime
# - IntradayStrategy.evaluate_entry expects a df with indicator columns present on last row if used.


@dataclass
class CostModel:
    slippage_ticks: float = 0.0         # price units per trade side
    fee_per_share: float = 0.0          # fee per share
    spread_widen_prob: float = 0.0      # probability of spread widening event
    spread_widen_ticks: float = 0.0     # additional slippage during widening
    latency_bars: int = 0               # bars between decision and execution
    seed: int = 42

    def apply_slippage(self, price: float, side: str, rng: random.Random) -> float:
        slip = self.slippage_ticks
        # occasional spread widening
        if rng.random() < self.spread_widen_prob:
            slip += self.spread_widen_ticks
        if side.upper() == "BUY" or side.upper() == "LONG":
            return price + slip
        return price - slip


@dataclass
class RiskLimits:
    capital: float
    risk_per_trade_pct: float
    daily_max_loss_pct: float
    max_leverage: float = 1.0
    max_position_risk_multiple: float = 1.0  # max initial R per trade


@dataclass
class TradeRecord:
    entry_time: datetime
    exit_time: datetime
    side: str
    entry: float
    exit: float
    stop: float
    target: float
    shares: float
    pnl: float
    r_multiple: float
    mae_r: float
    mfe_r: float
    duration_bars: int
    setup_name: str
    reasons: str
    regime: str


from src.data.database import DatabaseManager

class Backtester:
    def __init__(
        self,
        strategy,
        data_fetcher,
        config: Dict[str, Any],
        symbol: str,
        timeframe: str,
        start: Optional[str],
        end: Optional[str],
        rth_only: bool = True,
        csv_out: Optional[str] = None,
        cost_model: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        db_path: str = "data/database/trading_data.sqlite3",
    ):
        self.logger = logging.getLogger(__name__)
        self.strategy = strategy
        self.data_fetcher = data_fetcher
        self.db_manager = DatabaseManager(db_path)
        self.config = config or {}
        self.symbol = symbol
        self.timeframe = timeframe
        self.start = start
        self.end = end
        self.rth_only = rth_only
        self.csv_out = csv_out or "data/trades_master.csv"
        self.rng = random.Random(seed)

        # Risk
        capital = float(self.config.get("capital", 100000))
        risk_per_trade = float(self.config.get("risk_per_trade", 0.005))
        daily_max_loss = float(self.config.get("daily_max_loss", 0.015))
        self.risk = RiskLimits(
            capital=capital,
            risk_per_trade_pct=risk_per_trade,
            daily_max_loss_pct=daily_max_loss,
        )

        # Cost model
        cm = cost_model or {}
        self.cost = CostModel(
            slippage_ticks=float(cm.get("slippage_ticks", 0.01)),
            fee_per_share=float(cm.get("fee_per_share", 0.0)),
            spread_widen_prob=float(cm.get("spread_widen_prob", 0.0)),
            spread_widen_ticks=float(cm.get("spread_widen_ticks", 0.02)),
            latency_bars=int(cm.get("latency_bars", 0)),
            seed=seed,
        )

        # Runtime state
        self.trades: List[TradeRecord] = []
        self.daily_pnl = 0.0
        self.current_day = None

    def run(self):
        df = self._load_data()
        if df is None or df.empty:
            self.logger.warning("No data loaded for backtest.")
            return
        df = self._engineer_features(df)

        # Precompute daily bias using daily and 1h data where possible
        daily_bias_map = self._compute_daily_bias()

        # Iterate day-by-day
        for day, day_df in tqdm(df.groupby(df.index.date), desc="Backtesting"):
            # reset daily loss
            self.daily_pnl = 0.0
            self.current_day = day

            # optionally RTH filter 09:30–16:00
            if self.rth_only:
                day_df = self._filter_rth(day_df)
                if day_df.empty:
                    continue

            # Skip days with too few bars
            if len(day_df) < 10:
                continue

            open_positions: List[Dict[str, Any]] = []

            # Build state scaffold
            bias = daily_bias_map.get(day, {"bias_label": "UNCERTAIN", "bias_weight": 0.5})
            state: Dict[str, Any] = {
                "bias_label": bias.get("bias_label", "UNCERTAIN"),
                "bias_weight": float(bias.get("bias_weight", 0.5)),
                "config": {
                    "n_stop_atr": getattr(self.strategy, "n_stop_atr", 1.0),
                    "n_tp_atr": getattr(self.strategy, "n_tp_atr", 1.8),
                },
            }

            # Iterate bars
            for i in range(len(day_df)):
                # Daily max loss halt
                if self.daily_pnl <= -(self.risk.daily_max_loss_pct * self.risk.capital):
                    self.logger.info(f"{day} halted due to daily max loss.")
                    break

                window_df = day_df.iloc[: i + 1]
                recent = window_df.iloc[-1]

                # Manage existing positions first
                if open_positions:
                    open_positions = self._manage_positions(window_df, open_positions)

                # Evaluate new entry
                entry_sig = None
                try:
                    state["atr"] = float(recent.get("atr", np.nan))
                    entry_sig = self.strategy.evaluate_entry(window_df, state)
                except Exception as e:
                    self.logger.exception("Error in evaluate_entry: %s", e)

                if entry_sig:
                    # Convert to execution side and compute risk
                    side = "LONG" if entry_sig["side"].upper() == "LONG" else "SHORT"
                    entry_price = float(entry_sig["entry_price"])
                    stop_price = float(entry_sig["stop_price"])
                    target_price = float(entry_sig["target_price"])
                    size_mult = float(entry_sig.get("size_multiplier", 1.0))
                    atr = float(entry_sig.get("atr", recent.get("atr", np.nan)))

                    risk_per_share = abs(entry_price - stop_price)
                    if risk_per_share <= 0 or np.isnan(risk_per_share):
                        continue

                    nominal_risk = self.risk.capital * self.risk.risk_per_trade_pct
                    shares = max(0.0, (nominal_risk / risk_per_share) * size_mult)

                    # Safety: cap position by risk_multiple and leverage (simple: skip leverage enforcement here)
                    if (risk_per_share * shares) > (self.risk.max_position_risk_multiple * nominal_risk):
                        # scale down
                        scale = (self.risk.max_position_risk_multiple * nominal_risk) / max(1e-9, (risk_per_share * shares))
                        shares *= scale

                    # Execution latency: shift execution bar
                    exec_idx = i + max(0, self.cost.latency_bars)
                    if exec_idx >= len(day_df):
                        continue
                    exec_bar = day_df.iloc[exec_idx]
                    exec_price_raw = float(exec_bar["open"])  # assume next open
                    exec_price = self.cost.apply_slippage(exec_price_raw, "BUY" if side == "LONG" else "SELL", self.rng)

                    # Fees
                    entry_fee = shares * self.cost.fee_per_share

                    position = {
                        "id": f"{self.symbol}_{day}_{exec_idx}_{len(self.trades)+len(open_positions)}",
                        "side": side,
                        "entry_price": exec_price,
                        "stop_loss": stop_price,
                        "target": target_price,
                        "shares": shares,
                        "open_time": exec_bar.name.to_pydatetime() if hasattr(exec_bar, "name") else datetime.combine(day, time(9, 30)),
                        "breakeven_r": getattr(self.strategy, "breakeven_r", 1.0),
                        "atr": atr,
                        "setup_name": entry_sig.get("setup_name", ""),
                        "reasons": ";".join(entry_sig.get("reasons", [])),
                        "mae": 0.0,
                        "mfe": 0.0,
                        "initial_risk": risk_per_share,
                        "entry_fee": entry_fee,
                    }
                    open_positions.append(position)

                # If positions exist, check exits on this bar close
                if open_positions:
                    open_positions = self._check_exits(day_df.iloc[: i + 1], open_positions)

            # Force close any positions at day end at last close
            if open_positions:
                last_bar = day_df.iloc[-1]
                for pos in open_positions:
                    exit_price = float(last_bar["close"])
                    exit_fee = pos["shares"] * self.cost.fee_per_share
                    pnl = (exit_price - pos["entry_price"]) * pos["shares"] if pos["side"] == "LONG" else (pos["entry_price"] - exit_price) * pos["shares"]
                    pnl -= (pos.get("entry_fee", 0.0) + exit_fee)
                    r_mult = pnl / max(1e-9, (pos["initial_risk"] * pos["shares"]))
                    self.daily_pnl += pnl
                    self._record_trade(
                        entry_time=pos["open_time"],
                        exit_time=last_bar.name.to_pydatetime() if hasattr(last_bar, "name") else datetime.combine(day, time(16, 0)),
                        side=pos["side"],
                        entry=pos["entry_price"],
                        stop=pos["stop_loss"],
                        target=pos["target"],
                        exit=exit_price,
                        shares=pos["shares"],
                        pnl=pnl,
                        r_mult=r_mult,
                        mae_r=(pos["mae"] / max(1e-9, pos["initial_risk"])),
                        mfe_r=(pos["mfe"] / max(1e-9, pos["initial_risk"])),
                        duration_bars=len(day_df) - day_df.index.get_indexer([pos["open_time"]])[0] if hasattr(day_df.index, "get_indexer") else 0,
                        setup_name=pos.get("setup_name", ""),
                        reasons=pos.get("reasons", ""),
                        regime=self._regime_from_features(last_bar),
                    )
                open_positions = []

        # Write CSV if requested
        self._write_csv()

        # Print summary
        self._summary()

    # ---------------------------
    # Data and features
    # ---------------------------
    def _load_data(self) -> Optional[pd.DataFrame]:
        try:
            df = self.data_fetcher.fetch_data(self.symbol, self._normalize_timeframe(self.timeframe), start=self.start, end=self.end)
            if df is None or df.empty:
                return None
            # Ensure datetime index
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(self._tz())
                df = df.set_index("timestamp").sort_index()
            else:
                if not isinstance(df.index, pd.DatetimeIndex):
                    # try parse
                    df.index = pd.to_datetime(df.index, utc=True).tz_convert(self._tz())
                else:
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC").tz_convert(self._tz())
            # Local slice in case provider ignores start/end
            if self.start:
                df = df[df.index.date >= datetime.strptime(self.start, "%Y%m%d").date()]
            if self.end:
                df = df[df.index.date <= datetime.strptime(self.end, "%Y%m%d").date()]
            return df
        except Exception as e:
            self.logger.exception("Failed to load data: %s", e)
            return None

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # ATR
        period = int(self.config.get("strategy", {}).get("params", {}).get("atr_period", 14))
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        close = out["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        out["atr"] = tr.rolling(window=period, min_periods=1).mean()

        # EMA20
        out["ema20"] = close.ewm(span=20, adjust=False).mean()

        # EMA50
        out["ema50"] = close.ewm(span=50, adjust=False).mean()

        # MACD hist (12,26,9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        out["macd_hist"] = macd - signal

        # Bollinger bands 20,2
        sma20 = close.rolling(window=20, min_periods=1).mean()
        std20 = close.rolling(window=20, min_periods=1).std(ddof=0)
        out["bb_upper"] = sma20 + 2 * std20
        out["bb_lower"] = sma20 - 2 * std20

        # VWAP (session reset per day)
        out["vwap"] = self._session_vwap(out)

        # Simple volatility regime
        out["vol_ratio"] = (out["atr"] / out["close"]).clip(lower=0.0)

        return out

    def _session_vwap(self, df: pd.DataFrame) -> pd.Series:
        # Reset per day; vwap = cumsum(price*volume)/cumsum(volume)
        vwap = []
        cur_num = 0.0
        cur_den = 0.0
        cur_day = None
        for idx, row in df.iterrows():
            d = idx.date()
            if cur_day != d:
                cur_day = d
                cur_num = 0.0
                cur_den = 0.0
            price = float(row["close"])
            vol = float(row.get("volume", 1.0))
            cur_num += price * vol
            cur_den += vol if vol > 0 else 1.0
            vwap.append(cur_num / max(1e-9, cur_den))
        return pd.Series(vwap, index=df.index)

    def _filter_rth(self, day_df: pd.DataFrame) -> pd.DataFrame:
        # America/New_York session 09:30–16:00
        start_t = time(9, 30)
        end_t = time(16, 0)
        return day_df[(day_df.index.time >= start_t) & (day_df.index.time <= end_t)]

    def _compute_daily_bias(self) -> Dict[datetime.date, Dict[str, Any]]:
        # Fetch daily and 1h for SPY only per current strategy
        try:
            spy_daily = self.data_fetcher.fetch_data(self.symbol, "1day", limit=100)
            spy_1h = self.data_fetcher.fetch_data(self.symbol, "1hour", limit=100)
            # Normalize timestamp
            for d in (spy_daily, spy_1h):
                if d is not None and not d.empty:
                    if "timestamp" in d.columns:
                        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(self._tz())
                        d.set_index("timestamp", inplace=True)
                    else:
                        if d.index.tz is None:
                            d.index = d.index.tz_localize("UTC").tz_convert(self._tz())
            result = {}
            # Map bias per date using available daily close dates
            if spy_daily is None or spy_daily.empty:
                return result
            for idx in spy_daily.index:
                bias = self.strategy.evaluate_daily_bias_tjr_style(spy_daily.loc[:idx], None, None, spy_1h)
                result[idx.date()] = bias
            return result
        except Exception as e:
            self.logger.exception("Failed to compute daily bias: %s", e)
            return {}

    # ---------------------------
    # Position management and exits
    # ---------------------------
    def _manage_positions(self, window_df: pd.DataFrame, positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Delegate to strategy.manage_positions for breakeven/trailing logic if available
        try:
            managed = self.strategy.manage_positions(window_df, positions, risk_per_trade=self.config.get("risk_per_trade", 0.005))
            return managed if managed is not None else positions
        except Exception:
            # Fallback: simple breakeven move when +1R reached
            recent = window_df.iloc[-1]
            for pos in positions:
                if pos["side"] == "LONG":
                    move = float(recent["high"]) - pos["entry_price"]
                    if move >= pos["initial_risk"] and pos["stop_loss"] < pos["entry_price"]:
                        pos["stop_loss"] = pos["entry_price"]
                else:
                    move = pos["entry_price"] - float(recent["low"])
                    if move >= pos["initial_risk"] and pos["stop_loss"] > pos["entry_price"]:
                        pos["stop_loss"] = pos["entry_price"]
            return positions

    def _check_exits(self, window_df: pd.DataFrame, positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        recent = window_df.iloc[-1]
        exits: List[Tuple[int, Dict[str, Any], float, float]] = []
        # Update MAE/MFE
        for idx, pos in enumerate(positions):
            # track MAE/MFE in price distance
            if pos["side"] == "LONG":
                adverse = max(0.0, pos["entry_price"] - float(recent["low"]))
                favorable = max(0.0, float(recent["high"]) - pos["entry_price"])
            else:
                adverse = max(0.0, float(recent["high"]) - pos["entry_price"])
                favorable = max(0.0, pos["entry_price"] - float(recent["low"]))
            pos["mae"] = max(pos.get("mae", 0.0), adverse)
            pos["mfe"] = max(pos.get("mfe", 0.0), favorable)

            # Stop/Target checks at bar extremes
            hit_stop = False
            hit_target = False
            if pos["side"] == "LONG":
                if float(recent["low"]) <= pos["stop_loss"]:
                    hit_stop = True
                if float(recent["high"]) >= pos["target"]:
                    hit_target = True
            else:
                if float(recent["high"]) >= pos["stop_loss"]:
                    hit_stop = True
                if float(recent["low"]) <= pos["target"]:
                    hit_target = True

            if hit_stop or hit_target:
                # Assume exit at stop/target price with slippage and fee
                exit_side = "SELL" if pos["side"] == "LONG" else "BUY"
                px = pos["stop_loss"] if hit_stop else pos["target"]
                px_exec = self.cost.apply_slippage(px, exit_side, self.rng)
                exit_fee = pos["shares"] * self.cost.fee_per_share
                pnl = (px_exec - pos["entry_price"]) * pos["shares"] if pos["side"] == "LONG" else (pos["entry_price"] - px_exec) * pos["shares"]
                pnl -= (pos.get("entry_fee", 0.0) + exit_fee)
                r_mult = pnl / max(1e-9, (pos["initial_risk"] * pos["shares"]))
                self.daily_pnl += pnl
                exits.append((idx, pos, px_exec, pnl))
                self._record_trade(
                    entry_time=pos["open_time"],
                    exit_time=recent.name.to_pydatetime() if hasattr(recent, "name") else datetime.combine(self.current_day, time(16, 0)),
                    side=pos["side"],
                    entry=pos["entry_price"],
                    stop=pos["stop_loss"],
                    target=pos["target"],
                    exit=px_exec,
                    shares=pos["shares"],
                    pnl=pnl,
                    r_mult=r_mult,
                    mae_r=(pos["mae"] / max(1e-9, pos["initial_risk"])),
                    mfe_r=(pos["mfe"] / max(1e-9, pos["initial_risk"])),
                    duration_bars=len(window_df),
                    setup_name=pos.get("setup_name", ""),
                    reasons=pos.get("reasons", ""),
                    regime=self._regime_from_features(recent),
                )

        # Remove exited positions
        if exits:
            to_remove = sorted([i for i, _, _, _ in exits], reverse=True)
            for i in to_remove:
                positions.pop(i)
        return positions

    # ---------------------------
    # Utilities
    # ---------------------------
    def _record_trade(
        self,
        entry_time: datetime,
        exit_time: datetime,
        side: str,
        entry: float,
        stop: float,
        target: float,
        exit: float,
        shares: float,
        pnl: float,
        r_mult: float,
        mae_r: float,
        mfe_r: float,
        duration_bars: int,
        setup_name: str,
        reasons: str,
        regime: str,
    ):
        rec = TradeRecord(
            entry_time=entry_time,
            exit_time=exit_time,
            side=side,
            entry=entry,
            stop=stop,
            target=target,
            exit=exit,
            shares=shares,
            pnl=pnl,
            r_multiple=r_mult,
            mae_r=mae_r,
            mfe_r=mfe_r,
            duration_bars=duration_bars,
            setup_name=setup_name,
            reasons=reasons,
            regime=regime,
        )
        self.trades.append(rec)
        self.db_manager.log_trade_record(rec)

    def _write_csv(self):
        if not self.trades:
            self.logger.info("No trades to write.")
            return
        rows = []
        for t in self.trades:
            rows.append(
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "side": t.side,
                    "entry": t.entry,
                    "stop": t.stop,
                    "target": t.target,
                    "exit": t.exit,
                    "shares": t.shares,
                    "pnl": t.pnl,
                    "r_multiple": t.r_multiple,
                    "mae_r": t.mae_r,
                    "mfe_r": t.mfe_r,
                    "duration_bars": t.duration_bars,
                    "setup_name": t.setup_name,
                    "reasons": t.reasons,
                    "regime": t.regime,
                }
            )
        trades_df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(self.csv_out), exist_ok=True)
        if os.path.exists(self.csv_out):
            # append
            trades_df.to_csv(self.csv_out, mode="a", index=False, header=False)
        else:
            trades_df.to_csv(self.csv_out, index=False)
        self.logger.info(f"Wrote trades to {self.csv_out}")

    def _summary(self):
        if not self.trades:
            self.logger.info("No trades executed.")
            return
        df = pd.DataFrame([t.__dict__ for t in self.trades])
        trades = len(df)
        win_rate = float((df["pnl"] > 0).mean()) if trades > 0 else 0.0
        total_pnl = float(df["pnl"].sum())
        avg_r = float(df["r_multiple"].mean())
        self.logger.info(f"Summary - Trades: {trades} | WinRate: {win_rate*100:.2f}% | Total PnL: {total_pnl:.2f} | Avg R: {avg_r:.2f}")

    def _regime_from_features(self, row: pd.Series) -> str:
        vr = float(row.get("vol_ratio", 0.0))
        if vr >= 0.01:
            return "HIGH_VOL"
        if vr >= 0.005:
            return "MED_VOL"
        return "LOW_VOL"

    def _tz(self):
        # map timezone string from config if exists
        tz = self.config.get("timezone", "America/New_York")
        return tz

    def _normalize_timeframe(self, tf: str) -> str:
        # Allow 5min/15min/1h/1day or 5m/15m/1h/1day
        mapping = {"5m": "5min", "15m": "15min", "10m": "10min", "60m": "1hour", "1h": "1hour", "1d": "1day", "1day": "1day"}
        return mapping.get(tf, tf)
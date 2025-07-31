import argparse
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils.data_fetcher import DataFetcher
from src.strategies.intraday_strategy import IntradayStrategy
from src.execution.backtester import Backtester


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    import yaml

    with open(path, "r") as f:
        base = yaml.safe_load(f)
    # Merge RL overrides if present
    rl_over = {}
    if os.path.exists("config/rl.yaml"):
        with open("config/rl.yaml", "r") as f:
            rl_over = yaml.safe_load(f) or {}
    merged = base.copy()
    merged.update(rl_over or {})
    return merged


def metrics_from_trades_csv(csv_path: str) -> Dict[str, Any]:
    if not os.path.exists(csv_path):
        return {"error": f"CSV not found: {csv_path}"}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"error": "No trades in CSV"}

    trades = len(df)
    win_mask = df["pnl"] > 0
    loss_mask = ~win_mask

    # Primary metrics requested
    win_pct = float(win_mask.mean() * 100.0) if trades > 0 else 0.0
    loss_pct = float(loss_mask.mean() * 100.0) if trades > 0 else 0.0
    avg_win = float(df.loc[win_mask, "pnl"].mean()) if win_mask.any() else 0.0

    # Additional metrics retained
    total_pnl = float(df["pnl"].sum())
    avg_r = float(df["r_multiple"].mean()) if "r_multiple" in df.columns else float("nan")
    expectancy = float(df["pnl"].mean()) if trades > 0 else 0.0

    equity = df["pnl"].cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = float(dd.min()) if not dd.empty else 0.0
    mar = float(total_pnl / abs(max_dd)) if max_dd < 0 else float("inf")

    r = df["r_multiple"].dropna() if "r_multiple" in df.columns else pd.Series(dtype=float)
    tail_p95 = float(np.percentile(r, 95)) if len(r) > 0 else float("nan")
    tail_p99 = float(np.percentile(r, 99)) if len(r) > 0 else float("nan")

    small_losses = float(((r >= -1.0) & (r < 0.0)).mean()) if len(r) > 0 else float("nan")
    large_winners = float((r >= 1.5).mean()) if len(r) > 0 else float("nan")

    regime_stats = {}
    if "regime" in df.columns:
        for regime, g in df.groupby("regime"):
            regime_stats[regime] = {
                "trades": int(len(g)),
                "win_rate": float((g["pnl"] > 0).mean()),
                "avg_r": float(g["r_multiple"].mean()) if "r_multiple" in g.columns else float("nan"),
                "total_pnl": float(g["pnl"].sum()),
            }

    return {
        "Trades": trades,
        "WinRate": float((df["pnl"] > 0).mean()) if trades > 0 else 0.0,
        "TotalPnL": total_pnl,
        "AvgR": avg_r,
        "Expectancy": expectancy,
        "MaxDrawdown": max_dd,
        "MAR": mar,
        "R_p95": tail_p95,
        "R_p99": tail_p99,
        "SmallLossFrac": small_losses,
        "LargeWinnerFrac": large_winners,
        "Regime": regime_stats,
        # Explicit fields requested for downstream usage:
        "win_pct": win_pct,
        "loss_pct": loss_pct,
        "avg_win": avg_win,
    }


def run_backtest_and_evaluate(
    symbol: str,
    timeframe: str,
    start: Optional[str],
    end: Optional[str],
    rth_only: bool,
    csv_out: str,
    cost_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = load_config()
    strategy = IntradayStrategy(cfg)
    fetcher = DataFetcher(cfg)
    bt = Backtester(
        strategy=strategy,
        data_fetcher=fetcher,
        config=cfg,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        rth_only=rth_only,
        csv_out=csv_out,
        cost_model=cost_model or cfg.get("backtest_costs", {}),
        seed=int(cfg.get("seed", 42)),
    )
    bt.run()
    return metrics_from_trades_csv(csv_out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="SPY")
    p.add_argument("--timeframe", type=str, default="5min")
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--rth-only", action="store_true")
    p.add_argument("--csv-out", type=str, default="data/trades_master.csv")
    return p.parse_args()


def main():
    args = parse_args()
    res = run_backtest_and_evaluate(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        rth_only=bool(args.rth_only),
        csv_out=args.csv_out,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
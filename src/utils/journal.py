import os
import json
from datetime import datetime, timezone
from typing import Dict, Any


def _ensure_dir(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    """
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


class Journal:
    """
    JSONL trade journal specialized for live/paper executions.
    Writes one JSON dict per line.

    Minimal required fields in entry:
      - timestamp (UTC str, ISO8601)
      - mode ("live" or "paper")
      - symbol
      - side ("buy" or "sell")
      - qty (number)
      - entry_price (number)
      - strategy_name (str)
      - signal_reason (str)

    Optional fields:
      - exit_price (number)
      - stop_loss (number)
      - take_profit (number)
      - bias_snapshot (dict)
      - risk (number)
      - tags (list[str])
      - order_id (str)
      - broker (str)
    """

    REQUIRED_FIELDS = {
        "timestamp",
        "mode",
        "symbol",
        "side",
        "qty",
        "entry_price",
        "strategy_name",
        "signal_reason",
    }

    def __init__(self, mode: str, base_dir: str = "data/journal") -> None:
        """
        Initialize a Journal instance.
        Creates a date-based JSONL file: base_dir/YYYYMMDD/journal_YYYYMMDD.jsonl
        """
        self.mode = mode
        self.base_dir = base_dir

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        self.date_dir = os.path.join(self.base_dir, today)
        _ensure_dir(self.date_dir)

        self.filename = os.path.join(self.date_dir, f"journal_{today}.jsonl")

    def _validate_entry(self, entry: Dict[str, Any]) -> None:
        missing = [f for f in self.REQUIRED_FIELDS if f not in entry]
        if missing:
            raise ValueError(f"Journal entry missing required fields: {missing}")

        # Basic type/format sanity checks
        ts = entry.get("timestamp")
        if not isinstance(ts, str):
            raise ValueError("timestamp must be a UTC ISO8601 string")

        if entry.get("mode") not in ("live", "paper"):
            raise ValueError("mode must be 'live' or 'paper'")

        if not isinstance(entry.get("symbol"), str):
            raise ValueError("symbol must be a string")

        if entry.get("side") not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        # Numeric sanity checks
        for k in ("qty", "entry_price"):
            if not isinstance(entry.get(k), (int, float)):
                raise ValueError(f"{k} must be numeric")

        if not isinstance(entry.get("strategy_name"), str):
            raise ValueError("strategy_name must be a string")

        if not isinstance(entry.get("signal_reason"), str):
            raise ValueError("signal_reason must be a string")

        # Optional fields type checks
        if "exit_price" in entry and not isinstance(entry["exit_price"], (int, float)) and entry["exit_price"] is not None:
            raise ValueError("exit_price must be numeric or None")
        if "stop_loss" in entry and not isinstance(entry["stop_loss"], (int, float)) and entry["stop_loss"] is not None:
            raise ValueError("stop_loss must be numeric or None")
        if "take_profit" in entry and not isinstance(entry["take_profit"], (int, float)) and entry["take_profit"] is not None:
            raise ValueError("take_profit must be numeric or None")
        if "bias_snapshot" in entry and not isinstance(entry["bias_snapshot"], dict) and entry["bias_snapshot"] is not None:
            raise ValueError("bias_snapshot must be a dict or None")
        if "risk" in entry and not isinstance(entry["risk"], (int, float)) and entry["risk"] is not None:
            raise ValueError("risk must be numeric or None")
        if "tags" in entry:
            tags = entry["tags"]
            if tags is not None and not (isinstance(tags, list) and all(isinstance(t, str) for t in tags)):
                raise ValueError("tags must be a list of strings or None")
        if "order_id" in entry and not isinstance(entry["order_id"], str) and entry["order_id"] is not None:
            raise ValueError("order_id must be a string or None")
        if "broker" in entry and not isinstance(entry["broker"], str) and entry["broker"] is not None:
            raise ValueError("broker must be a string or None")

    def log_trade(self, entry: Dict[str, Any]) -> str:
        """
        Validate and append a trade entry to the JSONL file.
        Returns the absolute path to the journal file written.
        """
        self._validate_entry(entry)
        _ensure_dir(self.date_dir)

        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return os.path.abspath(self.filename)


def utc_now_iso() -> str:
    """
    Helper to produce UTC ISO8601 timestamp with 'Z'.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
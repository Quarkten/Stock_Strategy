import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

try:
    from src.utils.journal import Journal, utc_now_iso
except Exception:
    Journal = None  # Fallback if journal not available
    def utc_now_iso():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

class AlpacaExecutor:
    def __init__(self, config):
        self.api = tradeapi.REST(
            key_id=config.get('alpaca_api_key') or config.get('ALPACA_API_KEY') or "",
            secret_key=config.get('alpaca_secret_key') or config.get('ALPACA_SECRET_KEY') or "",
            base_url=config.get('alpaca_base_url'),
            api_version='v2'
        )
        self.logger = logging.getLogger(__name__)
        # Determine live vs paper for journaling
        base_url = (config.get('alpaca_base_url') or "").lower()
        self.mode = "paper" if "paper" in base_url else "live"

        # Initialize journal only for live/paper, never for backtests (guarded by caller passing is_backtest=False)
        self.journal = None
        try:
            if Journal is not None:
                self.journal = Journal(mode=self.mode)
        except Exception as e:
            self.logger.warning(f"Journal initialization failed: {e}")

    def execute_trade(self, symbol: str, action: str, size: float, *, price: Optional[float] = None,
                      rationale: Optional[Dict[str, Any]] = None, is_backtest: bool = False) -> Optional[Any]:
        """
        Execute a live/paper trade. When not backtest, append a journal entry with rationale.
        Parameters:
          - symbol, action ("buy"/"sell"), size
          - price: optional last/entry price context
          - rationale: optional dict including fields like:
              strategy_name, signal_reason, bias_snapshot, stop_loss, take_profit, risk, tags, broker
          - is_backtest: guard to disable journaling during backtests
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=size,
                side=action.lower(),
                type='market',
                time_in_force='gtc'
            )
            print(f"Successfully submitted {action} order for {size} shares of {symbol}.")

            # Journal only if live/paper and explicitly not backtest
            if not is_backtest and self.journal is not None:
                payload = {
                    "timestamp": utc_now_iso(),
                    "mode": self.mode,
                    "symbol": symbol,
                    "side": action.lower(),
                    "qty": float(size),
                    "entry_price": float(price) if price is not None else None,
                    "strategy_name": (rationale or {}).get("strategy_name", "UnknownStrategy"),
                    "signal_reason": (rationale or {}).get("signal_reason", "No rationale provided"),
                    "exit_price": None,
                    "stop_loss": (rationale or {}).get("stop_loss"),
                    "take_profit": (rationale or {}).get("take_profit"),
                    "bias_snapshot": (rationale or {}).get("bias_snapshot"),
                    "risk": (rationale or {}).get("risk"),
                    "tags": (rationale or {}).get("tags"),
                    "order_id": getattr(order, "id", None) if order is not None else None,
                    "broker": (rationale or {}).get("broker", "alpaca")
                }
                # Remove None entry_price to satisfy validator numeric check; ensure presence if available
                if payload["entry_price"] is None:
                    # If price wasn't provided, try to derive from order (best effort)
                    try:
                        payload["entry_price"] = float(getattr(order, "filled_avg_price", None)) if getattr(order, "filled_avg_price", None) else 0.0
                    except Exception:
                        payload["entry_price"] = 0.0

                file_path = self.journal.log_trade(payload)
                self.logger.info(f"Journal entry added at {file_path}")

            return order
        except APIError as e:
            print(f"Error executing trade: {e}")
            return None

def calculate_position_size(capital, risk_per_trade, entry_price, stop_loss):
    """
    Calculate position size based on risk parameters
    Args:
        capital: Total account capital
        risk_per_trade: Percentage of capital to risk (0-1)
        entry_price: Entry price
        stop_loss: Stop loss price
    Returns:
        Number of shares/units to trade
    """
    risk_amount = capital * risk_per_trade
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share <= 0:
        return 0
    return int(risk_amount / risk_per_share)

def calculate_stop_loss(pattern_type, entry_price, atr):
    """
    Pattern-specific stop loss calculation
    """
    if pattern_type in ['hammer', 'hanging_man']:
        return entry_price - 1.5 * atr
    elif pattern_type in ['engulfing', 'morning_star']:
        return entry_price - 2 * atr
    else:
        return entry_price - 3 * atr

def calculate_take_profit(pattern_type, entry_price, atr, risk_reward_ratio=2):
    stop_loss = calculate_stop_loss(pattern_type, entry_price, atr)
    risk = abs(entry_price - stop_loss)
    if pattern_type.startswith('bullish'):
        return entry_price + risk * risk_reward_ratio
    else:
        return entry_price - risk * risk_reward_ratio
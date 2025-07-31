
import numpy as np

def calculate_asymmetric_reward(pnl: float, drawdown: float, is_tail_win: bool) -> float:
    """
    Calculates a reward that encourages small losses and large wins.

    Args:
        pnl (float): The profit or loss for the current step.
        drawdown (float): The current drawdown.
        is_tail_win (bool): Whether the current win is a tail event.

    Returns:
        float: The calculated reward.
    """
    if pnl < 0:
        # Penalize large losses more heavily
        reward = pnl * (1 + drawdown)
    else:
        if is_tail_win:
            # Amplify large wins
            reward = pnl * 2
        else:
            # Standard reward for small wins
            reward = pnl

    return reward

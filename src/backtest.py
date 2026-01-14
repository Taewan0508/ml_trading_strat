import numpy as np
import pandas as pd

def generate_positions(probabilities, upper=0.55, lower=0.45):
    positions = np.where(
        probabilities > upper, 1,
        np.where(probabilities < lower, -1, 0)
    )
    return positions


def backtest(positions, returns, cost=0.0005):
    trades = np.abs(np.diff(positions, prepend=0))
    pnl = positions * returns - trades * cost

    equity = (1 + pnl).cumprod()

    sharpe = np.sqrt(252) * pnl.mean() / pnl.std()
    max_dd = (equity / equity.cummax() - 1).min()

    return {
        "equity": equity,
        "sharpe": sharpe,
        "max_drawdown": max_dd
    }



# Re-import necessary modules after state reset
import pandas as pd
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.optimize import Bounds, LinearConstraint

# === Step 1: Helper function to compute log returns from CSV-style OHLC data ===
def load_log_returns_from_ohlc_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return df


# === Step 2: MVO optimization objective (maximize Sharpe ratio) ===

@njit
def portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (port_return - risk_free_rate) / port_vol
    return -sharpe_ratio  # Negative because we want to minimize for optimization

# === Step 3: Constraint: weights must sum to 1 and be between 0 and 1 ===
def weight_constraints(n):
    bounds = Bounds(0, 1)
    linear_constraint = LinearConstraint(np.ones(n), lb=1, ub=1)
    return bounds, linear_constraint

# === Step 4: Optimization wrapper ===
def optimize_portfolio_from_weights(csv_path, init_weights, risk_free_rate=0.02):
    mean_returns = pd.read_csv("data/excess_mean.csv", index_col=0).squeeze()
    cov_matrix = pd.read_csv("data/excess_cov.csv", index_col=0)
    num_assets = len(mean_returns)

    bounds, constraint = weight_constraints(num_assets)

    result = minimize(
        fun=portfolio_metrics,
        x0=np.array(init_weights),
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=[constraint],
        options={'disp': True}
    )

    optimized_weights = result.x
    optimized_sharpe = -result.fun

    return {
        "optimized_weights": optimized_weights,
        "sharpe": optimized_sharpe,
        "expected_return": np.dot(optimized_weights, mean_returns),
        "expected_volatility": np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights))),
        "tickers": log_returns.columns.tolist()
    }

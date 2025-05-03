import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds, LinearConstraint

def portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (port_return - risk_free_rate) / port_vol
    return -sharpe_ratio  # Negative for minimization

def weight_constraints(n):
    bounds = Bounds(0, 1)
    linear_constraint = LinearConstraint(np.ones(n), lb=1, ub=1)
    return bounds, linear_constraint

def optimize_portfolio_from_weights(csv_path, init_weights, risk_free_rate=0.02, tickers=None):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if tickers is not None:
        df = df[tickers]

    mean_returns = df.mean() * 252
    cov_matrix = df.cov() * 252
    num_assets = len(mean_returns)

    bounds, constraint = weight_constraints(num_assets)

    result = minimize(
        fun=portfolio_metrics,
        x0=np.array(init_weights),
        args=(mean_returns.values, cov_matrix.values, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=[constraint],
        options={'disp': False}
    )

    optimized_weights = result.x
    optimized_sharpe = -result.fun

    return {
        "optimized_weights": optimized_weights,
        "sharpe": optimized_sharpe,
        "expected_return": np.dot(optimized_weights, mean_returns),
        "expected_volatility": np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights))),
        "tickers": tickers if tickers is not None else df.columns.tolist()
    }

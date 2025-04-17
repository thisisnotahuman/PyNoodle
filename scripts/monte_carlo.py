import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def monte_carlo_portfolio_optimization(csv_path, n_simulations=100_000, risk_free_rate=0.02, random_seed=42):
    np.random.seed(random_seed)

    # Load standardized log return data
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    mean_returns = pd.read_csv("data/excess_mean.csv", index_col=0).squeeze()
    cov_matrix = pd.read_csv("data/excess_cov.csv", index_col=0)

    num_assets = len(mean_returns)

    # Prepare simulation containers
    all_weights = []
    ret_arr = np.zeros(n_simulations)
    vol_arr = np.zeros(n_simulations)
    sharpe_arr = np.zeros(n_simulations)

    # Run simulation
    for i in range(n_simulations):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / vol

        all_weights.append(weights)
        ret_arr[i] = ret
        vol_arr[i] = vol
        sharpe_arr[i] = sharpe

    # Find optimal
    max_idx = sharpe_arr.argmax()
    optimal_weights = all_weights[max_idx]

    return {
        "tickers": df.columns.tolist(),
        "max_sharpe": sharpe_arr[max_idx],
        "expected_return": ret_arr[max_idx],
        "expected_volatility": vol_arr[max_idx],
        "optimal_weights": optimal_weights,
        "returns": ret_arr,
        "volatilities": vol_arr,
        "sharpes": sharpe_arr
    }

def plot_simulation(returns, volatilities, sharpes, max_idx):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=sharpes, cmap='viridis', alpha=0.5)
    plt.colorbar(label="Sharpe Ratio")
    plt.scatter(volatilities[max_idx], returns[max_idx], color='red', marker='*', s=200, label='Max Sharpe')
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.title("Monte Carlo Portfolio Simulation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

def monte_carlo_portfolio_optimization(
    csv_path,
    n_simulations=100_000,
    n_assets_to_select=50,
    risk_free_rate=0.02,
    random_seed=42,
    fixed_tickers=None
):
    np.random.seed(random_seed)

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if fixed_tickers is not None:
        selected_tickers = fixed_tickers
    else:
        full_tickers = df.columns.tolist()
        if len(full_tickers) < n_assets_to_select:
            raise ValueError("Asset pool smaller than number to select")
        selected_tickers = np.random.choice(full_tickers, size=n_assets_to_select, replace=False)

    df_selected = df[selected_tickers]
    mean_returns = df_selected.mean() * 252
    cov_matrix = df_selected.cov() * 252
    num_assets = len(selected_tickers)

    all_weights = []
    ret_arr = np.zeros(n_simulations)
    vol_arr = np.zeros(n_simulations)
    sharpe_arr = np.zeros(n_simulations)

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

    max_idx = sharpe_arr.argmax()
    optimal_weights = all_weights[max_idx]

    return {
        "tickers": selected_tickers,
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

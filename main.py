import os
import pandas as pd
from scripts.monte_carlo import monte_carlo_portfolio_optimization, monte_carlo_portfolio_optimization_opt
from scripts.optimize_mvo import optimize_portfolio_from_weights

# Load Processed Log Return Data
project_root = os.path.dirname(__file__)
data_path = os.path.join(project_root, "data/raw/processed.csv")
returns = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Use SHY as Risk-Free Proxy
if 'SHY' not in returns.columns:
    raise ValueError("Ticker 'SHY' not found in return data.")

risk_free = returns['SHY']  # This Series will be used as risk-free rate

def main():
    processed_data_path = "./data/raw/processed.csv"
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Data file not found: {processed_data_path}. Please run preprocess.py first.")

    NUM_TRIALS = 50   # Number of random portfolio samples
    N_ASSETS = 50     # Number of assets to select in each trial

    print(f"Running {NUM_TRIALS} portfolio samplings to find best asset subset...")

    best_result = None
    best_sharpe = -float("inf")

    # Outer loop: Try different random selections of N assets
    for seed in range(NUM_TRIALS):
        result = monte_carlo_portfolio_optimization_opt(
            csv_path=processed_data_path,
            n_simulations=5000,
            n_assets_to_select=N_ASSETS,
            risk_free_rate=0.02,
            random_seed=seed
        )
        print(f"Seed {seed} | Sharpe = {result['max_sharpe']:.4f}")
        if result["max_sharpe"] > best_sharpe:
            best_sharpe = result["max_sharpe"]
            best_result = result

    print("\n✅ Found best asset combination. Re-running high-precision MC and MVO...")

    # Stage 2: Re-run MC on the best ticker set with more simulations
    refined_result = monte_carlo_portfolio_optimization_opt(
        csv_path=processed_data_path,
        n_simulations=100_000,
        risk_free_rate=0.02,
        fixed_tickers=best_result["tickers"]
    )

    # Save MC result
    mc_row = {
        "sharpe": refined_result["max_sharpe"],
        "expected_return": refined_result["expected_return"],
        "expected_volatility": refined_result["expected_volatility"],
    }
    for ticker, weight in zip(refined_result["tickers"], refined_result["optimal_weights"]):
        mc_row[f"weight_{ticker}"] = weight

    # Run MVO based on refined MC weights
    opt = optimize_portfolio_from_weights(
        csv_path=processed_data_path,
        init_weights=refined_result['optimal_weights'],
        tickers=refined_result['tickers'],
        risk_free_rate=0.02
    )
    opt_row = {
        "sharpe": opt["sharpe"],
        "expected_return": opt["expected_return"],
        "expected_volatility": opt["expected_volatility"],
    }
    for ticker, weight in zip(opt['tickers'], opt['optimized_weights']):
        opt_row[f"weight_{ticker}"] = weight

    # Save results
    os.makedirs("./data", exist_ok=True)
    pd.DataFrame([mc_row]).to_csv("./data/best_mc_result.csv", index=False)
    pd.DataFrame([opt_row]).to_csv("./data/best_optimized_result.csv", index=False)

    print("\n✅ Saved best MC result to ./data/best_mc_result.csv")
    print("✅ Saved best MVO result to ./data/best_optimized_result.csv")

    print("\n=== Best Monte Carlo Portfolio ===")
    print(f"Sharpe Ratio:        {mc_row['sharpe']:.4f}")
    print(f"Expected Return:     {mc_row['expected_return']:.4%}")
    print(f"Expected Volatility: {mc_row['expected_volatility']:.4%}")

    print("\n=== Best Optimized Portfolio (MVO) ===")
    print(f"Sharpe Ratio:        {opt_row['sharpe']:.4f}")
    print(f"Expected Return:     {opt_row['expected_return']:.4%}")
    print(f"Expected Volatility: {opt_row['expected_volatility']:.4%}")

    print("\n=== Sharpe Ratio Improvement ===")
    improvement = opt_row['sharpe'] - mc_row['sharpe']
    print(f"Sharpe Increase:     {improvement:.4f}")

if __name__ == "__main__":
    main()
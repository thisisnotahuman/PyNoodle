import os
import pandas as pd
from scripts.monte_carlo import monte_carlo_portfolio_optimization_opt, monte_carlo_portfolio_optimization
from scripts.optimize_mvo import optimize_portfolio_from_weights

# Load Processed Log Return Data
project_root = os.path.dirname(__file__)
data_path = os.path.join(project_root, "data/raw/processed.csv")
returns = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Use SHY as Risk-Free Proxy
if 'SHY' not in returns.columns:
    raise ValueError("Ticker 'SHY' not found in return data.")
risk_free = returns['SHY']
print("risk_free",risk_free)

def main():
    processed_data_path = "./data/raw/processed.csv"
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Data file not found: {processed_data_path}. Please run preprocess.py first.")

    # Load data and remove 'SHY' column if present
    df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
    if "SHY" in df.columns:
        df = df.drop(columns=["SHY"])

    NUM_TRIALS = 50
    N_ASSETS = 50

    print(f"Running {NUM_TRIALS} portfolio samplings to find best asset subset...")

    best_results = []
    n_best_results = 10

    for seed in range(NUM_TRIALS):
        result = monte_carlo_portfolio_optimization_opt(
            df=df,
            n_simulations=5000,
            n_assets_to_select=N_ASSETS,
            risk_free_rate=risk_free.mean(),
            random_seed=seed
        )
        print(f"Seed {seed} | Sharpe = {result['max_sharpe']:.4f}")
        best_results.append(result)
        best_results = sorted(best_results, key=lambda x: x["max_sharpe"], reverse=True)[:n_best_results]

    print("\n✅ Running MVO on top n_best_results MC results...")

    refined_best_results = []

    # Stage 2: Re-run MC on the best ticker set with more simulations
    for i in range(n_best_results):
        refined_result = monte_carlo_portfolio_optimization_opt(
            df=df,
            n_simulations=100_000,
            risk_free_rate=risk_free.mean(),
            fixed_tickers=best_results[i]["tickers"]
        )
        refined_best_results.append(refined_result)

    refined_best_results = sorted(refined_best_results, key=lambda x: x["max_sharpe"], reverse=True)

    mvo_results = []
    for i, result in enumerate(best_results):
        opt = optimize_portfolio_from_weights(
            df=df,
            init_weights=result['optimal_weights'],
            tickers=result['tickers'],
            risk_free_rate=risk_free.mean()
        )
        print(f"MVO {i} | Sharpe = {opt['sharpe']:.4f}")
        mvo_results.append(opt)

    best_mvo = max(mvo_results, key=lambda x: x["sharpe"])

    opt_row = {
        "sharpe": best_mvo["sharpe"],
        "expected_return": best_mvo["expected_return"],
        "expected_volatility": best_mvo["expected_volatility"],
    }
    for ticker, weight in zip(best_mvo["tickers"], best_mvo["optimized_weights"]):
        opt_row[f"weight_{ticker}"] = weight

    os.makedirs("./data", exist_ok=True)
    pd.DataFrame([opt_row]).to_csv("./data/best_optimized_result.csv", index=False)

    print("\n✅ Saved best MVO result to ./data/best_optimized_result.csv")
    print("\n=== Best Optimized Portfolio (MVO from Top 10 MC Seeds) ===")
    print(f"Sharpe Ratio:        {opt_row['sharpe']:.4f}")
    print(f"Expected Return:     {opt_row['expected_return']:.4%}")
    print(f"Expected Volatility: {opt_row['expected_volatility']:.4%}")

if __name__ == "__main__":
    main()

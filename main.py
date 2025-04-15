import os
import pandas as pd
from script.monte_carlo import monte_carlo_portfolio_optimization
from script.optimize_mvo import optimize_portfolio_from_weights

def main():
    processed_data_path = "processed_50.csv"
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Data file not found: {processed_data_path}. Please run preprocess.py first.")

    NUM_SEEDS = 50  # Number of initial points to generate
    all_results = []         # Monte Carlo results
    all_optimized = []       # MVO optimized results

    print(f"Running {NUM_SEEDS} Monte Carlo simulations and MVO optimizations...")

    for seed in range(NUM_SEEDS):
        print(f"Simulation {seed+1}/{NUM_SEEDS} with random_seed={seed}")

        # Monte Carlo optimization
        result = monte_carlo_portfolio_optimization(
            csv_path=processed_data_path,
            n_simulations=100_000,
            risk_free_rate=0.02,
            random_seed=seed
        )

        # Record Monte Carlo result
        row = {
            "seed": seed,
            "sharpe": result["max_sharpe"],
            "expected_return": result["expected_return"],
            "expected_volatility": result["expected_volatility"],
        }
        for ticker, weight in zip(result['tickers'], result['optimal_weights']):
            row[f"weight_{ticker}"] = weight
        all_results.append(row)

        # Run MVO optimization starting from Monte Carlo result
        opt = optimize_portfolio_from_weights(
            csv_path=processed_data_path,
            init_weights=result['optimal_weights'],
            risk_free_rate=0.02
        )
        opt_row = {
            "seed": seed,
            "sharpe": opt["sharpe"],
            "expected_return": opt["expected_return"],
            "expected_volatility": opt["expected_volatility"],
        }
        for ticker, weight in zip(opt['tickers'], opt['optimized_weights']):
            opt_row[f"weight_{ticker}"] = weight
        all_optimized.append(opt_row)

    # Convert to DataFrame
    df_mc = pd.DataFrame(all_results)
    df_opt = pd.DataFrame(all_optimized)

    # Save results
    os.makedirs("./data", exist_ok=True)
    df_mc.to_csv("./data/initial_portfolios_for_mvo.csv", index=False)
    df_opt.to_csv("./data/optimized_portfolios.csv", index=False)
    print("\nSaved Monte Carlo results to ./data/initial_portfolios_for_mvo.csv")
    print("Saved optimized portfolios to ./data/optimized_portfolios.csv")

    # Compare best Sharpe from both
    best_mc = df_mc.loc[df_mc['sharpe'].idxmax()]
    best_opt = df_opt.loc[df_opt['sharpe'].idxmax()]

    print("\n=== Best Monte Carlo Portfolio ===")
    print(f"Seed: {best_mc['seed']}")
    print(f"Sharpe Ratio:        {best_mc['sharpe']:.4f}")
    print(f"Expected Return:     {best_mc['expected_return']:.4%}")
    print(f"Expected Volatility: {best_mc['expected_volatility']:.4%}")

    print("\n=== Best Optimized Portfolio (MVO) ===")
    print(f"Seed: {best_opt['seed']}")
    print(f"Sharpe Ratio:        {best_opt['sharpe']:.4f}")
    print(f"Expected Return:     {best_opt['expected_return']:.4%}")
    print(f"Expected Volatility: {best_opt['expected_volatility']:.4%}")

    print("\n=== Sharpe Ratio Improvement ===")
    improvement = best_opt['sharpe'] - best_mc['sharpe']
    print(f"Sharpe Increase:     {improvement:.4f}")

if __name__ == "__main__":
    main()

import sys
import os
from scripts.monte_carlo import monte_carlo_portfolio_optimization, plot_simulation

def main():
    # Path to preprocessed return data
    processed_data_path = "./data/raw/processed_50.csv"

    # Check if data exists
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Data file not found: {processed_data_path}. Please run preprocess.py first.")

    print("🚀 Running Monte Carlo Portfolio Optimization...")
    result = monte_carlo_portfolio_optimization(
        csv_path=processed_data_path,
        n_simulations=100_000,
        risk_free_rate=0.02
    )

    print("\n✅ Optimal Portfolio Allocation (Maximum Sharpe Ratio):")
    for ticker, weight in zip(result['tickers'], result['optimal_weights']):
        print(f"{ticker:<6}: {weight:.2%}")

    print(f"\n📈 Annualized Return:     {result['expected_return']:.2%}")
    print(f"📉 Annualized Volatility: {result['expected_volatility']:.2%}")
    print(f"⚖️  Sharpe Ratio:          {result['max_sharpe']:.2f}")

    print("\n📊 Plotting Efficient Frontier with Sharpe Ratio Color Coding...")
    plot_simulation(
        result['returns'],
        result['volatilities'],
        result['sharpes'],
        max_idx=result['sharpes'].argmax()
    )

if __name__ == "__main__":
    main()


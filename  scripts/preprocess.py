import pandas as pd
import numpy as np

def preprocess_data(input_path="data/raw/sp500_bonds_50.csv", output_path="data/raw/processed_50.csv"):
    # 1. Load raw CSV file
    df = pd.read_csv(input_path)
    
    # 2. Reshape to pivot table: rows = Date, columns = Ticker, values = Close prices
    pivot_df = df.pivot(index="Date", columns="Ticker", values="Close")
    pivot_df.index = pd.to_datetime(pivot_df.index)
    pivot_df = pivot_df.sort_index()

    # 3. Handle missing values: forward-fill first, then backward-fill
    pivot_df = pivot_df.ffill().bfill()

    # 4. Calculate log returns
    log_return = np.log(pivot_df / pivot_df.shift(1)).dropna()

    # 5. Standardize (z-score normalization)
    standardized = (log_return - log_return.mean()) / log_return.std()

    # 6. Save to CSV
    standardized.to_csv(output_path)
    print(f"Preprocessing complete. Saved to {output_path}")

if __name__ == "__main__":
    preprocess_data()

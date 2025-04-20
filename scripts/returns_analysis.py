import pandas as pd
import numpy as np
import time
from numba import njit

# === 1. Load Processed Log Return Data ===
data_path = "data/raw/processed_50.csv"
returns = pd.read_csv(data_path, index_col=0, parse_dates=True)

# === 2. Use SHY as Risk-Free Proxy ===
if 'SHY' not in returns.columns:
    raise ValueError("Ticker 'SHY' not found in return data. Cannot compute excess returns.")

risk_free = returns['SHY']  # Series

# === 3. Compute Excess Returns ===
excess_returns = returns.sub(risk_free, axis=0)
excess_returns.drop(columns='SHY', inplace=True)  # remove SHY from asset pool

# ---------------------------------------
# Part A: Baseline Pandas Implementation
# ---------------------------------------

start = time.time()
mu_baseline = excess_returns.mean() * 252
cov_baseline = excess_returns.cov() * 252
end = time.time()
print(f"\n🐼 Pandas Version Time: {end - start:.4f}s")

# ---------------------------------------
# Part B: Optimized Numba Implementation
# ---------------------------------------

@njit
def mean_annualized(arr):
    return np.mean(arr, axis=0) * 252

@njit
def cov_annualized(arr):
    n = arr.shape[0]
    mean = np.mean(arr, axis=0)
    centered = arr - mean
    cov = np.dot(centered.T, centered) / (n - 1)
    return cov * 252

X = excess_returns.to_numpy()

start = time.time()
mu_numba = mean_annualized(X)
cov_numba = cov_annualized(X)
end = time.time()
print(f"⚡ Numba Version Time: {end - start:.4f}s")

# ---------------------------------------
# Part C: Save Results & Compare Accuracy
# ---------------------------------------

# Save only the pandas version as it has labels
mu_baseline.to_csv("excess_mean.csv")
cov_baseline.to_csv("excess_cov.csv")
excess_returns.to_csv("excess_returns.csv")

print("\n Saved:")
print("- excess_mean.csv")
print("- excess_cov.csv")
print("- excess_returns.csv")

# Accuracy Check
mu_diff = np.abs(mu_numba - mu_baseline.values).max()
cov_diff = np.abs(cov_numba - cov_baseline.values).max()

print(f"\n Max Error in mu:  {mu_diff:.2e}")
print(f" Max Error in cov: {cov_diff:.2e}")
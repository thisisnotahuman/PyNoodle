import pandas as pd
import numpy as np
import time
from numba import njit
import os

# Load Processed Log Return Data
project_root = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(project_root, "data/raw/processed_50.csv")
returns = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Use SHY as Risk-Free Proxy
if 'SHY' not in returns.columns:
    raise ValueError("Ticker 'SHY' not found in return data. Cannot compute excess returns.")

risk_free = returns['SHY']  # Series

# Compute Excess Returns
excess_returns = returns.sub(risk_free, axis=0)
excess_returns.drop(columns='SHY', inplace=True)  # remove SHY from asset pool

# Baseline Pandas Implementation

start = time.time()
mu_baseline = excess_returns.mean() * 252
cov_baseline = excess_returns.cov() * 252
end = time.time()
print(f"\n Pandas Version Time: {end - start:.4f}s")


# Optimized Numba Implementation

@njit
def mean_annualized(arr):
    n = arr.shape[0]
    return arr.sum(axis=0) / n * 252


@njit
def cov_annualized(arr):
    n, d = arr.shape
    mean = np.zeros(d)

    # compute mean vector manually
    for j in range(d):
        for i in range(n):
            mean[j] += arr[i, j]
        mean[j] /= n

    # manually center the matrix
    centered = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            centered[i, j] = arr[i, j] - mean[j]

    # compute covariance matrix manually
    cov = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            for k in range(n):
                cov[i, j] += centered[k, i] * centered[k, j]
            cov[i, j] /= (n - 1)

    return cov * 252


X = excess_returns.to_numpy()

start = time.time()
mu_numba = mean_annualized(X)
cov_numba = cov_annualized(X)
end = time.time()
print(f" Numba Version Time: {end - start:.4f}s")

# Measure first Numba run (includes compile time)
start = time.time()
mu_numba = mean_annualized(X)
cov_numba = cov_annualized(X)
end = time.time()
print(f"\n Numba First Run (includes compile): {end - start:.4f}s")


# Save Numba Results & Compare Accuracy

# Convert Numba results back to Pandas with labels
mu_df = pd.Series(mu_numba, index=excess_returns.columns, name='Excess Mean')
cov_df = pd.DataFrame(cov_numba, index=excess_returns.columns, columns=excess_returns.columns)

# Save with original filenames (overwrite Pandas results)
mu_df.to_csv("excess_mean.csv")   # overwriting previous Pandas file
cov_df.to_csv("excess_cov.csv")   # overwriting previous Pandas file
excess_returns.to_csv("excess_returns.csv")  # keep this the same

print("\n Saved Numba Results:")
print("- excess_mean.csv")
print("- excess_cov.csv")
print("- excess_returns.csv")

# Accuracy Check (still optional, but compares Numba vs Pandas results)
mu_diff = np.abs(mu_numba - mu_baseline.values).max()
cov_diff = np.abs(cov_numba - cov_baseline.values).max()

print(f"\n Max Error in mu:  {mu_diff:.2e}")
print(f" Max Error in cov: {cov_diff:.2e}")

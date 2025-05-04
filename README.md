## 📁 Project Structure – Portfolio Optimization with Monte Carlo & MVO

This project implements a two-phase portfolio optimization strategy that combines **Monte Carlo simulations** with **Mean-Variance Optimization (MVO)**. The codebase is organized for clarity, modular development, and performance experimentation.

[Project Intro PPT](https://docs.google.com/presentation/d/1G4pCDC_eiN8FB5hZXEQbwiOKLtBF27I-a8a8lewox6E/edit?usp=sharing)

[Final Presentation PPT](https://docs.google.com/presentation/d/1MmyTgADte4OAgjfuZ0_GFawipkVd2WBXlHgYy-0_bmU/edit?usp=sharing)

[Github Link](https://github.com/thisisnotahuman/PyNoodle/tree/main)

### 🔧 Directory Layout

```python
PyNoodle/
├── data/
│   ├── raw/
│   │   └── preprocessed.csv          # Cleaned dataset
│   ├── best_mc_result.csv            # Best portfolio from Monte Carlo
│   ├── best_optimized_result.csv     # Optimized portfolio from MVO
│   ├── excess_returns.csv            # Raw excess log return data
│   ├── excess_mean.csv               # Precomputed mean returns
│   └── excess_cov.csv                # Precomputed covariance matrix
│
├── scripts/
│   ├── __init__.py
│   ├── fetch_data.py                 # Fetch historical prices
│   ├── preprocess.py                 # Clean & compute log returns
│   ├── monte_carlo.py                # Monte Carlo simulation
│   ├── optimize_mvo.py               # Mean-Variance Optimization (SLSQP)
│   ├── returns_analysis.py           # Compute & export annualized excess return stats
│   └── text_appendix/                # Benchmark experiments & comparisons
│       ├── test_base.ipynb           # time test for baseline code
│       ├── test_opt.ipynb            # time test for optimized code
│       └── pandas_vs_numba.ipynb     # time test for returns analysis
│
├── main.py                           # Pipeline entry point: MC + MVO
├── MC_sim_0.png                      # Example: single Monte Carlo run
├── MC_sim_standardized.png           # Cleaned plot for report
├── LICENSE
├── README.md
└── .gitignore
```


### ▶️ How to Run

Make sure you're in the project root directory, then run:

```bash
python main.py
```

This will:

Perform randomized Monte Carlo simulations across asset subsets.
Identify the best subset by Sharpe Ratio.
Refine it with high-resolution MC and MVO optimization.
Save the resulting portfolios to the data/raw/ directory.

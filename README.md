## 📁 Project Structure – Portfolio Optimization with Monte Carlo & MVO

This project implements a two-phase portfolio optimization strategy that combines **Monte Carlo simulations** with **Mean-Variance Optimization (MVO)**. The codebase is organized for clarity, modular development, and performance experimentation.

[Project Intro PPT](https://docs.google.com/presentation/d/1G4pCDC_eiN8FB5hZXEQbwiOKLtBF27I-a8a8lewox6E/edit?usp=sharing)

[Final Presentation PPT](https://docs.google.com/presentation/d/1MmyTgADte4OAgjfuZ0_GFawipkVd2WBXlHgYy-0_bmU/edit?usp=sharing)

[Github Link](https://github.com/thisisnotahuman/PyNoodle/tree/main)

### 🔧 Directory Layout

```python
PyNoodle/
├── data/
│ ├── raw/
│ │ ├── best_mc_result.csv # Best portfolio found via Monte Carlo
│ │ ├── best_optimized_result.csv # Optimized version via MVO
│ │ ├── excess_cov.csv # Precomputed covariance matrix
│ │ ├── excess_mean.csv # Precomputed mean returns
│ │ └── excess_returns.csv # Raw excess log return data
│
├── scripts/
│ ├── init.py
│ ├── fetch_data.py # Script for fetching/loading stock data
│ ├── monte_carlo.py # Monte Carlo logic (baseline, parallel, vectorized)
│ ├── optimize_mvo.py # Constrained optimizer (MVO)
│ ├── preprocess.py # Preprocessing script for return data
│ └── returns_analysis.py # Sharpe ratio & volatility visualization utilities
│
├── main.py # Main script to run MC + MVO workflow
├── MC_sim_0.png # Sample MC simulation plot
├── MC_sim_standardized.png # Cleaned plot for reporting
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

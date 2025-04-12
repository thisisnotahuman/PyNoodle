import yfinance as yf
import pandas as pd
from datetime import datetime

# s&p e.g.
stock_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "META", "TSLA", "NVDA", "UNH", "V",
    "HD", "JNJ", "PG", "PFE", "MA", "MRK", "XOM", "PEP", "KO", "ABBV",
    "AVGO", "LLY", "COST", "WMT", "CVX", "BAC", "TMO", "QCOM", "ACN", "INTC",
    "ADBE", "CRM", "AMD", "ABT", "NFLX", "NEE", "MCD", "DHR", "TXN", "LIN",
    "ORCL", "CMCSA", "UPS", "AMAT", "PM", "INTU", "HON", "IBM", "GE", "LOW",
    "ISRG", "RTX", "SBUX", "BLK", "CAT", "AXP", "BKNG", "DE", "CI", "SPGI",
    "GS", "MS", "CB", "ZTS", "PLD", "LRCX", "NOW", "MO", "ADI", "MDT",
    "T", "GILD", "VRTX", "ADI", "ADP", "USB", "TGT", "CL", "REGN", "SO",
    "ETN", "BDX", "NKE", "CSCO", "APD", "EL", "WM", "MAR", "ILMN", "ROST"
]

bond_tickers = ["TLT", "IEF", "SHY", "BND", "AGG"]

tickers = stock_tickers + bond_tickers

def fetch_all_assets(tickers, start="2014-01-01", end="2024-01-01", save_path="data/raw/sp500_bonds_50.csv"):
    df = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)
    all_data = []
    for ticker in tickers:
        temp = df[ticker].copy()
        temp['Ticker'] = ticker
        temp = temp.reset_index()
        all_data.append(temp)

    final_df = pd.concat(all_data)
    final_df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    fetch_all_assets(tickers)

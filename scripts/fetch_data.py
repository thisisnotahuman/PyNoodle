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
    "ETN", "BDX", "NKE", "CSCO", "APD", "EL", "WM", "MAR", "ILMN", "ROST", 
    "A", "AAL", "AEE", "AEP", "AES", "AFL", "AJG", "AKAM", "ALB", "ALL",
    "AME", "AMGN", "ANET", "AON", "APA", "ARE", "ATO", "AZO", "BALL", "BAX",
    "BBY", "BEN", "BIO", "BKR", "BMY", "BR", "BRO", "C", "CAG", "CAH",
    "CDW", "CE", "CF", "CHRW", "CINF", "CLX", "CNP", "COO", "CPB", "CPT",
    "CTLT", "CTRA", "CVS", "D", "DAL", "DD", "DG", "DLR", "DOV", "DRI",
    "DVN", "DXC", "ECL", "ED", "EFX", "EG", "EIX", "EMN", "EMR", "ENPH",
    "EOG", "EQR", "EQX", "ES", "ESS", "ETR", "EVRG", "EXC", "EXPD", "EXR",
    "F", "FAST", "FCX", "FDS", "FE", "FFIV", "FISV", "FLT", "FMC", "FRT",
    "FSLR", "FTNT", "GEN", "GL", "GLW", "HIG", "HLT", "HPE", "HPQ", "HRL",
    "HSIC", "HWM", "IEX", "IR", "IT", "JBHT", "JCI", "JKHY", "KEYS", "KHC",
    "KIM", "KLAC", "KMB", "KMI", "KMX", "KR", "L", "LDOS", "LEN", "LKQ",
    "LLY", "LLY", "LNT", "LUMN", "LW", "LYB", "MAA", "MCHP", "MKC", "MKTX",
    "MNST", "MSCI", "MTB", "MTCH", "NDAQ", "NEE", "NEM", "NFLX", "NI", "NOC",
    "NRG", "NTRS", "NUE", "NVDA", "NVR", "O", "ODFL", "OKE", "OTIS", "PFG",
    "PGR", "PH", "PNW", "PNC", "PNR", "PPG", "PPL", "PRU", "PSA", "PTC",
    "PWR", "QRVO", "RCL", "RE", "REG", "RF", "RHI", "RMD", "ROK", "ROL",
    "ROP", "RSG", "SBAC", "SEDG", "SEE", "SJM", "SLB", "SNA", "SNPS", "STT",
    "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "TAP", "TDG", "TEL", "TER"
]

bond_tickers = ["TLT", "IEF", "SHY", "BND", "AGG", "TIP", "IEI", "ITOT", "VGSH", "VGIT", "VGLT", "HYG", "IGSB", "LQD", "EDV"]

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

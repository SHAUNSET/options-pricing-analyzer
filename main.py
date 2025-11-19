# main.py
from src.option_chain_analyzer import preprocess_option_chain
import pandas as pd

def main():
    print("🔹 NSE Option Pricing Analyzer 🔹")
    symbols = [
        ("NIFTY", True),
        ("BANKNIFTY", True),
        ("RELIANCE", False),
        ("TCS", False)
    ]

    for symbol, is_index in symbols:
        print(f"\nProcessing symbol: {symbol}")
        df, spot_price = preprocess_option_chain(symbol, index=is_index)
        if df is None or df.empty:
            print(f"No options available for {symbol}.")
            continue

        print(f"Spot price for {symbol}: {spot_price}")
        print(df.head(10))
        df.to_excel(f"{symbol}_option_analysis.xlsx", index=False)
        print(f"Saved analysis to {symbol}_option_analysis.xlsx")

if __name__=="__main__":
    main()

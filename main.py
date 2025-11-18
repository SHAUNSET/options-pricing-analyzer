from src.option_chain_analyzer import preprocess_option_chain

if __name__ == "__main__":
    symbol = "AAPL"
    df = preprocess_option_chain(symbol)
    
    if df is not None:
        print(df.head(10))  # preview
        # Save to Excel
        try:
            df.to_excel(f"{symbol}_option_analysis.xlsx", index=False)
            print(f"Saved analysis to {symbol}_option_analysis.xlsx")
        except Exception as e:
            print(f"Error saving Excel file: {e}")

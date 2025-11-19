# src/data_fetcher.py
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def fetch_option_chain(symbol: str):
    """
    Fetch NSE option chain JSON for a symbol (index or equity).
    Returns the raw 'records' -> 'data' list or an empty list on failure.
    """
    symbol = symbol.upper()
    if symbol in ("NIFTY", "BANKNIFTY"):
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    else:
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

    try:
        session = requests.Session()
        # initial request to set cookies
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=5)
        resp = session.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        records = j.get("records", {}).get("data", [])
        return records
    except Exception as e:
        # upstream may block requests — caller should handle empty list
        print(f"[data_fetcher] fetch failed for {symbol}: {e}")
        return []

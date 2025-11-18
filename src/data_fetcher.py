import yfinance as yf

def get_option_chain(symbol):
    """Fetch the nearest expiry option chain for a symbol."""
    ticker = yf.Ticker(symbol)
    options = ticker.options
    if not options:
        return None, None, None
    nearest_expiry = options[0]
    chain = ticker.option_chain(nearest_expiry)
    return chain.calls, chain.puts, nearest_expiry

# src/option_chain_analyzer.py
import pandas as pd
from datetime import datetime
import yfinance as yf
from src.data_fetcher import get_option_chain
from src.black_scholes import bs_call_price, bs_put_price, bs_greeks
from src.utils import make_datetime_naive

def preprocess_option_chain(symbol):
    """
    Fetch, process, and compute Black-Scholes prices, Greeks, and signals
    for the option chain of the given stock symbol.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., "AAPL").
    
    Returns:
        pd.DataFrame: Processed option chain with BS prices, Greeks, and signals.
    """
    calls, puts, expiry = get_option_chain(symbol)
    if calls is None or puts is None:
        print("No option chain data available.")
        return None

    # Add option type and expiry
    calls['type'] = 'call'
    puts['type'] = 'put'
    expiry_ts = pd.to_datetime(expiry)
    calls['expiry'] = expiry_ts
    puts['expiry'] = expiry_ts

    # Combine calls and puts
    df = pd.concat([calls, puts], ignore_index=True)

    # Days to expiry
    today = pd.Timestamp(datetime.now().date())
    df['days_to_expiry'] = (df['expiry'].dt.tz_localize(None) - today).dt.days

    # Fetch actual spot price from Yahoo Finance
    try:
        ticker = yf.Ticker(symbol)
        spot_price = ticker.history(period='1d')['Close'].iloc[-1]
        print(f"Spot price for {symbol}: {spot_price}")
    except Exception as e:
        print(f"Error fetching spot price: {e}")
        spot_price = df['lastPrice'].iloc[0]  # fallback

    r = 0.05  # risk-free rate
    bs_prices, deltas, gammas, vegas, thetas, rhos, signals = [], [], [], [], [], [], []

    # Compute BS price, Greeks, and signal for each option
    for _, row in df.iterrows():
        K = row['strike']
        T = row['days_to_expiry'] / 365
        option_type = row['type']
        sigma = row['impliedVolatility'] if not pd.isna(row['impliedVolatility']) else 0.2

        # Black-Scholes price
        bs_price = (
            bs_call_price(spot_price, K, T, r, sigma) 
            if option_type == 'call' 
            else bs_put_price(spot_price, K, T, r, sigma)
        )
        bs_prices.append(bs_price)

        # Greeks
        delta, gamma, vega, theta, rho = bs_greeks(spot_price, K, T, r, sigma, option_type)
        deltas.append(delta)
        gammas.append(gamma)
        vegas.append(vega)
        thetas.append(theta)
        rhos.append(rho)

        # Signal logic
        diff = row['lastPrice'] - bs_price
        if abs(diff)/bs_price < 0.05:
            signals.append('Fair')
        elif diff > 0:
            signals.append('Overpriced')
        else:
            signals.append('Underpriced')

    # Add computed columns
    df['BS_price'] = bs_prices
    df['delta'] = deltas
    df['gamma'] = gammas
    df['vega'] = vegas
    df['theta'] = thetas
    df['rho'] = rhos
    df['signal'] = signals

    # Make datetime columns naive for Excel
    df = make_datetime_naive(df, ['lastTradeDate', 'expiry'])

    return df

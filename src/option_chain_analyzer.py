# src/option_chain_analyzer.py
"""
Robust option chain analyzer.

- Tries to fetch option chain from NSE via src.data_fetcher.fetch_option_chain()
- If NSE data is empty/unavailable, falls back to yfinance option chain
- Computes Black-Scholes prices and Greeks
- Solves for implied volatility when necessary (bisection)
- Returns DataFrame and spot price

DataFrame columns (final):
    strike, type, lastPrice, implied_vol, BS_price_iv, BS_price_user,
    delta_user, delta_iv, gamma_user, gamma_iv, vega_user, vega_iv,
    theta_user, theta_iv, rho_user, rho_iv, signal
"""

from typing import Tuple, Optional, List, Dict
import math
import pandas as pd

# Attempt to import fetch_option_chain from local data_fetcher
try:
    from src.data_fetcher import fetch_option_chain
except Exception:
    # If package import fails, define a stub to avoid hard crash; caller should handle empty data
    def fetch_option_chain(symbol: str):
        return []

# Try to import yfinance for fallback
try:
    import yfinance as yf
    _HAS_YFINANCE = True
except Exception:
    _HAS_YFINANCE = False

# ---------------------------
# Black-Scholes & Greeks
# ---------------------------
def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price_call(S: float, K: float, r: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

def bs_price_put(S: float, K: float, r: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

def bs_greeks(S: float, K: float, r: float, T: float, sigma: float, opt_type: str = "call"):
    """
    Returns (delta, gamma, vega, theta_per_day, rho)
    - theta returned as per-day value (divide by 365 applied)
    - vega returned as price change for +1.0 vol (e.g. if sigma=0.25 -> +1.0 is +100% vol). UI can scale if needed.
    """
    if T <= 0 or sigma <= 0:
        # degenerate: expiry or zero vol
        if opt_type == "call":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return delta, 0.0, 0.0, 0.0, 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = _norm_pdf(d1)

    if opt_type == "call":
        delta = _norm_cdf(d1)
        theta = ( - S * pdf_d1 * sigma / (2 * math.sqrt(T))
                  - r * K * math.exp(-r * T) * _norm_cdf(d2) )
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = ( - S * pdf_d1 * sigma / (2 * math.sqrt(T))
                  + r * K * math.exp(-r * T) * _norm_cdf(-d2) )

    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * pdf_d1 * math.sqrt(T)
    rho = K * T * math.exp(-r * T) * (_norm_cdf(d2) if opt_type == "call" else -_norm_cdf(-d2))

    theta_per_day = theta / 365.0
    return delta, gamma, vega, theta_per_day, rho

# ---------------------------
# Implied volatility solver (bisection)
# ---------------------------
def implied_volatility_from_price(market_price: float, S: float, K: float, r: float, T: float,
                                  opt_type: str = "call", tol: float = 1e-6, max_iter: int = 80) -> Optional[float]:
    """
    Use bisection to find sigma such that BS_price(sigma) ~= market_price.
    Returns sigma (annual) or None if unsolvable.
    """
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None

    # Lower/upper bounds for volatility
    low, high = 1e-6, 5.0  # 500% volatility upper bound
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        price = bs_price_call(S, K, r, T, mid) if opt_type == "call" else bs_price_put(S, K, r, T, mid)
        if math.isnan(price):
            return None
        diff = price - market_price
        if abs(diff) < tol:
            return mid
        # If model price at mid is higher than market, reduce volatility; otherwise increase
        if price > market_price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)

# ---------------------------
# Helper: Yahoo fallback fetch (if NSE fails)
# ---------------------------
def _fetch_option_chain_yahoo(symbol: str) -> Tuple[List[Dict], Optional[float]]:
    """
    Use yfinance to fetch option chain. Returns (rows, spot) where rows is a list of dicts similar to NSE entries.
    Each dict contains: strike, type, lastPrice, impliedVol (if available)
    """
    if not _HAS_YFINANCE:
        return [], None

    try:
        ticker = yf.Ticker(symbol)
        # best available spot:
        spot = None
        try:
            spot = float(ticker.info.get("regularMarketPrice") or ticker.history(period="1d")["Close"].iloc[-1])
        except Exception:
            spot = None

        rows = []
        expirations = ticker.options  # list of expiry strings
        if not expirations:
            return [], spot

        # choose nearest expiry (first)
        expiry = expirations[0]
        chain = ticker.option_chain(expiry)
        calls = chain.calls.to_dict(orient="records")
        puts = chain.puts.to_dict(orient="records")
        # unify format: strike, type, lastPrice, impliedVol if present
        for rec in calls:
            rows.append({
                "strike": float(rec.get("strike")),
                "type": "call",
                "lastPrice": float(rec.get("lastPrice") or 0.0),
                "impliedVolatility": float(rec.get("impliedVol") or 0.0)
            })
        for rec in puts:
            rows.append({
                "strike": float(rec.get("strike")),
                "type": "put",
                "lastPrice": float(rec.get("lastPrice") or 0.0),
                "impliedVolatility": float(rec.get("impliedVol") or 0.0)
            })
        return rows, spot
    except Exception:
        return [], None

# ---------------------------
# Main public function
# ---------------------------
def preprocess_option_chain(symbol: str,
                            opt_type: str = "call",
                            T: float = 30/365.0,
                            r: float = 0.06,
                            user_vol: Optional[float] = None) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Fetch option chain and compute:
      - implied_vol (market if available or solved)
      - BS price using implied vol (BS_price_iv)
      - BS price using user_vol (BS_price_user). If user_vol is None, user_vol fallback is implied vol
      - Greeks for both the implied vol and user vol (delta/gamma/vega/theta/rho)
      - signal comparing market lastPrice to BS_price_user (Underpriced / Overpriced / Fair / Unknown)

    Args:
      symbol: NSE or ticker symbol
      opt_type: "call" or "put"
      T: time to expiry in years (e.g., 30/365)
      r: annual risk-free rate (decimal, e.g., 0.06)
      user_vol: user-specified vol in percent (e.g., 25) or decimal (0.25). If >1 assumed percent.
    Returns:
      df, spot_price
    """

    # ensure string
    opt_type = opt_type.lower()
    if opt_type not in ("call", "put"):
        raise ValueError("opt_type must be 'call' or 'put'")

    # Normalize user_vol
    vol_user_val = None
    if user_vol is not None:
        try:
            vv = float(user_vol)
            vol_user_val = vv / 100.0 if vv > 1.0 else vv
        except Exception:
            vol_user_val = None

    # 1) Try NSE fetch (expected format: list of entries where each has CE/PE)
    raw = []
    try:
        raw = fetch_option_chain(symbol) or []
    except Exception:
        raw = []

    rows = []
    spot_price = None

    # If NSE returns a list of strike entries (records), attempt to parse
    if isinstance(raw, list) and raw:
        # attempt to find spot from records top-level 'underlyingValue' if present
        # entries may have 'underlyingValue' at top-level or inside CE/PE
        for entry in raw:
            if isinstance(entry, dict) and "underlyingValue" in entry and entry.get("underlyingValue"):
                try:
                    spot_price = float(entry.get("underlyingValue"))
                    break
                except Exception:
                    pass
        # fallback: check inside CE/PE
        if spot_price is None:
            for entry in raw:
                ce = entry.get("CE") if isinstance(entry, dict) else None
                pe = entry.get("PE") if isinstance(entry, dict) else None
                val = None
                if ce and isinstance(ce, dict):
                    val = ce.get("underlying") or ce.get("underlyingValue")
                if not val and pe and isinstance(pe, dict):
                    val = pe.get("underlying") or pe.get("underlyingValue")
                if val:
                    try:
                        spot_price = float(val)
                        break
                    except Exception:
                        pass

        # parse rows: pick option object per entry safely
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            option_obj = entry.get("CE") if opt_type == "call" else entry.get("PE")
            if not option_obj:
                continue

            # strike may be in option_obj or top-level entry
            strike = option_obj.get("strikePrice", entry.get("strikePrice"))
            if strike is None:
                continue
            try:
                strike_f = float(strike)
            except Exception:
                continue

            # last price & market iv (NSE often has 'lastPrice' and 'impliedVolatility')
            last_price = option_obj.get("lastPrice", 0.0)
            try:
                last_price_f = float(last_price or 0.0)
            except Exception:
                last_price_f = 0.0

            market_iv_raw = option_obj.get("impliedVolatility", option_obj.get("impliedVol", None))
            market_iv = None
            try:
                if market_iv_raw is not None:
                    ivf = float(market_iv_raw)
                    # NSE sometimes gives percent as 12.34 (not divided), sometimes already decimal; heuristic:
                    market_iv = ivf / 100.0 if ivf > 1.0 else ivf
            except Exception:
                market_iv = None

            rows.append({
                "strike": strike_f,
                "type": opt_type,
                "lastPrice": last_price_f,
                "market_iv": market_iv
            })

    # 2) If NSE failed or returned nothing, fallback to Yahoo via yfinance
    if not rows:
        yahoo_rows, yahoo_spot = _fetch_option_chain_yahoo(symbol) if _HAS_YFINANCE else ([], None)
        if yahoo_rows:
            spot_price = yahoo_spot if yahoo_spot else spot_price
            for rec in yahoo_rows:
                if rec.get("type") != opt_type:
                    continue
                strike_f = float(rec.get("strike"))
                last_price_f = float(rec.get("lastPrice") or 0.0)
                market_iv_raw = rec.get("impliedVolatility", None)
                market_iv = None
                try:
                    if market_iv_raw is not None:
                        ivf = float(market_iv_raw)
                        market_iv = ivf / 100.0 if ivf > 1.0 else ivf
                except Exception:
                    market_iv = None
                rows.append({
                    "strike": strike_f,
                    "type": opt_type,
                    "lastPrice": last_price_f,
                    "market_iv": market_iv
                })

    # If still nothing, return empty
    if not rows:
        return pd.DataFrame(), None

    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)

    # Spot fallback: if we never found spot_price, approximate with median strike (rough)
    if spot_price is None or spot_price == 0:
        spot_price = float(df["strike"].median())

    # Now compute implied vol (solve where market_iv missing), bs prices and greeks
    implied_vols = []
    bs_price_iv_list = []
    bs_price_user_list = []
    greeks_iv_list = []
    greeks_user_list = []

    for idx, row in df.iterrows():
        K = float(row["strike"])
        market_price = float(row["lastPrice"])
        market_iv = row.get("market_iv")

        # prefer market_iv; else attempt to solve implied vol from market price
        iv = None
        if market_iv and market_iv > 1e-6:
            iv = market_iv
        else:
            if market_price > 0:
                iv = implied_volatility_from_price(market_price, spot_price, K, r, T, opt_type)
        # fallback default
        if iv is None or iv <= 0:
            iv = vol_user_val if vol_user_val is not None else 0.20

        implied_vols.append(iv)

        # BS price using implied vol (market)
        bs_iv = bs_price_call(spot_price, K, r, T, iv) if opt_type == "call" else bs_price_put(spot_price, K, r, T, iv)
        bs_price_iv_list.append(bs_iv)

        # BS price using user vol if provided, otherwise use iv
        if vol_user_val is not None:
            bs_user = bs_price_call(spot_price, K, r, T, vol_user_val) if opt_type == "call" else bs_price_put(spot_price, K, r, T, vol_user_val)
        else:
            bs_user = bs_iv
        bs_price_user_list.append(bs_user)

        # Greeks implied
        g_iv = bs_greeks(spot_price, K, r, T, iv, opt_type)
        greeks_iv_list.append(g_iv)

        # Greeks user/model
        vol_for_user = vol_user_val if vol_user_val is not None else iv
        g_user = bs_greeks(spot_price, K, r, T, vol_for_user, opt_type)
        greeks_user_list.append(g_user)

    # attach to df
    df["implied_vol"] = implied_vols
    df["BS_price_iv"] = bs_price_iv_list
    df["BS_price_user"] = bs_price_user_list

    # Unpack greeks tuples
    g_iv_df = pd.DataFrame(greeks_iv_list, columns=["delta_iv", "gamma_iv", "vega_iv", "theta_iv", "rho_iv"])
    g_user_df = pd.DataFrame(greeks_user_list, columns=["delta_user", "gamma_user", "vega_user", "theta_user", "rho_user"])
    df = pd.concat([df, g_user_df, g_iv_df], axis=1)

    # Choose model price to compare signals: BS_price_user
    signals = []
    for _, rrow in df.iterrows():
        market_p = float(rrow["lastPrice"])
        bs_model = float(rrow["BS_price_user"] if "BS_price_user" in rrow else rrow.get("BS_price_iv", 0.0))
        if market_p <= 0 or bs_model <= 0:
            signals.append("Unknown")
            continue
        diff = market_p - bs_model
        rel = abs(diff) / (bs_model if bs_model != 0 else 1.0)
        if rel <= 0.05:
            signals.append("Fair")
        elif diff > 0:
            signals.append("Overpriced")
        else:
            signals.append("Underpriced")
    df["signal"] = signals

    # For convenience: set single 'BS_price' column to user's model price
    df["BS_price"] = df["BS_price_user"]

    # reorder columns for readability
    col_order = ["strike", "type", "lastPrice", "implied_vol", "BS_price_iv", "BS_price_user",
                 "BS_price",
                 "delta_user", "delta_iv",
                 "gamma_user", "gamma_iv",
                 "vega_user", "vega_iv",
                 "theta_user", "theta_iv",
                 "rho_user", "rho_iv",
                 "signal"]
    # keep only columns present
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order].copy()

    return df.reset_index(drop=True), float(spot_price)

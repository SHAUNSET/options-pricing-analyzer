# main_streamlit.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.option_chain_analyzer import preprocess_option_chain

st.set_page_config(page_title="NSE Option Pricing Analyzer ", layout="wide")
st.title("📊 NSE Option Pricing Analyzer")

st.markdown(
    "Compare market option prices with your Black–Scholes model (enter a model vol), "
    "see model vs market (implied) Greeks, and get clean buy/avoid/fair signals."
)

# -------------------------
# User inputs
# -------------------------
col1, col2, col3 = st.columns([2, 1.2, 1.2])
with col1:
    symbol = st.text_input("NSE Symbol (e.g., NIFTY, RELIANCE.NS):", "NIFTY").upper()
with col2:
    option_type = st.selectbox("Option type:", ["Call", "Put"])
with col3:
    time_to_expiry_days = st.number_input("Time to expiry (days):", min_value=1, max_value=365, value=30)

col4, col5 = st.columns([1.2, 1.2])
with col4:
    risk_free_pct = st.number_input("Risk-free rate (%):", value=6.5, step=0.1)
with col5:
    user_vol_input = st.text_input("Model volatility (%, leave blank to use market IV):", "")

st.write("Tip: enter volatility like `25` for 25% (or leave blank to use market-implied vols).")

# -------------------------
# Run analysis
# -------------------------
if st.button("Analyze"):
    T = time_to_expiry_days / 365.0
    r = risk_free_pct / 100.0

    # parse user vol
    user_vol_val = None
    if user_vol_input is not None and str(user_vol_input).strip() != "":
        try:
            v = float(user_vol_input)
            user_vol_val = v  # preprocess_option_chain handles percent/decimal
        except Exception:
            st.error("Model volatility parse error. Enter a number like 25 for 25%.")
            st.stop()

    with st.spinner("Fetching option chain and computing Greeks..."):
        try:
            df, spot = preprocess_option_chain(
                symbol,
                opt_type=option_type.lower(),
                T=T,
                r=r,
                user_vol=user_vol_val
            )
        except Exception as e:
            st.error(f"Analyzer crashed: {e}")
            st.stop()

    if df is None or df.empty:
        st.error("No option data found. Check symbol, NSE availability, or try a different expiry.")
        st.stop()

    # Basic info
    st.subheader(f"Spot (approx): {spot:.2f}")
    st.markdown(f"Analyzed strikes: {len(df)}  ·  Time to expiry: {time_to_expiry_days} days")

    # Top table
    st.markdown("### Option chain (top rows)")
    display_cols = ["strike", "lastPrice", "BS_price", "implied_vol", "signal"]
    st.dataframe(df[display_cols].head(40))

    # -------------------------
    # Market vs Model Price graph
    # -------------------------
    st.subheader("📈 Market Price vs Model (BS) Price")

    fig = go.Figure()
    # Market price line
    fig.add_trace(go.Scatter(x=df["strike"], y=df["lastPrice"],
                             mode="lines+markers", name="Market Price", line=dict(color="royalblue")))
    # Model BS price line (BS_price)
    fig.add_trace(go.Scatter(x=df["strike"], y=df["BS_price"],
                             mode="lines+markers", name="Model (BS) Price", line=dict(color="darkorange")))

    # Add colored markers per signal, ensure each signal shown once in legend
    color_map = {"Underpriced": "green", "Overpriced": "red", "Fair": "gold", "Unknown": "gray"}
    shown = set()
    for i, row in df.iterrows():
        s = row.get("signal", "Unknown")
        clr = color_map.get(s, "gray")
        showlegend = False
        if s not in shown:
            showlegend = True
            shown.add(s)
        fig.add_trace(go.Scatter(x=[row["strike"]], y=[row["lastPrice"]],
                                 mode="markers",
                                 marker=dict(color=clr, size=12, line=dict(color="black", width=1)),
                                 name=s, showlegend=showlegend))

    fig.update_layout(xaxis_title="Strike", yaxis_title="Price", template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**How to read:**  \n"
        "- **Blue line** = Market price (last traded)  \n"
        "- **Orange line** = Model Black–Scholes price (uses your model vol if provided)  \n"
        "- **Green dots** = Market < Model → Underpriced (buy candidate)  \n"
        "- **Red dots** = Market > Model → Overpriced (avoid buying or consider selling)  \n"
        "- **Gold dots** = Fair (within ±5% of model)."
    )

    # -------------------------
    # Greeks: Model vs Implied
    # -------------------------
    st.subheader("📊 Greeks — Model vs Market-Implied")

    greek_pairs = [
        ("delta_user", "delta_iv", "Delta (Δ)"),
        ("gamma_user", "gamma_iv", "Gamma (Γ)"),
        ("vega_user", "vega_iv", "Vega (ν)"),
        ("theta_user", "theta_iv", "Theta (Θ)"),
        ("rho_user", "rho_iv", "Rho (ρ)")
    ]

    for user_col, iv_col, label in greek_pairs:
        if user_col not in df.columns or iv_col not in df.columns:
            continue

        figg = go.Figure()
        figg.add_trace(go.Scatter(x=df["strike"], y=df[user_col],
                                  mode="lines+markers", name=f"Model {label}", line=dict(color="purple")))
        figg.add_trace(go.Scatter(x=df["strike"], y=df[iv_col],
                                  mode="lines+markers", name=f"Implied {label}", line=dict(color="teal")))
        figg.update_layout(title=f"{label}: Model vs Implied", xaxis_title="Strike", yaxis_title=label, template="plotly_white", height=420)
        st.plotly_chart(figg, use_container_width=True)

        # Short beginner-friendly interpretation
        if "Delta" in label or "delta" in label.lower():
            st.markdown(
                "**Delta (Δ)** — sensitivity of option price to underlying price.  \n"
                "- Closer to 1 (calls) / -1 (puts) → deep ITM; closer to 0 → deep OTM.  \n"
                "- If Market Δ > Model Δ, market expects stronger price sensitivity than your model."
            )
        elif "Gamma" in label:
            st.markdown(
                "**Gamma (Γ)** — rate of change of Delta.  \n"
                "- High Gamma → option's Delta moves quickly with underlying; bigger hedging needs and risk around ATM."
            )
        elif "Vega" in label:
            st.markdown(
                "**Vega (ν)** — sensitivity to volatility changes.  \n"
                "- If Implied Vega >> Model Vega, market places higher value on volatility than your model does."
            )
        elif "Theta" in label:
            st.markdown(
                "**Theta (Θ)** — time decay per day.  \n"
                "- Negative Θ means option loses value each day. Consider time decay when choosing expiry/strike."
            )
        elif "Rho" in label:
            st.markdown(
                "**Rho (ρ)** — sensitivity to interest rates.  \n"
                "- Usually small effect; more relevant for long-dated options."
            )

    # -------------------------
    # Quick recommendations
    # -------------------------
    st.subheader("⚡ Quick Signals & Recommendations")
    buys = df[df["signal"] == "Underpriced"]
    sells = df[df["signal"] == "Overpriced"]
    fair = df[df["signal"] == "Fair"]

    st.markdown(f"- Underpriced (buy candidates): **{len(buys)}** strikes")
    if not buys.empty:
        st.dataframe(buys[["strike", "lastPrice", "BS_price", "implied_vol"]].head(10))

    st.markdown(f"- Overpriced (consider sell / avoid buying): **{len(sells)}** strikes")
    if not sells.empty:
        st.dataframe(sells[["strike", "lastPrice", "BS_price", "implied_vol"]].head(10))

    st.markdown(f"- Fairly priced: **{len(fair)}** strikes")

    # -------------------------
    # Export to Excel
    # -------------------------
    outname = f"{symbol}_{option_type}_analysis.xlsx"
    df.to_excel(outname, index=False)
    st.success(f"Saved analysis to `{outname}` (project root)")

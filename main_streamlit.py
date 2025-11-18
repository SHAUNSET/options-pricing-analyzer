import streamlit as st
import pandas as pd
import plotly.express as px
from src.option_chain_analyzer import preprocess_option_chain

st.set_page_config(page_title="Option Pricing Analyzer", layout="wide")
st.title("📊 Option Pricing Analyzer")

# --- Input ---
symbol = st.text_input("Enter Stock Symbol:", "AAPL").upper()
filter_atm = st.checkbox("Show only ATM options (+/- 2 strikes)")

# --- Fetch data ---
if st.button("Fetch Option Chain"):
    with st.spinner(f"Fetching option chain for {symbol}..."):
        df = preprocess_option_chain(symbol)

        if df is None or df.empty:
            st.error("No option data available.")
        else:
            # --- ATM filter ---
            if filter_atm:
                spot_price = df['BS_price'].iloc[0]  # approximate ATM
                atm_strike = min(df['strike'], key=lambda x: abs(x - spot_price))
                df = df[df['strike'].isin([atm_strike-2.5, atm_strike, atm_strike+2.5])]

            # --- Color-coded signals for dataframe ---
            def highlight_signal(val):
                if val == 'Overpriced':
                    return 'background-color: #ff9999'  # red
                elif val == 'Underpriced':
                    return 'background-color: #99ff99'  # green
                else:
                    return ''
            st.subheader(f"Option Chain Table for {symbol}")
            st.dataframe(df.style.applymap(highlight_signal, subset=['signal']))

            # --- Download Excel ---
            excel_filename = f"{symbol}_option_analysis.xlsx"
            df.to_excel(excel_filename, index=False)
            st.success(f"Saved analysis to {excel_filename}")

            # --- Interactive Plots ---
            st.subheader("📈 Strike vs Prices")
            fig_price = px.scatter(
                df,
                x='strike',
                y=['lastPrice', 'BS_price'],
                color='signal',
                symbol='type',
                title='Market Price vs Black-Scholes Price',
                labels={'value': 'Price', 'strike': 'Strike'}
            )
            st.plotly_chart(fig_price, use_container_width=True)

            st.subheader("📊 Strike vs Greeks")
            greek_cols = ['delta', 'gamma', 'vega', 'theta', 'rho']
            for greek in greek_cols:
                fig_greek = px.scatter(
                    df,
                    x='strike',
                    y=greek,
                    color='signal',
                    symbol='type',
                    title=f'Strike vs {greek.capitalize()}',
                    labels={greek: greek.capitalize(), 'strike': 'Strike'}
                )
                st.plotly_chart(fig_greek, use_container_width=True)

# Options Pricing Analyzer

A Python tool to fetch option chains, compute Black-Scholes prices, Greeks, implied volatility, and generate signals (Fair, Overpriced, Underpriced). Export results to Excel for analysis.

---

## Features

* Fetch latest option chains for any stock symbol using Yahoo Finance
* Compute Black-Scholes option prices (calls & puts)
* Calculate option Greeks: Delta, Gamma, Vega, Theta, Rho
* Compute implied volatility
* Generate trading signals: Fair, Overpriced, Underpriced
* Export analyzed data to Excel

---

## Installation

1. Clone the repository:

```bash
git clone git@github.com:SHAUNSET/options-pricing-analyzer.git
```

2. Navigate to the project folder:

```bash
cd options-pricing-analyzer
```

3. Create and activate a virtual environment:

```bash
python -m venv venv
.\venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the main analysis

```bash
python main.py
```

* Fetches the option chain for the symbol defined in `main.py` (default: AAPL)
* Generates a dataframe with BS prices, Greeks, implied volatility, and signals
* Exports analysis to an Excel file `AAPL_option_analysis.xlsx`

### Optional: Run Streamlit dashboard

```bash
streamlit run main_streamlit.py
```

* Visualize option chains and analytics interactively

---

## Folder Structure

```
options-pricing-analyzer/
├─ src/
│  ├─ data_fetcher.py       # Fetch option chain from Yahoo Finance
│  ├─ black_scholes.py      # Black-Scholes formulas & Greeks
│  ├─ option_chain_analyzer.py # Preprocessing & analytics
│  └─ utils.py              # Helper functions
├─ data/                    # Optional local datasets
├─ main.py                  # Run analysis script
├─ main_streamlit.py        # Streamlit dashboard
├─ requirements.txt         # Python dependencies
└─ README.md
```

---

## Dependencies

* pandas
* numpy
* matplotlib
* yfinance
* scipy
* python-dotenv
* streamlit
* plotly
* scikit-learn

---

## License

MIT License

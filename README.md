# FactorLens: Systematic Portfolio Reconstructor

FactorLens is an institutional-grade tool designed to decompose and replicate the performance of investment funds using systematic factors. It allows users to understand the "Beta" drivers behind a fund's returns and identify whether "Alpha" is genuine or simply a result of factor exposure.

> **Status:** Early Stage / Experimental. This is a "proof of concept" and not a professional investment tool.

## 🚀 How it Works

The application breaks down returns into systematic style exposures through a simple pipeline:

1.  **Metadata Extraction:** Analyzes factsheets to infer strategy and benchmark.
2.  **Institutional Data Fetching:** Automatically retrieves the **Risk-Free Rate (RF)** and historical factor returns (Market, Value, Size, Momentum, Quality, Low-Volatility).
3.  **Advanced Regression Engine:** 
    *   Uses **Excess Returns** (`Return - RF`) to ensure statistically accurate Alpha.
    *   Supports **Monthly Resampling** to reduce daily noise and lead-lag effects.
    *   Runs **Rolling OLS Regression** to track style drift over different market cycles.
4.  **Replicating Portfolio:** Uses **Constrained Optimization** (Minimizing Tracking Error) to build an investable factor-based clone.

## 🌟 Key Features

*   **Alpha Accuracy:** Calculates genuine excess returns by adjusting for the risk-free rate.
*   **Leverage Support:** Can target exposures > 100% to replicate leveraged portfolios (e.g., Berkshire Hathaway style).
*   **Investable Proxies:** Uses specifically selected Factor ETFs (e.g., `QUAL`, `VLUE`, `MTUM`) as proxies for academic factors.
*   **Stability:** Monthly resampling provides more reliable factor loadings than daily data.

## ✅ What it CAN do

*   **Identify Style Drifts:** Visualize how a fund's factor tilts change over time.
*   **Factor Attribution:** Understand which drivers (Size, Value, Quality, etc.) are actually responsible for the returns.
*   **Synthetic Benchmarking:** Compare a manager against a "synthetic clone" rather than a simple market index.

## ❌ Current Limitations

*   **Long-Only Proxies:** While the engine is robust, it uses Long-only ETFs as proxies rather than pure academic Long/Short factors.
*   **Parsing Heuristics:** The PDF parser relies on text patterns; scanned or complex image-only PDFs may not be parsed correctly.

## 🛠️ Tech Stack

*   **Frontend:** Streamlit (Primary) / Flask (Legacy)
*   **Analysis:** Pandas, NumPy, Statsmodels, SciPy
*   **Visualization:** Plotly
*   **Data Sources:** yfinance, Kenneth French Data Library

---
*Disclaimer: This project is for educational purposes only. Past performance is no guarantee of future results.*

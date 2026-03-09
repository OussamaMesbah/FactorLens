import streamlit as st
import pandas as pd
import plotly.io as pio
from factor_engine import run_full_analysis, ASSET_CLASS_MAP
from pdf_parser import parse_factsheet
import tempfile
import os

st.set_page_config(page_title="FactorLens", layout="wide")

st.title("🚀 FactorLens")
st.markdown("""
Decompose any fund's performance into systematic factors (Value, Quality, Momentum, etc.) 
using institutional-grade regression techniques.
""")

with st.sidebar:
    st.header("Settings")
    
    # 1. File Upload
    uploaded_file = st.file_uploader("Upload Fund Factsheet (PDF)", type="pdf")
    
    # 2. Manual Ticker Input (Overrides PDF)
    ticker = st.text_input("Fund Ticker (e.g. BRK-B, ARKK, VOO)", "").upper()
    
    asset_class = st.selectbox("Asset Class", options=list(ASSET_CLASS_MAP.keys()), index=0)
    
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
    end_date = col2.date_input("End Date", value=pd.to_datetime("2024-12-31"))
    
    max_leverage = st.slider("Max Leverage", 1.0, 3.0, 1.0, 0.1)
    resample_freq = st.selectbox("Regression Frequency", options=["Monthly (Recommended)", "Daily"], index=0)
    freq_code = "ME" if "Monthly" in resample_freq else None

# --- Analysis Logic ---

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    with st.expander("📄 Metadata Extracted", expanded=False):
        metadata = parse_factsheet(tmp_path)
        st.json(metadata)
        if not ticker and "isins" in metadata and metadata["isins"]:
            st.info(f"Suggested ISIN found: {metadata['isins'][0]}. Please enter a ticker below if needed.")
    
    os.unlink(tmp_path) 

if st.button("Run Factor Analysis"):
    if not ticker:
        st.error("Please provide a ticker symbol.")
    else:
        with st.spinner(f"Fetching data and regressing {ticker}..."):
            try:
                results = run_full_analysis(
                    fund_ticker=ticker,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    asset_class=asset_class,
                    resample_freq=freq_code,
                    max_leverage=max_leverage
                )
                
                # --- Helper to parse JSON charts ---
                def get_fig(json_str):
                    return pio.from_json(json_str)

                # --- Display Results ---
                
                # 1. Top Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Ann. Return", f"{results['fund_metrics']['ann_return']}%")
                m2.metric("Sharpe Ratio", results['fund_metrics']['sharpe'])
                m3.metric("R-Squared", f"{results['full_ols']['r_squared']*100:.1f}%")
                m4.metric("Alpha (Ann.)", f"{results['full_ols']['alpha']}%")
                
                # 2. Charts
                st.subheader("Performance vs. Factor Clone")
                st.plotly_chart(get_fig(results['charts']['performance']), use_container_width=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Factor Exposures (Betas)")
                    st.plotly_chart(get_fig(results['charts']['factor_betas']), use_container_width=True)
                with c2:
                    st.subheader("Allocations Over Time")
                    st.plotly_chart(get_fig(results['charts']['factor_weights']), use_container_width=True)
                
                st.subheader("Risk Analysis")
                st.plotly_chart(get_fig(results['charts']['drawdown']), use_container_width=True)
                
                # 3. Tables
                with st.expander("View Regression Statistics"):
                    st.table(pd.DataFrame({
                        "Beta": results['full_ols']['betas'],
                        "T-Stat": results['full_ols']['t_stats'],
                        "P-Value": results['full_ols']['p_values']
                    }))
                    
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)
else:
    st.info("Upload a factsheet or enter a ticker in the sidebar to begin.")

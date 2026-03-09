import streamlit as st
import pandas as pd
import plotly.io as pio
from factor_engine import run_full_analysis, ASSET_CLASS_MAP
from pdf_parser import parse_factsheet
import tempfile
import os

st.set_page_config(page_title="FactorLens", layout="wide")

# --- Session State ---
if "ticker" not in st.session_state:
    st.session_state.ticker = ""

def set_ticker(t):
    st.session_state.ticker = t

st.title("🚀 FactorLens")
st.markdown("""
Decompose any fund's performance into systematic factors (Value, Quality, Momentum, etc.) 
using institutional-grade regression techniques.
""")

with st.sidebar:
    st.header("1. Data Input")
    
    # Explain PDF usage
    with st.expander("ℹ️ About Factsheet Upload", expanded=False):
        st.info("""
        **What it does:** Extracts the fund name, ISIN, and potential tickers to help you set up the analysis.
        
        **What it DOES NOT do:** It doesn't extract return data from the PDF tables. We use the **Ticker** to fetch clean, daily data from Yahoo Finance.
        """)
    
    uploaded_file = st.file_uploader("Upload Fund Factsheet (PDF)", type="pdf")
    
    # 2. Manual Ticker Input (Overrides PDF)
    ticker_input = st.text_input("Fund Ticker (Yahoo Finance)", value=st.session_state.ticker, key="ticker_input").upper()
    st.session_state.ticker = ticker_input
    
    asset_class = st.selectbox("Asset Class", options=list(ASSET_CLASS_MAP.keys()), index=0)
    
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
    end_date = col2.date_input("End Date", value=pd.to_datetime("2024-12-31"))
    
    st.markdown("---")
    st.subheader("2. Model Parameters")
    max_leverage = st.slider("Max Leverage", 1.0, 3.0, 1.0, 0.1, help="Target total exposure. Legends like Buffett often use ~1.7x.")
    long_only = st.checkbox("Long-Only Constraints", value=True, help="If unchecked, allows shorting factors to match fund profile (e.g. 'Junk' bets).")
    resample_freq = st.selectbox("Regression Frequency", options=["Monthly (Recommended)", "Daily"], index=0)
    freq_code = "ME" if "Monthly" in resample_freq else None

# --- Analysis Logic ---

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    with st.expander("📄 PDF Metadata Extracted", expanded=True):
        metadata = parse_factsheet(tmp_path)
        
        c1, c2 = st.columns(2)
        c1.write(f"**Fund:** {metadata.get('fund_name', 'Unknown')}")
        c1.write(f"**Asset Class:** {metadata.get('asset_class', 'Unknown')}")
        
        # Suggested Ticker logic
        found_ticker = metadata.get('ticker_hint', "")
        found_isins = metadata.get('isins', [])
        
        if found_ticker:
            c2.success(f"**Suggested Ticker:** `{found_ticker}`")
            if c2.button(f"Use {found_ticker}"):
                st.session_state.ticker = found_ticker
                st.rerun()
        elif found_isins:
            c2.info(f"**ISIN found:** `{found_isins[0]}`")
            st.caption("No clear ticker found. Please search for the Yahoo Finance ticker using the ISIN.")
        else:
            c2.warning("No ticker or ISIN detected. Please enter a ticker manually.")

        with st.expander("Raw Metadata"):
            st.json(metadata)
    
    os.unlink(tmp_path) 

if st.button("Run Factor Analysis"):
    ticker_to_run = st.session_state.ticker
    if not ticker_to_run:
        st.error("Please provide a ticker symbol in the sidebar (e.g. ARKK, BRK-B).")
    else:
        with st.spinner(f"Fetching data and regressing {ticker_to_run}..."):
            try:
                results = run_full_analysis(
                    fund_ticker=ticker_to_run,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    asset_class=asset_class,
                    resample_freq=freq_code,
                    max_leverage=max_leverage,
                    long_only=long_only
                )
                
                def get_fig(json_str):
                    return pio.from_json(json_str)

                # --- Display Results ---
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Ann. Return", f"{results['fund_metrics']['ann_return']}%")
                m2.metric("Sharpe Ratio", results['fund_metrics']['sharpe'])
                
                r2_val = results['full_ols']['r_squared']
                m3.metric("R-Squared (Fit)", f"{r2_val*100:.1f}%")
                m4.metric("Alpha (Ann.)", f"{results['full_ols']['alpha']}%")
                
                if r2_val > 0.8:
                    st.success(f"**High Fit ({r2_val*100:.1f}%)**: This fund's returns are largely driven by systematic style exposures.")
                elif r2_val > 0.5:
                    st.info(f"**Moderate Fit ({r2_val*100:.1f}%)**: Factors explain a significant portion, but the manager has unique bets.")
                else:
                    st.warning(f"**Low Fit ({r2_val*100:.1f}%)**: High idiosyncratic risk. The factors don't explain this manager well.")

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
                
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    with st.expander("Detailed Regression Statistics"):
                        st.table(pd.DataFrame({
                            "Beta": results['full_ols']['betas'],
                            "T-Stat": results['full_ols']['t_stats'],
                            "P-Value": results['full_ols']['p_values']
                        }))
                
                with col_b:
                    with st.expander("Methodology & Limitations"):
                        st.markdown(f"""
                        - **Frequency:** {results['methodology']['frequency']}
                        - **Constraint:** {"Long-Only" if long_only else "Long/Short"}
                        - **Leverage:** {max_leverage}x
                        
                        **Why doesn't it follow the curve perfectly?**
                        1. **Concentration:** Stock-specific risk in concentrated funds.
                        2. **ETF Proxies:** Investable ETFs are not "pure" academic factors.
                        3. **Selection Alpha:** True manager skill or idiosyncratic luck.
                        """)
                    
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)
else:
    if not st.session_state.ticker:
        st.info("Upload a factsheet to extract details or enter a Yahoo Finance ticker manually.")

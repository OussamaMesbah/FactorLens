"""
factor_engine.py
----------------
Downloads factor & ETF data, runs rolling OLS regression against fund returns,
builds the investable replicating portfolio, and generates Plotly chart JSON.
"""

from __future__ import annotations
import io, json, zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import statsmodels.api as sm
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Factor ETF universe
# ---------------------------------------------------------------------------

EQUITY_ETFS: dict[str, dict] = {
    "Market": {
        "ticker": "SPY",
        "name": "S&P 500 (Market)",
        "color": "#4C9BE8"
    },
    "Value": {
        "ticker": "VLUE",
        "name": "MSCI Value Factor",
        "color": "#F4B942"
    },
    "Size": {
        "ticker": "IWM",
        "name": "Russell 2000 (Small Cap)",
        "color": "#E85D75"
    },
    "Momentum": {
        "ticker": "MTUM",
        "name": "MSCI Momentum Factor",
        "color": "#8AC48E"
    },
    "Quality": {
        "ticker": "QUAL",
        "name": "MSCI Quality Factor",
        "color": "#B47AEA"
    },
    "LowVol": {
        "ticker": "USMV",
        "name": "MSCI Min Volatility",
        "color": "#5DD4C8"
    },
}

BOND_ETFS: dict[str, dict] = {
    "CoreBond": {
        "ticker": "AGG",
        "name": "US Aggregate Bond",
        "color": "#4C9BE8"
    },
    "InvGrade": {
        "ticker": "LQD",
        "name": "Investment Grade Corp",
        "color": "#F4B942"
    },
    "HighYield": {
        "ticker": "HYG",
        "name": "High Yield Bond",
        "color": "#E85D75"
    },
    "Treasury": {
        "ticker": "IEF",
        "name": "7-10yr Treasury",
        "color": "#8AC48E"
    },
    "TIPS": {
        "ticker": "TIP",
        "name": "TIPS (Inflation Linked)",
        "color": "#B47AEA"
    },
    "ShortBond": {
        "ticker": "SHY",
        "name": "1-3yr Treasury",
        "color": "#5DD4C8"
    },
}

MIXED_ETFS: dict[str, dict] = {
    **{
        k: v
        for k, v in EQUITY_ETFS.items()
    },
    "CoreBond": {
        "ticker": "AGG",
        "name": "US Aggregate Bond",
        "color": "#FF9F40"
    },
    "InvGrade": {
        "ticker": "LQD",
        "name": "Investment Grade Corp",
        "color": "#FF6384"
    },
    "HighYield": {
        "ticker": "HYG",
        "name": "High Yield Bond",
        "color": "#36A2EB"
    },
}

GLOBAL_EQUITY_ETFS: dict[str, dict] = {
    "US Market": {
        "ticker": "SPY",
        "name": "US S&P 500",
        "color": "#4C9BE8"
    },
    "Intl DM": {
        "ticker": "EFA",
        "name": "Intl Developed Markets",
        "color": "#F4B942"
    },
    "Emerging": {
        "ticker": "EEM",
        "name": "Emerging Markets",
        "color": "#E85D75"
    },
    "Value": {
        "ticker": "VLUE",
        "name": "MSCI Value Factor",
        "color": "#8AC48E"
    },
    "Momentum": {
        "ticker": "MTUM",
        "name": "MSCI Momentum Factor",
        "color": "#B47AEA"
    },
    "Quality": {
        "ticker": "QUAL",
        "name": "MSCI Quality Factor",
        "color": "#5DD4C8"
    },
    "LowVol": {
        "ticker": "USMV",
        "name": "MSCI Min Volatility",
        "color": "#FF9F40"
    },
}

ASSET_CLASS_MAP = {
    "equity": EQUITY_ETFS,
    "bond": BOND_ETFS,
    "fixed income": BOND_ETFS,
    "mixed": MIXED_ETFS,
    "balanced": MIXED_ETFS,
    "global equity": GLOBAL_EQUITY_ETFS,
    "global": GLOBAL_EQUITY_ETFS,
}

BENCHMARK_BY_CLASS = {
    "equity": "SPY",
    "bond": "AGG",
    "fixed income": "AGG",
    "mixed": "AOM",
    "balanced": "AOM",
    "global equity": "ACWI",
    "global": "ACWI",
}

# ---------------------------------------------------------------------------
# Factor descriptions
# ---------------------------------------------------------------------------

FACTOR_DESCRIPTIONS = {
    "Market": "Broad equity market exposure (market beta).",
    "Value":
    "Long cheap / short expensive stocks based on fundamentals (book-to-price, earnings yield).",
    "Size": "Long small-cap / short large-cap stocks — the small-cap premium.",
    "Momentum":
    "Long recent winners / short recent losers — trend-following within equities.",
    "Quality":
    "Long profitable, low-leverage companies / short unprofitable, high-leverage ones.",
    "LowVol":
    "Long low-volatility stocks — Betting Against Beta (BAB); exploits low-risk anomaly.",
    "CoreBond": "Duration exposure to investment-grade US bonds.",
    "InvGrade": "Credit spread on investment-grade corporate bonds.",
    "HighYield":
    "High-yield (junk) credit spread — compensation for default risk.",
    "Treasury": "Intermediate Treasury rate duration.",
    "TIPS": "Real rate duration plus inflation premium.",
    "ShortBond": "Short-duration (1-3 yr) Treasury exposure.",
    "Intl DM": "International developed-market equity beta.",
    "Emerging": "Emerging-market equity beta.",
    "US Market": "US equity market (S&P 500) beta.",
}

METHODOLOGY_CONTEXT: dict[str, str] = {
    "Market":
    "The primary driver of equity returns. Most legendary funds rely on a core market beta.",
    "Value":
    "The classic strategy of buying quality assets when they are undervalued by the market.",
    "Size":
    "Harvesting the premium associated with smaller, less liquid companies.",
    "Momentum":
    "Capturing the tendency of recent winners to continue outperforming.",
    "Quality":
    "Investing in companies with durable business models, high profitability, and low debt.",
    "LowVol":
    "Exploiting the low-risk anomaly where low-volatility stocks offer superior risk-adjusted returns.",
    "CoreBond":
    "Duration and interest rate sensitivity, typically the main driver of fixed income funds.",
    "InvGrade":
    "Harvesting credit spreads from investment-grade corporate debt.",
    "HighYield":
    "Capturing the default risk premium from non-investment grade (junk) bonds.",
}

# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------


def _download_ff_factors_daily() -> pd.DataFrame:
    """Download Fama-French 3-factor + Momentum daily data from Kenneth French's website."""
    urls = [
        ("FF3",
         "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
         ),
        ("MOM",
         "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
         ),
    ]
    frames = {}
    for label, url in urls:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                fname = [
                    n for n in zf.namelist()
                    if n.endswith(".CSV") or n.endswith(".csv")
                ][0]
                with zf.open(fname) as f:
                    raw = f.read().decode("latin-1")
            # Skip header lines before the data starts
            lines = raw.split("\n")
            data_start = 0
            for i, line in enumerate(lines):
                stripped = line.strip().replace(",", "")
                if stripped.lstrip("-").isdigit() and len(stripped) == 8:
                    data_start = i
                    break
            csv_text = "\n".join(lines[data_start:])
            df = pd.read_csv(io.StringIO(csv_text), header=0, index_col=0)
            df.index = pd.to_datetime(df.index,
                                      format="%Y%m%d",
                                      errors="coerce")
            df = df[df.index.notna()]
            df = df.apply(pd.to_numeric,
                          errors="coerce") / 100  # convert % to decimal
            frames[label] = df
        except Exception as e:
            print(f"Warning: Could not download {label} factors: {e}")

    if "FF3" in frames and "MOM" in frames:
        result = frames["FF3"].join(frames["MOM"], how="inner")
        # Standardize column names
        rename = {}
        for col in result.columns:
            cu = col.strip().upper()
            if "MKT" in cu or "MKT-RF" in cu:
                rename[col] = "Mkt-RF"
            elif "SMB" in cu:
                rename[col] = "SMB"
            elif "HML" in cu:
                rename[col] = "HML"
            elif "RF" == cu:
                rename[col] = "RF"
            elif "MOM" in cu or "WML" in cu or "UMD" in cu:
                rename[col] = "Mom"
        result = result.rename(columns=rename)
        return result
    elif "FF3" in frames:
        return frames["FF3"]
    return pd.DataFrame()


def fetch_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices for a list of tickers from yfinance."""
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(tickers,
                          start=start,
                          end=end,
                          auto_adjust=True,
                          progress=False,
                          threads=True)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]] if "Close" in raw.columns else raw
        prices.index = pd.to_datetime(prices.index)
        prices = prices.dropna(how="all")
        return prices
    except Exception as e:
        print(f"Error fetching prices for {tickers}: {e}")
        return pd.DataFrame()


def fetch_fund_returns(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch daily returns for a fund ticker."""
    prices = fetch_price_data([ticker], start, end)
    if prices.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'. "
                         "Check the ticker symbol and try again.")
    col = ticker if ticker in prices.columns else prices.columns[0]
    series = prices[col].dropna()
    if len(series) < 60:
        raise ValueError(f"Less than 60 trading days of data for '{ticker}'. "
                         "Try a longer date range.")
    returns = series.pct_change().dropna()
    returns.name = ticker
    return returns


# ---------------------------------------------------------------------------
# Regression & portfolio construction
# ---------------------------------------------------------------------------


def _align(fund: pd.Series,
           factors: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Align fund returns and factor returns on the same dates."""
    common = fund.index.intersection(factors.index)
    return fund.loc[common], factors.loc[common]


def run_ols(fund_returns: pd.Series,
            factor_returns: pd.DataFrame,
            add_intercept: bool = True,
            annualization: int = 252) -> dict:
    """Run OLS regression of fund returns on factor returns."""
    y, X = _align(fund_returns, factor_returns)
    if add_intercept:
        X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X, missing="drop").fit()
    result = {
        "alpha": model.params.get("const", 0.0) * annualization,  # annualised
        "betas": {
            k: v
            for k, v in model.params.items() if k != "const"
        },
        "t_stats": {
            k: v
            for k, v in model.tvalues.items() if k != "const"
        },
        "p_values": {
            k: v
            for k, v in model.pvalues.items() if k != "const"
        },
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n_obs": int(model.nobs),
        "residuals": model.resid,
        "fitted": model.fittedvalues,
    }
    return result


def constrained_weights(fund_returns: pd.Series,
                        factor_returns: pd.DataFrame,
                        max_leverage: float = 1.0) -> dict[str, float]:
    """
    Minimise tracking error between fund and replicating portfolio.
    Weights are constrained to [0, max_leverage] and sum to total allocation.
    We allow sum to be <= max_leverage.
    """
    y, X = _align(fund_returns, factor_returns)
    X_np = X.values
    y_np = y.values
    n_factors = X_np.shape[1]

    def tracking_error(w):
        repl = X_np @ w
        return np.sum((y_np - repl)**2)

    # Constraint: sum(weights) = max_leverage
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - max_leverage}]
    bounds = [(0.0, max_leverage)] * n_factors
    w0 = np.ones(n_factors) * (max_leverage / n_factors)

    res = minimize(tracking_error,
                   w0,
                   method="SLSQP",
                   bounds=bounds,
                   constraints=constraints,
                   options={
                       "ftol": 1e-9,
                       "maxiter": 1000
                   })
    if res.success:
        return dict(zip(factor_returns.columns, res.x))
    # Fallback: normalise positive OLS betas
    ols = run_ols(fund_returns, factor_returns, add_intercept=False)
    betas = np.array(
        [ols["betas"].get(c, 0.0) for c in factor_returns.columns])
    betas = np.clip(betas, 0, None)
    total = betas.sum()
    if total > 0:
        betas = betas * (max_leverage / total)
    return dict(zip(factor_returns.columns, betas))


def resample_returns(returns: pd.DataFrame | pd.Series,
                     freq: str = "ME") -> pd.DataFrame | pd.Series:
    """Resample daily returns to a lower frequency (e.g. Monthly)."""
    if returns.empty:
        return returns
    # Compound returns: (1+r1)(1+r2)... - 1
    return (1 + returns).resample(freq).prod() - 1


def rolling_regression(fund_returns: pd.Series,
                       factor_etf_returns: pd.DataFrame,
                       window_days: int = 252,
                       rebalance_freq: str = "QE",
                       max_leverage: float = 1.0) -> pd.DataFrame:
    """
    Compute constrained portfolio weights at each rebalancing date using a
    rolling window of `window_days` prior daily returns.
    """
    # Align
    common = fund_returns.index.intersection(factor_etf_returns.index)
    fund = fund_returns.loc[common]
    factors = factor_etf_returns.loc[common]

    # Rebalancing dates
    rebal_dates = pd.date_range(start=fund.index[window_days],
                                end=fund.index[-1],
                                freq=rebalance_freq)
    # Snap to actual trading days
    rebal_dates = pd.Index([
        fund.index[fund.index <= d][-1]
        if len(fund.index[fund.index <= d]) > 0 else None for d in rebal_dates
    ]).dropna()

    rows = []
    for rebal_date in rebal_dates:
        pos = fund.index.get_loc(rebal_date)
        start_pos = max(0, pos - window_days + 1)
        y_window = fund.iloc[start_pos:pos + 1]
        X_window = factors.iloc[start_pos:pos + 1]
        if len(y_window) < 60:
            continue
        try:
            w = constrained_weights(y_window, X_window, max_leverage=max_leverage)
            w["date"] = rebal_date
            rows.append(w)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("date")
    return df


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------


def simulate_portfolio(weights_history: pd.DataFrame,
                       etf_prices: pd.DataFrame,
                       initial_capital: float = 100.0) -> pd.Series:
    """
    Simulate replicating portfolio daily NAV given quarterly weight history
    and daily ETF price data.
    """
    if weights_history.empty:
        return pd.Series(dtype=float)

    # Daily ETF returns
    etf_returns = etf_prices.pct_change().dropna()
    factors = [c for c in weights_history.columns if c in etf_returns.columns]
    etf_returns = etf_returns[factors]
    weights_history = weights_history[factors]

    # Align
    all_dates = etf_returns.index
    rebal_dates = weights_history.index

    nav = initial_capital
    nav_series = {}
    current_weights = weights_history.iloc[0]

    for date in all_dates:
        if date in rebal_dates:
            current_weights = weights_history.loc[date]
        day_ret = (etf_returns.loc[date] * current_weights).sum()
        nav *= (1 + day_ret)
        nav_series[date] = nav

    return pd.Series(nav_series)


def compute_rebalancing_events(weights_history: pd.DataFrame,
                               etf_info: dict) -> list[dict]:
    """
    For each rebalancing date, compute which ETFs were bought, sold, or held.
    Returns a list of events sorted by date.
    """
    if weights_history.empty or len(weights_history) < 2:
        return []

    events = []
    prev_weights = None
    for date, row in weights_history.iterrows():
        current = row.to_dict()
        event = {
            "date": date.strftime("%Y-%m-%d"),
            "holdings": [],
        }
        for factor, weight in current.items():
            info = etf_info.get(factor, {"ticker": factor, "name": factor})
            ticker = info["ticker"]
            prev_w = prev_weights.get(factor, 0.0) if prev_weights else 0.0
            delta = weight - prev_w
            threshold = 0.005  # 0.5% change threshold
            if prev_weights is None:
                action = "BUY" if weight > threshold else "SKIP"
            elif delta > threshold:
                action = "BUY"
            elif delta < -threshold:
                action = "SELL"
            else:
                action = "HOLD"

            if action != "SKIP" and (weight > 0.005 or action in ("SELL", )):
                event["holdings"].append({
                    "factor": factor,
                    "ticker": ticker,
                    "name": info["name"],
                    "weight": round(weight * 100, 2),
                    "prev_weight": round(prev_w * 100, 2),
                    "delta": round(delta * 100, 2),
                    "action": action,
                })
        events.append(event)
        prev_weights = current
    return events


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def performance_metrics(returns: pd.Series, rf_annual: float = 0.04) -> dict:
    """Compute annualised return, vol, Sharpe, max drawdown."""
    if returns is None or len(returns) < 2:
        return {}
    rf_daily = (1 + rf_annual)**(1 / 252) - 1
    excess = returns - rf_daily
    ann_ret = (1 + returns).prod()**(252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = excess.mean() / returns.std() * np.sqrt(252) if returns.std(
    ) > 0 else 0
    nav = (1 + returns).cumprod()
    drawdown = (nav / nav.cummax() - 1)
    max_dd = drawdown.min()
    return {
        "ann_return": round(ann_ret * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd * 100, 2),
        "total_return": round(((1 + returns).prod() - 1) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0F1117",
    plot_bgcolor="#0F1117",
    font=dict(family="Inter, sans-serif", color="#E0E0E0"),
    margin=dict(l=50, r=30, t=60, b=50),
    legend=dict(bgcolor="rgba(255,255,255,0.05)",
                bordercolor="#333",
                borderwidth=1,
                font=dict(size=11)),
    hovermode="x unified",
)


def chart_performance(fund_nav: pd.Series, repl_nav: pd.Series,
                      bench_nav: pd.Series, fund_name: str,
                      bench_name: str) -> str:
    fig = go.Figure()
    # Normalise all to 100
    for nav, label, color, dash in [
        (fund_nav, fund_name, "#4C9BE8", "solid"),
        (repl_nav, "Replicating Portfolio", "#F4B942", "dash"),
        (bench_nav, bench_name, "#888888", "dot"),
    ]:
        if nav is not None and len(nav) > 0:
            n = nav / nav.iloc[0] * 100
            fig.add_trace(
                go.Scatter(
                    x=n.index,
                    y=n.values,
                    name=label,
                    line=dict(color=color, dash=dash, width=2),
                    hovertemplate="%{y:.1f}<extra>" + label + "</extra>",
                ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="Cumulative Performance (Base = 100)",
                      yaxis_title="NAV (rebased 100)",
                      xaxis_title="")
    return fig.to_json()


def chart_factor_weights(weights_history: pd.DataFrame, etf_info: dict) -> str:
    """Stacked area chart of factor weights over time."""
    if weights_history.empty:
        return "{}"
    fig = go.Figure()
    for factor in weights_history.columns:
        info = etf_info.get(factor, {"name": factor, "color": "#888"})
        fig.add_trace(
            go.Scatter(
                x=weights_history.index,
                y=(weights_history[factor] * 100).round(2),
                name=f"{factor} ({info['ticker']})"
                if "ticker" in info else factor,
                stackgroup="one",
                line=dict(color=info.get("color", "#888"), width=0.5),
                fillcolor=info.get("color", "#888"),
                hovertemplate="%{y:.1f}%<extra>" + factor + "</extra>",
            ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="Factor Weights Over Time (%)",
                      yaxis=dict(title="Allocation (%)", range=[0, 105]),
                      xaxis_title="")
    return fig.to_json()


def chart_factor_betas(full_ols: dict, etf_info: dict) -> str:
    """Horizontal bar chart of full-sample factor betas."""
    betas = {k: v for k, v in full_ols.get("betas", {}).items()}
    if not betas:
        return "{}"
    factors = list(betas.keys())
    values = [betas[f] for f in factors]
    colors = [etf_info.get(f, {}).get("color", "#888") for f in factors]
    descriptions = [FACTOR_DESCRIPTIONS.get(f, "") for f in factors]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=factors,
            orientation="h",
            marker_color=colors,
            customdata=descriptions,
            hovertemplate=("<b>%{y}</b><br>Beta: %{x:.3f}"
                           "<br>%{customdata}<extra></extra>"),
        ))
    fig.add_vline(x=0, line_color="#555", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="Factor Exposures (Full-Sample OLS Betas)",
                      xaxis_title="Beta Coefficient",
                      yaxis_title="")
    return fig.to_json()


def chart_drawdown(fund_ret: pd.Series, repl_ret: pd.Series,
                   fund_name: str) -> str:
    """Drawdown chart."""
    fig = go.Figure()
    for ret, label, color in [
        (fund_ret, fund_name, "#4C9BE8"),
        (ret_p, "Replicating Portfolio", "#F4B942") if (ret_p := repl_ret) is not None else (None, "", ""),
    ]:
        if ret is not None and len(ret) > 0:
            nav = (1 + ret).cumprod()
            dd = (nav / nav.cummax() - 1) * 100
            fig.add_trace(
                go.Scatter(
                    x=dd.index,
                    y=dd.values,
                    name=label,
                    fill="tozeroy",
                    line=dict(color=color, width=1.5),
                    hovertemplate="%{y:.1f}%<extra>" + label + "</extra>",
                ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="Drawdown (%)",
                      yaxis_title="Drawdown from Peak (%)",
                      xaxis_title="")
    return fig.to_json()


def chart_rolling_corr(fund_ret: pd.Series, repl_ret: pd.Series,
                       fund_name: str) -> str:
    """36-month rolling correlation between fund and replicating portfolio."""
    common = fund_ret.index.intersection(repl_ret.index)
    if len(common) < 252:
        return "{}"
    combined = pd.DataFrame({
        "fund": fund_ret.loc[common],
        "repl": repl_ret.loc[common]
    })
    rolling_corr = combined["fund"].rolling(252).corr(
        combined["repl"]).dropna()
    fig = go.Figure(
        go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr.values,
            fill="tozeroy",
            line=dict(color="#8AC48E", width=2),
            hovertemplate="Corr: %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=
        f"252-Day Rolling Correlation: {fund_name} vs. Replicating Portfolio",
        yaxis=dict(title="Correlation", range=[-0.2, 1.1]),
        xaxis_title="")
    return fig.to_json()


# ---------------------------------------------------------------------------
# Master analysis function
# ---------------------------------------------------------------------------


def run_full_analysis(fund_ticker: str,
                      start: str,
                      end: str,
                      asset_class: str = "equity",
                      fund_name: str = None,
                      rebal_freq: str = "QE",
                      window_days: int = 252,
                      resample_freq: str = "ME",
                      max_leverage: float = 1.0) -> dict:
    """
    Run the complete factor replication analysis.
    Returns a dict holding all results, metrics, and chart JSON.
    """
    if fund_name is None:
        fund_name = fund_ticker.upper()

    # Pick factor ETF universe
    ac = asset_class.lower()
    etf_info = ASSET_CLASS_MAP.get(ac, EQUITY_ETFS)
    bench_ticker = BENCHMARK_BY_CLASS.get(ac, "SPY")

    # ---- 1. Fetch data ----
    fund_returns = fetch_fund_returns(fund_ticker, start, end)
    ff_data = _download_ff_factors_daily()
    
    all_tickers = list({v["ticker"]
                        for v in etf_info.values()}
                       | {bench_ticker, fund_ticker})
    prices = fetch_price_data(all_tickers, start, end)

    if prices.empty:
        raise ValueError(
            "Could not download price data. Check your internet connection.")

    factor_tickers = [v["ticker"] for v in etf_info.values()]
    factor_tickers = [t for t in factor_tickers if t in prices.columns]
    factor_prices = prices[factor_tickers]
    factor_returns = factor_prices.pct_change().dropna()

    # Create a reverse map: ticker -> factor name
    ticker_to_factor = {v["ticker"]: k for k, v in etf_info.items()}
    factor_returns.columns = [
        ticker_to_factor.get(c, c) for c in factor_returns.columns
    ]

    bench_returns = prices[bench_ticker].pct_change().dropna(
    ) if bench_ticker in prices.columns else None

    # ---- 2. Align & Calculate Excess Returns ----
    # Join everything to ensure same dates
    all_rets = pd.DataFrame({"fund": fund_returns}).join(factor_returns, how="inner")
    if bench_returns is not None:
        all_rets = all_rets.join(pd.Series(bench_returns, name="bench"), how="left")
    
    if not ff_data.empty and "RF" in ff_data.columns:
        all_rets = all_rets.join(ff_data["RF"], how="inner")
        rf = all_rets["RF"]
        # Calculate excess returns
        for col in all_rets.columns:
            if col != "RF":
                all_rets[col] = all_rets[col] - rf
    else:
        # Fallback if FF data fails
        all_rets["RF"] = 0.0

    # ---- 3. Optional Resampling (Monthly for stability) ----
    if resample_freq:
        # Regression data (X and y)
        y_reg = resample_returns(all_rets["fund"], freq=resample_freq)
        X_reg = resample_returns(all_rets.drop(columns=["fund", "RF", "bench"], errors="ignore"), freq=resample_freq)
        ann_factor = 12 if "M" in resample_freq.upper() else 252
    else:
        y_reg = all_rets["fund"]
        X_reg = all_rets.drop(columns=["fund", "RF", "bench"], errors="ignore")
        ann_factor = 252

    # ---- 4. Full-sample OLS regression ----
    full_ols = run_ols(y_reg, X_reg, annualization=ann_factor)

    # ---- 5. Rolling regression / rebalancing ----
    weights_history = rolling_regression(fund_returns,
                                         factor_returns,
                                         window_days=window_days,
                                         rebalance_freq=rebal_freq,
                                         max_leverage=max_leverage)

    # ---- 6. Simulate replicating portfolio ----
    # Map factor names back to tickers for simulate_portfolio
    factor_prices_renamed = factor_prices.copy()
    factor_prices_renamed.columns = [
        ticker_to_factor.get(c, c) for c in factor_prices_renamed.columns
    ]

    repl_nav = simulate_portfolio(weights_history, factor_prices_renamed)

    # ---- 7. Build return series ----
    fund_nav_series = (1 + fund_returns).cumprod() * 100
    repl_returns = repl_nav.pct_change().dropna(
    ) if not repl_nav.empty else pd.Series(dtype=float)
    bench_nav_series = (1 + bench_returns).cumprod(
    ) * 100 if bench_returns is not None else None

    # ---- 8. Performance metrics ----
    fund_metrics = performance_metrics(fund_returns)
    repl_metrics = performance_metrics(repl_returns)
    bench_metrics = performance_metrics(
        bench_returns) if bench_returns is not None else {}

    # ---- 9. Rebalancing events ----
    rebal_events = compute_rebalancing_events(weights_history, etf_info)

    # ---- 10. Build charts ----
    charts = {
        "performance":
        chart_performance(fund_nav_series, repl_nav, bench_nav_series,
                          fund_name, bench_ticker),
        "factor_weights":
        chart_factor_weights(weights_history, etf_info),
        "factor_betas":
        chart_factor_betas(full_ols, etf_info),
        "drawdown":
        chart_drawdown(fund_returns, repl_returns, fund_name),
        "rolling_corr":
        chart_rolling_corr(fund_returns, repl_returns, fund_name),
    }

    # ---- 11. Latest holdings ----
    latest_weights = weights_history.iloc[-1].to_dict(
    ) if not weights_history.empty else {}
    latest_holdings = [{
        "factor": k,
        "ticker": etf_info.get(k, {}).get("ticker", k),
        "name": etf_info.get(k, {}).get("name", k),
        "weight": round(v * 100, 2),
        "description": FACTOR_DESCRIPTIONS.get(k, ""),
        "methodology_context": METHODOLOGY_CONTEXT.get(k, ""),
    } for k, v in sorted(latest_weights.items(), key=lambda x: -x[1])
                       if v > 0.005]

    return {
        "fund_name": fund_name,
        "fund_ticker": fund_ticker,
        "bench_ticker": bench_ticker,
        "asset_class": asset_class,
        "period": {
            "start": fund_returns.index[0].strftime("%Y-%m-%d"),
            "end": fund_returns.index[-1].strftime("%Y-%m-%d")
        },
        "n_trading_days": len(fund_returns),
        "fund_metrics": fund_metrics,
        "repl_metrics": repl_metrics,
        "bench_metrics": bench_metrics,
        "full_ols": {
            "alpha": round(full_ols["alpha"] * 100, 3),
            "r_squared": round(full_ols["r_squared"], 4),
            "adj_r_squared": round(full_ols["adj_r_squared"], 4),
            "betas": {
                k: round(v, 4)
                for k, v in full_ols["betas"].items()
            },
            "t_stats": {
                k: round(v, 3)
                for k, v in full_ols["t_stats"].items()
            },
            "p_values": {
                k: round(v, 4)
                for k, v in full_ols["p_values"].items()
            },
        },
        "latest_holdings": latest_holdings,
        "rebalancing_events": rebal_events,
        "n_rebalances": len(rebal_events),
        "charts": charts,
        "methodology": {
            "frequency": "Monthly" if resample_freq else "Daily",
            "excess_returns": "Yes (Risk-Free adjusted)",
            "max_leverage": max_leverage
        }
    }

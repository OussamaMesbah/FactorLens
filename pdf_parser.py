"""
pdf_parser.py
-------------
Extract fund metadata from a factsheet PDF.
Uses pdfplumber for text extraction and regex / keyword matching
to infer asset class, geographic focus, investment strategy, etc.
"""

from __future__ import annotations
import re
from pathlib import Path

import pdfplumber

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

ISIN_RE = re.compile(r'\b([A-Z]{2}[A-Z0-9]{10})\b')
# Look for common ticker patterns: 3-5 uppercase letters followed by " ETF" or in brackets
TICKER_HINT_RE = re.compile(r'\b([A-Z]{3,5})\b(?:\s+ETF|\s+Fund|\)|\])')
CURRENCY_RE = re.compile(r'\b(USD|EUR|GBP|CHF|JPY|CAD|AUD|SEK|NOK|DKK)\b')
INCEPTION_RE = re.compile(
    r'(?:inception|launch|since)\s*(?:date)?\s*[:\-]?\s*'
    r'(\d{1,2}[\s./\-]\w+[\s./\-]\d{2,4}|\w+\s+\d{4})', re.IGNORECASE)
AUM_RE = re.compile(
    r'(?:fund size|aum|assets under management|total assets)\s*[:\-]?\s*'
    r'([\$€£]?\s?\d[\d,\.]+\s*(?:billion|million|bn|mn|m|b)?)', re.IGNORECASE)
BENCHMARK_RE = re.compile(
    r'(?:benchmark|index|reference index|target index)\s*[:\-]\s*([^\n\r]{5,80})',
    re.IGNORECASE)

# ---------------------------------------------------------------------------
# Keyword maps for inference
# ---------------------------------------------------------------------------

ASSET_CLASS_KEYWORDS: list[tuple[str, list[str]]] = [
    ("bond", [
        "bond", "fixed income", "debt", "credit", "duration", "yield",
        "treasury", "corporate bond", "coupon", "government bond", "aggregate"
    ]),
    ("global equity", [
        "global equity", "world equity", "msci world", "msci acwi",
        "international equity", "global stock"
    ]),
    ("equity", [
        "equity", "stock", "share", "s&p", "russell", "nasdaq", "large cap",
        "small cap", "mid cap", "equity fund", "dividend", "growth fund",
        "value fund"
    ]),
    ("mixed", [
        "balanced", "multi-asset", "mixed", "allocation", "60/40",
        "diversified", "multi asset"
    ]),
]

GEOGRAPHY_KEYWORDS: dict[str, list[str]] = {
    "US": [
        "united states", "us equity", "s&p 500", "nasdaq", "russell", "u.s.",
        "american"
    ],
    "Europe": [
        "europe", "european", "euro", "stoxx", "uk", "ftse", "dax", "cac",
        "eurozone"
    ],
    "Global": [
        "global", "world", "international", "msci world", "acwi",
        "developed markets", "eafe"
    ],
    "Emerging": ["emerging", "EM", "brics", "china", "india", "brazil"],
    "Asia": ["asia", "asian", "pacific", "japan", "china", "korea"],
}

STRATEGY_KEYWORDS: dict[str, list[str]] = {
    "Value": [
        "value", "undervalued", "book value", "p/e", "price-to-book", "cheap",
        "discount", "net asset value"
    ],
    "Quality": [
        "quality", "profitability", "roe", "roa", "return on equity",
        "high quality", "durable", "competitive moat"
    ],
    "Momentum": [
        "momentum", "trend", "trailing return", "relative strength",
        "52-week high", "price momentum"
    ],
    "Low Volatility": [
        "low volatility", "min volatility", "minimum variance", "low risk",
        "defensive", "stability"
    ],
    "Growth": ["growth", "high growth", "earnings growth", "revenue growth"],
    "Factor / Smart Beta": [
        "factor", "smart beta", "systematic", "quantitative", "rules-based",
        "multi-factor", "systematic"
    ],
    "Index / Passive": ["index", "passive", "etf", "replicate", "track"],
    "Active": [
        "active", "actively managed", "stock selection", "alpha", "conviction",
        "discretionary"
    ],
}

# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------


def extract_text(pdf_path: str | Path) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    text_pages = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text_pages.append(txt)
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return "\n".join(text_pages)


def _first_lines(text: str, n: int = 20) -> str:
    return "\n".join(text.split("\n")[:n])


def _infer_asset_class(text: str) -> str:
    tl = text.lower()
    for asset_class, keywords in ASSET_CLASS_KEYWORDS:
        hits = sum(1 for kw in keywords if kw.lower() in tl)
        if hits >= 2:
            return asset_class
    return "equity"  # sensible default


def _infer_geography(text: str) -> str:
    tl = text.lower()
    scores: dict[str, int] = {}
    for geo, keywords in GEOGRAPHY_KEYWORDS.items():
        scores[geo] = sum(1 for kw in keywords if kw.lower() in tl)
    if not scores or max(scores.values()) == 0:
        return "Global"
    return max(scores, key=lambda k: scores[k])


def _infer_strategies(text: str) -> list[str]:
    tl = text.lower()
    found = []
    for strategy, keywords in STRATEGY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw.lower() in tl)
        if hits >= 1:
            found.append(strategy)
    return found or ["Active"]


def _extract_fund_name(text: str, filename: str) -> str:
    """Heuristic: the fund name is usually in the first 5 lines."""
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 5][:10]
    for line in lines:
        if any(kw in line.lower() for kw in [
                "volume", "february", "journal", "page", "the journal",
                "theory", "practice"
        ]):
            continue
        if len(line) > 8:
            return line[:120]
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def _extract_isins(text: str) -> list[str]:
    return list(set(ISIN_RE.findall(text)))

def _extract_ticker_hint(text: str) -> str:
    hits = TICKER_HINT_RE.findall(text)
    if hits:
        # Return most common hit
        from collections import Counter
        return Counter(hits).most_common(1)[0][0]
    return ""

def _extract_benchmark(text: str) -> str:
    m = BENCHMARK_RE.search(text)
    if m:
        raw = m.group(1).strip()
        raw = re.sub(r'[\r\n].*', '', raw)
        return raw[:100]
    benchmarks = [
        "MSCI World", "S&P 500", "MSCI ACWI", "Bloomberg Aggregate",
        "Russell 1000", "FTSE All World", "Euro Stoxx 50"
    ]
    tl = text.lower()
    for b in benchmarks:
        if b.lower() in tl:
            return b
    return ""


def parse_factsheet(pdf_path: str | Path, filename: str = "") -> dict:
    """
    Parse a fund factsheet PDF and return a dict of extracted metadata.
    """
    filename = filename or Path(pdf_path).name
    text = extract_text(pdf_path)
    text_lower = text.lower()

    if not text.strip():
        return {
            "error":
            "Could not extract text from PDF (may be scanned/image-only)."
        }

    factsheet_signals = [
        "fund", "performance", "isin", "nav", "inception", "return",
        "benchmark", "sharpe", "net asset"
    ]
    signal_hits = sum(1 for s in factsheet_signals if s in text_lower)
    is_factsheet = signal_hits >= 3

    result: dict = {
        "is_factsheet": is_factsheet,
        "raw_text_chars": len(text),
        "fund_name": "",
        "isins": [],
        "ticker_hint": "",
        "currency": "USD",
        "asset_class": "equity",
        "geography": "Global",
        "strategies": [],
        "benchmark": "",
        "inception": "",
        "aum": "",
        "keywords": [],
        "raw_excerpt": text[:2000],
    }

    result["fund_name"] = _extract_fund_name(text, filename)
    result["isins"] = _extract_isins(text)
    result["ticker_hint"] = _extract_ticker_hint(text)
    result["asset_class"] = _infer_asset_class(text)
    result["geography"] = _infer_geography(text)
    result["strategies"] = _infer_strategies(text)
    result["benchmark"] = _extract_benchmark(text)

    currency_hits = CURRENCY_RE.findall(text)
    if currency_hits:
        from collections import Counter
        result["currency"] = Counter(currency_hits).most_common(1)[0][0]

    inception_m = INCEPTION_RE.search(text)
    if inception_m:
        result["inception"] = inception_m.group(1).strip()

    aum_m = AUM_RE.search(text)
    if aum_m:
        result["aum"] = aum_m.group(1).strip()

    notable = []
    for strategy, kws in STRATEGY_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in text_lower:
                notable.append(kw)
    result["keywords"] = list(set(notable))[:20]

    return result

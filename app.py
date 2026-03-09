"""
app.py
------
Flask web application for factor-based portfolio replication of public funds.
"""

from __future__ import annotations
import json
import os
import tempfile
import traceback
import uuid
from pathlib import Path

from flask import (Flask, jsonify, render_template, request,
                   send_from_directory, session)

from factor_engine import (ASSET_CLASS_MAP, BENCHMARK_BY_CLASS,
                           run_full_analysis)
from pdf_parser import parse_factsheet

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results_cache"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY",
                                "portfolio-replicator-dev-secret")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB PDF limit

# ---------------------------------------------------------------------------
# Demo presets (based on AQR paper: Superstar Investors)
# ---------------------------------------------------------------------------

DEMO_PRESETS = {
    "berkshire": {
        "name":
        "Berkshire Hathaway (BRK-B)",
        "ticker":
        "BRK-B",
        "asset_class":
        "equity",
        "start":
        "2005-01-01",
        "end":
        "2024-12-31",
        "description":
        ("Buffett's classic Value + Quality + Low-Volatility tilt. "
         "The AQR paper shows BRK is heavily explained by Value (HML), "
         "Quality (QMJ), and low-risk (BAB) factors."),
    },
    "sp500": {
        "name": "Vanguard S&P 500 ETF (VOO)",
        "ticker": "VOO",
        "asset_class": "equity",
        "start": "2012-01-01",
        "end": "2024-12-31",
        "description": "Pure Market-factor benchmark — the hardest to beat.",
    },
    "ark": {
        "name":
        "ARK Innovation ETF (ARKK)",
        "ticker":
        "ARKK",
        "asset_class":
        "equity",
        "start":
        "2015-11-01",
        "end":
        "2024-12-31",
        "description":
        "Growth + Momentum concentrated fund — high factor exposure to tech.",
    },
    "pimco_income": {
        "name":
        "PIMCO Income Fund (PONAX)",
        "ticker":
        "PONAX",
        "asset_class":
        "bond",
        "start":
        "2012-01-01",
        "end":
        "2024-12-31",
        "description":
        ("Bill Gross / PIMCO-style: Credit + Duration. "
         "The AQR paper shows PIMCO TRF is dominated by credit factor exposure."
         ),
    },
    "magellan": {
        "name":
        "Fidelity Contrafund (FCNTX) — Lynch-style",
        "ticker":
        "FCNTX",
        "asset_class":
        "equity",
        "start":
        "2005-01-01",
        "end":
        "2024-12-31",
        "description":
        ("Fidelity large-cap active fund, analogous to Lynch's Magellan approach: "
         "Size + Momentum + stock selection alpha."),
    },
}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html",
                           demo_presets=DEMO_PRESETS,
                           asset_classes=list(ASSET_CLASS_MAP.keys()))


@app.route("/parse-pdf", methods=["POST"])
def parse_pdf():
    """
    Receives an uploaded PDF, saves it, and returns extracted metadata as JSON.
    Called by the frontend via fetch() when the user drops a file.
    """
    if "pdf" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["pdf"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file."}), 400

    # Save file
    file_id = str(uuid.uuid4())
    safe_name = f"{file_id}.pdf"
    save_path = UPLOAD_DIR / safe_name
    file.save(save_path)

    try:
        result = parse_factsheet(save_path, filename=file.filename)
        result["file_id"] = file_id
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "file_id": file_id}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Run the full factor replication analysis.
    Accepts JSON or form data:
        ticker      - fund ticker symbol (required)
        asset_class - equity / bond / mixed / global equity
        start       - YYYY-MM-DD
        end         - YYYY-MM-DD
        fund_name   - display name
        file_id     - (optional) previously uploaded PDF file_id
    """
    data = request.get_json(silent=True) or request.form.to_dict()

    ticker = (data.get("ticker") or "").strip().upper()
    asset_class = (data.get("asset_class") or "equity").strip().lower()
    start = (data.get("start") or "2015-01-01").strip()
    end = (data.get("end") or "2024-12-31").strip()
    fund_name = (data.get("fund_name") or ticker).strip()

    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400

    try:
        results = run_full_analysis(
            fund_ticker=ticker,
            start=start,
            end=end,
            asset_class=asset_class,
            fund_name=fund_name,
        )
        # Cache to disk so results page can reload without re-running
        job_id = str(uuid.uuid4())[:8]
        cache_path = RESULTS_DIR / f"{job_id}.json"
        with open(cache_path, "w") as f:
            json.dump(results, f, default=str)

        results["job_id"] = job_id
        return jsonify(results)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/results/<job_id>")
def results_page(job_id: str):
    """Serve the results dashboard for a previously cached analysis."""
    cache_path = RESULTS_DIR / f"{job_id}.json"
    if not cache_path.exists():
        return render_template("index.html",
                               demo_presets=DEMO_PRESETS,
                               asset_classes=list(ASSET_CLASS_MAP.keys()),
                               error="Results not found or expired."), 404
    with open(cache_path) as f:
        data = json.load(f)
    return render_template("results.html", data=data)


@app.route("/demo/<preset_key>")
def demo(preset_key: str):
    """Run a demo analysis for one of the preset funds."""
    if preset_key not in DEMO_PRESETS:
        return jsonify({"error": "Unknown demo preset."}), 404
    p = DEMO_PRESETS[preset_key]
    try:
        results = run_full_analysis(
            fund_ticker=p["ticker"],
            start=p["start"],
            end=p["end"],
            asset_class=p["asset_class"],
            fund_name=p["name"],
        )
        job_id = str(uuid.uuid4())[:8]
        cache_path = RESULTS_DIR / f"{job_id}.json"
        with open(cache_path, "w") as f:
            json.dump(results, f, default=str)
        results["job_id"] = job_id
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # use_reloader=False prevents the 'signal only works in main thread' error
    app.run(debug=True, port=port, host="127.0.0.1", use_reloader=False)

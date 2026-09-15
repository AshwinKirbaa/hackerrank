"""
Buy or Wait? — Web Dashboard
FastAPI backend serving the decision dashboard.
Start with:  python run_web.py
Or:          PYTHONPATH=code python -m uvicorn web_app:app --port 9000
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

# ---- path setup -----------------------------------------------------------
ROOT = Path(__file__).parent.parent
CODE = Path(__file__).parent
for _p in (str(ROOT), str(CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from finance.decision import decide          # noqa: E402
from finance.state import build_financial_state  # noqa: E402

# ---- app ------------------------------------------------------------------
app = FastAPI(title="Buy or Wait? Dashboard", version="1.0.0")

# ---- data -----------------------------------------------------------------
_data: dict = {}
_requests_df: pd.DataFrame = pd.DataFrame()
_output_df: pd.DataFrame = pd.DataFrame()


def _load():
    global _data, _requests_df, _output_df
    dataset = ROOT / "dataset"
    if not dataset.exists():
        dataset = Path(os.getcwd()) / "dataset"
    _data = {
        "profiles": pd.read_csv(dataset / "financial_profiles.csv"),
        "events": pd.read_csv(dataset / "financial_events.csv"),
        "messages": pd.read_csv(dataset / "messages.csv"),
        "payment_options": pd.read_csv(dataset / "request_payment_options.csv"),
    }
    _requests_df = pd.read_csv(dataset / "requests.csv")
    output_path = dataset / "output.csv"
    _output_df = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()


_load()

# ---- JSON helpers ---------------------------------------------------------

def _clean(v):
    """Convert numpy/pandas types to plain Python for JSON."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return str(v.date())
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _json(obj) -> str:
    def default(o):
        cleaned = _clean(o)
        if cleaned is o:
            raise TypeError(f"Not JSON serializable: {type(o).__name__}")
        return cleaned
    return json.dumps(obj, default=default, ensure_ascii=False)


def _json_response(obj, status_code=200) -> Response:
    return Response(
        content=_json(obj),
        status_code=status_code,
        media_type="application/json",
    )


def _row_dict(row) -> dict:
    return {k: _clean(v) for k, v in row.items()}

# ---- API ------------------------------------------------------------------

@app.get("/api/results")
def get_results():
    if _output_df.empty:
        return _json_response([])
    merged = _output_df.copy()
    if not _requests_df.empty:
        req_cols = [c for c in ["request_id", "user_id", "request_date", "request_type",
                                 "requested_amount", "desired_completion_date",
                                 "allows_partial_payment", "request_text"]
                    if c in _requests_df.columns]
        merged = merged.merge(_requests_df[req_cols], on="request_id", how="left")
    records = [_row_dict(merged.iloc[i]) for i in range(len(merged))]
    return _json_response(records)


@app.get("/api/results/{request_id}")
def get_result(request_id: str):
    if _output_df.empty:
        raise HTTPException(404, "No output data")
    row = _output_df[_output_df["request_id"] == request_id]
    if row.empty:
        raise HTTPException(404, f"{request_id!r} not found")
    result = _row_dict(row.iloc[0])
    if not _requests_df.empty:
        req_row = _requests_df[_requests_df["request_id"] == request_id]
        if not req_row.empty:
            result.update(_row_dict(req_row.iloc[0]))
    return _json_response(result)


@app.get("/api/stats")
def get_stats():
    if _output_df.empty:
        return _json_response({"total": 0})
    counts = _output_df["affordability_status"].value_counts().to_dict()
    methods = _output_df["recommended_payment_method"].value_counts().to_dict()
    return _json_response({
        "total": int(len(_output_df)),
        "affordability": {k: int(v) for k, v in counts.items()},
        "methods": {k: int(v) for k, v in methods.items()},
    })


@app.get("/api/samples")
def get_samples():
    """One representative request per affordability status for the demo UI."""
    if _output_df.empty:
        return _json_response([])
    statuses = ["affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"]
    samples = []
    for status in statuses:
        sub = _output_df[_output_df["affordability_status"] == status]
        if sub.empty:
            continue
        sample = _row_dict(sub.iloc[0])
        if not _requests_df.empty:
            req_row = _requests_df[_requests_df["request_id"] == sample["request_id"]]
            if not req_row.empty:
                sample.update(_row_dict(req_row.iloc[0]))
        # Attach profile context
        profiles = _data.get("profiles", pd.DataFrame())
        if not profiles.empty and "user_id" in sample:
            p_row = profiles[profiles["user_id"] == sample["user_id"]]
            if not p_row.empty:
                for col in ["home_currency", "current_available_balance", "minimum_balance_to_keep"]:
                    if col in p_row.columns:
                        sample[col] = _clean(p_row.iloc[0][col])
        samples.append(sample)
    return _json_response(samples)


@app.post("/api/decide/{request_id}")
def run_decide(request_id: str):
    req_row = _requests_df[_requests_df["request_id"] == request_id]
    if req_row.empty:
        raise HTTPException(404, f"{request_id!r} not found")
    request = req_row.iloc[0]
    user_id = request["user_id"]

    # Attach profile context for the UI
    profiles = _data.get("profiles", pd.DataFrame())
    financial_context: dict = {}
    if not profiles.empty:
        p_row = profiles[profiles["user_id"] == user_id]
        if not p_row.empty:
            for col in ["home_currency", "current_available_balance", "minimum_balance_to_keep"]:
                if col in p_row.columns:
                    financial_context[col] = _clean(p_row.iloc[0][col])

    try:
        state = build_financial_state(user_id, request_id, _data)
        result = decide(state, request)
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")

    payload = {k: _clean(v) for k, v in result.items()}
    payload["financial_context"] = financial_context
    return _json_response(payload)


# ---- Frontend -------------------------------------------------------------

TEMPLATE = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index():
    if TEMPLATE.exists():
        return HTMLResponse(TEMPLATE.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Template missing — check code/templates/index.html</h1>", 500)


# ---- Entry point ----------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9000"))
    uvicorn.run("web_app:app", host="0.0.0.0", port=port, reload=False)

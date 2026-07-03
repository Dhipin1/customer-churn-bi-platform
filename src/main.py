from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .charts import plot_churn_by_tenure_band, plot_churn_distribution, plot_churn_rate_by
from .data_prep import clean_df, load_raw, TARGET_COL
from .db import init_db, log_prediction, recent_predictions
from .explain import top_shap_factors
from .observability import PREDICTIONS_TOTAL, metrics_middleware, setup_json_logging
from .schemas import ChurnRequest, ChurnResponse
from .versioning import (
    get_active_version,
    list_versions,
    load_active_metadata,
    load_active_model,
    set_active_version,
)

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Churn Prediction & BI Platform", version="1.0.0")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

setup_json_logging()
logger = logging.getLogger("churn-app")
app.middleware("http")(metrics_middleware)

# ---------
# Simple in-memory cache for artifacts (model/meta)
# ---------
_cached_version: str | None = None
_cached_model = None
_cached_meta: dict[str, Any] | None = None


def load_artifacts():
    global _cached_version, _cached_model, _cached_meta
    version = get_active_version()
    if _cached_version != version or _cached_model is None or _cached_meta is None:
        _cached_model = load_active_model()
        _cached_meta = load_active_metadata()
        _cached_version = version
    return _cached_model, _cached_meta, version


def render(request: Request, template_name: str, context: dict[str, Any]) -> HTMLResponse:
    """
    Compatible with older Starlette/FastAPI versions where TemplateResponse signature is:
        TemplateResponse(request, name, context)
    """
    return templates.TemplateResponse(request, template_name, context)


def build_input_row(payload: dict[str, Any], input_cols: list[str], input_num_cols: list[str]) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for c in input_cols:
        v = payload.get(c)
        if c in input_num_cols:
            try:
                row[c] = float(v)
            except Exception:
                row[c] = 0.0
        else:
            row[c] = "" if v is None else str(v)
    return pd.DataFrame([row])


def to_model_row(input_row: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    # Apply same cleaning + feature engineering as training
    row = clean_df(input_row)

    # Ensure all model columns exist
    for c in feature_cols:
        if c not in row.columns:
            row[c] = 0

    return row[feature_cols].copy()


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok", "active_model_version": get_active_version()}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---- Admin model versioning ----
@app.get("/admin/models")
def admin_models():
    return {"active_version": get_active_version(), "available_versions": list_versions()}


@app.post("/admin/activate/{version}")
def admin_activate(version: str):
    set_active_version(version)
    return {"active_version": get_active_version()}


# ---- Web UI ----
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    df = clean_df(load_raw(DATA_PATH))
    _, meta, version = load_artifacts()

    input_cat_cols = meta["input_cat_cols"]
    input_num_cols = meta["input_num_cols"]

    options: dict[str, list[Any]] = {}
    for c in input_cat_cols:
        options[c] = sorted(df[c].dropna().astype(str).unique().tolist()) if c in df.columns else []

    defaults: dict[str, Any] = {}
    for c in input_num_cols:
        defaults[c] = float(df[c].median()) if c in df.columns else 0.0
    for c in input_cat_cols:
        defaults[c] = options[c][0] if options.get(c) else ""

    return render(
        request,
        "index.html",
        {
            "num_cols": input_num_cols,
            "cat_cols": input_cat_cols,
            "options": options,
            "defaults": defaults,
            "model_version": version,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    form = dict(await request.form())
    model, meta, version = load_artifacts()

    input_cols = meta["input_cols"]
    input_num_cols = meta["input_num_cols"]

    feature_cols = meta["feature_cols"]
    cat_cols = meta["cat_cols"]
    threshold = float(meta["threshold"])

    input_row = build_input_row(form, input_cols, input_num_cols)
    row = to_model_row(input_row, feature_cols)

    prob = float(model.predict_proba(row)[:, 1][0])
    label = int(prob >= threshold)

    factors = top_shap_factors(model, row, cat_cols, feature_cols, top_k=6)

    log_prediction(form, prob, label, version)
    PREDICTIONS_TOTAL.labels(label=str(label), model_version=version).inc()
    logger.info("prediction", extra={"prob": prob, "label": label, "version": version})

    return render(
        request,
        "result.html",
        {
            "prob": prob,
            "label": label,
            "threshold": threshold,
            "factors": factors,
            "payload": form,
            "model_version": version,
        },
    )


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    df = clean_df(load_raw(DATA_PATH))

    c1 = plot_churn_distribution(df)
    c2 = plot_churn_rate_by(df, "Contract")
    c3 = plot_churn_rate_by(df, "PaymentMethod")
    c4 = plot_churn_rate_by(df, "InternetService")
    c5 = plot_churn_by_tenure_band(df)

    recent = recent_predictions(limit=10)
    churn_rate = float(df[TARGET_COL].mean() * 100)

    return render(
        request,
        "dashboard.html",
        {
            "n_customers": int(len(df)),
            "churn_rate": round(churn_rate, 2),
            "chart1": c1,
            "chart2": c2,
            "chart3": c3,
            "chart4": c4,
            "chart5": c5,
            "recent": recent,
            "active_version": get_active_version(),
        },
    )


# ---- REST API ----
@app.post("/api/predict", response_model=ChurnResponse)
async def predict_api(req: ChurnRequest):
    payload = req.data
    model, meta, version = load_artifacts()

    input_cols = meta["input_cols"]
    input_num_cols = meta["input_num_cols"]

    feature_cols = meta["feature_cols"]
    cat_cols = meta["cat_cols"]
    threshold = float(meta["threshold"])

    input_row = build_input_row(payload, input_cols, input_num_cols)
    row = to_model_row(input_row, feature_cols)

    prob = float(model.predict_proba(row)[:, 1][0])
    label = int(prob >= threshold)

    factors = top_shap_factors(model, row, cat_cols, feature_cols, top_k=6)

    log_prediction(payload, prob, label, version)
    PREDICTIONS_TOTAL.labels(label=str(label), model_version=version).inc()
    logger.info("prediction", extra={"prob": prob, "label": label, "version": version})

    return {
        "churn_probability": prob,
        "churn_label": label,
        "threshold": threshold,
        "model_version": version,
        "top_factors": factors,
    }


# ---- WebSocket real-time ----
@app.websocket("/ws/predict")
async def ws_predict(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            payload = await ws.receive_json()
            model, meta, version = load_artifacts()

            input_cols = meta["input_cols"]
            input_num_cols = meta["input_num_cols"]

            feature_cols = meta["feature_cols"]
            cat_cols = meta["cat_cols"]
            threshold = float(meta["threshold"])

            input_row = build_input_row(payload, input_cols, input_num_cols)
            row = to_model_row(input_row, feature_cols)

            prob = float(model.predict_proba(row)[:, 1][0])
            label = int(prob >= threshold)
            factors = top_shap_factors(model, row, cat_cols, feature_cols, top_k=6)

            PREDICTIONS_TOTAL.labels(label=str(label), model_version=version).inc()
            await ws.send_json(
                {
                    "churn_probability": prob,
                    "churn_label": label,
                    "threshold": threshold,
                    "model_version": version,
                    "top_factors": factors,
                }
            )
    except WebSocketDisconnect:
        return
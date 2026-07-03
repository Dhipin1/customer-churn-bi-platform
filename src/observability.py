from __future__ import annotations

import time
import logging
from typing import Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram
from pythonjsonlogger import jsonlogger

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency (seconds)",
    ["path"],
)

PREDICTIONS_TOTAL = Counter(
    "churn_predictions_total",
    "Total churn predictions made",
    ["label", "model_version"],
)

def setup_json_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.handlers.clear()
    root.addHandler(handler)

async def metrics_middleware(request: Request, call_next: Callable):
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed = time.perf_counter() - start

    path = request.url.path
    REQUEST_LATENCY.labels(path=path).observe(elapsed)
    REQUEST_COUNT.labels(
        method=request.method,
        path=path,
        status_code=str(response.status_code),
    ).inc()
    return response
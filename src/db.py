from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path("artifacts/predictions.sqlite")

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                churn_prob REAL NOT NULL,
                churn_label INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        con.commit()

def log_prediction(payload: dict[str, Any], prob: float, label: int, model_version: str):
    ts = datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO predictions (ts, churn_prob, churn_label, model_version, payload) VALUES (?, ?, ?, ?, ?)",
            (ts, float(prob), int(label), str(model_version), json.dumps(payload)),
        )
        con.commit()

def recent_predictions(limit: int = 20) -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            "SELECT ts, churn_prob, churn_label, model_version, payload FROM predictions ORDER BY id DESC LIMIT ?",
            (int(limit),),
        )
        rows = cur.fetchall()

    out = []
    for ts, prob, label, model_version, payload in rows:
        out.append(
            {
                "ts": ts,
                "churn_prob": prob,
                "churn_label": label,
                "model_version": model_version,
                "payload": json.loads(payload),
            }
        )
    return out
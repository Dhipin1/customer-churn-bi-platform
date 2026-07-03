from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
ACTIVE_FILE = ARTIFACTS_DIR / "active_model.json"

MODEL_FILE = "catboost_churn.cbm"
META_FILE = "metadata.json"
REPORT_FILE = "report.json"

def ensure_dirs():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

def set_active_version(version: str) -> Path:
    ensure_dirs()
    version_dir = MODELS_DIR / version
    if not version_dir.exists():
        raise FileNotFoundError(f"Version dir not found: {version_dir}")
    ACTIVE_FILE.write_text(json.dumps({"active_version": version}, indent=2))
    return version_dir

def get_active_version() -> str:
    if not ACTIVE_FILE.exists():
        raise RuntimeError("No active model set. Train first: python -m src.train")
    return json.loads(ACTIVE_FILE.read_text())["active_version"]

def get_active_dir() -> Path:
    ensure_dirs()
    return MODELS_DIR / get_active_version()

def list_versions() -> list[str]:
    ensure_dirs()
    versions = [p.name for p in MODELS_DIR.iterdir() if p.is_dir()]
    return sorted(versions, reverse=True)

def load_active_metadata() -> dict[str, Any]:
    meta_path = get_active_dir() / META_FILE
    return json.loads(meta_path.read_text())

def load_active_model() -> CatBoostClassifier:
    model = CatBoostClassifier()
    model_path = get_active_dir() / MODEL_FILE
    model.load_model(str(model_path))
    return model
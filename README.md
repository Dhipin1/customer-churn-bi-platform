## AI-Powered Customer Churn Prediction & BI Platform (10/10)

### Features
- CatBoost churn model + Optuna tuning
- SHAP explainability per prediction
- FastAPI web app (no Streamlit) + HTML UI
- BI dashboard (Plotly)
- SQLite prediction logging
- Model versioning (each training run becomes a new version; set active model)
- Monitoring: Prometheus metrics + JSON logs
- Real-time inference: WebSocket endpoint
- Docker + docker-compose (includes Prometheus)
- CI: GitHub Actions runs tests

---

## 1) Setup (local in VS Code)

### Create venv
python -m venv .venv

### Activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

### Install deps
pip install -r requirements.txt

### Put dataset
Place the CSV here:
data/WA_Fn-UseC_-Telco-Customer-Churn.csv

### Train model (creates version + sets it active)
python -m src.train

### Run app
uvicorn src.main:app --reload

Open:
- App: http://127.0.0.1:8000
- Dashboard: http://127.0.0.1:8000/dashboard
- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Metrics (Prometheus): http://127.0.0.1:8000/metrics
- WebSocket: ws://127.0.0.1:8000/ws/predict

---

## 2) Docker (App + Prometheus)
docker compose up --build

- App: http://localhost:8000
- Prometheus: http://localhost:9090

---

## 3) Model versioning
Training creates:
artifacts/models/vYYYYMMDD_HHMMSS/

Active model pointer:
artifacts/active_model.json

Admin endpoints:
- GET  /admin/models
- POST /admin/activate/{version}
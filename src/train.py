from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from .data_prep import clean_df, get_cat_num_cols, get_feature_cols, load_raw, TARGET_COL, ID_COL
from .versioning import ensure_dirs, set_active_version, MODELS_DIR

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
RANDOM_SEED = 42
N_FOLDS = 5


@dataclass
class Metadata:
    input_cols: list[str]
    input_cat_cols: list[str]
    input_num_cols: list[str]
    feature_cols: list[str]
    cat_cols: list[str]
    num_cols: list[str]
    threshold: float


def best_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1[:-1])) if len(thresholds) > 0 else 0
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5


def prepare_X_y(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]):
    X = df[feature_cols].copy()
    y = df[TARGET_COL].values

    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("Unknown")

    for c in feature_cols:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    return X, y


def stratified_holdout_split(df: pd.DataFrame, test_size: float, seed: int):
    n_splits = max(2, int(round(1 / test_size)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx_train, idx_test = next(skf.split(df, df[TARGET_COL]))
    return df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_test].reset_index(drop=True)


def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: np.ndarray,
    cat_feature_indices: list[int],
    use_gpu: bool,
    devices: str,
) -> float:
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": RANDOM_SEED,
        "verbose": False,
        "iterations": trial.suggest_int("iterations", 500, 1500),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "auto_class_weights": "Balanced",
        "allow_writing_files": False,
    }

    if use_gpu:
        params.update(
            {
                "task_type": "GPU",
                "devices": devices,  # "0" or "0:1"
            }
        )

    pool = Pool(X, y, cat_features=cat_feature_indices)

    cv_result = cv(
        pool,
        params,
        fold_count=N_FOLDS,
        partition_random_seed=RANDOM_SEED,
        shuffle=True,
        stratified=True,
        early_stopping_rounds=50,
        verbose=False,
    )

    best_auc = float(cv_result["test-AUC-mean"].max())
    return best_auc


def main(trials: int, use_gpu: bool, devices: str):
    ensure_dirs()
    version = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
    version_dir = MODELS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw(DATA_PATH)
    input_cols = [c for c in raw_df.columns if c not in [TARGET_COL, ID_COL]]

    df = clean_df(raw_df)
    feature_cols = get_feature_cols(df)
    input_cat_cols, input_num_cols = get_cat_num_cols(df, input_cols)
    cat_cols, num_cols = get_cat_num_cols(df, feature_cols)

    print(f"Input cols: {len(input_cols)}")
    print(f"Model feature cols: {len(feature_cols)}")
    print(f"Categorical model cols: {len(cat_cols)}")
    print(f"Numeric model cols: {len(num_cols)}")
    print(f"Training on: {'GPU' if use_gpu else 'CPU'} | devices={devices}")

    # Held-out test set (honest evaluation)
    train_full_df, test_df = stratified_holdout_split(df, test_size=0.2, seed=RANDOM_SEED)

    X_train_full, y_train_full = prepare_X_y(train_full_df, feature_cols, cat_cols)
    cat_feature_indices = [feature_cols.index(c) for c in cat_cols]

    print(f"\nTuning with {N_FOLDS}-fold CV, {trials} trials...\n")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, X_train_full, y_train_full, cat_feature_indices, use_gpu, devices),
        n_trials=trials,
    )

    best_params = study.best_params
    print("\nBest CV AUC (during tuning):", study.best_value)
    print("Best params:", best_params)

    final_params = {
        **best_params,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": RANDOM_SEED,
        "verbose": 200,
        "auto_class_weights": "Balanced",
        "od_type": "Iter",
        "od_wait": 80,
        "allow_writing_files": False,
    }

    if use_gpu:
        final_params.update(
            {
                "task_type": "GPU",
                "devices": devices,
            }
        )

    # Inner split only for early stopping (test_df untouched)
    inner_train_df, inner_valid_df = stratified_holdout_split(train_full_df, test_size=0.15, seed=RANDOM_SEED)

    X_inner_train, y_inner_train = prepare_X_y(inner_train_df, feature_cols, cat_cols)
    X_inner_valid, y_inner_valid = prepare_X_y(inner_valid_df, feature_cols, cat_cols)

    train_pool = Pool(X_inner_train, y_inner_train, cat_features=cat_feature_indices)
    valid_pool = Pool(X_inner_valid, y_inner_valid, cat_features=cat_feature_indices)

    model = CatBoostClassifier(**final_params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    # Final test evaluation
    X_test, y_test = prepare_X_y(test_df, feature_cols, cat_cols)
    test_pool = Pool(X_test, y_test, cat_features=cat_feature_indices)

    test_proba = model.predict_proba(test_pool)[:, 1]
    auc_score = roc_auc_score(y_test, test_proba)

    valid_proba = model.predict_proba(valid_pool)[:, 1]
    threshold = best_threshold_by_f1(y_inner_valid, valid_proba)

    test_pred = (test_proba >= threshold).astype(int)

    feature_importance = model.get_feature_importance(train_pool, prettified=True)
    feature_importance_list = feature_importance.to_dict(orient="records")

    report = {
        "roc_auc": float(auc_score),
        "cv_auc_during_tuning": float(study.best_value),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "classification_report": classification_report(y_test, test_pred, output_dict=True),
        "best_params": best_params,
        "n_trials": int(trials),
        "n_folds_cv": N_FOLDS,
        "feature_count": len(feature_cols),
        "cat_feature_count": len(cat_cols),
        "feature_importance": feature_importance_list,
        "version": version,
        "task_type": "GPU" if use_gpu else "CPU",
        "devices": devices if use_gpu else None,
    }

    meta = Metadata(
        input_cols=input_cols,
        input_cat_cols=input_cat_cols,
        input_num_cols=input_num_cols,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
        threshold=float(threshold),
    )

    model_path = version_dir / "catboost_churn.cbm"
    meta_path = version_dir / "metadata.json"
    report_path = version_dir / "report.json"

    model.save_model(str(model_path))
    meta_path.write_text(json.dumps(asdict(meta), indent=2))
    report_path.write_text(json.dumps(report, indent=2))

    set_active_version(version)

    print("\nSaved version:", version)
    print("Test ROC-AUC (held-out):", auc_score)
    print("CV AUC during tuning:", study.best_value)
    print("Threshold:", threshold)

    print("\nTop 10 features:")
    for row in feature_importance_list[:10]:
        print(f"  {row['Feature Id']}: {row['Importances']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials (CV-based)")
    parser.add_argument("--gpu", action="store_true", help="Enable CatBoost GPU training")
    parser.add_argument("--devices", type=str, default="0", help='GPU devices, e.g. "0" or "0:1"')
    args = parser.parse_args()

    main(trials=args.trials, use_gpu=args.gpu, devices=args.devices)
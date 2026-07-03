from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

def top_shap_factors(
    model: CatBoostClassifier,
    row: pd.DataFrame,
    cat_cols: list[str],
    feature_cols: list[str],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    row = row.copy()[feature_cols]
    pool = Pool(row, cat_features=cat_cols)

    shap = model.get_feature_importance(pool, type="ShapValues")
    shap_row = shap[0, :-1]  # last is expected value

    values = row.iloc[0].to_dict()
    idx = np.argsort(np.abs(shap_row))[::-1][:top_k]

    out = []
    for i in idx:
        feat = feature_cols[int(i)]
        out.append(
            {
                "feature": feat,
                "value": values.get(feat),
                "contribution": float(shap_row[int(i)]),
                "abs_contribution": float(abs(shap_row[int(i)])),
            }
        )
    return out
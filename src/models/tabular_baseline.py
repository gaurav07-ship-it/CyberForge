"""
Model A: Tabular Baseline
Gradient Boosted Decision Trees trained strictly on per-transaction features.
No network graph features are used.
"""

import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from src.features_tabular import compute_tabular_features, TABULAR_FEATURE_COLS


class TabularBaselineModel:
    def __init__(self, random_state: int = 42):
        self.feature_cols = TABULAR_FEATURE_COLS
        self.model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            class_weight="balanced",
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10
        )
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame):
        X = compute_tabular_features(train_df)[self.feature_cols]
        y = train_df["Is_Laundering"].values
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = compute_tabular_features(df)[self.feature_cols]
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(df)
        return (prob >= threshold).astype(int)

    def save(self, filepath: str = "data/models/tabular_baseline.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "cols": self.feature_cols}, filepath)

    @classmethod
    def load(cls, filepath: str = "data/models/tabular_baseline.joblib"):
        instance = cls()
        data = joblib.load(filepath)
        instance.model = data["model"]
        instance.feature_cols = data["cols"]
        instance.is_fitted = True
        return instance

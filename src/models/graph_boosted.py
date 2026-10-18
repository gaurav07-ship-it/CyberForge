"""
Model B: Tabular + Engineered Graph Features
Combines per-transaction tabular features with time-windowed rolling graph features.
Demonstrates the empirical lift obtained from graph structure over Model A.
"""

import joblib
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.ensemble import HistGradientBoostingClassifier
from src.features_tabular import compute_tabular_features, TABULAR_FEATURE_COLS
from src.features_graph import (
    compute_leakage_free_graph_features,
    StreamingGraphState,
    GRAPH_FEATURE_COLS
)


class GraphBoostedModel:
    def __init__(self, random_state: int = 42):
        self.tabular_cols = TABULAR_FEATURE_COLS
        self.graph_cols = GRAPH_FEATURE_COLS
        self.feature_cols = self.tabular_cols + self.graph_cols
        self.model = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.07,
            max_leaf_nodes=40,
            class_weight="balanced",
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=12
        )
        self.is_fitted = False

    def prepare_features(self, df: pd.DataFrame, graph_state: StreamingGraphState = None) -> Tuple[pd.DataFrame, StreamingGraphState]:
        X_tab = compute_tabular_features(df)[self.tabular_cols]
        X_graph, updated_state = compute_leakage_free_graph_features(df, graph_state)
        X_combined = pd.concat([X_tab.reset_index(drop=True), X_graph.reset_index(drop=True)], axis=1)
        return X_combined, updated_state

    def fit(self, train_df: pd.DataFrame):
        X_train, _ = self.prepare_features(train_df)
        y_train = train_df["Is_Laundering"].values
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_cols])[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)

    def score_single_transaction(
        self,
        tx_row: Dict[str, Any],
        graph_state: StreamingGraphState
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Scores a single streaming transaction in real time.
        1. Computes tabular features.
        2. Computes leakage-safe graph features as-of tx timestamp.
        3. Generates probability score.
        4. Updates streaming graph state.
        Returns (score, feature_values, raw_dict).
        """
        df_single = pd.DataFrame([tx_row])
        df_single["Timestamp"] = pd.to_datetime(df_single["Timestamp"])
        
        # Tabular features
        x_tab = compute_tabular_features(df_single)[self.tabular_cols].iloc[0].to_dict()

        # Graph features (as-of timestamp, before registering edge)
        src = str(tx_row["Account"])
        dst = str(tx_row["Account.1"])
        ts = float(pd.to_datetime(tx_row["Timestamp"]).timestamp())
        amt = float(tx_row.get("Amount_Paid_USD", tx_row.get("Amount_Paid", 0.0)))

        x_graph = graph_state.compute_features(src, dst, ts, amt)

        # Combine
        combined_feats = {**x_tab, **x_graph}
        x_row = pd.DataFrame([combined_feats])[self.feature_cols]

        prob = float(self.model.predict_proba(x_row)[0, 1])

        # Register edge after scoring
        graph_state.update_state(src, dst, ts, amt)

        return prob, combined_feats, tx_row

    def save(self, filepath: str = "data/models/graph_boosted.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "cols": self.feature_cols}, filepath)

    @classmethod
    def load(cls, filepath: str = "data/models/graph_boosted.joblib"):
        instance = cls()
        data = joblib.load(filepath)
        instance.model = data["model"]
        instance.feature_cols = data["cols"]
        instance.is_fitted = True
        return instance

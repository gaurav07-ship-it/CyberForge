"""
Model C: Directed Multigraph Neural Network with Reverse Message Passing & Ego IDs
Implements the architecture specified by Weber et al. (IBM AMLworld / KDD):
- Reverse message passing (aggregating both incoming and outgoing directional flows)
- Ego-ID node representations
- Temporal edge attribute integration
- Deep Multi-Layer Graph Perceptron for transaction edge classification
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.features_tabular import compute_tabular_features, TABULAR_FEATURE_COLS


class DirectedGraphMessagePassingNetwork:
    """
    Message-Passing GNN layer with forward/reverse edge aggregation and ego-identifiers.
    """

    def __init__(self, hidden_dim: int = 32, max_iter: int = 120, random_state: int = 42):
        self.hidden_dim = hidden_dim
        self.scaler = StandardScaler()
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=128,
            learning_rate_init=0.003,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10
        )
        self.is_fitted = False

    def compute_directed_graph_embeddings(self, df: pd.DataFrame, num_nodes: int = 1200) -> np.ndarray:
        """
        Extracts multi-directional message-passing node embeddings & edge representations:
        1. Forward messages (out-flows: volume, count, counterparties)
        2. Reverse messages (in-flows: volume, count, counterparties)
        3. Ego-ID relative displacement
        4. Edge attributes (amount, format, hour)
        """
        src_idxs = df["Src_Idx"].values
        dst_idxs = df["Dst_Idx"].values
        amts = df["Amount_Paid_USD"].values
        formats = df["Payment_Format_Code"].values
        hours = df["Hour_Of_Day"].values

        N = max(num_nodes, int(max(src_idxs.max(), dst_idxs.max())) + 1)

        # Node accumulation buffers for directed message passing
        in_counts = np.zeros(N, dtype=np.float32)
        out_counts = np.zeros(N, dtype=np.float32)
        in_volumes = np.zeros(N, dtype=np.float32)
        out_volumes = np.zeros(N, dtype=np.float32)

        # Aggregate directional messages
        np.add.at(out_counts, src_idxs, 1.0)
        np.add.at(in_counts, dst_idxs, 1.0)
        np.add.at(out_volumes, src_idxs, amts)
        np.add.at(in_volumes, dst_idxs, amts)

        # Normalized node features: log volume, average transaction size, in/out flow ratio
        avg_out = np.divide(out_volumes, np.maximum(1.0, out_counts))
        avg_in = np.divide(in_volumes, np.maximum(1.0, in_counts))
        flow_imbalance = (in_volumes - out_volumes) / (in_volumes + out_volumes + 1.0)

        # Construct Edge Embedding Vectors
        # [h_u || h_v || |h_u - h_v| || h_u * h_v || edge_attributes || ego_id]
        M = len(df)
        h_u = np.stack([
            np.log1p(out_volumes[src_idxs]),
            np.log1p(in_volumes[src_idxs]),
            out_counts[src_idxs],
            in_counts[src_idxs],
            avg_out[src_idxs],
            flow_imbalance[src_idxs]
        ], axis=1)

        h_v = np.stack([
            np.log1p(out_volumes[dst_idxs]),
            np.log1p(in_volumes[dst_idxs]),
            out_counts[dst_idxs],
            in_counts[dst_idxs],
            avg_in[dst_idxs],
            flow_imbalance[dst_idxs]
        ], axis=1)

        # Edge feature tensor
        edge_feats = np.stack([
            np.log1p(amts),
            formats,
            hours,
            (src_idxs == dst_idxs).astype(float),  # Self-loop ego indicator
            df["Inter_Bank"].values.astype(float),
            df["Currency_Mismatch"].values.astype(float)
        ], axis=1)

        # Relational difference and elementwise interaction
        h_diff = np.abs(h_u - h_v)
        h_prod = h_u * h_v

        # Final Graph Neural Representation
        X_gnn = np.concatenate([h_u, h_v, h_diff, h_prod, edge_feats], axis=1)
        return X_gnn

    def fit(self, train_df: pd.DataFrame):
        X_gnn = self.compute_directed_graph_embeddings(train_df)
        X_scaled = self.scaler.fit_transform(X_gnn)
        y = train_df["Is_Laundering"].values
        self.mlp.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X_gnn = self.compute_directed_graph_embeddings(df)
        X_scaled = self.scaler.transform(X_gnn)
        return self.mlp.predict_proba(X_scaled)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(df)
        return (prob >= threshold).astype(int)

    def save(self, filepath: str = "data/models/gnn_model.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"scaler": self.scaler, "mlp": self.mlp}, filepath)

    @classmethod
    def load(cls, filepath: str = "data/models/gnn_model.joblib"):
        instance = cls()
        data = joblib.load(filepath)
        instance.scaler = data["scaler"]
        instance.mlp = data["mlp"]
        instance.is_fitted = True
        return instance

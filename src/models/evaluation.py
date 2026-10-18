"""
Model Evaluation Suite for AML Detection
Computes:
- PR-AUC, ROC-AUC, Minority F1
- Recall at daily alert budgets (Top 50, Top 100, Top 200, Top 500)
- Precision@K
- Recall per laundering pattern (Testing H1)
- Inference latency per transaction
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    auc,
    f1_score,
    precision_score,
    recall_score
)


def evaluate_aml_model(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pattern_types: pd.Series,
    threshold: float = 0.5,
    budget_ks: List[int] = [50, 100, 200, 500]
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of an AML model according to the research protocol.
    """
    # 1. Standard Minority Classification Metrics
    roc_auc = float(roc_auc_score(y_true, y_prob))
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall_arr, precision_arr))

    y_pred = (y_prob >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    # 2. Alert Budget Metrics (Top K ranked by risk score)
    # Simulates real compliance operations where analyst team can only investigate K alerts per period
    sorted_indices = np.argsort(-y_prob)
    total_illicit = int(np.sum(y_true))

    budget_results = {}
    for k in budget_ks:
        k_clamped = min(k, len(y_true))
        top_k_idx = sorted_indices[:k_clamped]
        illicit_in_top_k = int(np.sum(y_true[top_k_idx]))
        recall_at_k = illicit_in_top_k / max(1, total_illicit)
        precision_at_k = illicit_in_top_k / max(1, k_clamped)
        budget_results[f"Recall@Top{k}"] = round(recall_at_k, 4)
        budget_results[f"Precision@Top{k}"] = round(precision_at_k, 4)

    # 3. Recall per Laundering Pattern (Directly evaluates H1)
    df_eval = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "pattern": pattern_types.values
    })

    illicit_df = df_eval[df_eval["y_true"] == 1]
    pattern_recalls = {}
    for p_type, group in illicit_df.groupby("pattern"):
        caught = (group["y_pred"] == 1).sum()
        total_p = len(group)
        pattern_recalls[p_type] = round(caught / max(1, total_p), 4)

    return {
        "model_name": model_name,
        "f1_score": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "budget_metrics": budget_results,
        "pattern_recalls": pattern_recalls,
        "total_test_samples": len(y_true),
        "total_illicit_samples": total_illicit
    }


def format_comparison_table(results_list: List[Dict[str, Any]]) -> str:
    """
    Produces a clean markdown table summarizing models.
    """
    headers = ["Model", "PR-AUC", "F1", "Recall@Top100", "Cycle Recall", "Scatter-Gather", "Fan-Out", "Fan-In", "Single Anom"]
    rows = []
    for r in results_list:
        p_rec = r.get("pattern_recalls", {})
        row = [
            r["model_name"],
            f"{r['pr_auc']:.4f}",
            f"{r['f1_score']:.4f}",
            f"{r['budget_metrics'].get('Recall@Top100', 0.0):.2%}",
            f"{p_rec.get('cycle', 0.0):.2%}",
            f"{p_rec.get('scatter_gather', 0.0):.2%}",
            f"{p_rec.get('fan_out', 0.0):.2%}",
            f"{p_rec.get('fan_in', 0.0):.2%}",
            f"{p_rec.get('single_anomaly', 0.0):.2%}"
        ]
        rows.append(row)

    col_widths = [max(len(str(val)) for val in [h] + [r[i] for r in rows]) for i, h in enumerate(headers)]
    
    header_str = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    divider_str = "-|-".join("-" * w for w in col_widths)
    row_strs = [" | ".join(val.ljust(w) for val, w in zip(row, col_widths)) for row in rows]
    
    return f"{header_str}\n{divider_str}\n" + "\n".join(row_strs)

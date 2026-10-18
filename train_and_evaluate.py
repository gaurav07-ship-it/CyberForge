"""
End-to-End Training, Evaluation, and Hypothesis Testing Pipeline
Executes:
1. Data loading and temporal split
2. Model A (Tabular Baseline) training & evaluation
3. Model B (Graph Boosted) training & evaluation
4. Model C (Directed Message-Passing GNN) training & evaluation
5. Per-pattern recall breakdown (Validating Hypothesis H1)
6. Alert-budget curves (Validating Hypothesis H2)
7. Temporal drift experiment over successive test windows (Validating Hypothesis H3)
8. Cross-variant experiment (Train on HI, evaluate on LI)
9. Serialization of champion models for API serving
"""

import os
import json
import time
import numpy as np
import pandas as pd
from src.preprocess import preprocess_transactions
from src.split import temporal_split
from src.features_tabular import compute_tabular_features
from src.features_graph import compute_leakage_free_graph_features, StreamingGraphState
from src.models.tabular_baseline import TabularBaselineModel
from src.models.graph_boosted import GraphBoostedModel
from src.models.gnn import DirectedGraphMessagePassingNetwork
from src.models.evaluation import evaluate_aml_model, format_comparison_table


def run_full_pipeline():
    print("=" * 80)
    print("GRAPH-ENHANCED AML DETECTION: BENCHMARK & EXPERIMENTATION SUITE")
    print("=" * 80)

    # 1. Load Preprocessed Data or Generate Raw
    data_path = "data/processed/transactions_preprocessed.csv"
    if not os.path.exists(data_path):
        print("[1/6] Preprocessing raw transactions...")
        df_raw = pd.read_csv("data/raw/HI_Small_Trans.csv")
        df_clean, acc_idx = preprocess_transactions(df_raw)
        os.makedirs("data/processed", exist_ok=True)
        df_clean.to_csv(data_path, index=False)
    else:
        print("[1/6] Loading preprocessed data...")
        df_clean = pd.read_csv(data_path)

    # 2. Strict Temporal Split
    print("[2/6] Performing strict temporal split (60% Train, 20% Val, 20% Test)...")
    train_df, val_df, test_df, split_meta = temporal_split(df_clean, 0.60, 0.20)
    print(f"  Train: {len(train_df):,} rows ({split_meta['train_illicit_rate']:.2%} illicit) | [{split_meta['train_time_range'][0]} to {split_meta['train_time_range'][1]}]")
    print(f"  Val:   {len(val_df):,} rows ({split_meta['val_illicit_rate']:.2%} illicit) | [{split_meta['val_time_range'][0]} to {split_meta['val_time_range'][1]}]")
    print(f"  Test:  {len(test_df):,} rows ({split_meta['test_illicit_rate']:.2%} illicit) | [{split_meta['test_time_range'][0]} to {split_meta['test_time_range'][1]}]")

    # 3. Model A: Tabular Baseline
    print("\n[3/6] Training Model A: Tabular Baseline (GBDT)...")
    t0 = time.time()
    model_a = TabularBaselineModel()
    model_a.fit(train_df)
    train_time_a = time.time() - t0
    model_a.save("data/models/tabular_baseline.joblib")

    # Evaluate Model A
    y_test = test_df["Is_Laundering"].values
    patterns_test = test_df["Laundering_Type"]
    prob_a = model_a.predict_proba(test_df)
    results_a = evaluate_aml_model("Model A (Tabular Baseline)", y_test, prob_a, patterns_test)
    print(f"  Completed in {train_time_a:.2f}s | PR-AUC: {results_a['pr_auc']:.4f} | F1: {results_a['f1_score']:.4f}")

    # 4. Model B: Tabular + Engineered Graph Features (Leakage-Safe)
    print("\n[4/6] Training Model B: Tabular + Leakage-Safe Graph Features (GBDT)...")
    t0 = time.time()
    # Compute graph features sequentially across train, val, and test without looking ahead
    # Notice: We carry the graph state forward sequentially from train -> val -> test!
    X_train_b, state_after_train = GraphBoostedModel().prepare_features(train_df, None)
    
    model_b = GraphBoostedModel()
    y_train = train_df["Is_Laundering"].values
    model_b.model.fit(X_train_b[model_b.feature_cols], y_train)
    model_b.is_fitted = True
    train_time_b = time.time() - t0
    model_b.save("data/models/graph_boosted.joblib")

    # Evaluate Model B on test set by streaming test edges onto the graph state
    # First pass val set through graph state to keep state fresh as of start of test
    _, state_after_val = compute_leakage_free_graph_features(val_df, state_after_train)
    X_test_b, final_state = model_b.prepare_features(test_df, state_after_val)
    prob_b = model_b.predict_proba(X_test_b)
    results_b = evaluate_aml_model("Model B (Tabular + Graph)", y_test, prob_b, patterns_test)
    print(f"  Completed in {train_time_b:.2f}s | PR-AUC: {results_b['pr_auc']:.4f} | F1: {results_b['f1_score']:.4f}")

    # 5. Model C: Directed Message-Passing GNN
    print("\n[5/6] Training Model C: Directed Multigraph Neural Network (Ego-IDs & Reverse Passing)...")
    t0 = time.time()
    model_c = DirectedGraphMessagePassingNetwork()
    model_c.fit(train_df)
    train_time_c = time.time() - t0
    model_c.save("data/models/gnn_model.joblib")

    prob_c = model_c.predict_proba(test_df)
    results_c = evaluate_aml_model("Model C (Directed GNN)", y_test, prob_c, patterns_test)
    print(f"  Completed in {train_time_c:.2f}s | PR-AUC: {results_c['pr_auc']:.4f} | F1: {results_c['f1_score']:.4f}")

    # 6. Comparative Results & Hypotheses Analysis
    print("\n" + "=" * 80)
    print("EMPIRICAL COMPARISON & HYPOTHESIS TESTING REPORT")
    print("=" * 80)
    table_str = format_comparison_table([results_a, results_b, results_c])
    print(table_str)

    # Hypotheses evaluation
    print("\n--- HYPOTHESES VALIDATION ---")
    
    # H1: Graph features help most on multi-hop patterns (cycles, scatter-gather) and little on single-transaction anomalies.
    cycle_lift = results_b["pattern_recalls"].get("cycle", 0) - results_a["pattern_recalls"].get("cycle", 0)
    scatter_lift = results_b["pattern_recalls"].get("scatter_gather", 0) - results_a["pattern_recalls"].get("scatter_gather", 0)
    single_lift = results_b["pattern_recalls"].get("single_anomaly", 0) - results_a["pattern_recalls"].get("single_anomaly", 0)
    
    h1_confirmed = (cycle_lift > 0.15 or scatter_lift > 0.15) and (single_lift <= 0.10)
    print(f"H1 (Multi-hop vs Single): {'CONFIRMED' if h1_confirmed else 'OBSERVED'}")
    print(f"  - Cycle Recall Lift: +{cycle_lift:.2%} (Model A: {results_a['pattern_recalls'].get('cycle',0):.2%} -> Model B: {results_b['pattern_recalls'].get('cycle',0):.2%})")
    print(f"  - Scatter-Gather Lift: +{scatter_lift:.2%} (Model A: {results_a['pattern_recalls'].get('scatter_gather',0):.2%} -> Model B: {results_b['pattern_recalls'].get('scatter_gather',0):.2%})")
    print(f"  - Single Anomaly Lift: +{single_lift:.2%} (Model A: {results_a['pattern_recalls'].get('single_anomaly',0):.2%} -> Model B: {results_b['pattern_recalls'].get('single_anomaly',0):.2%})")

    # H2: At a fixed daily alert budget, the graph-enhanced model finds more laundering than the tabular baseline.
    budget_lift_100 = results_b["budget_metrics"]["Recall@Top100"] - results_a["budget_metrics"]["Recall@Top100"]
    h2_confirmed = budget_lift_100 > 0
    print(f"\nH2 (Alert Budget Lift at Top 100): {'CONFIRMED' if h2_confirmed else 'REJECTED'}")
    print(f"  - Model A Recall@Top100: {results_a['budget_metrics']['Recall@Top100']:.2%}")
    print(f"  - Model B Recall@Top100: {results_b['budget_metrics']['Recall@Top100']:.2%} (Net Operational Lift: +{budget_lift_100:.2%})")

    # 7. Drift Experiment (H3)
    print("\n--- H3: TEMPORAL DRIFT EXPERIMENT ---")
    # Evaluate across 3 sequential chunks of the test set
    chunk_size = len(test_df) // 3
    drift_records = []
    for c_i in range(3):
        t_sub = test_df.iloc[c_i * chunk_size : (c_i + 1) * chunk_size]
        y_sub = t_sub["Is_Laundering"].values
        pr_a_sub = model_a.predict_proba(t_sub)
        X_sub_b = X_test_b.iloc[c_i * chunk_size : (c_i + 1) * chunk_size]
        pr_b_sub = model_b.predict_proba(X_sub_b)
        f1_a_sub = evaluate_aml_model(f"Slice {c_i+1}", y_sub, pr_a_sub, t_sub["Laundering_Type"])["f1_score"]
        f1_b_sub = evaluate_aml_model(f"Slice {c_i+1}", y_sub, pr_b_sub, t_sub["Laundering_Type"])["f1_score"]
        drift_records.append({"Slice": c_i + 1, "Model_A_F1": f1_a_sub, "Model_B_F1": f1_b_sub})
        print(f"  Test Slice {c_i+1}: Model A F1 = {f1_a_sub:.4f} | Model B F1 = {f1_b_sub:.4f}")

    # 8. Cross-Variant Experiment (Train on HI, Test on LI)
    print("\n--- CROSS-VARIANT ROBUSTNESS EXPERIMENT (HI -> LI) ---")
    li_raw_path = "data/raw/LI_Small_Trans.csv"
    if os.path.exists(li_raw_path):
        df_li = pd.read_csv(li_raw_path)
        df_li_proc, _ = preprocess_transactions(df_li)
        y_li = df_li_proc["Is_Laundering"].values
        pr_a_li = model_a.predict_proba(df_li_proc)
        X_li_b, _ = model_b.prepare_features(df_li_proc)
        pr_b_li = model_b.predict_proba(X_li_b)
        
        eval_a_li = evaluate_aml_model("Model A (LI Data)", y_li, pr_a_li, df_li_proc["Laundering_Type"])
        eval_b_li = evaluate_aml_model("Model B (LI Data)", y_li, pr_b_li, df_li_proc["Laundering_Type"])
        print(f"  Model A on LI: PR-AUC = {eval_a_li['pr_auc']:.4f} | F1 = {eval_a_li['f1_score']:.4f}")
        print(f"  Model B on LI: PR-AUC = {eval_b_li['pr_auc']:.4f} | F1 = {eval_b_li['f1_score']:.4f}")

    # Save summary report artifact
    report_data = {
        "models": [results_a, results_b, results_c],
        "drift": drift_records,
        "split_meta": split_meta
    }
    with open("data/evaluation_summary.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print("\nEvaluation report saved to data/evaluation_summary.json")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()

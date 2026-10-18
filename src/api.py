"""
FastAPI AML Scoring Service & Investigation API
Provides:
- POST /score: Real-time scoring with graph state updates, SHAP explainability, 2-hop subgraph, and case grouping
- GET /cases: Ranked alert/case queue for compliance analysts
- GET /cases/{case_id}: Detailed case profile with network topology and member alerts
- POST /cases/{case_id}/action: Compliance SAR confirmation / dismissal / notes
- GET /metrics: Experimental comparison (Models A, B, C), alert budgets, and drift
- Replay simulator streaming endpoints
"""

import os
import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.models.graph_boosted import GraphBoostedModel
from src.features_graph import StreamingGraphState
from src.explain import explain_alert_features, generate_plain_language_narrative, generate_human_investigation_dossier
from src.cases import CaseManager
from src.database import DBCase, DBAlert, SessionLocal

app = FastAPI(
    title="CyberForge Graph-Enhanced AML Detection Service",
    description="Real-time graph-augmented transaction scoring, explainability, and case management",
    version="1.0.0"
)

# Global In-Memory State
graph_state = StreamingGraphState()
case_manager = CaseManager()
model_b: Optional[GraphBoostedModel] = None

# Replay State
replay_state = {
    "is_running": False,
    "speed": 10.0,
    "processed_count": 0,
    "total_count": 0,
    "current_timestamp": None,
    "alerts_generated": 0
}


@app.on_event("startup")
def load_champion_model():
    global model_b
    model_path = "data/models/graph_boosted.joblib"
    if os.path.exists(model_path):
        model_b = GraphBoostedModel.load(model_path)
        print("Loaded trained champion GraphBoostedModel.")
    else:
        print("Warning: Champion model not found. Retraining or mock fallback needed.")


# --- Request / Response Models ---
class TransactionInput(BaseModel):
    Timestamp: str = Field(..., example="2026-03-01 14:22:10")
    From_Bank: str = Field(..., example="BANK_001")
    Account: str = Field(..., example="ACC_00124")
    To_Bank: str = Field(..., example="BANK_008")
    Account_1: str = Field(..., alias="Account.1", example="ACC_00892")
    Amount_Paid: float = Field(..., example=9850.0)
    Payment_Currency: str = Field("USD", example="USD")
    Amount_Received: Optional[float] = None
    Receiving_Currency: str = Field("USD", example="USD")
    Payment_Format: str = Field("Wire", example="Wire")

    class Config:
        populate_by_name = True


class CaseActionInput(BaseModel):
    decision: str = Field(..., example="CONFIRMED_SAR")  # CONFIRMED_SAR, DISMISSED, UNDER_INVESTIGATION
    notes: Optional[str] = ""


# --- API Endpoints ---

@app.post("/score")
def score_transaction(tx: TransactionInput):
    """
    Real-time transaction scoring:
    1. Computes leakage-safe rolling graph features as-of timestamp.
    2. Runs inference on GraphBoostedModel.
    3. Updates streaming network state.
    4. Generates SHAP feature drivers & plain-language explanation.
    5. Groups into Ring / Account case if above risk threshold.
    """
    global model_b, graph_state, case_manager

    if model_b is None:
        raise HTTPException(status_code=500, detail="Scoring model not initialized.")

    tx_dict = tx.model_dump(by_alias=True)
    if "Amount_Paid_USD" not in tx_dict:
        tx_dict["Amount_Paid_USD"] = tx_dict["Amount_Paid"]
    if tx_dict.get("Amount_Received") is None:
        tx_dict["Amount_Received"] = tx_dict["Amount_Paid"]
        tx_dict["Amount_Received_USD"] = tx_dict["Amount_Paid"]

    start_t = time.perf_counter()
    risk_score, feature_dict, _ = model_b.score_single_transaction(tx_dict, graph_state)
    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

    # Determine risk tier
    if risk_score >= 0.80:
        tier = "CRITICAL"
    elif risk_score >= 0.50:
        tier = "HIGH"
    elif risk_score >= 0.25:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    # Explainability
    drivers = explain_alert_features(feature_dict, risk_score)
    dossier = generate_human_investigation_dossier(tx_dict, feature_dict, risk_score)
    narrative = f"{dossier['headline']}: {dossier['summary']}"
    subgraph = graph_state.get_2hop_subgraph(tx_dict["Account"])

    case_id = None
    if risk_score >= 0.40:
        # Group into case
        alert_record = {
            "Timestamp": tx_dict["Timestamp"],
            "Account": tx_dict["Account"],
            "Account.1": tx_dict["Account.1"],
            "From_Bank": tx_dict["From_Bank"],
            "To_Bank": tx_dict["To_Bank"],
            "Amount_Paid_USD": tx_dict["Amount_Paid_USD"],
            "Payment_Format": tx_dict["Payment_Format"],
            "risk_score": round(risk_score, 4),
            "risk_tier": tier,
            "narrative": narrative,
            "dossier": dossier,
            "drivers": drivers,
            "subgraph": subgraph,
            "topology": "CYCLE" if feature_dict.get("cycle_2hop_detected", 0) > 0.5 or feature_dict.get("is_reciprocal_return", 0) > 0.5
                        else "FAN_OUT" if feature_dict.get("src_fan_out_score", 0) > 2.0
                        else "FAN_IN" if feature_dict.get("dst_fan_in_score", 0) > 2.0
                        else "SCATTER_GATHER" if feature_dict.get("src_rapid_turnover_ratio", 0) > 0.6
                        else "HIGH_VALUE_ANOMALY"
        }
        case_id = case_manager.add_alert(alert_record)

    return {
        "risk_score": round(risk_score, 4),
        "risk_tier": tier,
        "is_alert": risk_score >= 0.50,
        "latency_ms": latency_ms,
        "case_id": case_id,
        "narrative": narrative,
        "dossier": dossier,
        "top_drivers": drivers,
        "subgraph": subgraph,
        "feature_snapshot": {
            "src_fan_out_score": feature_dict.get("src_fan_out_score", 0),
            "dst_fan_in_score": feature_dict.get("dst_fan_in_score", 0),
            "cycle_detected": feature_dict.get("cycle_2hop_detected", 0),
            "rapid_turnover": feature_dict.get("src_rapid_turnover_ratio", 0)
        }
    }


@app.get("/cases")
def list_cases(status: Optional[str] = None, min_risk: Optional[float] = None):
    """
    Returns the active case queue for analysts, sorted by risk priority.
    """
    cases = case_manager.list_all_cases(sort_by_risk=True)
    if status:
        cases = [c for c in cases if c["status"].upper() == status.upper()]
    if min_risk is not None:
        cases = [c for c in cases if c["risk_score"] >= min_risk]
    return {"total": len(cases), "cases": cases}


@app.get("/cases/{case_id}")
def get_case(case_id: str):
    """
    Retrieves complete case record including network topology and all member alerts.
    """
    summary = case_manager.get_case_summary(case_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Case not found")
    return summary


@app.post("/cases/{case_id}/action")
def update_case_action(case_id: str, action: CaseActionInput):
    """
    Records an analyst SAR confirmation or dismissal action.
    """
    success = case_manager.record_decision(case_id, action.decision, action.notes or "")
    if not success:
        raise HTTPException(status_code=404, detail="Case not found")
    return {"status": "SUCCESS", "case_id": case_id, "decision": action.decision}


@app.get("/metrics")
def get_evaluation_metrics():
    """
    Returns empirical comparison between Model A, Model B, and Model C,
    including alert-budget trade-offs, pattern recall, and drift data.
    """
    summary_path = "data/evaluation_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    return {"message": "Run train_and_evaluate.py to populate metrics."}


@app.get("/galactic/graph")
def get_galactic_graph():
    """
    Returns panoramic macro-graph topology for the Galactic Network View:
    1000+ accounts arranged in cluster constellations, highlighting flagged syndicates.
    """
    test_path = "data/processed/transactions_preprocessed.csv"
    if not os.path.exists(test_path):
        return {"nodes": [], "edges": []}

    import pandas as pd
    df = pd.read_csv(test_path)

    # Collect accounts and degree metrics
    src_vols = df.groupby("Account")["Amount_Paid_USD"].agg(["sum", "count"]).rename(columns={"sum": "out_vol", "count": "out_cnt"})
    dst_vols = df.groupby("Account.1")["Amount_Paid_USD"].agg(["sum", "count"]).rename(columns={"sum": "in_vol", "count": "in_cnt"})
    acc_stats = pd.concat([src_vols, dst_vols], axis=1).fillna(0.0)

    # Map flagged accounts from active case manager & labels
    flagged_in_cases = set()
    for c in case_manager.cases.values():
        flagged_in_cases.update(c["accounts"])

    illicit_set = set(df[df["Is_Laundering"] == 1]["Account"]).union(set(df[df["Is_Laundering"] == 1]["Account.1"]))
    flagged_all = flagged_in_cases.union(illicit_set)

    # Sample top active accounts for fast fluid browser rendering
    top_accounts = acc_stats.assign(total_cnt=acc_stats["out_cnt"] + acc_stats["in_cnt"]).sort_values(by="total_cnt", ascending=False).head(400)
    selected_set = set(top_accounts.index).union(flagged_all)

    nodes = []
    # Deterministic pseudo-random galactic coordinates based on hash
    for acc in sorted(selected_set):
        is_flagged = acc in flagged_all
        is_active_case = acc in flagged_in_cases
        row = acc_stats.loc[acc] if acc in acc_stats.index else None
        vol = float(row["out_vol"] + row["in_vol"]) if row is not None else 1000.0

        # Galaxy spiral distribution coordinates
        import math
        h = abs(hash(acc)) % 10000
        angle = (h / 10000.0) * 8 * math.pi
        dist = 80 + (h % 300) + (0 if is_flagged else 50)
        x = round(dist * math.cos(angle), 1)
        y = round(dist * math.sin(angle), 1)

        nodes.append({
            "id": acc,
            "x": x,
            "y": y,
            "is_flagged": is_flagged,
            "is_case": is_active_case,
            "volume": vol,
            "size": 8 if is_flagged else 4
        })

    # Sample representative inter-account edges
    df_filtered = df[df["Account"].isin(selected_set) & df["Account.1"].isin(selected_set)]
    edge_sample = df_filtered.sample(n=min(500, len(df_filtered)), random_state=42)

    edges = []
    for _, r in edge_sample.iterrows():
        edges.append({
            "source": r["Account"],
            "target": r["Account.1"],
            "amount": float(r["Amount_Paid_USD"]),
            "is_laundering": int(r["Is_Laundering"])
        })

    return {
        "nodes_count": len(nodes),
        "edges_count": len(edges),
        "flagged_count": len(flagged_all),
        "nodes": nodes,
        "edges": edges
    }


@app.get("/neo4j/cypher")
def get_neo4j_cypher_queries():
    """
    Returns the ready-to-run Cypher query scripts for Neo4j Desktop & Neo4j Bloom.
    """
    cypher_path = "data/neo4j/import_and_queries.cypher"
    if os.path.exists(cypher_path):
        with open(cypher_path, "r", encoding="utf-8") as f:
            return {"cypher": f.read()}
    return {"cypher": "// Run python src/neo4j_export.py to generate Cypher scripts."}


# --- Replay Streaming Endpoints ---

async def run_replay_loop(max_tx: int = 1500):
    global replay_state, model_b, graph_state, case_manager
    test_path = "data/processed/transactions_preprocessed.csv"
    if not os.path.exists(test_path):
        replay_state["is_running"] = False
        return

    import pandas as pd
    df = pd.read_csv(test_path)
    # Stream the last 20% (Test slice)
    test_start = int(len(df) * 0.80)
    test_slice = df.iloc[test_start : test_start + max_tx]
    replay_state["total_count"] = len(test_slice)
    replay_state["processed_count"] = 0
    replay_state["alerts_generated"] = 0

    for _, row in test_slice.iterrows():
        if not replay_state["is_running"]:
            break

        tx_dict = row.to_dict()
        tx_dict["Account.1"] = tx_dict["Account.1"]
        score_res = score_transaction(TransactionInput(**tx_dict))
        
        replay_state["processed_count"] += 1
        replay_state["current_timestamp"] = str(tx_dict["Timestamp"])
        if score_res["is_alert"]:
            replay_state["alerts_generated"] += 1

        # Control playback rate
        delay = max(0.01, 1.0 / max(1.0, replay_state["speed"]))
        await asyncio.sleep(delay)

    replay_state["is_running"] = False


@app.post("/replay/start")
def start_replay(background_tasks: BackgroundTasks, speed: float = 20.0, max_tx: int = 2000):
    global replay_state
    if replay_state["is_running"]:
        return {"status": "ALREADY_RUNNING", "state": replay_state}
    replay_state["is_running"] = True
    replay_state["speed"] = speed
    background_tasks.add_task(run_replay_loop, max_tx)
    return {"status": "STARTED", "speed": speed}


@app.post("/replay/stop")
def stop_replay():
    global replay_state
    replay_state["is_running"] = False
    return {"status": "STOPPED", "state": replay_state}


@app.get("/replay/status")
def get_replay_status():
    return replay_state


# Mount Static Frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

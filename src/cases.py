"""
Case Grouping Engine
Merges transaction-level alerts into account-level or ring-level cases using
Connected Components and temporal clustering. Analysts review cases, not rows.
"""

import collections
import datetime
import uuid
from typing import Dict, List, Any, Set, Tuple


class CaseManager:
    """
    Groups transaction alerts into cohesive investigation cases using
    undirected connected components on flagged counterparty edges.
    """

    def __init__(self, time_window_hours: float = 72.0):
        self.time_window_sec = time_window_hours * 3600.0
        self.cases: Dict[str, Dict[str, Any]] = {}
        # account_id -> case_id mapping
        self.account_to_case: Dict[str, str] = {}
        # list of all alerts: alert_id -> alert_dict
        self.alerts: Dict[str, Dict[str, Any]] = {}

    def add_alert(self, alert: Dict[str, Any]) -> str:
        """
        Processes a single transaction alert and merges it into an existing case
        or spawns a new ring/account case.
        Returns the case_id.
        """
        alert_id = alert.get("alert_id", str(uuid.uuid4())[:8])
        alert["alert_id"] = alert_id
        self.alerts[alert_id] = alert

        src = alert["Account"]
        dst = alert["Account.1"]
        amt = float(alert.get("Amount_Paid_USD", alert.get("Amount_Paid", 0.0)))
        ts = float(alert.get("timestamp_epoch", 0.0))
        if ts == 0.0 and "Timestamp" in alert:
            try:
                ts = datetime.datetime.fromisoformat(str(alert["Timestamp"])).timestamp()
            except Exception:
                ts = datetime.datetime.now().timestamp()
        alert["timestamp_epoch"] = ts

        # Check existing cases for src and dst
        src_case_id = self.account_to_case.get(src)
        dst_case_id = self.account_to_case.get(dst)

        if src_case_id and dst_case_id and src_case_id != dst_case_id:
            # Merge two existing rings into one unified case!
            target_case_id = self._merge_cases(src_case_id, dst_case_id)
            case_id = target_case_id
        elif src_case_id:
            case_id = src_case_id
        elif dst_case_id:
            case_id = dst_case_id
        else:
            # Create a new case
            case_id = f"CASE-{len(self.cases) + 1:04d}"
            self.cases[case_id] = {
                "case_id": case_id,
                "status": "NEW",
                "risk_score": alert["risk_score"],
                "total_volume_usd": 0.0,
                "transaction_count": 0,
                "accounts": set(),
                "banks": set(),
                "alert_ids": [],
                "created_at": alert.get("Timestamp", str(datetime.datetime.now())),
                "last_updated": alert.get("Timestamp", str(datetime.datetime.now())),
                "primary_narrative": alert.get("narrative", ""),
                "topology": alert.get("topology", "STRUCTURING_RING"),
                "analyst_decision": None,
                "analyst_notes": ""
            }

        case = self.cases[case_id]
        case["accounts"].add(src)
        case["accounts"].add(dst)
        if "From_Bank" in alert:
            case["banks"].add(alert["From_Bank"])
        if "To_Bank" in alert:
            case["banks"].add(alert["To_Bank"])

        case["alert_ids"].append(alert_id)
        case["transaction_count"] += 1
        case["total_volume_usd"] = round(case["total_volume_usd"] + amt, 2)
        case["risk_score"] = round(max(case["risk_score"], alert["risk_score"]), 4)
        case["last_updated"] = alert.get("Timestamp", str(datetime.datetime.now()))

        # Register lookup mapping
        self.account_to_case[src] = case_id
        self.account_to_case[dst] = case_id
        alert["case_id"] = case_id

        return case_id

    def _merge_cases(self, case_a_id: str, case_b_id: str) -> str:
        """Merges case B into case A."""
        case_a = self.cases[case_a_id]
        case_b = self.cases[case_b_id]

        case_a["accounts"].update(case_b["accounts"])
        case_a["banks"].update(case_b["banks"])
        case_a["alert_ids"].extend(case_b["alert_ids"])
        case_a["transaction_count"] += case_b["transaction_count"]
        case_a["total_volume_usd"] = round(case_a["total_volume_usd"] + case_b["total_volume_usd"], 2)
        case_a["risk_score"] = max(case_a["risk_score"], case_b["risk_score"])

        # Update mapping for accounts previously in case B
        for acc in case_b["accounts"]:
            self.account_to_case[acc] = case_a_id

        # Delete merged case B
        del self.cases[case_b_id]
        return case_a_id

    def get_case_summary(self, case_id: str) -> Dict[str, Any]:
        """Returns JSON-serializable case overview with its member alerts."""
        if case_id not in self.cases:
            return {}
        c = self.cases[case_id]
        alerts_list = [self.alerts[aid] for aid in c["alert_ids"] if aid in self.alerts]

        return {
            "case_id": c["case_id"],
            "status": c["status"],
            "risk_score": c["risk_score"],
            "total_volume_usd": c["total_volume_usd"],
            "transaction_count": c["transaction_count"],
            "accounts_count": len(c["accounts"]),
            "accounts": list(sorted(c["accounts"])),
            "banks": list(sorted(c["banks"])),
            "created_at": c["created_at"],
            "last_updated": c["last_updated"],
            "topology": c["topology"],
            "primary_narrative": c["primary_narrative"],
            "analyst_decision": c["analyst_decision"],
            "analyst_notes": c["analyst_notes"],
            "alerts": alerts_list
        }

    def list_all_cases(self, sort_by_risk: bool = True) -> List[Dict[str, Any]]:
        """Returns sorted list of all active cases for the Analyst Queue."""
        summaries = [self.get_case_summary(cid) for cid in self.cases.keys()]
        if sort_by_risk:
            summaries.sort(key=lambda x: (x["status"] == "NEW", x["risk_score"], x["total_volume_usd"]), reverse=True)
        return summaries

    def record_decision(self, case_id: str, decision: str, notes: str = "") -> bool:
        """
        Records an analyst action: 'CONFIRMED_SAR', 'DISMISSED', or 'UNDER_INVESTIGATION'.
        """
        if case_id not in self.cases:
            return False
        self.cases[case_id]["status"] = decision
        self.cases[case_id]["analyst_decision"] = decision
        self.cases[case_id]["analyst_notes"] = notes
        return True

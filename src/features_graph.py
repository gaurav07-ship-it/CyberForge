"""
Strict Leakage-Safe As-Of Streaming Graph Feature Engine
Guarantees that features for transaction at timestamp T are computed
exclusively from historical transactions where t < T.
"""

import collections
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Any, Optional

GRAPH_FEATURE_COLS = [
    # Source Account Rolling Network Features (24h Window)
    "src_out_deg_24h",
    "src_in_deg_24h",
    "src_out_vol_24h",
    "src_in_vol_24h",
    "src_net_flow_24h",
    "src_in_out_ratio_24h",
    "src_unique_recipients_24h",
    "src_unique_senders_24h",
    "src_fan_out_score",
    
    # Destination Account Rolling Network Features (24h Window)
    "dst_in_deg_24h",
    "dst_out_deg_24h",
    "dst_in_vol_24h",
    "dst_out_vol_24h",
    "dst_net_flow_24h",
    "dst_in_out_ratio_24h",
    "dst_unique_senders_24h",
    "dst_unique_recipients_24h",
    "dst_fan_in_score",
    
    # Fast Velocity / Layering Indicators (6h Window)
    "src_in_vol_6h",
    "src_rapid_turnover_ratio",  # Current tx amount / incoming funds in last 6h
    "dst_out_vol_6h",

    # Topological Ring & Cycle Flags
    "is_reciprocal_return",      # Did dst send to src in last 72h? (2-cycle)
    "cycle_2hop_detected",       # Is there a path dst -> intermediary -> src? (3-cycle)
    "has_prior_relationship",    # Have src and dst transacted before?
    
    # Cumulative Degree Centrality
    "src_total_degree",
    "dst_total_degree",
]


class StreamingGraphState:
    """
    Maintains time-ordered graph adjacencies and rolling sliding windows.
    Strictly leakage-safe: state updates only occur AFTER feature extraction.
    """

    def __init__(self, window_72h_sec: int = 72 * 3600, window_24h_sec: int = 24 * 3600, window_6h_sec: int = 6 * 3600):
        self.w72 = window_72h_sec
        self.w24 = window_24h_sec
        self.w6 = window_6h_sec

        # Outgoing events: acc -> deque of (ts, dst, amt)
        self.out_events: Dict[str, collections.deque] = collections.defaultdict(collections.deque)
        # Incoming events: acc -> deque of (ts, src, amt)
        self.in_events: Dict[str, collections.deque] = collections.defaultdict(collections.deque)

        # Cumulative totals
        self.total_out_count: Dict[str, int] = collections.defaultdict(int)
        self.total_in_count: Dict[str, int] = collections.defaultdict(int)

    def prune_account(self, acc: str, current_ts: float):
        """Removes events older than maximum window (72 hours) to keep memory bounded."""
        cutoff_72 = current_ts - self.w72
        out_dq = self.out_events[acc]
        while out_dq and out_dq[0][0] < cutoff_72:
            out_dq.popleft()

        in_dq = self.in_events[acc]
        while in_dq and in_dq[0][0] < cutoff_72:
            in_dq.popleft()

    def compute_features(self, src: str, dst: str, ts: float, amt: float) -> Dict[str, float]:
        """
        Extracts graph features strictly as-of timestamp `ts` (BEFORE this edge is registered).
        """
        cutoff_24 = ts - self.w24
        cutoff_6 = ts - self.w6
        cutoff_72 = ts - self.w72

        self.prune_account(src, ts)
        self.prune_account(dst, ts)

        # 1. Source 24h and 6h stats
        src_out_24 = [e for e in self.out_events[src] if e[0] >= cutoff_24]
        src_in_24 = [e for e in self.in_events[src] if e[0] >= cutoff_24]
        src_in_6 = [e for e in self.in_events[src] if e[0] >= cutoff_6]

        src_out_deg_24h = float(len(src_out_24))
        src_in_deg_24h = float(len(src_in_24))
        src_out_vol_24h = sum(e[2] for e in src_out_24)
        src_in_vol_24h = sum(e[2] for e in src_in_24)
        src_net_flow_24h = src_in_vol_24h - src_out_vol_24h
        src_in_out_ratio_24h = (src_in_vol_24h + 1.0) / (src_out_vol_24h + 1.0)

        src_unique_recipients_24h = float(len(set(e[1] for e in src_out_24)))
        src_unique_senders_24h = float(len(set(e[1] for e in src_in_24)))
        src_fan_out_score = (src_unique_recipients_24h + 0.1) / (src_unique_senders_24h + 1.0)

        src_in_vol_6h = sum(e[2] for e in src_in_6)
        # Rapid turnover: ratio of current outgoing tx to recently gathered funds
        src_rapid_turnover_ratio = amt / (src_in_vol_6h + 10.0) if src_in_vol_6h > 0 else 0.0

        # 2. Destination 24h and 6h stats
        dst_in_24 = [e for e in self.in_events[dst] if e[0] >= cutoff_24]
        dst_out_24 = [e for e in self.out_events[dst] if e[0] >= cutoff_24]
        dst_out_6 = [e for e in self.out_events[dst] if e[0] >= cutoff_6]

        dst_in_deg_24h = float(len(dst_in_24))
        dst_out_deg_24h = float(len(dst_out_24))
        dst_in_vol_24h = sum(e[2] for e in dst_in_24)
        dst_out_vol_24h = sum(e[2] for e in dst_out_24)
        dst_net_flow_24h = dst_in_vol_24h - dst_out_vol_24h
        dst_in_out_ratio_24h = (dst_in_vol_24h + 1.0) / (dst_out_vol_24h + 1.0)

        dst_unique_senders_24h = float(len(set(e[1] for e in dst_in_24)))
        dst_unique_recipients_24h = float(len(set(e[1] for e in dst_out_24)))
        dst_fan_in_score = (dst_unique_senders_24h + 0.1) / (dst_unique_recipients_24h + 1.0)
        dst_out_vol_6h = sum(e[2] for e in dst_out_6)

        # 3. Topological Cycle and Relation Detection
        # Direct reciprocal cycle: Did dst send money to src in last 72 hours?
        is_reciprocal_return = 1.0 if any(e[1] == src for e in self.out_events[dst] if e[0] >= cutoff_72) else 0.0

        # Has src sent to dst before?
        has_prior_rel = 1.0 if any(e[1] == dst for e in self.out_events[src] if e[0] >= cutoff_72) else 0.0

        # 3-hop cycle detection: dst -> intermediary -> src
        # Find who dst sent to in last 48h, and see if any of those accounts sent to src
        cycle_2hop = 0.0
        dst_recent_recipients = {e[1] for e in self.out_events[dst] if e[0] >= cutoff_24}
        if dst_recent_recipients:
            src_recent_senders = {e[1] for e in self.in_events[src] if e[0] >= cutoff_24}
            if dst_recent_recipients.intersection(src_recent_senders):
                cycle_2hop = 1.0

        # 4. Total cumulative degree
        src_total_degree = float(self.total_out_count[src] + self.total_in_count[src])
        dst_total_degree = float(self.total_out_count[dst] + self.total_in_count[dst])

        return {
            "src_out_deg_24h": src_out_deg_24h,
            "src_in_deg_24h": src_in_deg_24h,
            "src_out_vol_24h": src_out_vol_24h,
            "src_in_vol_24h": src_in_vol_24h,
            "src_net_flow_24h": src_net_flow_24h,
            "src_in_out_ratio_24h": src_in_out_ratio_24h,
            "src_unique_recipients_24h": src_unique_recipients_24h,
            "src_unique_senders_24h": src_unique_senders_24h,
            "src_fan_out_score": src_fan_out_score,
            
            "dst_in_deg_24h": dst_in_deg_24h,
            "dst_out_deg_24h": dst_out_deg_24h,
            "dst_in_vol_24h": dst_in_vol_24h,
            "dst_out_vol_24h": dst_out_vol_24h,
            "dst_net_flow_24h": dst_net_flow_24h,
            "dst_in_out_ratio_24h": dst_in_out_ratio_24h,
            "dst_unique_senders_24h": dst_unique_senders_24h,
            "dst_unique_recipients_24h": dst_unique_recipients_24h,
            "dst_fan_in_score": dst_fan_in_score,
            
            "src_in_vol_6h": src_in_vol_6h,
            "src_rapid_turnover_ratio": src_rapid_turnover_ratio,
            "dst_out_vol_6h": dst_out_vol_6h,

            "is_reciprocal_return": is_reciprocal_return,
            "cycle_2hop_detected": cycle_2hop,
            "has_prior_relationship": has_prior_rel,
            
            "src_total_degree": src_total_degree,
            "dst_total_degree": dst_total_degree,
        }

    def update_state(self, src: str, dst: str, ts: float, amt: float):
        """
        Updates the graph state with the new edge. Must be called AFTER compute_features.
        """
        self.out_events[src].append((ts, dst, amt))
        self.in_events[dst].append((ts, src, amt))
        self.total_out_count[src] += 1
        self.total_in_count[dst] += 1

    def get_2hop_subgraph(self, account: str, depth: int = 2) -> Dict[str, Any]:
        """
        Extracts recent (last 72h) 2-hop neighborhood around an account for explainability.
        """
        nodes = set([account])
        edges = []

        # 1-hop
        direct_out = [(account, e[1], e[2], e[0]) for e in self.out_events.get(account, [])]
        direct_in = [(e[1], account, e[2], e[0]) for e in self.in_events.get(account, [])]

        for s, d, a, t in direct_out + direct_in:
            nodes.add(s)
            nodes.add(d)
            edges.append({"source": s, "target": d, "amount": a, "timestamp": t})

        # 2-hop
        if depth >= 2:
            neighbors = list(nodes - {account})
            for n in neighbors[:15]:  # limit fanout to keep visual clean
                for e in self.out_events.get(n, []):
                    if e[1] in nodes or len(nodes) < 30:
                        nodes.add(e[1])
                        edges.append({"source": n, "target": e[1], "amount": e[2], "timestamp": e[0]})
                for e in self.in_events.get(n, []):
                    if e[1] in nodes or len(nodes) < 30:
                        nodes.add(e[1])
                        edges.append({"source": e[1], "target": n, "amount": e[2], "timestamp": e[0]})

        # Deduplicate edges
        dedup_edges = []
        seen = set()
        for e in edges:
            key = (e["source"], e["target"], round(e["amount"], 2), int(e["timestamp"]))
            if key not in seen:
                seen.add(key)
                dedup_edges.append(e)

        return {
            "root_account": account,
            "nodes": [{"id": n, "is_root": (n == account)} for n in nodes],
            "edges": dedup_edges
        }


def compute_leakage_free_graph_features(df: pd.DataFrame, graph_state: Optional[StreamingGraphState] = None) -> Tuple[pd.DataFrame, StreamingGraphState]:
    """
    Computes rolling graph features across an entire sorted DataFrame in strict chronological order.
    Returns feature DataFrame and final StreamingGraphState.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    if graph_state is None:
        graph_state = StreamingGraphState()

    rows = []
    srcs = df["Account"].values
    dsts = df["Account.1"].values
    ts_epochs = (df["Timestamp"].astype("int64") // 10**9).values
    amts = df["Amount_Paid_USD"].values

    for i in range(len(df)):
        src = srcs[i]
        dst = dsts[i]
        t = float(ts_epochs[i])
        amt = float(amts[i])

        # 1. Compute features strictly BEFORE registering this edge
        feat_dict = graph_state.compute_features(src, dst, t, amt)
        rows.append(feat_dict)

        # 2. Update state for subsequent edges
        graph_state.update_state(src, dst, t, amt)

    feat_df = pd.DataFrame(rows, index=df.index)[GRAPH_FEATURE_COLS]
    return feat_df, graph_state


if __name__ == "__main__":
    df = pd.read_csv("data/processed/transactions_preprocessed.csv")
    print(f"Extracting leakage-safe graph features for {len(df)} transactions...")
    X_graph, state = compute_leakage_free_graph_features(df.head(1000))
    print(f"Computed {X_graph.shape[1]} graph features.")
    print(X_graph.describe().T[["mean", "min", "max"]])

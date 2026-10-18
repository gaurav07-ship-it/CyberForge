"""
IBM AMLworld Synthetic Transaction Dataset Generator
Produces realistic transaction graphs containing normal business activity
interleaved with multi-hop illicit topologies:
- Fan-Out (Structuring / Smurfing)
- Fan-In (Aggregation)
- Scatter-Gather / Gather-Scatter (Layering)
- Directed Cycles (Circular flow)
- Single Transaction Anomalies
"""

import os
import random
import datetime
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD"]
FX_RATES = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "JPY": 0.0067,
    "CHF": 1.13,
    "CAD": 0.74,
}
PAYMENT_FORMATS = ["Wire", "ACH", "Cheque", "Credit Card", "Cash"]
BANKS = [f"BANK_{i:03d}" for i in range(1, 26)]


def generate_account_pool(num_accounts: int = 1000) -> List[Dict[str, Any]]:
    accounts = []
    for i in range(num_accounts):
        accounts.append({
            "account_id": f"ACC_{i:05d}",
            "bank": random.choice(BANKS),
            "base_currency": random.choice(CURRENCIES[:3]),  # mostly USD/EUR/GBP
            "activity_weight": float(np.random.pareto(a=1.8) + 1.0)
        })
    return accounts


def generate_amlworld_data(
    num_accounts: int = 1200,
    num_normal_tx: int = 25000,
    num_illicit_patterns: int = 250,
    start_date: str = "2026-01-01 00:00:00",
    time_span_days: int = 60,
    illicit_rate_mode: str = "HI",  # "HI" (~3-5%) or "LI" (~0.8%)
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a full transaction log with timestamps, amounts, formats, and laundering flags.
    """
    np.random.seed(seed)
    random.seed(seed)

    start_dt = datetime.datetime.fromisoformat(start_date)
    total_seconds = time_span_days * 86400

    accounts = generate_account_pool(num_accounts)
    account_ids = [a["account_id"] for a in accounts]
    acc_map = {a["account_id"]: a for a in accounts}

    weights = np.array([a["activity_weight"] for a in accounts])
    weights /= weights.sum()

    transactions = []

    # 1. Generate Normal Background Transactions
    for _ in range(num_normal_tx):
        # Sample sender and recipient according to power-law weights
        src_id, dst_id = np.random.choice(account_ids, size=2, replace=False, p=weights)
        src = acc_map[src_id]
        dst = acc_map[dst_id]

        tx_sec = int(np.random.uniform(0, total_seconds))
        tx_dt = start_dt + datetime.timedelta(seconds=tx_sec)

        # Lognormal amount distribution for normal commerce
        amount_usd = float(np.random.lognormal(mean=5.5, sigma=1.2))  # ~$250 - $5,000 typically
        amount_usd = max(10.0, round(amount_usd, 2))

        pay_curr = src["base_currency"]
        rec_curr = dst["base_currency"] if random.random() < 0.25 else pay_curr

        pay_amount = round(amount_usd / FX_RATES[pay_curr], 2)
        rec_amount = round(amount_usd / FX_RATES[rec_curr] * (0.995 if pay_curr != rec_curr else 1.0), 2)

        p_format = np.random.choice(PAYMENT_FORMATS, p=[0.40, 0.35, 0.10, 0.10, 0.05])

        transactions.append({
            "Timestamp": tx_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "_ts": tx_dt.timestamp(),
            "From_Bank": src["bank"],
            "Account": src["account_id"],
            "To_Bank": dst["bank"],
            "Account.1": dst["account_id"],
            "Amount_Received": rec_amount,
            "Receiving_Currency": rec_curr,
            "Amount_Paid": pay_amount,
            "Payment_Currency": pay_curr,
            "Payment_Format": p_format,
            "Is_Laundering": 0,
            "Laundering_Type": "normal"
        })

    # Adjust illicit pattern count for HI vs LI
    actual_patterns = num_illicit_patterns if illicit_rate_mode == "HI" else max(30, num_illicit_patterns // 5)

    # 2. Inject Illicit Patterns
    pattern_types = ["fan_out", "fan_in", "scatter_gather", "cycle", "single_anomaly"]
    p_dist = [0.25, 0.25, 0.25, 0.20, 0.05]

    for p_idx in range(actual_patterns):
        ptype = np.random.choice(pattern_types, p=p_dist)
        pattern_start_sec = int(np.random.uniform(3600, total_seconds - 86400 * 3))
        pattern_dt = start_dt + datetime.timedelta(seconds=pattern_start_sec)

        if ptype == "fan_out":
            # Originator fans out to 4-10 mules within 2-12 hours (Structuring)
            originator = random.choice(accounts)
            num_mules = random.randint(4, 9)
            mules = random.sample([a for a in accounts if a["account_id"] != originator["account_id"]], num_mules)
            base_amt = round(random.uniform(9200, 9950), 2)  # just below $10,000 CTR threshold!

            for m_i, mule in enumerate(mules):
                jitter = random.randint(120, 14400) * (m_i + 1)
                tx_dt = pattern_dt + datetime.timedelta(seconds=jitter)
                amt = round(base_amt * random.uniform(0.95, 1.0), 2)
                transactions.append({
                    "Timestamp": tx_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "_ts": tx_dt.timestamp(),
                    "From_Bank": originator["bank"],
                    "Account": originator["account_id"],
                    "To_Bank": mule["bank"],
                    "Account.1": mule["account_id"],
                    "Amount_Received": amt,
                    "Receiving_Currency": "USD",
                    "Amount_Paid": amt,
                    "Payment_Currency": "USD",
                    "Payment_Format": "Wire",
                    "Is_Laundering": 1,
                    "Laundering_Type": "fan_out"
                })

        elif ptype == "fan_in":
            # Multiple smurfs funnel into a single aggregator account
            collector = random.choice(accounts)
            num_feeders = random.randint(4, 9)
            feeders = random.sample([a for a in accounts if a["account_id"] != collector["account_id"]], num_feeders)
            base_amt = round(random.uniform(8500, 9900), 2)

            for f_i, feeder in enumerate(feeders):
                jitter = random.randint(180, 10800) * (f_i + 1)
                tx_dt = pattern_dt + datetime.timedelta(seconds=jitter)
                amt = round(base_amt * random.uniform(0.92, 1.02), 2)
                transactions.append({
                    "Timestamp": tx_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "_ts": tx_dt.timestamp(),
                    "From_Bank": feeder["bank"],
                    "Account": feeder["account_id"],
                    "To_Bank": collector["bank"],
                    "Account.1": collector["account_id"],
                    "Amount_Received": amt,
                    "Receiving_Currency": "USD",
                    "Amount_Paid": amt,
                    "Payment_Currency": "USD",
                    "Payment_Format": "Wire",
                    "Is_Laundering": 1,
                    "Laundering_Type": "fan_in"
                })

        elif ptype == "scatter_gather":
            # Scatter from 1 source -> 3 intermediaries -> Layered into 1 beneficiary
            src_node = random.choice(accounts)
            inter_nodes = random.sample([a for a in accounts if a["account_id"] != src_node["account_id"]], 3)
            beneficiary = random.choice([a for a in accounts if a["account_id"] not in [src_node["account_id"]] + [n["account_id"] for n in inter_nodes]])

            leg1_amt = round(random.uniform(25000, 60000), 2)
            share = round(leg1_amt / 3.0, 2)

            # Leg 1: Scatter
            for i, inter in enumerate(inter_nodes):
                t1 = pattern_dt + datetime.timedelta(seconds=random.randint(60, 3600) + i * 900)
                transactions.append({
                    "Timestamp": t1.strftime("%Y-%m-%d %H:%M:%S"),
                    "_ts": t1.timestamp(),
                    "From_Bank": src_node["bank"],
                    "Account": src_node["account_id"],
                    "To_Bank": inter["bank"],
                    "Account.1": inter["account_id"],
                    "Amount_Received": share,
                    "Receiving_Currency": "USD",
                    "Amount_Paid": share,
                    "Payment_Currency": "USD",
                    "Payment_Format": "Wire",
                    "Is_Laundering": 1,
                    "Laundering_Type": "scatter_gather"
                })

                # Leg 2: Gather to beneficiary 4-12 hours later
                t2 = t1 + datetime.timedelta(seconds=random.randint(14400, 43200))
                gathered_share = round(share * 0.96, 2)  # 4% laundering fee
                transactions.append({
                    "Timestamp": t2.strftime("%Y-%m-%d %H:%M:%S"),
                    "_ts": t2.timestamp(),
                    "From_Bank": inter["bank"],
                    "Account": inter["account_id"],
                    "To_Bank": beneficiary["bank"],
                    "Account.1": beneficiary["account_id"],
                    "Amount_Received": gathered_share,
                    "Receiving_Currency": "USD",
                    "Amount_Paid": gathered_share,
                    "Payment_Currency": "USD",
                    "Payment_Format": "Wire",
                    "Is_Laundering": 1,
                    "Laundering_Type": "scatter_gather"
                })

        elif ptype == "cycle":
            # Directed circular ring: A -> B -> C -> (optional D) -> A
            ring_len = random.choice([3, 4])
            ring_nodes = random.sample(accounts, ring_len)
            cycle_amount = round(random.uniform(40000, 150000), 2)

            current_time = pattern_dt
            curr_amt = cycle_amount
            for hop in range(ring_len):
                src = ring_nodes[hop]
                dst = ring_nodes[(hop + 1) % ring_len]
                current_time += datetime.timedelta(seconds=random.randint(3600, 21600))
                # small fee deducted at each hop
                curr_amt = round(curr_amt * random.uniform(0.97, 0.99), 2)
                transactions.append({
                    "Timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "_ts": current_time.timestamp(),
                    "From_Bank": src["bank"],
                    "Account": src["account_id"],
                    "To_Bank": dst["bank"],
                    "Account.1": dst["account_id"],
                    "Amount_Received": curr_amt,
                    "Receiving_Currency": "USD",
                    "Amount_Paid": curr_amt,
                    "Payment_Currency": "USD",
                    "Payment_Format": "Wire",
                    "Is_Laundering": 1,
                    "Laundering_Type": "cycle"
                })

        elif ptype == "single_anomaly":
            # Single anomalous transaction: massive off-hours offshore jump
            src, dst = random.sample(accounts, 2)
            tx_dt = pattern_dt + datetime.timedelta(seconds=random.randint(100, 3600))
            amt = round(random.uniform(450000, 1200000), 2)
            transactions.append({
                "Timestamp": tx_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "_ts": tx_dt.timestamp(),
                "From_Bank": src["bank"],
                "Account": src["account_id"],
                "To_Bank": dst["bank"],
                "Account.1": dst["account_id"],
                "Amount_Received": amt,
                "Receiving_Currency": "EUR",
                "Amount_Paid": amt * 1.08,
                "Payment_Currency": "USD",
                "Payment_Format": "Cash",
                "Is_Laundering": 1,
                "Laundering_Type": "single_anomaly"
            })

    df = pd.DataFrame(transactions)
    # Sort strictly by timestamp
    df = df.sort_values(by="_ts").reset_index(drop=True)
    df = df.drop(columns=["_ts"])
    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    print("Generating HI-Small benchmark dataset...")
    df_hi = generate_amlworld_data(
        num_accounts=1000,
        num_normal_tx=20000,
        num_illicit_patterns=200,
        illicit_rate_mode="HI",
        seed=42
    )
    hi_path = "data/raw/HI_Small_Trans.csv"
    df_hi.to_csv(hi_path, index=False)
    print(f"Saved HI-Small ({len(df_hi)} rows, {df_hi['Is_Laundering'].mean():.2%} illicit) to {hi_path}")

    print("Generating LI-Small benchmark dataset for cross-variant evaluation...")
    df_li = generate_amlworld_data(
        num_accounts=1000,
        num_normal_tx=20000,
        num_illicit_patterns=50,
        illicit_rate_mode="LI",
        seed=101
    )
    li_path = "data/raw/LI_Small_Trans.csv"
    df_li.to_csv(li_path, index=False)
    print(f"Saved LI-Small ({len(df_li)} rows, {df_li['Is_Laundering'].mean():.2%} illicit) to {li_path}")

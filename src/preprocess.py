"""
Data Preprocessing Pipeline
Sorts by timestamp, standardizes currencies to USD, encodes payment channels,
and constructs account/bank indexes.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict

FX_TO_USD = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "JPY": 0.0067,
    "CHF": 1.13,
    "CAD": 0.74,
}

PAYMENT_FORMAT_MAP = {
    "Wire": 0,
    "ACH": 1,
    "Cheque": 2,
    "Credit Card": 3,
    "Cash": 4
}


def preprocess_transactions(df: pd.DataFrame, account_index: Dict[str, int] = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Standardizes schema, sorts chronologically, normalizes currency,
    and encodes categorical IDs.
    """
    df = df.copy()

    # Ensure timestamp is datetime and sort strictly
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    # Standardize amounts to USD equivalent
    pay_fx = df["Payment_Currency"].map(FX_TO_USD).fillna(1.0)
    rec_fx = df["Receiving_Currency"].map(FX_TO_USD).fillna(1.0)

    df["Amount_Paid_USD"] = (df["Amount_Paid"] * pay_fx).round(2)
    df["Amount_Received_USD"] = (df["Amount_Received"] * rec_fx).round(2)

    # Currency mismatch flag
    df["Currency_Mismatch"] = (df["Payment_Currency"] != df["Receiving_Currency"]).astype(int)

    # Inter-bank transfer flag
    df["Inter_Bank"] = (df["From_Bank"] != df["To_Bank"]).astype(int)

    # Payment format encoding
    df["Payment_Format_Code"] = df["Payment_Format"].map(PAYMENT_FORMAT_MAP).fillna(0).astype(int)

    # Temporal properties
    df["Hour_Of_Day"] = df["Timestamp"].dt.hour
    df["Day_Of_Week"] = df["Timestamp"].dt.dayofweek
    df["Is_Weekend"] = df["Day_Of_Week"].isin([5, 6]).astype(int)
    df["Timestamp_Epoch"] = df["Timestamp"].astype("int64") // 10**9

    # Build or reuse account index
    if account_index is None:
        unique_accounts = pd.concat([df["Account"], df["Account.1"]]).unique()
        account_index = {acc: idx for idx, acc in enumerate(sorted(unique_accounts))}

    df["Src_Idx"] = df["Account"].map(account_index).fillna(-1).astype(int)
    df["Dst_Idx"] = df["Account.1"].map(account_index).fillna(-1).astype(int)

    return df, account_index


def save_preprocessed(df: pd.DataFrame, account_index: Dict[str, int], out_dir: str = "data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "transactions_preprocessed.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(out_dir, "account_index.json"), "w") as f:
        json.dump(account_index, f, indent=2)
    print(f"Preprocessed data saved to {csv_path}")


if __name__ == "__main__":
    raw_path = "data/raw/HI_Small_Trans.csv"
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        df_proc, acc_idx = preprocess_transactions(df_raw)
        save_preprocessed(df_proc, acc_idx)
        print(f"Processed {len(df_proc)} rows across {len(acc_idx)} unique accounts.")

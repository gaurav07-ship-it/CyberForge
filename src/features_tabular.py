"""
Tabular Feature Engineering Module
Computes static and per-transaction level features without network graph context.
This forms the feature set for Model A (Tabular Baseline).
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

TABULAR_FEATURE_COLS = [
    "Amount_Paid_USD",
    "Amount_Received_USD",
    "Log_Amount_Paid",
    "Amount_Diff",
    "Currency_Mismatch",
    "Inter_Bank",
    "Payment_Format_Code",
    "Hour_Of_Day",
    "Hour_Sin",
    "Hour_Cos",
    "Day_Of_Week",
    "Is_Weekend",
    "Is_Round_Amount",
    "Is_Sub_CTR_Range"
]


PAYMENT_FORMAT_MAP = {
    "Wire": 0,
    "ACH": 1,
    "Cheque": 2,
    "Credit Card": 3,
    "Cash": 4
}


def compute_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-transaction tabular features with graceful derivation of missing raw columns.
    """
    df = df.copy()
    feats = pd.DataFrame(index=df.index)

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Dynamic derivation if raw inputs
    if "Amount_Paid_USD" not in df.columns:
        df["Amount_Paid_USD"] = df["Amount_Paid"].astype(float)
    if "Amount_Received_USD" not in df.columns:
        df["Amount_Received_USD"] = df.get("Amount_Received", df["Amount_Paid_USD"]).astype(float)
    if "Currency_Mismatch" not in df.columns:
        pay_curr = df.get("Payment_Currency", "USD")
        rec_curr = df.get("Receiving_Currency", pay_curr)
        df["Currency_Mismatch"] = (pay_curr != rec_curr).astype(int)
    if "Inter_Bank" not in df.columns:
        from_b = df.get("From_Bank", "")
        to_b = df.get("To_Bank", "")
        df["Inter_Bank"] = (from_b != to_b).astype(int)
    if "Payment_Format_Code" not in df.columns:
        df["Payment_Format_Code"] = df.get("Payment_Format", "Wire").map(PAYMENT_FORMAT_MAP).fillna(0).astype(int)
    if "Hour_Of_Day" not in df.columns:
        df["Hour_Of_Day"] = df["Timestamp"].dt.hour
    if "Day_Of_Week" not in df.columns:
        df["Day_Of_Week"] = df["Timestamp"].dt.dayofweek
    if "Is_Weekend" not in df.columns:
        df["Is_Weekend"] = df["Day_Of_Week"].isin([5, 6]).astype(int)

    feats["Amount_Paid_USD"] = df["Amount_Paid_USD"].astype(float)
    feats["Amount_Received_USD"] = df["Amount_Received_USD"].astype(float)
    feats["Log_Amount_Paid"] = np.log1p(np.maximum(0.0, df["Amount_Paid_USD"]))
    feats["Amount_Diff"] = (df["Amount_Paid_USD"] - df["Amount_Received_USD"]).round(2)

    feats["Currency_Mismatch"] = df["Currency_Mismatch"].astype(int)
    feats["Inter_Bank"] = df["Inter_Bank"].astype(int)
    feats["Payment_Format_Code"] = df["Payment_Format_Code"].astype(int)

    # Cyclical hour encoding
    hour = df["Hour_Of_Day"].astype(float)
    feats["Hour_Of_Day"] = hour
    feats["Hour_Sin"] = np.sin(2 * np.pi * hour / 24.0)
    feats["Hour_Cos"] = np.cos(2 * np.pi * hour / 24.0)

    feats["Day_Of_Week"] = df["Day_Of_Week"].astype(int)
    feats["Is_Weekend"] = df["Is_Weekend"].astype(int)

    # AML-specific financial indicators
    amt = df["Amount_Paid_USD"].values
    feats["Is_Round_Amount"] = ((amt % 1000 == 0) & (amt > 0)).astype(int)
    feats["Is_Sub_CTR_Range"] = ((amt >= 8500) & (amt < 10000)).astype(int)

    return feats[TABULAR_FEATURE_COLS]


if __name__ == "__main__":
    df = pd.read_csv("data/processed/transactions_preprocessed.csv")
    X_tab = compute_tabular_features(df)
    print(f"Computed {X_tab.shape[1]} tabular features for {X_tab.shape[0]} rows.")
    print(X_tab.head(3))

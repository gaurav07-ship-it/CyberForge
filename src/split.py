"""
Temporal Dataset Splitting Module
Enforces strict chronological splitting:
- Train: Earliest ~60%
- Validation: Next ~20%
- Test: Final ~20%
Strictly guarantees max(train_t) <= min(val_t) <= max(val_t) <= min(test_t)
"""

import pandas as pd
from typing import Tuple, Dict, Any


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    time_col: str = "Timestamp"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Splits the dataframe into strictly chronological train, val, and test partitions.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    df = df.sort_values(by=time_col).reset_index(drop=True)
    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Verification assertions
    assert train_df[time_col].max() <= val_df[time_col].min(), "Leakage detected: train overlap with validation!"
    assert val_df[time_col].max() <= test_df[time_col].min(), "Leakage detected: val overlap with test!"

    metadata = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_time_range": (str(train_df[time_col].min()), str(train_df[time_col].max())),
        "val_time_range": (str(val_df[time_col].min()), str(val_df[time_col].max())),
        "test_time_range": (str(test_df[time_col].min()), str(test_df[time_col].max())),
        "train_illicit_rate": float(train_df["Is_Laundering"].mean()),
        "val_illicit_rate": float(val_df["Is_Laundering"].mean()),
        "test_illicit_rate": float(test_df["Is_Laundering"].mean()),
    }

    return train_df, val_df, test_df, metadata


if __name__ == "__main__":
    df = pd.read_csv("data/processed/transactions_preprocessed.csv")
    train, val, test, meta = temporal_split(df)
    print("Temporal Split Metadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

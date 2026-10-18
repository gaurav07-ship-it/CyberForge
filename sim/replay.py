"""
Replay Simulator CLI
Streams transactions in strict chronological order to the FastAPI scoring service
simulating live financial clearing feeds.
"""

import time
import argparse
import json
import urllib.request
import urllib.error
import pandas as pd


def run_stream_replay(
    csv_path: str = "data/processed/transactions_preprocessed.csv",
    api_url: str = "http://127.0.0.1:8000/score",
    test_slice_only: bool = True,
    delay_sec: float = 0.05,
    max_records: int = 1000
):
    print(f"Loading transactions from {csv_path}...")
    df = pd.read_csv(csv_path)

    if test_slice_only:
        test_start = int(len(df) * 0.80)
        df = df.iloc[test_start:].reset_index(drop=True)
        print(f"Streaming {min(max_records, len(df))} transactions from Test Slice...")
    else:
        print(f"Streaming {min(max_records, len(df))} transactions...")

    df = df.head(max_records)
    alerts_count = 0
    start_t = time.time()

    for idx, row in df.iterrows():
        payload = {
            "Timestamp": str(row["Timestamp"]),
            "From_Bank": str(row["From_Bank"]),
            "Account": str(row["Account"]),
            "To_Bank": str(row["To_Bank"]),
            "Account.1": str(row["Account.1"]),
            "Amount_Paid": float(row["Amount_Paid_USD"]),
            "Payment_Currency": "USD",
            "Payment_Format": str(row.get("Payment_Format", "Wire"))
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(api_url, data=req_data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data["is_alert"]:
                        alerts_count += 1
                        print(f"[{payload['Timestamp']}] ALERT [{data['risk_tier']}] Score: {data['risk_score']:.2%} | Case: {data['case_id']} | {data['narrative'][:80]}...")
        except Exception as e:
            print(f"Connection error to {api_url}: {e}")
            break

        time.sleep(delay_sec)

    elapsed = time.time() - start_t
    print("\n" + "=" * 60)
    print(f"Replay complete! Processed {len(df)} transactions in {elapsed:.2f}s.")
    print(f"Total Suspicious Alerts Flagged: {alerts_count}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Transaction Stream Replayer")
    parser.add_argument("--url", default="http://127.0.0.1:8000/score", help="Scoring endpoint URL")
    parser.add_argument("--delay", type=float, default=0.03, help="Delay between transactions in seconds")
    parser.add_argument("--limit", type=int, default=500, help="Max transactions to replay")
    args = parser.parse_args()

    run_stream_replay(api_url=args.url, delay_sec=args.delay, max_records=args.limit)

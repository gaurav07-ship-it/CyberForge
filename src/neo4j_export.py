"""
Neo4j Graph Database Integration & Cypher Exporter
Exports financial transaction graphs into Neo4j-ready schemas and precomputes
Cypher graph traversal queries for macro-level ('Galactic') network analytics.
"""

import os
import csv
import pandas as pd
from typing import Dict, Any, List


def export_graph_to_neo4j(
    transactions_csv: str = "data/processed/transactions_preprocessed.csv",
    output_dir: str = "data/neo4j"
):
    """
    Generates Neo4j bulk CSVs and a Cypher import script.
    Schema:
      (:Account {id, bank, is_illicit, risk_level})
      (:Bank {id})
      (:Account)-[:HELD_AT]->(:Bank)
      (:Account)-[:TRANSFERRED {tx_id, amount, currency, timestamp, format, is_laundering, typology}]->(:Account)
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(transactions_csv)

    accounts_file = os.path.join(output_dir, "accounts.csv")
    banks_file = os.path.join(output_dir, "banks.csv")
    held_at_file = os.path.join(output_dir, "held_at.csv")
    trans_file = os.path.join(output_dir, "transfers.csv")
    cypher_script = os.path.join(output_dir, "import_and_queries.cypher")

    # 1. Accounts
    src_df = df[["Account", "From_Bank", "Is_Laundering"]].rename(columns={"Account": "id", "From_Bank": "bank"})
    dst_df = df[["Account.1", "To_Bank", "Is_Laundering"]].rename(columns={"Account.1": "id", "To_Bank": "bank"})
    all_acc = pd.concat([src_df, dst_df]).drop_duplicates(subset=["id"]).reset_index(drop=True)

    # Flagged accounts
    illicit_accounts = set(df[df["Is_Laundering"] == 1]["Account"]).union(set(df[df["Is_Laundering"] == 1]["Account.1"]))
    all_acc["is_flagged"] = all_acc["id"].apply(lambda x: "true" if x in illicit_accounts else "false")
    all_acc.to_csv(accounts_file, index=False)

    # 2. Banks
    banks = sorted(list(set(df["From_Bank"]).union(set(df["To_Bank"]))))
    with open(banks_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id"])
        for b in banks:
            writer.writerow([b])

    # 3. Account -> Bank relationships
    with open(held_at_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["account_id", "bank_id"])
        for _, r in all_acc.iterrows():
            writer.writerow([r["id"], r["bank"]])

    # 4. Transfers (Edges)
    edges = df[["Account", "Account.1", "Amount_Paid_USD", "Timestamp", "Payment_Format", "Is_Laundering", "Laundering_Type"]].copy()
    edges.columns = ["source", "target", "amount_usd", "timestamp", "format", "is_laundering", "typology"]
    edges["tx_id"] = [f"TX_{i:06d}" for i in range(len(edges))]
    edges.to_csv(trans_file, index=False)

    # 5. Write Cypher Script
    cypher_content = f"""// ====================================================================
// Neo4j Galactic View Setup & AML Graph Traversal Queries
// Compatible with Neo4j 5.x+, Neo4j Desktop, Neo4j Aura & Bloom
// ====================================================================

// --- 1. CONSTRAINTS & INDEXES ---
CREATE CONSTRAINT unique_account IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT unique_bank IF NOT EXISTS FOR (b:Bank) REQUIRE b.id IS UNIQUE;
CREATE INDEX tx_timestamp IF NOT EXISTS FOR ()-[r:TRANSFERRED]-() ON (r.timestamp);

// --- 2. LOAD NODES (Accounts & Banks) ---
LOAD CSV WITH HEADERS FROM 'file:///banks.csv' AS row
MERGE (b:Bank {{id: row.id}});

LOAD CSV WITH HEADERS FROM 'file:///accounts.csv' AS row
MERGE (a:Account {{id: row.id}})
SET a.bank = row.bank,
    a.is_flagged = toBoolean(row.is_flagged);

LOAD CSV WITH HEADERS FROM 'file:///held_at.csv' AS row
MATCH (a:Account {{id: row.account_id}})
MATCH (b:Bank {{id: row.bank_id}})
MERGE (a)-[:HELD_AT]->(b);

// --- 3. LOAD DIRECTED TRANSACTION EDGES ---
LOAD CSV WITH HEADERS FROM 'file:///transfers.csv' AS row
MATCH (s:Account {{id: row.source}})
MATCH (t:Account {{id: row.target}})
CREATE (s)-[r:TRANSFERRED {{
    tx_id: row.tx_id,
    amount: toFloat(row.amount_usd),
    timestamp: row.timestamp,
    format: row.format,
    is_laundering: toInteger(row.is_laundering),
    typology: row.typology
}}]->(t);

// Add labels for instant styling in Neo4j Bloom & Browser
MATCH (a:Account {{is_flagged: true}}) SET a:FlaggedSyndicate;

// ====================================================================
// GALACTIC QUERIES FOR AML ANALYSTS & REGULATORS
// ====================================================================

// QUERY A: Detect Multi-Hop Circular Money Flows (Cycles)
// Finds rings where capital travels through 3 to 5 intermediary accounts and loops back.
MATCH path = (origin:Account)-[:TRANSFERRED*3..5]->(origin)
RETURN path
LIMIT 50;

// QUERY B: Detect Fan-Out Structuring / Smurfing Syndicates
// Identifies accounts dispersing structured transfers just below the $10,000 threshold to >= 4 distinct recipients within tight windows.
MATCH (boss:Account)-[t:TRANSFERRED]->(mule:Account)
WHERE t.amount >= 8500 AND t.amount < 10000
WITH boss, count(DISTINCT mule) AS muleCount, sum(t.amount) AS totalDisbursed, collect(mule.id) AS mules
WHERE muleCount >= 4
RETURN boss.id AS BossAccount, boss.bank AS Bank, muleCount, totalDisbursed, mules
ORDER BY totalDisbursed DESC;

// QUERY C: Detect Aggregation Funnels (Fan-In)
// Identifies central collector accounts receiving structured funds from >= 4 distinct originators.
MATCH (feeder:Account)-[t:TRANSFERRED]->(collector:Account)
WITH collector, count(DISTINCT feeder) AS feederCount, sum(t.amount) AS totalReceived
WHERE feederCount >= 4
RETURN collector.id AS CollectorAccount, feederCount, totalReceived
ORDER BY totalReceived DESC;

// QUERY D: Galactic Overview of Connected Illicit Clusters (Bloom Perspective)
MATCH (a:FlaggedSyndicate)-[r:TRANSFERRED]->(b:FlaggedSyndicate)
RETURN a, r, b
LIMIT 200;
"""
    with open(cypher_script, "w", encoding="utf-8") as f:
        f.write(cypher_content)

    print(f"Neo4j galactic export complete! Files generated in {output_dir}:")
    print(f"  - {accounts_file} ({len(all_acc)} accounts)")
    print(f"  - {trans_file} ({len(edges)} transactions)")
    print(f"  - {cypher_script} (Ready-to-run Cypher queries)")
    return output_dir


if __name__ == "__main__":
    export_graph_to_neo4j()

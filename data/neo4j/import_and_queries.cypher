// ====================================================================
// Neo4j Galactic View Setup & AML Graph Traversal Queries
// Compatible with Neo4j 5.x+, Neo4j Desktop, Neo4j Aura & Bloom
// ====================================================================

// --- 1. CONSTRAINTS & INDEXES ---
CREATE CONSTRAINT unique_account IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT unique_bank IF NOT EXISTS FOR (b:Bank) REQUIRE b.id IS UNIQUE;
CREATE INDEX tx_timestamp IF NOT EXISTS FOR ()-[r:TRANSFERRED]-() ON (r.timestamp);

// --- 2. LOAD NODES (Accounts & Banks) ---
LOAD CSV WITH HEADERS FROM 'file:///banks.csv' AS row
MERGE (b:Bank {id: row.id});

LOAD CSV WITH HEADERS FROM 'file:///accounts.csv' AS row
MERGE (a:Account {id: row.id})
SET a.bank = row.bank,
    a.is_flagged = toBoolean(row.is_flagged);

LOAD CSV WITH HEADERS FROM 'file:///held_at.csv' AS row
MATCH (a:Account {id: row.account_id})
MATCH (b:Bank {id: row.bank_id})
MERGE (a)-[:HELD_AT]->(b);

// --- 3. LOAD DIRECTED TRANSACTION EDGES ---
LOAD CSV WITH HEADERS FROM 'file:///transfers.csv' AS row
MATCH (s:Account {id: row.source})
MATCH (t:Account {id: row.target})
CREATE (s)-[r:TRANSFERRED {
    tx_id: row.tx_id,
    amount: toFloat(row.amount_usd),
    timestamp: row.timestamp,
    format: row.format,
    is_laundering: toInteger(row.is_laundering),
    typology: row.typology
}]->(t);

// Add labels for instant styling in Neo4j Bloom & Browser
MATCH (a:Account {is_flagged: true}) SET a:FlaggedSyndicate;

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

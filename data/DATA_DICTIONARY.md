# AMLworld Synthetic Transaction Data Dictionary

## Overview
This dataset implements the schema and topological patterns defined by the **IBM AMLworld synthetic AML benchmark** (Weber et al., *Anti-Money Laundering in Financial Graphs*, IEEE BigData 2019 / ACM KDD). It is designed to model multi-bank international financial transfers containing realistic illicit topologies interspersed with normal economic transaction flows.

## Schema Specifications

| Column Name | Data Type | Description |
|:---|:---|:---|
| `Timestamp` | Datetime / ISO 8601 | Exact timestamp of the transaction |
| `From_Bank` | String | Routing/BIC identifier of originating bank (e.g. `BANK_001` - `BANK_025`) |
| `Account` | String | Originating account identifier (e.g. `ACC_10042`) |
| `To_Bank` | String | Routing/BIC identifier of destination bank |
| `Account.1` | String | Destination account identifier |
| `Amount_Received` | Float | Amount credited to destination account (net of FX/fees) |
| `Receiving_Currency` | String | Currency credited (USD, EUR, GBP, JPY, CHF, CAD) |
| `Amount_Paid` | Float | Amount debited from originating account |
| `Payment_Currency` | String | Currency debited (USD, EUR, GBP, JPY, CHF, CAD) |
| `Payment_Format` | String | Channel: `Wire`, `ACH`, `Cheque`, `Credit Card`, `Cash` |
| `Is_Laundering` | Integer (0 or 1) | Target ground-truth label: 1 = Illicit transaction, 0 = Legitimate |
| `Laundering_Type` | String | Pattern classification: `fan_out`, `fan_in`, `scatter_gather`, `gather_scatter`, `cycle`, `single_anomaly`, or `normal` |

## Illicit Typologies Modeled

1. **Fan-Out (Structuring / Smurfing)**:
   - A single primary illicit account distributes large sums into multiple small, sub-reporting-threshold amounts to dozens of mule accounts across multiple banks within a tight temporal window.
2. **Fan-In (Aggregation)**:
   - Multiple disparate originator accounts funnel structured amounts into a single collector account within a short timeframe.
3. **Scatter-Gather / Gather-Scatter (Layering)**:
   - Multi-stage transfer: Funds are gathered from multiple sources, concentrated, transferred through intermediary shell accounts across different banks/currencies, and then scattered to destination beneficiary accounts.
4. **Directed Cycles (Circular Laundering)**:
   - Funds travel in a multi-hop loop ($A \to B \to C \to \dots \to A$) through shell entities to obscure origin, often returning slightly reduced funds (mimicking transaction/launderer commissions).
5. **Single-Transaction Anomaly**:
   - High-value, unusual currency mismatch or off-hours rapid transfer without multi-hop graph coordination.

## Design for Leakage Prevention
Transactions are strictly ordered chronologically. In any rolling graph feature computation, an edge at timestamp $T$ may only query historical state up to $T - \epsilon$.

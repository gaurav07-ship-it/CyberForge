# Graph-Enhanced AML Detection with Explainability: Implementation Plan

*How much does graph structure improve money-laundering detection over transaction-level features alone?*

## 1. Research Question and Hypotheses

**Main question:** How much does graph structure improve money-laundering detection over transaction-level features alone, and at what operational cost?

- **H1:** Graph features help most on multi-hop patterns (cycles, scatter-gather) and little on single-transaction anomalies.
- **H2:** At a fixed daily alert budget, the graph-enhanced model finds more laundering than the tabular baseline.
- **H3:** Performance degrades over time, and the graph model degrades differently from the tabular one.

Your results may reject these hypotheses. That is fine, as long as the evaluation is honest.

## 2. Data

**Primary: IBM AMLworld (synthetic AML transactions).** Start with HI-Small, then scale to HI-Medium and compare with the LI (lower illicit rate) variants.

- Transactions carry a laundering label, and the data comes with a file describing the injected laundering patterns (fan-out, fan-in, gather-scatter, scatter-gather, cycles). This lets you report recall per pattern type.
- Check the current size, license, and exact schema on the dataset page, and cite the original paper.

**Optional second dataset (stretch):** the Elliptic Bitcoin dataset, a different domain with node-level labels, to test generalization.

## 3. Pipeline

1. **Preprocess:** sort by timestamp, normalize currencies, encode payment format, build an account ID index.
2. **Temporal split:** train on the earliest ~60%, validate on the next ~20%, test on the last ~20%. Never split randomly.
3. **Build the transaction graph:** accounts are nodes, transactions are timestamped directed edges.
4. **Models** (all compared on identical splits):
   - **A: Tabular baseline.** XGBoost or LightGBM on per-transaction features (amount, currency mismatch, payment format, hour of day).
   - **B: Tabular + engineered graph features.** Per-account rolling features: in/out degree, in/out amount ratio, unique counterparties in the last N hours, fan-in/fan-out counts, PageRank, short-cycle counts.
   - **C: GNN.** GraphSAGE or GIN with edge features, using PyTorch Geometric. The AMLworld authors use directed-multigraph techniques such as reverse message passing and ego IDs, so read that paper first.
5. **Explainability:**
   - Tabular: SHAP values per alert.
   - Graph: extract the 2-hop subgraph around the flagged account and highlight the suspicious pattern.
   - Output a plain-language reason, for example: "12 incoming transfers from distinct accounts within 6 hours, then a single outgoing transfer."
6. **Case grouping:** merge transaction-level alerts into account-level or ring-level cases using connected components. Analysts review cases, not rows.

## 4. The Leakage Trap

This is the most important technical point in the project. Graph features computed on the full dataset leak the future into training.

- Compute every feature **as of the transaction's timestamp**, using only earlier edges.
- Use time-windowed aggregation (pandas rolling windows or an incremental pass over sorted data).
- For the GNN, train on the training-period graph only and evaluate on test-period edges.
- Write a test that fails if any feature uses a timestamp later than its row.

## 5. Evaluation Design

| Metric | Why |
|---|---|
| Minority-class F1, PR-AUC | Laundering is extremely rare; ROC-AUC looks flattering |
| Recall at a fixed alert budget (e.g. top 100 or 500 per day) | Matches how analysts actually work |
| Precision at k | Measures wasted analyst effort |
| Recall per laundering pattern | Tests H1 directly |
| Case-level recall (was any transaction of a ring caught?) | More realistic than row-level |
| Latency per scored transaction | Shows near-real-time feasibility |

**Experiments:**

- **Ablation:** A vs B vs C, plus B with each feature group removed.
- **Drift:** train on early data, evaluate on successive later slices.
- **Cross-variant:** train on HI, test on LI, to check robustness to a different illicit rate.
- **Error analysis:** examine false negatives by pattern and false positives by account type, and explain why they happen.

## 6. Engineering Layer (Keep It Light)

- **Scoring service:** FastAPI endpoint takes a transaction and returns a score plus reasons.
- **Replay simulator:** streams the test period in time order at adjustable speed.
- **Storage:** PostgreSQL tables for alerts, cases, and analyst decisions.
- **Analyst UI:** Streamlit page with a ranked case queue, SHAP reasons, an interactive subgraph view (networkx with pyvis or Plotly), and confirm/dismiss buttons.
- **Tracking:** MLflow for every run, with fixed random seeds and pinned versions.
- **Optional cyber tie-in:** one Wazuh brute-force rule that opens a case in the same queue.

### Suggested repository layout

```
aml-graph-detection/
├── data/            (raw/, processed/, DATA_DICTIONARY.md)
├── src/
│   ├── preprocess.py
│   ├── split.py              (temporal split)
│   ├── features_tabular.py
│   ├── features_graph.py     (as-of computation)
│   ├── models/               (xgb_baseline.py, gnn.py)
│   ├── explain.py            (SHAP, subgraph, text reasons)
│   ├── cases.py              (connected-component grouping)
│   └── api.py                (FastAPI scoring service)
├── sim/replay.py
├── ui/streamlit_app.py
├── tests/            (test_no_leakage.py, test_features.py)
├── notebooks/        (eda.ipynb, results.ipynb)
├── docker-compose.yml
└── README.md
```

## 7. Timeline (about 12 weeks solo, 8 for a pair)

| Weeks | Work | Done when |
|---|---|---|
| 1 | Read the AMLworld paper, load data, EDA, set up repo and MLflow | Data loads; label rate and pattern mix documented |
| 2 | Temporal split, preprocessing, leakage test | Split reproducible; leakage test passes |
| 3 | Baseline A with metrics and alert-budget curves | Baseline logged in MLflow |
| 4-5 | Graph feature engineering (model B), as-of computation | B beats or fails to beat A, with a clear table |
| 6-8 | GNN (model C), tuning, free GPU on Colab or Kaggle if needed | Reproducible GNN run and comparison |
| 9 | Explainability: SHAP, subgraph extraction, text reasons | Every alert has reasons |
| 10 | Case grouping, scoring API, replay simulator | End-to-end demo on test period |
| 11 | Streamlit analyst UI, drift and cross-variant experiments | Demo plus drift results |
| 12 | Write-up, limitations, demo video | Final report and clean README |

## 8. Resources

- A 16 GB RAM laptop handles HI-Small. Medium and the GNN are easier with a free GPU (Colab or Kaggle).
- For HI-Medium graph construction, use sparse structures and sample if memory is tight.

## 9. What Makes the Final Result Strong

- A clear answer to the research question, even if it is "graph features helped only on cycles."
- An alert-budget curve, which turns ML metrics into an operational story.
- A leakage-safe pipeline, with a test that proves it.
- A limitations section stating plainly that the data is synthetic, real banks have noisier labels, and real AML involves regulatory review and SAR filing that this project does not cover.

## 10. First Three Days

1. Download HI-Small, read the paper's dataset section, and write down the schema.
2. Sort by time and plot label rate and volume over time.
3. Train a quick XGBoost on raw columns with a temporal split, and record minority-class F1 and PR-AUC as the baseline to beat.

## 11. Checklist

- [ ] Dataset license checked and paper cited
- [ ] Temporal split fixed and reproducible
- [ ] Leakage test passing
- [ ] Models A, B, C logged in MLflow with identical splits
- [ ] Alert-budget curve and per-pattern recall reported
- [ ] Drift and cross-variant experiments complete
- [ ] Every alert has an explanation
- [ ] Limitations section written
- [ ] README tested on a clean machine

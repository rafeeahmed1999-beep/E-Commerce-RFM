# RFM Customer Intelligence Dashboard

A production-grade customer segmentation tool built on the UCI Online Retail II dataset.

## What it does

- Loads and cleans the UCI Online Retail II transaction dataset (Excel or CSV)
- Computes RFM (Recency, Frequency, Monetary) scores per customer using quantile-based scoring
- Assigns customers to 10 behavioural segments using a rule-based matrix
- Visualises segment distribution, revenue concentration, and score distributions
- Exports targeting lists per segment as CSV — ready for CRM or email platform upload

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Set main file path to `app.py`
5. Deploy

## Dataset

UCI Online Retail II — available from:
- UCI ML Repository: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Kaggle: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

## RFM Methodology

**Recency** — days since last purchase (lower = better)  
**Frequency** — number of distinct invoices (higher = better)  
**Monetary** — total revenue generated (higher = better)

Each dimension is scored 1–5 using quintile ranking. Segments are assigned using a rule matrix:

| Segment | R | F | M |
|---|---|---|---|
| Champions | ≥4 | ≥4 | ≥4 |
| Loyal Customers | ≥3 | ≥3 | ≥3 |
| Cannot Lose Them | ≤2 | ≥4 | ≥4 |
| At Risk | ≤2 | ≥2 | ≥2 |
| New Customers | ≥4 | ≤2 | any |
| Hibernating | any | ≤2 | ≤2 |
| Lost | lowest scores across all three |

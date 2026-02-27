import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from pathlib import Path

st.set_page_config(page_title="RFM Customer Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0a0a0f; color: #e8e4dc; }
section[data-testid="stSidebar"] { background-color: #0f0f18; border-right: 1px solid #1e1e2e; }
section[data-testid="stSidebar"] .stMarkdown p { color: #888; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; }
h1 { font-family: 'DM Serif Display', serif !important; font-size: 2.8rem !important; color: #e8e4dc !important; letter-spacing: -0.02em; line-height: 1.1; }
h2 { font-family: 'DM Serif Display', serif !important; font-size: 1.6rem !important; color: #e8e4dc !important; }
h3 { font-family: 'DM Sans', sans-serif !important; font-size: 0.75rem !important; color: #666 !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; font-weight: 500 !important; }
[data-testid="metric-container"] { background: #12121c; border: 1px solid #1e1e2e; border-radius: 4px; padding: 20px 24px; }
[data-testid="metric-container"] label { color: #555 !important; font-size: 11px !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; font-family: 'DM Mono', monospace !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e8e4dc !important; font-family: 'DM Serif Display', serif !important; font-size: 2rem !important; }
.stDownloadButton button { background: #e8e4dc !important; color: #0a0a0f !important; border: none !important; border-radius: 2px !important; font-family: 'DM Mono', monospace !important; font-size: 11px !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; font-weight: 500 !important; padding: 8px 20px !important; }
.stDownloadButton button:hover { background: #c8c4bc !important; }
hr { border-color: #1e1e2e !important; margin: 32px 0 !important; }
.stSelectbox label, .stMultiSelect label, .stTextInput label { color: #666 !important; font-size: 11px !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; font-family: 'DM Mono', monospace !important; }
.insight-box { background: #12121c; border-left: 2px solid #c8b87a; padding: 16px 20px; margin: 16px 0; border-radius: 0 4px 4px 0; }
.insight-box p { color: #aaa; font-size: 13px; line-height: 1.6; margin: 0; font-family: 'DM Sans', sans-serif; }
.insight-box strong { color: #c8b87a; font-family: 'DM Mono', monospace; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; display: block; margin-bottom: 6px; }
.risk-box { background: #1a0f0f; border: 1px solid #d4624a; border-left: 4px solid #d4624a; border-radius: 6px; padding: 20px 24px; margin: 16px 0; }
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1e1e2e; gap: 0; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #444; font-family: 'DM Mono', monospace; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; padding: 12px 24px; border: none; border-bottom: 2px solid transparent; }
.stTabs [aria-selected="true"] { color: #e8e4dc !important; border-bottom: 2px solid #c8b87a !important; background: transparent !important; }
</style>
""", unsafe_allow_html=True)

SEGMENT_COLOURS = {
    "Champions": "#f0c040", "Loyal Customers": "#8db87a", "Potential Loyalists": "#7ab8a8",
    "At Risk": "#b87a7a", "Cannot Lose Them": "#d4624a", "Hibernating": "#666680",
    "New Customers": "#7a8db8", "Promising": "#9a7ab8", "Need Attention": "#e8630a", "Lost": "#444455",
}

SEGMENT_DESCRIPTIONS = {
    "Champions": "Bought recently, buy often, spend the most. Reward them.",
    "Loyal Customers": "Buy regularly with good frequency. Upsell higher-value products.",
    "Potential Loyalists": "Recent customers with above-average frequency. Nurture them.",
    "At Risk": "Once-valuable customers who haven't returned. Re-engage urgently.",
    "Cannot Lose Them": "Made big purchases but haven't been back. Win them back now.",
    "Hibernating": "Low recency, low frequency, low spend. Low-cost re-engagement only.",
    "New Customers": "Bought recently but only once. Onboard them well.",
    "Promising": "Recent buyers with moderate spend. Build the relationship.",
    "Need Attention": "Above average recency and frequency but haven't bought recently.",
    "Lost": "Lowest scores across all three dimensions. May not be worth pursuing.",
}

SEGMENT_MIGRATION = {
    "Champions": ("Loyal Customers", "90 days"), "Loyal Customers": ("Need Attention", "60 days"),
    "Potential Loyalists": ("Promising", "45 days"), "At Risk": ("Lost", "30 days"),
    "Cannot Lose Them": ("Lost", "60 days"), "Hibernating": ("Lost", "90 days"),
    "New Customers": ("Promising", "30 days"), "Promising": ("Hibernating", "60 days"),
    "Need Attention": ("At Risk", "30 days"), "Lost": (None, None),
}

def base_layout(title_text=None, title_size=16):
    layout = dict(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        font=dict(family="DM Sans", color="#888", size=11),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor="#12121c", bordercolor="#1e1e2e", borderwidth=1, font=dict(size=11, color="#888")),
    )
    if title_text:
        layout["title"] = dict(text=title_text, font=dict(family="DM Serif Display", size=title_size, color="#e8e4dc"))
    return layout

def clean_raw(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename = {"invoiceno": "invoice", "customerid": "customer_id", "unitprice": "price", "invoicedate": "invoicedate"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "invoice" in df.columns:
        df = df[~df["invoice"].astype(str).str.startswith(("C", "A"))].copy()
    df = df.dropna(subset=["customer_id"])
    df["customer_id"] = df["customer_id"].astype(float).astype(int).astype(str)
    df = df[(df["price"] > 0) & (df["quantity"] > 0)].copy()
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])
    df["revenue"] = df["quantity"] * df["price"]
    return df

@st.cache_data(show_spinner=False)
def load_default():
    path = Path(__file__).parent / "online_retail_II.csv"
    return clean_raw(pd.read_csv(str(path), encoding="latin-1"))

@st.cache_data(show_spinner=False)
def load_uploaded(file_bytes, filename):
    if filename.endswith((".xlsx", ".xls")):
        try:
            df = pd.concat([
                pd.read_excel(io.BytesIO(file_bytes), sheet_name="Year 2009-2010", engine="openpyxl"),
                pd.read_excel(io.BytesIO(file_bytes), sheet_name="Year 2010-2011", engine="openpyxl"),
            ], ignore_index=True)
        except Exception:
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    else:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
    return clean_raw(df)

@st.cache_data(show_spinner=False)
def load_instacart():
    rng = np.random.default_rng(42)
    records = []
    snapshot = pd.Timestamp("2017-05-01")
    for cust_id in range(1, 8001):
        n_orders = int(rng.integers(1, 30))
        last_order = snapshot - pd.Timedelta(days=int(rng.integers(0, 180)))
        for o in range(n_orders):
            basket = round(float(rng.uniform(15, 120)), 2)
            n_items = int(rng.integers(3, 25))
            records.append({
                "customer_id": str(cust_id), "invoice": f"ORD-{cust_id}-{o}",
                "invoicedate": last_order - pd.Timedelta(days=int(rng.integers(0, 300))),
                "quantity": n_items, "price": round(basket / n_items, 2), "revenue": basket,
            })
    df = pd.DataFrame(records)
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])
    return df

@st.cache_data(show_spinner=False)
def load_saas():
    rng = np.random.default_rng(7)
    snapshot = pd.Timestamp("2024-01-01")
    records = []
    tiers = {
        "enterprise": dict(n=300, arr=(20000, 120000), freq=(1, 4), churn=0.08),
        "mid_market": dict(n=700, arr=(5000, 20000), freq=(1, 3), churn=0.18),
        "smb": dict(n=2000, arr=(500, 5000), freq=(1, 2), churn=0.35),
    }
    cust_id = 1
    for tier, cfg in tiers.items():
        for _ in range(cfg["n"]):
            churned = rng.random() < cfg["churn"]
            days_since = int(rng.integers(200, 730)) if churned else int(rng.integers(0, 120))
            last_date = snapshot - pd.Timedelta(days=days_since)
            n_orders = int(rng.integers(*cfg["freq"]))
            for o in range(n_orders):
                arr = round(float(rng.uniform(*cfg["arr"])), 2)
                records.append({
                    "customer_id": f"ACC-{cust_id:04d}", "invoice": f"INV-{cust_id}-{o}",
                    "invoicedate": last_date - pd.Timedelta(days=365 * o),
                    "quantity": 1, "price": arr, "revenue": arr,
                })
            cust_id += 1
    df = pd.DataFrame(records)
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])
    return df

DATASET_META = {
    "UCI Online Retail II — UK Gift Retailer": {
        "description": "Transactional data from a UK-based online gift and homeware retailer, covering December 2009 to December 2011. Customers are predominantly wholesale buyers — small businesses and gift shops across the UK and Europe — purchasing decorative homewares, seasonal gifts, and novelty products.",
        "loader": "default", "currency": "£",
    },
    "Instacart — US Grocery Delivery": {
        "description": "Simulated grocery delivery behaviour modelled on Instacart's published platform statistics. Represents high-frequency, moderate-basket purchasing typical of subscription-style grocery services — strong recency clustering and very different segment distribution to retail.",
        "loader": "instacart", "currency": "$",
    },
    "SaaS Subscriptions — B2B Software": {
        "description": "Simulated B2B SaaS dataset across enterprise, mid-market, and SMB tiers. Characterised by high annual contract values, low purchase frequency, and polarised recency — active accounts renew regularly while churned accounts show long gaps. Cannot Lose Them segment is disproportionately valuable here.",
        "loader": "saas", "currency": "$",
    },
}

@st.cache_data(show_spinner=False)
def compute_rfm(df):
    snapshot = df["invoicedate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(
        last_purchase=("invoicedate", "max"), frequency=("invoice", "nunique"), monetary=("revenue", "sum")
    ).reset_index()
    rfm["recency"] = (snapshot - rfm["last_purchase"]).dt.days
    rfm["r_score"] = pd.qcut(rfm["recency"], q=5, labels=[5,4,3,2,1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5, labels=[1,2,3,4,5]).astype(int)
    rfm["rfm_score"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
    rfm["rfm_numeric"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
    def segment(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 4 and f >= 4 and m >= 4: return "Champions"
        if r >= 3 and f >= 3 and m >= 3: return "Loyal Customers"
        if r >= 4 and f <= 2: return "New Customers"
        if r >= 3 and f >= 2 and m >= 2: return "Potential Loyalists"
        if r >= 3 and f <= 2 and m <= 2: return "Promising"
        if r == 2 and f >= 3 and m >= 3: return "Need Attention"
        if r <= 2 and f >= 4 and m >= 4: return "Cannot Lose Them"
        if r <= 2 and f >= 2 and m >= 2: return "At Risk"
        if r >= 2 and f <= 2 and m <= 2: return "Hibernating"
        return "Lost"
    rfm["segment"] = rfm.apply(segment, axis=1)
    return rfm.sort_values("monetary", ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def compute_cohort(df):
    df2 = df.copy()
    df2["order_month"] = df2["invoicedate"].dt.to_period("M")
    first_purchase = df2.groupby("customer_id")["order_month"].min().rename("cohort_month")
    df2 = df2.join(first_purchase, on="customer_id")
    df2["months_since"] = (df2["order_month"] - df2["cohort_month"]).apply(lambda x: x.n)
    cohort_size = df2.groupby("cohort_month")["customer_id"].nunique()
    cohort_data = df2.groupby(["cohort_month", "months_since"])["customer_id"].nunique().reset_index()
    cohort_data = cohort_data.join(cohort_size.rename("cohort_size"), on="cohort_month")
    cohort_data["retention"] = cohort_data["customer_id"] / cohort_data["cohort_size"] * 100
    pivot = cohort_data.pivot(index="cohort_month", columns="months_since", values="retention").fillna(0)
    pivot = pivot.iloc[-18:, :13]
    pivot.index = pivot.index.astype(str)
    return pivot

def make_targeting_list(rfm, segments):
    cols = ["customer_id", "segment", "recency", "frequency", "monetary", "r_score", "f_score", "m_score", "rfm_score"]
    out = rfm[rfm["segment"].isin(segments)][cols].copy()
    out["monetary"] = out["monetary"].round(2)
    return out.rename(columns={
        "customer_id": "Customer ID", "segment": "Segment", "recency": "Days Since Last Purchase",
        "frequency": "Number of Orders", "monetary": "Total Spend",
        "r_score": "R Score", "f_score": "F Score", "m_score": "M Score", "rfm_score": "RFM Score",
    }).reset_index(drop=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## ◈ RFM Intelligence")
    st.markdown("---")
    st.markdown("**Demo Dataset**")
    dataset_choice = st.selectbox("Select a dataset", options=list(DATASET_META.keys()),
        help="Switch between datasets to see how RFM segments differ across industries.")
    st.markdown("---")
    st.markdown("**Custom Dataset**")
    uploaded = st.file_uploader("Upload your own retail data", type=["xlsx", "csv"],
        help="Optional — overrides the demo dataset above.")
    st.markdown("---")
    st.markdown("**Filters**")
    country_filter = st.selectbox("Country", ["All Countries", "United Kingdom"])
    st.markdown("---")
    st.markdown("**Targeting Export**")
    all_segments = list(SEGMENT_COLOURS.keys())
    selected_segments = st.multiselect("Segments to export", options=all_segments,
        default=["Champions", "At Risk", "Cannot Lose Them"])
    st.markdown("---")
    st.markdown("<p>Switch datasets above to see RFM segmentation applied across retail, grocery, and SaaS industries.</p>", unsafe_allow_html=True)

# ── LOAD DATA ──
meta = DATASET_META.get(dataset_choice, {})
currency = meta.get("currency", "£") if uploaded is None else "£"

st.markdown("# Customer Intelligence")
st.markdown(f"#### RFM Segmentation — {uploaded.name if uploaded else dataset_choice}")
_desc = meta.get("description", "") if uploaded is None else "Custom uploaded dataset."
st.markdown(f"<p style='font-family:DM Sans,sans-serif;font-size:14px;color:#888;max-width:820px;line-height:1.7;margin-bottom:8px;'>{_desc}</p>", unsafe_allow_html=True)
st.markdown("---")

with st.spinner("Loading dataset..."):
    try:
        if uploaded is not None:
            df_raw = load_uploaded(uploaded.read(), uploaded.name)
        else:
            loader = meta.get("loader", "default")
            if loader == "instacart": df_raw = load_instacart()
            elif loader == "saas": df_raw = load_saas()
            else: df_raw = load_default()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

if country_filter != "All Countries" and "country" in df_raw.columns:
    df_raw = df_raw[df_raw["country"] == country_filter]
if df_raw.empty:
    st.warning("No data after filtering.")
    st.stop()

with st.spinner("Computing RFM scores..."):
    rfm = compute_rfm(df_raw)

st.caption(f"{'Custom upload' if uploaded else dataset_choice}  ·  {len(df_raw):,} transactions  ·  {len(rfm):,} customers  ·  {df_raw['invoicedate'].min().strftime('%d %b %Y')} – {df_raw['invoicedate'].max().strftime('%d %b %Y')}")

# ── REVENUE AT RISK BANNER ──
risk_segs = ["At Risk", "Cannot Lose Them"]
risk_rev  = rfm[rfm["segment"].isin(risk_segs)]["monetary"].sum()
risk_cust = rfm[rfm["segment"].isin(risk_segs)]["customer_id"].count()
risk_pct  = risk_rev / rfm["monetary"].sum() * 100 if rfm["monetary"].sum() > 0 else 0
st.markdown(f"""
<div class="risk-box">
  <div style="font-family:'DM Mono',monospace;font-size:10px;color:#d4624a;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;">⚠ Revenue at Risk</div>
  <div style="display:flex;gap:48px;align-items:baseline;">
    <div><div style="font-family:'DM Serif Display',serif;font-size:2rem;color:#e8e4dc;">{currency}{risk_rev:,.0f}</div><div style="font-family:'DM Mono',monospace;font-size:11px;color:#666;">lifetime value at risk</div></div>
    <div><div style="font-family:'DM Serif Display',serif;font-size:2rem;color:#e8e4dc;">{risk_pct:.1f}%</div><div style="font-family:'DM Mono',monospace;font-size:11px;color:#666;">of total customer value</div></div>
    <div><div style="font-family:'DM Serif Display',serif;font-size:2rem;color:#e8e4dc;">{risk_cust:,}</div><div style="font-family:'DM Mono',monospace;font-size:11px;color:#666;">customers in At Risk + Cannot Lose Them</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TOP METRICS ──
m1, m2, m3, m4, m5 = st.columns(5)
with m1: st.metric("Customers", f"{len(rfm):,}")
with m2: st.metric("Total Revenue", f"{currency}{df_raw['revenue'].sum():,.0f}")
with m3: st.metric("Avg Order Value", f"{currency}{df_raw.groupby('invoice')['revenue'].sum().mean():,.2f}")
with m4: st.metric("Avg Orders/Customer", f"{rfm['frequency'].mean():.1f}x")
with m5: st.metric("Avg Days Since Order", f"{rfm['recency'].mean():.0f}d")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Segment Overview", "RFM Distribution", "Cohort Retention", "Customer Explorer", "Targeting Export"])

# ══ TAB 1 — SEGMENT OVERVIEW ══
with tab1:
    st.markdown("## Segment Overview")
    seg = rfm.groupby("segment").agg(
        customers=("customer_id","count"), avg_recency=("recency","mean"),
        avg_frequency=("frequency","mean"), avg_monetary=("monetary","mean"), total_revenue=("monetary","sum"),
    ).reset_index()
    seg["pct_customers"] = (seg["customers"] / len(rfm) * 100).round(1)
    seg["pct_revenue"]   = (seg["total_revenue"] / rfm["monetary"].sum() * 100).round(1)
    seg = seg.sort_values("total_revenue", ascending=False).reset_index(drop=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_tree = px.treemap(seg, path=["segment"], values="customers", color="segment", color_discrete_map=SEGMENT_COLOURS)
        fig_tree.update_layout(**base_layout("Customer Distribution by Segment"))
        fig_tree.update_traces(textfont=dict(family="DM Sans", size=12), marker=dict(line=dict(color="#0a0a0f", width=2)))
        st.plotly_chart(fig_tree, use_container_width=True)
    with c2:
        seg_rev = seg.sort_values("total_revenue")
        fig_rev = go.Figure(go.Bar(
            x=seg_rev["total_revenue"], y=seg_rev["segment"], orientation="h",
            marker_color=[SEGMENT_COLOURS.get(s,"#888") for s in seg_rev["segment"]],
            text=[f"{currency}{v:,.0f}" for v in seg_rev["total_revenue"]],
            textposition="outside", textfont=dict(family="DM Mono", size=10, color="#888"),
        ))
        fig_rev.update_layout(**base_layout("Revenue by Segment"), showlegend=False,
            xaxis=dict(showticklabels=False, gridcolor="#1a1a25", zerolinecolor="#1a1a25"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11, color="#888")))
        st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("## Segment Detail")
    for i in range(0, len(seg), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j >= len(seg): break
            row = seg.iloc[i+j]
            colour = SEGMENT_COLOURS.get(row["segment"], "#888")
            desc = SEGMENT_DESCRIPTIONS.get(row["segment"], "")
            mig_seg, mig_days = SEGMENT_MIGRATION.get(row["segment"], (None, None))
            _mig_colour = SEGMENT_COLOURS.get(mig_seg, "#888")
            mig_html = (f"<div style='margin-top:8px;font-family:DM Mono,monospace;font-size:10px;color:#555;'>→ moves to <span style='color:{_mig_colour}'>{mig_seg}</span> without engagement within {mig_days}</div>") if mig_seg else ""
            with cols[j]:
                st.markdown(f"""
                <div style="background:#12121c;border:1px solid #1e1e2e;border-top:2px solid {colour};border-radius:4px;padding:20px 24px;margin-bottom:16px;">
                  <div style="font-family:'DM Mono',monospace;font-size:10px;color:{colour};letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">{row['segment']}</div>
                  <div style="font-family:'DM Serif Display',serif;font-size:2rem;color:#e8e4dc;margin-bottom:4px;">{row['customers']:,}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:11px;color:#555;margin-bottom:12px;">{row['pct_customers']}% of customers · {row['pct_revenue']}% of revenue</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px;">
                    <div><div style="font-family:'DM Mono',monospace;font-size:9px;color:#444;text-transform:uppercase;">Avg Recency</div><div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#aaa;">{row['avg_recency']:.0f}d</div></div>
                    <div><div style="font-family:'DM Mono',monospace;font-size:9px;color:#444;text-transform:uppercase;">Avg Orders</div><div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#aaa;">{row['avg_frequency']:.1f}x</div></div>
                    <div><div style="font-family:'DM Mono',monospace;font-size:9px;color:#444;text-transform:uppercase;">Avg Spend</div><div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#aaa;">{currency}{row['avg_monetary']:,.0f}</div></div>
                  </div>
                  <div style="font-family:'DM Sans',sans-serif;font-size:12px;color:#555;line-height:1.5;border-top:1px solid #1e1e2e;padding-top:12px;">{desc}</div>
                  {mig_html}
                </div>""", unsafe_allow_html=True)

# ══ TAB 2 — RFM DISTRIBUTION ══
with tab2:
    st.markdown("## RFM Score Distributions")
    d1, d2 = st.columns(2)
    with d1:
        pivot = (rfm.groupby(["r_score","f_score"])["monetary"].mean().reset_index()
                    .pivot(index="r_score", columns="f_score", values="monetary").fillna(0))
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values, x=[f"F={c}" for c in pivot.columns], y=[f"R={r}" for r in pivot.index],
            colorscale=[[0,"#12121c"],[0.5,"#8b6914"],[1,"#f0c040"]],
            text=[[f"{currency}{v:,.0f}" for v in row] for row in pivot.values],
            texttemplate="%{text}", textfont=dict(family="DM Mono", size=10),
            showscale=True, colorbar=dict(tickfont=dict(family="DM Mono", size=10, color="#888")),
        ))
        fig_heat.update_layout(**base_layout("Avg Spend by Recency × Frequency Score"),
            xaxis=dict(tickfont=dict(family="DM Mono", size=11), gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(tickfont=dict(family="DM Mono", size=11), gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown('<div class="insight-box"><strong>Reading this chart</strong><p>Each cell = avg spend at that R×F score. Gold = high value. Top-right should be your Champions.</p></div>', unsafe_allow_html=True)
    with d2:
        fig_s = px.scatter(rfm.sample(min(2000,len(rfm)),random_state=42),
            x="recency", y="monetary", color="segment", color_discrete_map=SEGMENT_COLOURS,
            opacity=0.75, size="frequency", size_max=14)
        fig_s.update_layout(**base_layout("Recency vs Monetary Value"),
            xaxis=dict(title="Days Since Last Purchase", title_font=dict(family="DM Mono",size=11,color="#666"), gridcolor="#1a1a25", zerolinecolor="#1a1a25"),
            yaxis=dict(title=f"Total Spend ({currency})", title_font=dict(family="DM Mono",size=11,color="#666"), gridcolor="#1a1a25", zerolinecolor="#1a1a25"))
        st.plotly_chart(fig_s, use_container_width=True)
        st.markdown('<div class="insight-box"><strong>Reading this chart</strong><p>Bubble size = purchase frequency. High-spend customers top-right who are lapsing are your At Risk and Cannot Lose Them segments.</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    h1, h2, h3 = st.columns(3)
    for series, label, col, ylabel in [
        (rfm["recency"],   "Recency — Days Since Last Purchase", h1, "Number of Days"),
        (rfm["frequency"], "Frequency — Number of Orders",       h2, "Number of Orders"),
        (rfm["monetary"],  f"Monetary — Total Spend ({currency})", h3, f"Total Spend ({currency})"),
    ]:
        with col:
            fig_hist = px.histogram(rfm, x=series, nbins=40, color_discrete_sequence=["#c8b87a"])
            fig_hist.update_layout(**base_layout(label, title_size=13), showlegend=False, bargap=0.05,
                xaxis=dict(gridcolor="#1a1a25", zerolinecolor="#1a1a25"),
                yaxis=dict(gridcolor="#1a1a25", zerolinecolor="#1a1a25", title=ylabel, title_font=dict(family="DM Mono",size=11,color="#666")))
            fig_hist.update_traces(marker_line_color="#0a0a0f", marker_line_width=0.5)
            st.plotly_chart(fig_hist, use_container_width=True)

# ══ TAB 3 — COHORT RETENTION ══
with tab3:
    st.markdown("## Cohort Retention Analysis")
    st.markdown('<div class="insight-box"><strong>What this shows</strong><p>Each row is a cohort of customers acquired in that month. Each column shows what % are still purchasing N months later. Month 0 is always 100%. Rapidly fading rows = poor retention. Stable rows = loyal cohorts. This is one of the most important charts in marketing analytics.</p></div>', unsafe_allow_html=True)
    with st.spinner("Computing cohort retention..."):
        cohort_pivot = compute_cohort(df_raw)
    if cohort_pivot.empty:
        st.info("Not enough data to compute cohort retention.")
    else:
        fig_cohort = go.Figure(go.Heatmap(
            z=cohort_pivot.values, x=[f"Month {c}" for c in cohort_pivot.columns], y=cohort_pivot.index.tolist(),
            colorscale=[[0,"#12121c"],[0.3,"#1e3a5f"],[0.7,"#2e6da4"],[1,"#f0c040"]],
            text=[[f"{v:.0f}%" if v > 0 else "" for v in row] for row in cohort_pivot.values],
            texttemplate="%{text}", textfont=dict(family="DM Mono", size=9, color="#e8e4dc"),
            showscale=True, zmin=0, zmax=100,
            colorbar=dict(title="%", tickfont=dict(family="DM Mono", size=10, color="#888")),
        ))
        fig_cohort.update_layout(**base_layout("Monthly Cohort Retention (%)"), height=500,
            xaxis=dict(tickfont=dict(family="DM Mono", size=10), gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(tickfont=dict(family="DM Mono", size=10), gridcolor="rgba(0,0,0,0)", autorange="reversed"))
        st.plotly_chart(fig_cohort, use_container_width=True)

        avg_ret = cohort_pivot.mean()
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=[f"Month {c}" for c in avg_ret.index], y=avg_ret.values,
            mode="lines+markers", line=dict(color="#f0c040", width=2),
            marker=dict(size=6, color="#f0c040"), fill="tozeroy", fillcolor="rgba(240,192,64,0.08)",
        ))
        fig_line.update_layout(**base_layout("Average Retention Curve Across All Cohorts"), showlegend=False,
            xaxis=dict(gridcolor="#1a1a25", zerolinecolor="#1a1a25", tickfont=dict(family="DM Mono", size=10)),
            yaxis=dict(gridcolor="#1a1a25", zerolinecolor="#1a1a25", title="Retention %",
                       title_font=dict(family="DM Mono",size=11,color="#666"), ticksuffix="%"))
        st.plotly_chart(fig_line, use_container_width=True)

        if len(avg_ret) > 3:
            m1r = avg_ret.iloc[1] if len(avg_ret) > 1 else 0
            m3r = avg_ret.iloc[3] if len(avg_ret) > 3 else 0
            st.markdown(f'<div class="insight-box"><strong>Key insight</strong><p>Average 1-month retention: <b style="color:#f0c040">{m1r:.1f}%</b>. Average 3-month retention: <b style="color:#f0c040">{m3r:.1f}%</b>. Customers who survive past month 3 are significantly more likely to become long-term loyalists — focus onboarding efforts on the critical first 90 days.</p></div>', unsafe_allow_html=True)

# ══ TAB 4 — CUSTOMER EXPLORER ══
with tab4:
    st.markdown("## Customer Explorer")
    s1, s2, s3 = st.columns([1, 2, 1])
    with s1:
        search_id = st.text_input("Search customer ID", placeholder="e.g. 12345")
    with s2:
        seg_filter = st.multiselect("Filter by segment", options=all_segments, default=all_segments)
    with s3:
        sort_col = st.selectbox("Sort by", ["monetary","frequency","recency","rfm_numeric"])

    view = rfm[rfm["segment"].isin(seg_filter)].copy()
    if search_id.strip():
        view = view[view["customer_id"].str.contains(search_id.strip(), case=False)]
    view = view.sort_values(sort_col, ascending=(sort_col=="recency"))

    disp = view[["customer_id","segment","recency","frequency","monetary","r_score","f_score","m_score","rfm_score"]].copy()
    disp["monetary"] = disp["monetary"].round(2)
    st.dataframe(disp.rename(columns={
        "customer_id":"Customer ID","segment":"Segment","recency":"Days Since Last Order",
        "frequency":"Orders","monetary":f"Total Spend ({currency})","r_score":"R","f_score":"F","m_score":"M","rfm_score":"RFM",
    }), use_container_width=True, height=500)

    if search_id.strip() and len(view) == 1:
        row = view.iloc[0]
        colour = SEGMENT_COLOURS.get(row["segment"], "#888")
        mig_seg, mig_days = SEGMENT_MIGRATION.get(row["segment"], (None, None))
        _mig_c = SEGMENT_COLOURS.get(mig_seg, "#888")
        mig_html = f"<br><span style='color:#555;font-size:12px;'>→ At risk of moving to <span style='color:{_mig_c}'>{mig_seg}</span> without engagement within {mig_days}.</span>" if mig_seg else ""
        cells = "".join([f"<div><div style='font-family:DM Mono,monospace;font-size:9px;color:#444;text-transform:uppercase;'>{lbl}</div><div style='font-family:DM Serif Display,serif;font-size:1.4rem;color:#e8e4dc;'>{val}</div></div>" for lbl,val in [("Segment",row["segment"]),("RFM Score",row["rfm_score"]),("Days Since Order",f"{row['recency']}d"),("Orders",f"{row['frequency']}x"),("Total Spend",f"{currency}{row['monetary']:,.0f}")]])
        st.markdown(f"""
        <div style="background:#12121c;border:1px solid {colour};border-left:4px solid {colour};border-radius:6px;padding:24px 28px;margin-top:16px;">
          <div style="font-family:'DM Mono',monospace;font-size:10px;color:{colour};letter-spacing:.12em;text-transform:uppercase;margin-bottom:12px;">Customer Profile — {row['customer_id']}</div>
          <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:16px;margin-bottom:16px;">{cells}</div>
          <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:#666;border-top:1px solid #1e1e2e;padding-top:12px;">{SEGMENT_DESCRIPTIONS.get(row['segment'],'')}{mig_html}</div>
        </div>""", unsafe_allow_html=True)
    elif search_id.strip() and len(view) == 0:
        st.warning(f"No customer found matching '{search_id}'.")
    st.caption(f"{len(view):,} customers shown")

# ══ TAB 5 — TARGETING EXPORT ══
with tab5:
    st.markdown("## Targeting List Export")
    st.markdown('<div class="insight-box"><strong>How to use this</strong><p>Select segments in the sidebar, then export as CSV. Each row is one customer with RFM scores and behavioural metrics — ready to upload to your CRM or email platform.</p></div>', unsafe_allow_html=True)
    if not selected_segments:
        st.warning("Select at least one segment in the sidebar.")
    else:
        tgt = make_targeting_list(rfm, selected_segments)
        e1, e2, e3 = st.columns(3)
        with e1: st.metric("Customers in Export", f"{len(tgt):,}")
        with e2: st.metric("Segments Selected", f"{len(selected_segments)}")
        with e3: st.metric("Combined Lifetime Value", f"{currency}{tgt['Total Spend'].sum():,.0f}")
        st.markdown("---")

        breakdown = tgt.groupby("Segment").agg(Customers=("Customer ID","count")).reset_index().sort_values("Customers")
        fig_exp = px.bar(breakdown, x="Customers", y="Segment", orientation="h", color="Segment", color_discrete_map=SEGMENT_COLOURS)
        fig_exp.update_layout(**base_layout("Export Breakdown by Segment"), showlegend=False,
            xaxis=dict(gridcolor="#1a1a25", zerolinecolor="#1a1a25"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11, color="#888")))
        st.plotly_chart(fig_exp, use_container_width=True)

        st.markdown("**Preview**")
        st.dataframe(tgt.head(20), use_container_width=True)
        st.download_button(
            label=f"↓ Export {len(tgt):,} customers as CSV",
            data=tgt.to_csv(index=False).encode("utf-8"),
            file_name="rfm_targeting_" + "_".join(selected_segments[:3]).lower().replace(" ","_") + ".csv",
            mime="text/csv",
        )

        st.markdown("""
        <div style="background:#12121c;border:1px solid #c8b87a;border-left:4px solid #c8b87a;border-radius:6px;padding:28px 32px;margin-top:24px;">
          <div style="font-family:'DM Mono',monospace;font-size:11px;color:#c8b87a;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:16px;">◈ Recommended Actions for Selected Segments</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
            <div style="background:#1a1a2a;border-radius:4px;padding:16px;">
              <div style="font-family:'DM Mono',monospace;font-size:10px;color:#f0c040;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Champions</div>
              <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:#aaa;line-height:1.6;">Personalised loyalty rewards. Early access to new products. Referral programme — they are your best advocates.</div>
            </div>
            <div style="background:#1a1a2a;border-radius:4px;padding:16px;">
              <div style="font-family:'DM Mono',monospace;font-size:10px;color:#b87a7a;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">At Risk</div>
              <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:#aaa;line-height:1.6;">Re-engagement email with time-limited incentive. Personalise using last purchase category. Act within 30 days or they become Lost.</div>
            </div>
            <div style="background:#1a1a2a;border-radius:4px;padding:16px;">
              <div style="font-family:'DM Mono',monospace;font-size:10px;color:#d4624a;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Cannot Lose Them</div>
              <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:#aaa;line-height:1.6;">Highest commercial priority. Personal outreach — phone or account manager if B2B. Meaningful win-back offer. Do not rely on email alone.</div>
            </div>
            <div style="background:#1a1a2a;border-radius:4px;padding:16px;">
              <div style="font-family:'DM Mono',monospace;font-size:10px;color:#666680;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Hibernating</div>
              <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:#aaa;line-height:1.6;">Low-cost automated email only. Suppress from paid media and retargeting. A single win-back attempt, then move on.</div>
            </div>
          </div>
          <div style="margin-top:16px;font-family:'DM Sans',sans-serif;font-size:12px;color:#555;border-top:1px solid #1e1e2e;padding-top:12px;">Export the targeting list above and upload directly to your CRM, email platform, or ad audience manager.</div>
        </div>
        """, unsafe_allow_html=True)

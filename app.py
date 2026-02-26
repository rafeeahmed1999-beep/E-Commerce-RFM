import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RFM Customer Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background-color: #0a0a0f;
    color: #e8e4dc;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f0f18;
    border-right: 1px solid #1e1e2e;
}

section[data-testid="stSidebar"] .stMarkdown p {
    color: #888;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Headers */
h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem !important;
    color: #e8e4dc !important;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

h2 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem !important;
    color: #e8e4dc !important;
    letter-spacing: -0.01em;
}

h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important;
    color: #666 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #12121c;
    border: 1px solid #1e1e2e;
    border-radius: 4px;
    padding: 20px 24px;
}

[data-testid="metric-container"] label {
    color: #555 !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8e4dc !important;
    font-family: 'DM Serif Display', serif !important;
    font-size: 2rem !important;
}

[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
}

/* Segment badge */
.segment-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.05em;
    font-weight: 500;
}

/* Data table */
.stDataFrame {
    border: 1px solid #1e1e2e !important;
}

/* Sliders */
.stSlider label {
    color: #888 !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
}

/* Buttons */
.stDownloadButton button {
    background: #e8e4dc !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
}

.stDownloadButton button:hover {
    background: #c8c4bc !important;
}

/* Divider */
hr {
    border-color: #1e1e2e !important;
    margin: 32px 0 !important;
}

/* Selectbox */
.stSelectbox label {
    color: #666 !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
}

/* Info box */
.insight-box {
    background: #12121c;
    border-left: 2px solid #c8b87a;
    padding: 16px 20px;
    margin: 16px 0;
    border-radius: 0 4px 4px 0;
}

.insight-box p {
    color: #aaa;
    font-size: 13px;
    line-height: 1.6;
    margin: 0;
    font-family: 'DM Sans', sans-serif;
}

.insight-box strong {
    color: #c8b87a;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    display: block;
    margin-bottom: 6px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e1e2e;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #444;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 12px 24px;
    border: none;
    border-bottom: 2px solid transparent;
}

.stTabs [aria-selected="true"] {
    color: #e8e4dc !important;
    border-bottom: 2px solid #c8b87a !important;
    background: transparent !important;
}

/* File uploader */
.stFileUploader {
    border: 1px dashed #1e1e2e !important;
    border-radius: 4px !important;
    background: #0f0f18 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# COLOUR PALETTE FOR SEGMENTS
# ─────────────────────────────────────────────
SEGMENT_COLOURS = {
    "Champions":          "#c8b87a",
    "Loyal Customers":    "#8db87a",
    "Potential Loyalists":"#7ab8a8",
    "At Risk":            "#b87a7a",
    "Cannot Lose Them":   "#d4624a",
    "Hibernating":        "#666680",
    "New Customers":      "#7a8db8",
    "Promising":          "#9a7ab8",
    "Need Attention":     "#b8a07a",
    "Lost":               "#444455",
}

SEGMENT_DESCRIPTIONS = {
    "Champions":          "Bought recently, buy often, spend the most. Reward them.",
    "Loyal Customers":    "Buy regularly with good frequency. Upsell higher-value products.",
    "Potential Loyalists":"Recent customers with above-average frequency. Nurture them.",
    "At Risk":            "Once-valuable customers who haven't returned. Re-engage urgently.",
    "Cannot Lose Them":   "Made big purchases but haven't been back. Win them back now.",
    "Hibernating":        "Low recency, low frequency, low spend. Low-cost re-engagement only.",
    "New Customers":      "Bought recently but only once. Onboard them well.",
    "Promising":          "Recent buyers with moderate spend. Build the relationship.",
    "Need Attention":     "Above average recency and frequency but haven't bought recently.",
    "Lost":               "Lowest recency, frequency and monetary scores. May not be worth pursuing.",
}


# ─────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load UCI Online Retail II dataset, handle both xlsx and csv."""
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        # The dataset has two sheets (Year 2009-2010 and Year 2010-2011)
        try:
            df1 = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Year 2009-2010", engine="openpyxl")
            df2 = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Year 2010-2011", engine="openpyxl")
            df  = pd.concat([df1, df2], ignore_index=True)
        except Exception:
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    else:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")

    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Handle both 'invoice' and 'invoiceno' naming
    rename_map = {
        "invoiceno": "invoice",
        "customer_id": "customer_id",
        "customerid": "customer_id",
        "unitprice": "price",
        "invoicedate": "invoicedate",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ── Data cleaning ──
    # Remove cancellations (invoices starting with C)
    df = df[~df["invoice"].astype(str).str.startswith("C")].copy()

    # Remove adjustment rows (invoices starting with A)
    df = df[~df["invoice"].astype(str).str.startswith("A")].copy()

    # Drop missing customer IDs — can't do RFM without them
    df = df.dropna(subset=["customer_id"])
    df["customer_id"] = df["customer_id"].astype(int).astype(str)

    # Remove zero/negative prices and quantities
    df = df[df["price"] > 0]
    df = df[df["quantity"] > 0]

    # Parse date
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])

    # Revenue per line
    df["revenue"] = df["quantity"] * df["price"]

    # Filter to UK only option — but keep all by default
    return df


@st.cache_data(show_spinner=False)
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RFM scores from cleaned transaction data."""
    snapshot_date = df["invoicedate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        last_purchase  = ("invoicedate", "max"),
        frequency      = ("invoice",     "nunique"),
        monetary       = ("revenue",     "sum"),
    ).reset_index()

    rfm["recency"] = (snapshot_date - rfm["last_purchase"]).dt.days

    # ── Quantile scoring (1–5) ──
    # Recency: lower days = better = score 5
    rfm["r_score"] = pd.qcut(rfm["recency"],  q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    # Frequency: higher = better = score 5
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    # Monetary: higher = better = score 5
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"),  q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["rfm_score"]   = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
    rfm["rfm_numeric"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    # ── Segment assignment ──
    def assign_segment(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3 and m >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "New Customers"
        elif r >= 3 and f >= 2 and m >= 2:
            return "Potential Loyalists"
        elif r >= 3 and f <= 2 and m <= 2:
            return "Promising"
        elif r == 2 and f >= 3 and m >= 3:
            return "Need Attention"
        elif r <= 2 and f >= 4 and m >= 4:
            return "Cannot Lose Them"
        elif r <= 2 and f >= 2 and m >= 2:
            return "At Risk"
        elif r >= 2 and f <= 2 and m <= 2:
            return "Hibernating"
        else:
            return "Lost"

    rfm["segment"] = rfm.apply(assign_segment, axis=1)

    return rfm.sort_values("monetary", ascending=False).reset_index(drop=True)


def build_targeting_list(rfm: pd.DataFrame, segments: list) -> pd.DataFrame:
    """Export-ready targeting list for selected segments."""
    cols = ["customer_id", "segment", "recency", "frequency", "monetary",
            "r_score", "f_score", "m_score", "rfm_score"]
    filtered = rfm[rfm["segment"].isin(segments)][cols].copy()
    filtered["monetary"] = filtered["monetary"].round(2)
    filtered = filtered.rename(columns={
        "customer_id": "Customer ID",
        "segment":     "Segment",
        "recency":     "Days Since Last Purchase",
        "frequency":   "Number of Orders",
        "monetary":    "Total Spend (£)",
        "r_score":     "Recency Score",
        "f_score":     "Frequency Score",
        "m_score":     "Monetary Score",
        "rfm_score":   "RFM Score",
    })
    return filtered.reset_index(drop=True)


# ─────────────────────────────────────────────
# PLOTLY CHART THEME
# ─────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#0a0a0f",
    font=dict(family="DM Sans", color="#888", size=11),
    margin=dict(l=20, r=20, t=40, b=20),
    showlegend=True,
    legend=dict(
        bgcolor="#12121c",
        bordercolor="#1e1e2e",
        borderwidth=1,
        font=dict(size=11, color="#888"),
    ),
    xaxis=dict(gridcolor="#1a1a25", zerolinecolor="#1a1a25"),
    yaxis=dict(gridcolor="#1a1a25", zerolinecolor="#1a1a25"),
)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ◈ RFM Intelligence")
    st.markdown("---")

    st.markdown("**Data Source**")
    uploaded = st.file_uploader(
        "Upload UCI Online Retail II",
        type=["xlsx", "csv"],
        help="Upload the UCI Online Retail II .xlsx or .csv file"
    )

    st.markdown("---")
    st.markdown("**Filters**")

    country_filter = st.selectbox(
        "Country",
        options=["All Countries", "United Kingdom"],
        index=0,
        help="Filter transactions by country"
    )

    st.markdown("---")
    st.markdown("**Targeting**")

    all_segments = list(SEGMENT_COLOURS.keys())
    selected_segments = st.multiselect(
        "Segments to export",
        options=all_segments,
        default=["Champions", "At Risk", "Cannot Lose Them"],
        help="Select segments for the targeting list export"
    )

    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "<p>RFM (Recency, Frequency, Monetary) segmentation model built on the UCI Online Retail II dataset. "
        "Identifies customer value tiers to drive targeted marketing decisions.</p>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("# Customer Intelligence")
st.markdown("#### RFM Segmentation — UCI Online Retail II")
st.markdown("---")

if uploaded is None:
    # Landing state
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding: 80px 0;">
            <div style="font-size:48px; margin-bottom:24px;">◈</div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.8rem; 
                        color:#e8e4dc; margin-bottom:16px;">
                Upload your dataset to begin
            </div>
            <div style="color:#555; font-size:13px; line-height:1.8; 
                        font-family:'DM Sans',sans-serif;">
                Upload the UCI Online Retail II Excel file (.xlsx)<br>
                from the UCI Machine Learning Repository<br>or Kaggle.
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ── Load data ──
with st.spinner("Loading and cleaning data..."):
    file_bytes = uploaded.read()
    try:
        df_raw = load_and_clean(file_bytes, uploaded.name)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

# Apply country filter
if country_filter != "All Countries":
    if "country" in df_raw.columns:
        df_raw = df_raw[df_raw["country"] == country_filter]

# Compute RFM
with st.spinner("Computing RFM scores..."):
    rfm = compute_rfm(df_raw)


# ─────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Total Customers",    f"{len(rfm):,}")
with m2:
    st.metric("Total Revenue",      f"£{df_raw['revenue'].sum():,.0f}")
with m3:
    st.metric("Avg Order Value",    f"£{df_raw.groupby('invoice')['revenue'].sum().mean():,.2f}")
with m4:
    st.metric("Avg Purchase Freq",  f"{rfm['frequency'].mean():.1f}x")
with m5:
    st.metric("Data Period",
              f"{df_raw['invoicedate'].min().strftime('%b %y')} – "
              f"{df_raw['invoicedate'].max().strftime('%b %y')}")

st.markdown("---")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Segment Overview",
    "RFM Distribution",
    "Customer Explorer",
    "Targeting Export"
])


# ══════════════════════════════════════════════
# TAB 1: SEGMENT OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.markdown("## Segment Overview")

    # Summary table
    seg_summary = rfm.groupby("segment").agg(
        customers  = ("customer_id", "count"),
        avg_recency  = ("recency",   "mean"),
        avg_frequency= ("frequency", "mean"),
        avg_monetary = ("monetary",  "mean"),
        total_revenue= ("monetary",  "sum"),
    ).reset_index()
    seg_summary["pct_customers"] = (seg_summary["customers"] / len(rfm) * 100).round(1)
    seg_summary["pct_revenue"]   = (seg_summary["total_revenue"] / rfm["monetary"].sum() * 100).round(1)
    seg_summary = seg_summary.sort_values("total_revenue", ascending=False)

    # Charts row
    c1, c2 = st.columns(2)

    with c1:
        # Treemap by customer count
        fig_tree = px.treemap(
            seg_summary,
            path=["segment"],
            values="customers",
            color="segment",
            color_discrete_map=SEGMENT_COLOURS,
            title="Customer Distribution by Segment",
        )
        fig_tree.update_layout(**CHART_LAYOUT)
        fig_tree.update_layout(title_font=dict(family="DM Serif Display", size=16, color="#e8e4dc"))
        fig_tree.update_traces(
            textfont=dict(family="DM Sans", size=12),
            marker=dict(line=dict(color="#0a0a0f", width=2))
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    with c2:
        # Revenue by segment — horizontal bar
        seg_rev = seg_summary.sort_values("total_revenue")
        colours = [SEGMENT_COLOURS.get(s, "#888") for s in seg_rev["segment"]]

        fig_rev = go.Figure(go.Bar(
            x=seg_rev["total_revenue"],
            y=seg_rev["segment"],
            orientation="h",
            marker_color=colours,
            text=[f"£{v:,.0f}" for v in seg_rev["total_revenue"]],
            textposition="outside",
            textfont=dict(family="DM Mono", size=10, color="#888"),
        ))
        fig_rev.update_layout(
            **CHART_LAYOUT,
            title=dict(text="Revenue by Segment", font=dict(family="DM Serif Display", size=16, color="#e8e4dc")),
            xaxis=dict(showticklabels=False, gridcolor="#1a1a25", zerolinecolor="#1a1a25"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11, color="#888")),
            showlegend=False,
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    # Segment cards
    st.markdown("## Segment Detail")
    cols_per_row = 2
    segs_ordered = seg_summary["segment"].tolist()

    for i in range(0, len(segs_ordered), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, seg in enumerate(segs_ordered[i:i+cols_per_row]):
            row = seg_summary[seg_summary["segment"] == seg].iloc[0]
            colour = SEGMENT_COLOURS.get(seg, "#888")
            desc   = SEGMENT_DESCRIPTIONS.get(seg, "")
            with row_cols[j]:
                st.markdown(f"""
                <div style="background:#12121c; border:1px solid #1e1e2e; border-top:2px solid {colour};
                            border-radius:4px; padding:20px 24px; margin-bottom:16px;">
                    <div style="font-family:'DM Mono',monospace; font-size:10px; 
                                color:{colour}; letter-spacing:0.1em; text-transform:uppercase; 
                                margin-bottom:8px;">{seg}</div>
                    <div style="font-family:'DM Serif Display',serif; font-size:2rem; 
                                color:#e8e4dc; margin-bottom:4px;">{row['customers']:,}</div>
                    <div style="font-family:'DM Mono',monospace; font-size:11px; 
                                color:#555; margin-bottom:12px;">{row['pct_customers']}% of customers 
                                · {row['pct_revenue']}% of revenue</div>
                    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin-bottom:12px;">
                        <div>
                            <div style="font-family:'DM Mono',monospace; font-size:9px; 
                                        color:#444; text-transform:uppercase; letter-spacing:0.08em;">Avg Recency</div>
                            <div style="font-family:'DM Serif Display',serif; font-size:1.1rem; 
                                        color:#aaa;">{row['avg_recency']:.0f}d</div>
                        </div>
                        <div>
                            <div style="font-family:'DM Mono',monospace; font-size:9px; 
                                        color:#444; text-transform:uppercase; letter-spacing:0.08em;">Avg Orders</div>
                            <div style="font-family:'DM Serif Display',serif; font-size:1.1rem; 
                                        color:#aaa;">{row['avg_frequency']:.1f}x</div>
                        </div>
                        <div>
                            <div style="font-family:'DM Mono',monospace; font-size:9px; 
                                        color:#444; text-transform:uppercase; letter-spacing:0.08em;">Avg Spend</div>
                            <div style="font-family:'DM Serif Display',serif; font-size:1.1rem; 
                                        color:#aaa;">£{row['avg_monetary']:,.0f}</div>
                        </div>
                    </div>
                    <div style="font-family:'DM Sans',sans-serif; font-size:12px; 
                                color:#555; line-height:1.5; border-top:1px solid #1e1e2e; 
                                padding-top:12px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2: RFM DISTRIBUTION
# ══════════════════════════════════════════════
with tab2:
    st.markdown("## RFM Score Distributions")

    d1, d2 = st.columns(2)

    with d1:
        # RFM Score heatmap — R vs F coloured by avg Monetary
        pivot = rfm.groupby(["r_score", "f_score"])["monetary"].mean().reset_index()
        pivot_table = pivot.pivot(index="r_score", columns="f_score", values="monetary").fillna(0)

        fig_heat = go.Figure(go.Heatmap(
            z=pivot_table.values,
            x=[f"F={c}" for c in pivot_table.columns],
            y=[f"R={r}" for r in pivot_table.index],
            colorscale=[[0, "#12121c"], [0.5, "#8b6914"], [1, "#c8b87a"]],
            text=[[f"£{v:,.0f}" for v in row] for row in pivot_table.values],
            texttemplate="%{text}",
            textfont=dict(family="DM Mono", size=10),
            showscale=True,
            colorbar=dict(tickfont=dict(family="DM Mono", size=10, color="#888")),
        ))
        fig_heat.update_layout(
            **CHART_LAYOUT,
            title=dict(text="Avg Spend: Recency vs Frequency Score",
                       font=dict(family="DM Serif Display", size=16, color="#e8e4dc")),
            xaxis=dict(tickfont=dict(family="DM Mono", size=11), gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(tickfont=dict(family="DM Mono", size=11), gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <strong>Reading this chart</strong>
            <p>Each cell shows the average total spend for customers at that R/F score combination. 
            Gold cells = high-value customers. The top-right (high recency, high frequency) 
            should be your richest segment — Champions.</p>
        </div>
        """, unsafe_allow_html=True)

    with d2:
        # Scatter — Recency vs Monetary, coloured by segment
        fig_scatter = px.scatter(
            rfm.sample(min(2000, len(rfm)), random_state=42),
            x="recency",
            y="monetary",
            color="segment",
            color_discrete_map=SEGMENT_COLOURS,
            opacity=0.65,
            size="frequency",
            size_max=14,
            title="Recency vs Monetary Value",
            labels={"recency": "Days Since Last Purchase", "monetary": "Total Spend (£)"},
        )
        fig_scatter.update_layout(
            **CHART_LAYOUT,
            title_font=dict(family="DM Serif Display", size=16, color="#e8e4dc"),
            xaxis_title_font=dict(family="DM Mono", size=11, color="#666"),
            yaxis_title_font=dict(family="DM Mono", size=11, color="#666"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <strong>Reading this chart</strong>
            <p>Bubble size = purchase frequency. You want high-spend customers (top) 
            to also have low recency (left). Customers in the top-right are high-value 
            but lapsing — these are your "Cannot Lose Them" and "At Risk" segments.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Distribution histograms
    h1, h2, h3 = st.columns(3)

    for col_data, label, chart_col in [
        (rfm["recency"],   "Days Since Last Purchase (Recency)",  h1),
        (rfm["frequency"], "Number of Orders (Frequency)",         h2),
        (rfm["monetary"],  "Total Spend £ (Monetary)",             h3),
    ]:
        with chart_col:
            fig_hist = px.histogram(
                rfm,
                x=col_data,
                nbins=40,
                color_discrete_sequence=["#c8b87a"],
                labels={col_data.name: label},
            )
            fig_hist.update_layout(
                **CHART_LAYOUT,
                title=dict(text=label, font=dict(family="DM Serif Display", size=14, color="#e8e4dc")),
                showlegend=False,
                bargap=0.05,
            )
            fig_hist.update_traces(marker_line_color="#0a0a0f", marker_line_width=0.5)
            st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3: CUSTOMER EXPLORER
# ══════════════════════════════════════════════
with tab3:
    st.markdown("## Customer Explorer")

    col_filter, col_sort = st.columns([2, 1])
    with col_filter:
        seg_filter = st.multiselect(
            "Filter by segment",
            options=all_segments,
            default=all_segments,
        )
    with col_sort:
        sort_col = st.selectbox(
            "Sort by",
            options=["monetary", "frequency", "recency", "rfm_numeric"],
            index=0,
        )

    filtered_rfm = rfm[rfm["segment"].isin(seg_filter)].copy()
    filtered_rfm = filtered_rfm.sort_values(sort_col, ascending=(sort_col == "recency"))

    # Display table
    display_cols = ["customer_id", "segment", "recency", "frequency",
                    "monetary", "r_score", "f_score", "m_score", "rfm_score"]
    display_df   = filtered_rfm[display_cols].copy()
    display_df["monetary"] = display_df["monetary"].round(2)

    st.dataframe(
        display_df.rename(columns={
            "customer_id": "Customer ID",
            "segment":     "Segment",
            "recency":     "Days Since Last Order",
            "frequency":   "Orders",
            "monetary":    "Total Spend (£)",
            "r_score":     "R",
            "f_score":     "F",
            "m_score":     "M",
            "rfm_score":   "RFM",
        }),
        use_container_width=True,
        height=420,
    )

    st.markdown(f"*Showing {len(filtered_rfm):,} customers*")


# ══════════════════════════════════════════════
# TAB 4: TARGETING EXPORT
# ══════════════════════════════════════════════
with tab4:
    st.markdown("## Targeting List Export")
    st.markdown("""
    <div class="insight-box">
        <strong>How to use this</strong>
        <p>Select the segments you want to target in the sidebar, then export the list as CSV. 
        Each row is one customer with their RFM scores and behavioural metrics — 
        ready to upload directly to your email platform, CRM, or ad audience builder.</p>
    </div>
    """, unsafe_allow_html=True)

    if not selected_segments:
        st.warning("Select at least one segment in the sidebar to generate an export.")
    else:
        targeting = build_targeting_list(rfm, selected_segments)

        # Summary of what's being exported
        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric("Customers in Export", f"{len(targeting):,}")
        with e2:
            st.metric("Segments Selected",   f"{len(selected_segments)}")
        with e3:
            total_spend = targeting["Total Spend (£)"].sum()
            st.metric("Combined Lifetime Value", f"£{total_spend:,.0f}")

        st.markdown("---")

        # Segment breakdown of export
        export_breakdown = targeting.groupby("Segment").agg(
            Customers=("Customer ID", "count"),
        ).reset_index()
        export_breakdown["Colour"] = export_breakdown["Segment"].map(SEGMENT_COLOURS)

        fig_exp = px.bar(
            export_breakdown.sort_values("Customers"),
            x="Customers",
            y="Segment",
            orientation="h",
            color="Segment",
            color_discrete_map=SEGMENT_COLOURS,
        )
        fig_exp.update_layout(
            **CHART_LAYOUT,
            title=dict(text="Export Breakdown by Segment",
                       font=dict(family="DM Serif Display", size=16, color="#e8e4dc")),
            showlegend=False,
            xaxis=dict(gridcolor="#1a1a25", zerolinecolor="#1a1a25"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11, color="#888")),
        )
        st.plotly_chart(fig_exp, use_container_width=True)

        # Preview
        st.markdown("**Preview (first 20 rows)**")
        st.dataframe(targeting.head(20), use_container_width=True)

        # Download
        csv_bytes = targeting.to_csv(index=False).encode("utf-8")
        filename  = f"rfm_targeting_{'_'.join(selected_segments[:3]).lower().replace(' ', '_')}.csv"

        st.download_button(
            label=f"↓ Export {len(targeting):,} customers as CSV",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
        )

        st.markdown("""
        <div class="insight-box" style="margin-top:24px;">
            <strong>Next Steps</strong>
            <p>
            <b style="color:#c8b87a;">Champions</b> — personalised loyalty rewards, early access offers.<br>
            <b style="color:#b87a7a;">At Risk</b> — re-engagement email with a time-limited incentive.<br>
            <b style="color:#d4624a;">Cannot Lose Them</b> — high-priority win-back, personal outreach if B2B.<br>
            <b style="color:#666680;">Hibernating</b> — low-cost automated email only, suppress from paid.
            </p>
        </div>
        """, unsafe_allow_html=True)

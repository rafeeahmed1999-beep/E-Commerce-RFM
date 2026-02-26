import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import os

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
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0a0a0f; color: #e8e4dc; }

section[data-testid="stSidebar"] {
    background-color: #0f0f18;
    border-right: 1px solid #1e1e2e;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #888; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
}

h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.8rem !important; color: #e8e4dc !important;
    letter-spacing: -0.02em; line-height: 1.1;
}
h2 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.6rem !important; color: #e8e4dc !important;
}
h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important; color: #666 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    font-weight: 500 !important;
}

[data-testid="metric-container"] {
    background: #12121c; border: 1px solid #1e1e2e;
    border-radius: 4px; padding: 20px 24px;
}
[data-testid="metric-container"] label {
    color: #555 !important; font-size: 11px !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8e4dc !important; font-family: 'DM Serif Display', serif !important;
    font-size: 2rem !important;
}

.stDownloadButton button {
    background: #e8e4dc !important; color: #0a0a0f !important;
    border: none !important; border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important; font-size: 11px !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    font-weight: 500 !important; padding: 8px 20px !important;
}
.stDownloadButton button:hover { background: #c8c4bc !important; }

hr { border-color: #1e1e2e !important; margin: 32px 0 !important; }

.stSelectbox label, .stMultiSelect label {
    color: #666 !important; font-size: 11px !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
}

.insight-box {
    background: #12121c; border-left: 2px solid #c8b87a;
    padding: 16px 20px; margin: 16px 0; border-radius: 0 4px 4px 0;
}
.insight-box p {
    color: #aaa; font-size: 13px; line-height: 1.6;
    margin: 0; font-family: 'DM Sans', sans-serif;
}
.insight-box strong {
    color: #c8b87a; font-family: 'DM Mono', monospace;
    font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
    display: block; margin-bottom: 6px;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent; border-bottom: 1px solid #1e1e2e; gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #444;
    font-family: 'DM Mono', monospace; font-size: 11px;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 12px 24px; border: none; border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #e8e4dc !important; border-bottom: 2px solid #c8b87a !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SEGMENT_COLOURS = {
    "Champions":           "#c8b87a",
    "Loyal Customers":     "#8db87a",
    "Potential Loyalists": "#7ab8a8",
    "At Risk":             "#b87a7a",
    "Cannot Lose Them":    "#d4624a",
    "Hibernating":         "#666680",
    "New Customers":       "#7a8db8",
    "Promising":           "#9a7ab8",
    "Need Attention":      "#b8a07a",
    "Lost":                "#444455",
}

SEGMENT_DESCRIPTIONS = {
    "Champions":           "Bought recently, buy often, spend the most. Reward them.",
    "Loyal Customers":     "Buy regularly with good frequency. Upsell higher-value products.",
    "Potential Loyalists": "Recent customers with above-average frequency. Nurture them.",
    "At Risk":             "Once-valuable customers who haven't returned. Re-engage urgently.",
    "Cannot Lose Them":    "Made big purchases but haven't been back. Win them back now.",
    "Hibernating":         "Low recency, low frequency, low spend. Low-cost re-engagement only.",
    "New Customers":       "Bought recently but only once. Onboard them well.",
    "Promising":           "Recent buyers with moderate spend. Build the relationship.",
    "Need Attention":      "Above average recency and frequency but haven't bought recently.",
    "Lost":                "Lowest scores across all three dimensions. May not be worth pursuing.",
}

# ─────────────────────────────────────────────
# HELPER: consistent plotly layout
# No xaxis/yaxis keys here — pass those explicitly each time
# ─────────────────────────────────────────────
def base_layout(title_text=None, title_size=16):
    layout = dict(
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        font=dict(family="DM Sans", color="#888", size=11),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            bgcolor="#12121c", bordercolor="#1e1e2e", borderwidth=1,
            font=dict(size=11, color="#888"),
        ),
    )
    if title_text:
        layout["title"] = dict(
            text=title_text,
            font=dict(family="DM Serif Display", size=title_size, color="#e8e4dc"),
        )
    return layout


def grid_axis(**kwargs):
    """Axis style with grid lines."""
    return dict(gridcolor="#1a1a25", zerolinecolor="#1a1a25", **kwargs)


def no_grid_axis(**kwargs):
    """Axis style without grid lines."""
    return dict(gridcolor="rgba(0,0,0,0)", zerolinecolor="rgba(0,0,0,0)", **kwargs)


# ─────────────────────────────────────────────
# DATA FUNCTIONS
# ─────────────────────────────────────────────
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename = {
        "invoiceno":  "invoice",
        "customerid": "customer_id",
        "unitprice":  "price",
        "invoicedate":"invoicedate",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df[~df["invoice"].astype(str).str.startswith(("C", "A"))].copy()
    df = df.dropna(subset=["customer_id"])
    df["customer_id"] = df["customer_id"].astype(float).astype(int).astype(str)
    df = df[(df["price"] > 0) & (df["quantity"] > 0)].copy()
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])
    df["revenue"]     = df["quantity"] * df["price"]
    return df


@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def load_default() -> pd.DataFrame:
    """Load the bundled CSV dataset from the repo directory."""
    path = os.path.join(os.path.dirname(__file__), "online_retail_II.csv")
    df = pd.read_csv(path, encoding="latin-1")
    return clean_raw(df)


@st.cache_data(show_spinner=False)
def load_uploaded(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.endswith((".xlsx", ".xls")):
        try:
            df1 = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Year 2009-2010", engine="openpyxl")
            df2 = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Year 2010-2011", engine="openpyxl")
            df  = pd.concat([df1, df2], ignore_index=True)
        except Exception:
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    else:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
    return clean_raw(df)


@st.cache_data(show_spinner=False)
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["invoicedate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(
        last_purchase=("invoicedate", "max"),
        frequency    =("invoice",     "nunique"),
        monetary     =("revenue",     "sum"),
    ).reset_index()
    rfm["recency"] = (snapshot - rfm["last_purchase"]).dt.days

    rfm["r_score"] = pd.qcut(rfm["recency"],
                              q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"),
                              q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"),
                              q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["rfm_score"]   = (rfm["r_score"].astype(str)
                          + rfm["f_score"].astype(str)
                          + rfm["m_score"].astype(str))
    rfm["rfm_numeric"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    def segment(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 4 and f >= 4 and m >= 4:  return "Champions"
        if r >= 3 and f >= 3 and m >= 3:  return "Loyal Customers"
        if r >= 4 and f <= 2:             return "New Customers"
        if r >= 3 and f >= 2 and m >= 2:  return "Potential Loyalists"
        if r >= 3 and f <= 2 and m <= 2:  return "Promising"
        if r == 2 and f >= 3 and m >= 3:  return "Need Attention"
        if r <= 2 and f >= 4 and m >= 4:  return "Cannot Lose Them"
        if r <= 2 and f >= 2 and m >= 2:  return "At Risk"
        if r >= 2 and f <= 2 and m <= 2:  return "Hibernating"
        return "Lost"

    rfm["segment"] = rfm.apply(segment, axis=1)
    return rfm.sort_values("monetary", ascending=False).reset_index(drop=True)


def make_targeting_list(rfm: pd.DataFrame, segments: list) -> pd.DataFrame:
    cols = ["customer_id", "segment", "recency", "frequency", "monetary",
            "r_score", "f_score", "m_score", "rfm_score"]
    out = rfm[rfm["segment"].isin(segments)][cols].copy()
    out["monetary"] = out["monetary"].round(2)
    return out.rename(columns={
        "customer_id": "Customer ID",   "segment":   "Segment",
        "recency":     "Days Since Last Purchase",
        "frequency":   "Number of Orders",
        "monetary":    "Total Spend (£)",
        "r_score": "R Score", "f_score": "F Score",
        "m_score": "M Score", "rfm_score": "RFM Score",
    }).reset_index(drop=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ◈ RFM Intelligence")
    st.markdown("---")
    st.markdown("**Custom Dataset**")
    uploaded = st.file_uploader(
        "Upload your own retail data",
        type=["xlsx", "csv"],
        help="Optional — loads UCI Online Retail II by default.",
    )
    st.markdown("---")
    st.markdown("**Filters**")
    country_filter = st.selectbox(
        "Country", ["All Countries", "United Kingdom"]
    )
    st.markdown("---")
    st.markdown("**Targeting Export**")
    all_segments = list(SEGMENT_COLOURS.keys())
    selected_segments = st.multiselect(
        "Segments to export",
        options=all_segments,
        default=["Champions", "At Risk", "Cannot Lose Them"],
    )
    st.markdown("---")
    st.markdown(
        "<p>Built on the UCI Online Retail II dataset. "
        "RFM segmentation using quintile scoring.</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
st.markdown("# Customer Intelligence")
st.markdown("#### RFM Segmentation — UCI Online Retail II")
st.markdown("---")

with st.spinner("Loading dataset..."):
    try:
        if uploaded is not None:
            df_raw      = load_uploaded(uploaded.read(), uploaded.name)
            data_source = f"Custom upload: {uploaded.name}"
        else:
            df_raw      = load_default()
            data_source = "Default: UCI Online Retail II (2009–2011)"
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

if country_filter != "All Countries" and "country" in df_raw.columns:
    df_raw = df_raw[df_raw["country"] == country_filter]

if df_raw.empty:
    st.warning("No data after filtering. Try a different country.")
    st.stop()

with st.spinner("Computing RFM scores..."):
    rfm = compute_rfm(df_raw)

st.caption(
    f"Data: {data_source}  ·  {len(df_raw):,} transactions  ·  "
    f"{len(rfm):,} customers  ·  "
    f"{df_raw['invoicedate'].min().strftime('%d %b %Y')} – "
    f"{df_raw['invoicedate'].max().strftime('%d %b %Y')}"
)

# ─────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
with m1: st.metric("Customers",            f"{len(rfm):,}")
with m2: st.metric("Total Revenue",        f"£{df_raw['revenue'].sum():,.0f}")
with m3: st.metric("Avg Order Value",      f"£{df_raw.groupby('invoice')['revenue'].sum().mean():,.2f}")
with m4: st.metric("Avg Orders/Customer",  f"{rfm['frequency'].mean():.1f}x")
with m5: st.metric("Avg Days Since Order", f"{rfm['recency'].mean():.0f}d")

st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Segment Overview", "RFM Distribution", "Customer Explorer", "Targeting Export"
])

# ══════════════════════════════════════════
# TAB 1 — SEGMENT OVERVIEW
# ══════════════════════════════════════════
with tab1:
    st.markdown("## Segment Overview")

    seg = rfm.groupby("segment").agg(
        customers    =("customer_id", "count"),
        avg_recency  =("recency",     "mean"),
        avg_frequency=("frequency",   "mean"),
        avg_monetary =("monetary",    "mean"),
        total_revenue=("monetary",    "sum"),
    ).reset_index()
    seg["pct_customers"] = (seg["customers"]     / len(rfm)              * 100).round(1)
    seg["pct_revenue"]   = (seg["total_revenue"] / rfm["monetary"].sum() * 100).round(1)
    seg = seg.sort_values("total_revenue", ascending=False).reset_index(drop=True)

    c1, c2 = st.columns(2)

    with c1:
        fig_tree = px.treemap(
            seg, path=["segment"], values="customers",
            color="segment", color_discrete_map=SEGMENT_COLOURS,
        )
        fig_tree.update_layout(**base_layout("Customer Distribution by Segment"))
        fig_tree.update_traces(
            textfont=dict(family="DM Sans", size=12),
            marker=dict(line=dict(color="#0a0a0f", width=2)),
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    with c2:
        seg_rev = seg.sort_values("total_revenue")
        fig_rev = go.Figure(go.Bar(
            x=seg_rev["total_revenue"],
            y=seg_rev["segment"],
            orientation="h",
            marker_color=[SEGMENT_COLOURS.get(s, "#888") for s in seg_rev["segment"]],
            text=[f"£{v:,.0f}" for v in seg_rev["total_revenue"]],
            textposition="outside",
            textfont=dict(family="DM Mono", size=10, color="#888"),
        ))
        fig_rev.update_layout(
            **base_layout("Revenue by Segment"),
            showlegend=False,
            xaxis=dict(showticklabels=False,
                       gridcolor="#1a1a25", zerolinecolor="#1a1a25"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)",
                       tickfont=dict(size=11, color="#888")),
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("## Segment Detail")
    for i in range(0, len(seg), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j >= len(seg):
                break
            row    = seg.iloc[i + j]
            colour = SEGMENT_COLOURS.get(row["segment"], "#888")
            desc   = SEGMENT_DESCRIPTIONS.get(row["segment"], "")
            with cols[j]:
                st.markdown(f"""
                <div style="background:#12121c;border:1px solid #1e1e2e;
                            border-top:2px solid {colour};border-radius:4px;
                            padding:20px 24px;margin-bottom:16px;">
                  <div style="font-family:'DM Mono',monospace;font-size:10px;
                              color:{colour};letter-spacing:.1em;
                              text-transform:uppercase;margin-bottom:8px;">
                    {row['segment']}</div>
                  <div style="font-family:'DM Serif Display',serif;
                              font-size:2rem;color:#e8e4dc;margin-bottom:4px;">
                    {row['customers']:,}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:11px;
                              color:#555;margin-bottom:12px;">
                    {row['pct_customers']}% of customers
                    · {row['pct_revenue']}% of revenue</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                              gap:8px;margin-bottom:12px;">
                    <div>
                      <div style="font-family:'DM Mono',monospace;font-size:9px;
                                  color:#444;text-transform:uppercase;">
                        Avg Recency</div>
                      <div style="font-family:'DM Serif Display',serif;
                                  font-size:1.1rem;color:#aaa;">
                        {row['avg_recency']:.0f}d</div>
                    </div>
                    <div>
                      <div style="font-family:'DM Mono',monospace;font-size:9px;
                                  color:#444;text-transform:uppercase;">
                        Avg Orders</div>
                      <div style="font-family:'DM Serif Display',serif;
                                  font-size:1.1rem;color:#aaa;">
                        {row['avg_frequency']:.1f}x</div>
                    </div>
                    <div>
                      <div style="font-family:'DM Mono',monospace;font-size:9px;
                                  color:#444;text-transform:uppercase;">
                        Avg Spend</div>
                      <div style="font-family:'DM Serif Display',serif;
                                  font-size:1.1rem;color:#aaa;">
                        £{row['avg_monetary']:,.0f}</div>
                    </div>
                  </div>
                  <div style="font-family:'DM Sans',sans-serif;font-size:12px;
                              color:#555;line-height:1.5;
                              border-top:1px solid #1e1e2e;padding-top:12px;">
                    {desc}</div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 2 — RFM DISTRIBUTION
# ══════════════════════════════════════════
with tab2:
    st.markdown("## RFM Score Distributions")

    d1, d2 = st.columns(2)

    with d1:
        pivot = (rfm.groupby(["r_score", "f_score"])["monetary"]
                    .mean().reset_index()
                    .pivot(index="r_score", columns="f_score", values="monetary")
                    .fillna(0))
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"F={c}" for c in pivot.columns],
            y=[f"R={r}" for r in pivot.index],
            colorscale=[[0, "#12121c"], [0.5, "#8b6914"], [1, "#c8b87a"]],
            text=[[f"£{v:,.0f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont=dict(family="DM Mono", size=10),
            showscale=True,
            colorbar=dict(tickfont=dict(family="DM Mono", size=10, color="#888")),
        ))
        fig_heat.update_layout(
            **base_layout("Avg Spend by Recency × Frequency Score"),
            xaxis=dict(tickfont=dict(family="DM Mono", size=11),
                       gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(tickfont=dict(family="DM Mono", size=11),
                       gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
          <strong>Reading this chart</strong>
          <p>Each cell = avg total spend at that R×F score combination.
          Gold = high value. Top-right (high recency + high frequency)
          should be your Champions.</p>
        </div>""", unsafe_allow_html=True)

    with d2:
        fig_scatter = px.scatter(
            rfm.sample(min(2000, len(rfm)), random_state=42),
            x="recency", y="monetary",
            color="segment", color_discrete_map=SEGMENT_COLOURS,
            opacity=0.65, size="frequency", size_max=14,
            title="Recency vs Monetary Value",
        )
        fig_scatter.update_layout(
            **base_layout("Recency vs Monetary Value"),
            xaxis=dict(title="Days Since Last Purchase",
                       title_font=dict(family="DM Mono", size=11, color="#666"),
                       gridcolor="#1a1a25", zerolinecolor="#1a1a25"),
            yaxis=dict(title="Total Spend (£)",
                       title_font=dict(family="DM Mono", size=11, color="#666"),
                       gridcolor="#1a1a25", zerolinecolor="#1a1a25"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
          <strong>Reading this chart</strong>
          <p>Bubble size = purchase frequency. High-spend customers top-right
          are lapsing — At Risk and Cannot Lose Them segments.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    h1, h2, h3 = st.columns(3)
    for series, label, col in [
        (rfm["recency"],   "Recency — Days Since Last Purchase", h1),
        (rfm["frequency"], "Frequency — Number of Orders",       h2),
        (rfm["monetary"],  "Monetary — Total Spend (£)",         h3),
    ]:
        with col:
            fig_hist = px.histogram(rfm, x=series, nbins=40,
                                    color_discrete_sequence=["#c8b87a"])
            fig_hist.update_layout(
                **base_layout(label, title_size=13),
                showlegend=False,
                bargap=0.05,
                xaxis=dict(gridcolor="#1a1a25", zerolinecolor="#1a1a25"),
                yaxis=dict(gridcolor="#1a1a25", zerolinecolor="#1a1a25"),
            )
            fig_hist.update_traces(
                marker_line_color="#0a0a0f", marker_line_width=0.5)
            st.plotly_chart(fig_hist, use_container_width=True)

# ══════════════════════════════════════════
# TAB 3 — CUSTOMER EXPLORER
# ══════════════════════════════════════════
with tab3:
    st.markdown("## Customer Explorer")

    cf, cs = st.columns([2, 1])
    with cf:
        seg_filter = st.multiselect(
            "Filter by segment", options=all_segments, default=all_segments)
    with cs:
        sort_col = st.selectbox(
            "Sort by", ["monetary", "frequency", "recency", "rfm_numeric"])

    view = (rfm[rfm["segment"].isin(seg_filter)]
              .sort_values(sort_col, ascending=(sort_col == "recency"))
              .copy())

    disp = view[["customer_id", "segment", "recency", "frequency",
                 "monetary", "r_score", "f_score", "m_score", "rfm_score"]].copy()
    disp["monetary"] = disp["monetary"].round(2)
    st.dataframe(
        disp.rename(columns={
            "customer_id": "Customer ID", "segment": "Segment",
            "recency":     "Days Since Last Order",
            "frequency":   "Orders",
            "monetary":    "Total Spend (£)",
            "r_score": "R", "f_score": "F",
            "m_score": "M", "rfm_score": "RFM",
        }),
        use_container_width=True, height=480,
    )
    st.caption(f"{len(view):,} customers shown")

# ══════════════════════════════════════════
# TAB 4 — TARGETING EXPORT
# ══════════════════════════════════════════
with tab4:
    st.markdown("## Targeting List Export")
    st.markdown("""
    <div class="insight-box">
      <strong>How to use this</strong>
      <p>Select segments in the sidebar, then export as CSV.
      Each row is one customer with RFM scores and behavioural metrics —
      ready to upload to your CRM or email platform.</p>
    </div>""", unsafe_allow_html=True)

    if not selected_segments:
        st.warning("Select at least one segment in the sidebar.")
    else:
        tgt = make_targeting_list(rfm, selected_segments)

        e1, e2, e3 = st.columns(3)
        with e1: st.metric("Customers in Export",    f"{len(tgt):,}")
        with e2: st.metric("Segments Selected",      f"{len(selected_segments)}")
        with e3: st.metric("Combined Lifetime Value",
                            f"£{tgt['Total Spend (£)'].sum():,.0f}")

        st.markdown("---")

        breakdown = (tgt.groupby("Segment")
                       .agg(Customers=("Customer ID", "count"))
                       .reset_index()
                       .sort_values("Customers"))
        fig_exp = px.bar(
            breakdown, x="Customers", y="Segment", orientation="h",
            color="Segment", color_discrete_map=SEGMENT_COLOURS,
        )
        fig_exp.update_layout(
            **base_layout("Export Breakdown by Segment"),
            showlegend=False,
            xaxis=dict(gridcolor="#1a1a25", zerolinecolor="#1a1a25"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)",
                       tickfont=dict(size=11, color="#888")),
        )
        st.plotly_chart(fig_exp, use_container_width=True)

        st.markdown("**Preview**")
        st.dataframe(tgt.head(20), use_container_width=True)

        st.download_button(
            label=f"↓ Export {len(tgt):,} customers as CSV",
            data=tgt.to_csv(index=False).encode("utf-8"),
            file_name=(
                "rfm_targeting_"
                + "_".join(selected_segments[:3]).lower().replace(" ", "_")
                + ".csv"
            ),
            mime="text/csv",
        )

        st.markdown("""
        <div class="insight-box" style="margin-top:24px;">
          <strong>Recommended Actions</strong>
          <p>
          <b style="color:#c8b87a;">Champions</b> —
            loyalty rewards, early access, referral programme.<br>
          <b style="color:#b87a7a;">At Risk</b> —
            re-engagement email with a time-limited incentive.<br>
          <b style="color:#d4624a;">Cannot Lose Them</b> —
            high-priority win-back, personal outreach.<br>
          <b style="color:#666680;">Hibernating</b> —
            low-cost automated email only; suppress from paid media.
          </p>
        </div>""", unsafe_allow_html=True)

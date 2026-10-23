"""
app.py  -  Streamlit dashboard for the Data Analysis & Visualization project
----------------------------------------------------------------------------
Same pipeline as main.py:
1. Generate raw data  2. Clean  3. Preprocess for ML  4. Visualize

Run:  streamlit run app.py
"""

import io
import os
import glob
import contextlib

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.generate_data import generate_raw_dataset
from src.cleaning import clean_dataset
from src.preprocessing import prepare_for_modeling
from src.visualize import run_all

PAL = ["#22d3ee", "#a78bfa", "#f472b6", "#fbbf24", "#34d399", "#fb923c", "#60a5fa", "#e879f9"]
RAW, CLEAN, MODEL = "data/raw_customer_data.csv", "data/cleaned_customer_data.csv", "data/model_ready_data.csv"

st.set_page_config(page_title="Data Insights Studio", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.6rem;max-width:1400px}
.hero{padding:22px 26px;border-radius:18px;margin-bottom:18px;border:1px solid #243056;
 background:radial-gradient(600px 200px at 90% 0%,#22d3ee22,transparent),
 radial-gradient(500px 200px at 0% 100%,#a78bfa22,transparent),#141b34}
.hero h1{margin:0;font-size:2rem;letter-spacing:-.5px;
 background:linear-gradient(90deg,#22d3ee,#a78bfa,#f472b6);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{margin:4px 0 0;color:#8b96c4}
.kpi{background:#141b34;border:1px solid #243056;border-left:4px solid var(--a);border-radius:14px;padding:14px 18px}
.kpi small{color:#8b96c4;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em}
.kpi b{display:block;font-size:1.7rem;letter-spacing:-.5px}
.kpi span{color:#8b96c4;font-size:.78rem}
.step{padding:6px 0;color:#8b96c4}.step.ok{color:#34d399}
div[data-baseweb="tab-list"]{gap:6px}
button[data-baseweb="tab"]{background:#141b34;border-radius:10px 10px 0 0;padding:8px 18px}
section[data-testid="stSidebar"]{background:#0e1428;border-right:1px solid #243056}
section[data-testid="stSidebar"] .stButton>button{width:100%;font-weight:700}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,#22d3ee,#a78bfa);color:#0b1020;border:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.hero h1{font-size:clamp(1.35rem,5vw,2rem)}
div[data-baseweb="tab-list"]{overflow-x:auto}
.modebar-container{display:none!important}
.st-key-run_main{display:none}
@media (max-width:640px){
 .block-container{padding:1rem .7rem 3rem}
 .hero{padding:16px}
 .kpi{padding:10px 14px}.kpi b{font-size:1.35rem}
 button[data-baseweb="tab"]{padding:6px 12px;font-size:.85rem}
 [data-testid="stDataFrame"]{font-size:.8rem}
 .st-key-run_main{display:block}
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- pipeline
def run_pipeline(progress):
    """Same steps as main.py; returns the 3 dataframes and the console log."""
    buf = io.StringIO()
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    with contextlib.redirect_stdout(buf):
        progress.progress(5, "Step 1/4 · Generating raw dataset")
        print("=" * 60); print("STEP 1: Generating raw dataset"); print("=" * 60)
        raw_df = generate_raw_dataset()
        raw_df.to_csv(RAW, index=False)
        print(f"Raw dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns\n")

        progress.progress(30, "Step 2/4 · Cleaning dataset")
        print("=" * 60); print("STEP 2: Cleaning dataset"); print("=" * 60)
        cleaned_df = clean_dataset(raw_df)
        cleaned_df.to_csv(CLEAN, index=False)
        print()

        progress.progress(55, "Step 3/4 · Preprocessing for ML")
        print("=" * 60); print("STEP 3: Preprocessing for ML (encoding + normalization)"); print("=" * 60)
        model_df = prepare_for_modeling(cleaned_df)
        model_df.to_csv(MODEL, index=False)
        print(f"Model-ready dataset shape: {model_df.shape}\n")

        progress.progress(75, "Step 4/4 · Generating visualizations")
        print("=" * 60); print("STEP 4: Generating visualizations"); print("=" * 60)
        run_all(cleaned_df)
        print("\nPipeline complete. See data/ and outputs/ folders.")
    progress.progress(100, "Done ✔")
    return raw_df, cleaned_df, model_df, buf.getvalue()


def load_existing():
    rd = lambda p: pd.read_csv(p) if os.path.exists(p) else None
    return rd(RAW), rd(CLEAN), rd(MODEL)


def detect(df):
    num, cat, dt = [], [], None
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            (num if s.nunique() > 12 else cat).append(c)
        else:
            if dt is None and any(k in c.lower() for k in ("date", "time")):
                parsed = pd.to_datetime(s, errors="coerce")
                if parsed.notna().mean() > .8:
                    dt = c
                    continue
            if s.nunique() <= 30:
                cat.append(c)
    return num, cat, dt


def style(fig, h=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      colorway=PAL, height=h, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color="#e8ecff"), title_font_size=15, dragmode=False,
                      legend=dict(orientation="h", y=-0.2, x=0))
    fig.update_xaxes(gridcolor="#243056", zerolinecolor="#243056")
    fig.update_yaxes(gridcolor="#243056", zerolinecolor="#243056")
    return fig


def kpi(items, label, value, sub, color):
    items.append(f'<div class="kpi" style="--a:{color}"><small>{label}</small><b>{value}</b><span>{sub}</span></div>')


# ---------------------------------------------------------------- state
if "dfs" not in st.session_state:
    r, c, m = load_existing()
    st.session_state.dfs = (r, c, m)
    st.session_state.log = "Click 'Run full pipeline' in the sidebar to start."

def launch(slot):
    bar = slot.progress(0, "Starting…")
    try:
        *dfs, log = run_pipeline(bar)
        st.session_state.dfs, st.session_state.log = tuple(dfs), log
        st.toast("Pipeline complete!", icon="✅")
    except Exception as e:
        st.error(f"Pipeline failed: {e}")


with st.sidebar:
    st.markdown("### 📊 Data Studio")
    st.caption("Clean · Preprocess · Visualize")
    run_side = st.button("▶  Run full pipeline", type="primary", key="run_side")
    prog_side = st.empty()
    if run_side:
        launch(prog_side)
    st.markdown("---")

st.markdown("""<div class="hero"><h1>Customer Data Insights Studio</h1>
<p>Messy data → clean data → ML-ready features → interactive visualizations</p></div>""", unsafe_allow_html=True)

# visible only on phones (CSS), where the sidebar is collapsed
run_main = st.button("▶  Run full pipeline", type="primary", key="run_main")
prog_main = st.empty()
if run_main:
    launch(prog_main)

raw_df, clean_df, model_df = st.session_state.dfs

if clean_df is None:
    st.info("No data yet. Use **Run full pipeline** in the sidebar to generate, clean and visualize the dataset.")
    st.stop()

num, cat, dt = detect(clean_df)
if dt:
    clean_df = clean_df.copy()
    clean_df[dt] = pd.to_datetime(clean_df[dt], errors="coerce")

# ---------------------------------------------------------------- filters
with st.sidebar:
    st.markdown("**Filters**")
    fdf = clean_df
    for c in cat[:3]:
        opts = sorted(fdf[c].dropna().astype(str).unique())
        pick = st.multiselect(c, opts, default=[])
        if pick:
            fdf = fdf[fdf[c].astype(str).isin(pick)]
    if num:
        lo, hi = float(clean_df[num[0]].min()), float(clean_df[num[0]].max())
        if lo < hi:
            a, b = st.slider(num[0], lo, hi, (lo, hi))
            fdf = fdf[fdf[num[0]].between(a, b)]
    st.caption(f"{len(fdf):,} of {len(clean_df):,} rows selected")

# ---------------------------------------------------------------- tabs
t1, t2, t3, t4, t5, t6 = st.tabs(["🏠 Overview", "📈 Explore", "🗂 Data", "🖼 Chart gallery", "🖥 Console", "📏 Metrics"])

with t1:
    k = []
    rawn = len(raw_df) if raw_df is not None else len(clean_df)
    rawm = int(raw_df.isna().sum().sum()) if raw_df is not None else 0
    cm = int(clean_df.isna().sum().sum())
    kpi(k, "Raw records", f"{rawn:,}", "before cleaning", PAL[0])
    kpi(k, "Clean records", f"{len(clean_df):,}", f"{rawn - len(clean_df):,} rows removed", PAL[4])
    kpi(k, "Missing values", f"{cm:,}", f"was {rawm:,} in raw data", PAL[2])
    kpi(k, "Features", f"{clean_df.shape[1]}", f"{len(num)} numeric · {len(cat)} categorical", PAL[3])
    kpi(k, "ML-ready shape", f"{model_df.shape[0]:,}×{model_df.shape[1]}" if model_df is not None else "—",
        "encoded + normalized", PAL[1])
    st.markdown('<div class="kpis">' + "".join(k) + '</div>', unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("##### Pipeline")
        for s in ["Generate raw data", "Clean (duplicates, outliers, text, dates, missing)",
                  "Preprocess (one-hot + min-max)", "Visualize (Matplotlib + Plotly)"]:
            st.markdown(f'<div class="step ok">● {s}</div>', unsafe_allow_html=True)
    with c2:
        if raw_df is not None:
            m = pd.DataFrame({"Stage": ["Raw", "Cleaned"], "Missing values": [rawm, cm]})
            st.plotly_chart(style(px.bar(m, x="Stage", y="Missing values", color="Stage", title="Missing values: before vs after",
                                         color_discrete_sequence=[PAL[2], PAL[4]]), 260))

with t2:
    if not num and not cat:
        st.warning("No numeric or categorical columns detected.")
    a, b = st.columns(2)
    if num:
        n1 = a.selectbox("Numeric column", num, key="n1")
        a.plotly_chart(style(px.histogram(fdf, x=n1, nbins=40, marginal="box", title=f"Distribution of {n1}",
                                          color_discrete_sequence=[PAL[0]])))
    if cat:
        c1 = b.selectbox("Category column", cat, key="c1")
        vc = fdf[c1].astype(str).value_counts().head(12).reset_index()
        vc.columns = [c1, "count"]
        b.plotly_chart(style(px.bar(vc, x="count", y=c1, orientation="h", color=c1, title=f"{c1} counts")))
    a, b = st.columns(2)
    if cat:
        c2 = a.selectbox("Share of", cat, index=min(1, len(cat) - 1), key="c2")
        vc = fdf[c2].astype(str).value_counts().head(10).reset_index()
        vc.columns = [c2, "count"]
        a.plotly_chart(style(px.pie(vc, names=c2, values="count", hole=.55, title=f"{c2} share")))
    if num and cat:
        n2 = b.selectbox("Numeric", num, index=min(1, len(num) - 1), key="n2")
        c3 = b.selectbox("Group by", cat, key="c3")
        b.plotly_chart(style(px.box(fdf, x=c3, y=n2, color=c3, title=f"{n2} by {c3}")))
    if len(num) >= 2:
        st.markdown("##### Relationship")
        x, y, col = st.columns(3)
        sx = x.selectbox("X", num, key="sx")
        sy = y.selectbox("Y", num, index=1, key="sy")
        sc = col.selectbox("Colour by", ["—"] + cat, key="sc")
        samp = fdf.sample(min(len(fdf), 2500), random_state=1) if len(fdf) else fdf
        st.plotly_chart(style(px.scatter(samp, x=sx, y=sy, color=None if sc == "—" else sc, opacity=.7,
                                         title=f"{sx} vs {sy}"), 360))
    if dt and num:
        tn = st.selectbox("Trend of", num, key="tn")
        ts = fdf.dropna(subset=[dt]).set_index(dt)[tn].resample("MS").mean().reset_index()
        st.plotly_chart(style(px.area(ts, x=dt, y=tn, title=f"Monthly average {tn}", color_discrete_sequence=[PAL[1]]), 340))
    if len(num) >= 2:
        corr = fdf[num[:8]].corr()
        st.plotly_chart(style(px.imshow(corr, text_auto=".2f", color_continuous_scale=["#f472b6", "#141b34", "#22d3ee"],
                                        zmin=-1, zmax=1, title="Correlation heatmap"), 400))

with t3:
    pick = st.radio("Dataset", ["Cleaned", "Raw", "Model-ready"], horizontal=True)
    d = {"Cleaned": clean_df, "Raw": raw_df, "Model-ready": model_df}[pick]
    if d is None:
        st.info("This dataset isn't available yet.")
    else:
        st.dataframe(d.head(1000), height=380)
        st.download_button(f"⬇ Download {pick} CSV", d.to_csv(index=False).encode(), f"{pick.lower()}_data.csv", "text/csv")
        with st.expander("Summary statistics"):
            st.dataframe(d.describe(include="all").T)

with t4:
    imgs = sorted(glob.glob("outputs/*.png"))
    if not imgs:
        st.info("No Matplotlib charts found in outputs/ yet. Run the pipeline.")
    for i in range(0, len(imgs), 3):
        cols = st.columns(3)
        for col, p in zip(cols, imgs[i:i + 3]):
            col.image(p, caption=os.path.splitext(os.path.basename(p))[0].replace("_", " ").title())

with t5:
    st.code(st.session_state.log, language="text")

with t6:
    from metrics import inconsistency_report, benchmark, verdicts
    if raw_df is None:
        st.info("Run the pipeline first - metrics compare raw vs cleaned data.")
    else:
        rep = inconsistency_report(raw_df, clean_df)
        m1, m2, m3 = st.columns(3)
        m1.metric("Inconsistencies before", f"{rep['before_total']:,}")
        m2.metric("After cleaning", f"{rep['after_total']:,}")
        m3.metric("Reduction", f"{rep['reduction_pct']}%")
        long = rep["table"].melt("Problem type", var_name="Stage", value_name="Count")
        st.plotly_chart(style(px.bar(long, x="Problem type", y="Count", color="Stage", barmode="group",
                                     title="Data problems: before vs after cleaning",
                                     color_discrete_sequence=[PAL[2], PAL[4]]), 340))
        if st.button("⏱ Run speed + reuse benchmark", key="bench_btn"):
            with st.spinner("Benchmarking clean_dataset()..."):
                st.session_state.bench = benchmark(raw_df, clean_dataset)
        bn = st.session_state.get("bench")
        if bn:
            b1, b2, b3 = st.columns(3)
            b1.metric("Speed", f"{bn['records_per_second']:,} rec/s")
            b2.metric("Median clean time", f"{bn['median_seconds']} s")
            b3.metric("Code-reuse saving (estimate)", f"{bn['effort_reduction_pct']}%")
        for line in verdicts(rep, bn):
            st.markdown(f"- {line}")

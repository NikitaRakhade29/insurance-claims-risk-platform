
import os

import pandas as pd
import psycopg2
import streamlit as st

PG_CONN_STR = os.environ.get(
    "DATABASE_URL", "postgresql://airflow:airflow@localhost:5432/airflow"
)

st.set_page_config(page_title="Claims Risk Console", layout="wide", initial_sidebar_state="expanded")

# --- Design system: dark ops-console theme, semantic risk color, technical type pairing ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* hide Streamlit chrome */
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; height: 0; }
.stDeployButton { display: none; }

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

/* page title */
.console-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2.4rem;
    color: #E7EAEE; margin-bottom: 0.1rem; }
.console-subtitle { font-family: 'Inter', sans-serif; color: #8B93A3; font-size: 0.95rem; margin-bottom: 1.6rem; }

/* metric cards */
div[data-testid="stMetric"] {
    background: #1B212B; border: 1px solid #2A313D; border-radius: 10px;
    padding: 1rem 1.1rem; border-left: 3px solid #4C8DFF;
}
div[data-testid="stMetric"] label { color: #8B93A3 !important; font-size: 0.78rem !important;
    text-transform: uppercase; letter-spacing: 0.04em; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace; color: #E7EAEE; }

/* semantic accent per metric column, by order: total / red / yellow / green */
div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetric"] { border-left-color: #E5484D; }
div[data-testid="column"]:nth-of-type(3) div[data-testid="stMetric"] { border-left-color: #F5A623; }
div[data-testid="column"]:nth-of-type(4) div[data-testid="stMetric"] { border-left-color: #3ECF8E; }

/* expander (claim card) */
div[data-testid="stExpander"] {
    background: #1B212B; border: 1px solid #2A313D !important; border-radius: 10px;
    margin-bottom: 0.5rem; overflow: hidden;
}
div[data-testid="stExpander"] summary { font-family: 'JetBrains Mono', monospace; font-size: 0.92rem; }
div[data-testid="stExpander"] summary:hover { background: #222A36; }

/* risk chip badges */
.chip { display:inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.03em; margin-right: 0.4rem; }
.chip-red { background: rgba(229,72,77,0.15); color: #F27074; border: 1px solid rgba(229,72,77,0.4); }
.chip-yellow { background: rgba(245,166,35,0.15); color: #F6B94D; border: 1px solid rgba(245,166,35,0.4); }
.chip-green { background: rgba(62,207,142,0.15); color: #5EDBA5; border: 1px solid rgba(62,207,142,0.4); }
.chip-neutral { background: rgba(76,141,255,0.15); color: #7CA8FF; border: 1px solid rgba(76,141,255,0.4); }

/* rationale box */
.rationale-box { background: #161B23; border-left: 3px solid #4C8DFF; border-radius: 6px;
    padding: 0.8rem 1rem; margin: 0.6rem 0; font-size: 0.92rem; color: #C7CDD6; }

.data-label { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #8B93A3; }
.data-value { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #E7EAEE; }

section[data-testid="stSidebar"] { background: #161B23; border-right: 1px solid #2A313D; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_summary() -> pd.DataFrame:
    conn = psycopg2.connect(PG_CONN_STR)
    df = pd.read_sql("SELECT * FROM claims.claim_risk_summary", conn)
    conn.close()
    return df


st.markdown('<div class="console-title">Claims Risk Console</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="console-subtitle">Fraud risk · billing anomaly · customer risk — with AI-generated adjuster rationale</div>',
    unsafe_allow_html=True,
)

df = load_summary()

if df.empty:
    st.warning("No scored claims yet. Run the scoring pipeline first (models/score_all.py).")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total claims", len(df))
col2.metric("Red flags", int((df.triage_flag == "RED").sum()))
col3.metric("Yellow flags", int((df.triage_flag == "YELLOW").sum()))
col4.metric("Green (clean)", int((df.triage_flag == "GREEN").sum()))

st.write("")

with st.sidebar:
    st.markdown("### Filters")
    flag_filter = st.multiselect(
        "Triage flag", options=["RED", "YELLOW", "GREEN"], default=["RED", "YELLOW"]
    )
    policy_filter = st.multiselect(
        "Base policy", options=sorted(df.base_policy.dropna().unique().tolist())
    )

filtered = df[df.triage_flag.isin(flag_filter)] if flag_filter else df
if policy_filter:
    filtered = filtered[filtered.base_policy.isin(policy_filter)]

filtered = filtered.sort_values("fraud_probability", ascending=False)

st.markdown(f"### Triage queue — {len(filtered)} claims")

flag_icon = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}
chip_class = {"RED": "chip-red", "YELLOW": "chip-yellow", "GREEN": "chip-green"}

for _, row in filtered.head(100).iterrows():
    icon = flag_icon.get(row.triage_flag, "⚪")
    label = (
        f"{icon}  {row.claim_id}  ·  {row.vehicle_category or 'N/A'}  ·  "
        f"fraud risk {row.fraud_probability:.1%}"
        if pd.notna(row.fraud_probability)
        else f"{icon}  {row.claim_id}  ·  not yet scored"
    )
    with st.expander(label):
        chip = chip_class.get(row.triage_flag, "chip-neutral")
        st.markdown(f'<span class="chip {chip}">{row.triage_flag}</span>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud probability", f"{row.fraud_probability:.1%}" if pd.notna(row.fraud_probability) else "—")
        c2.metric("Anomaly score", f"{row.anomaly_score:.1%}" if pd.notna(row.anomaly_score) else "—")
        c3.metric("Customer risk index", f"{row.customer_risk_index:.1%}" if pd.notna(row.customer_risk_index) else "—")

        st.markdown(
            f'<span class="data-label">FAULT</span> <span class="data-value">{row.fault}</span>'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;'
            f'<span class="data-label">BASE POLICY</span> <span class="data-value">{row.base_policy}</span>',
            unsafe_allow_html=True,
        )

        if pd.notna(row.rationale_text):
            st.markdown(f'<div class="rationale-box">{row.rationale_text}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="chip chip-neutral">{row.recommended_action}</span>',
                unsafe_allow_html=True,
            )
            if row.flagged_reasons:
                for reason in row.flagged_reasons:
                    st.markdown(f'<span class="data-label">— {reason}</span>', unsafe_allow_html=True)
        else:
            st.caption("No AI rationale generated yet for this claim (below triage threshold, or pending batch).")

if len(filtered) > 100:
    st.caption(f"Showing top 100 of {len(filtered)} matching claims, sorted by fraud probability.")

import os
import pandas as pd
import psycopg2
import streamlit as st

PG_CONN_STR = os.environ.get("DATABASE_URL", "postgresql://airflow:airflow@localhost:5432/airflow")

st.set_page_config(
    page_title="Insurance Claims Risk & Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; height: 0; }
.stDeployButton { display: none; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #0B0F17;
    color: #E2E8F0;
}

.hero-header {
    background: linear-gradient(135deg, #111827 0%, #1F2937 100%);
    border: 1px solid #374151;
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60A5FA, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    color: #9CA3AF;
    font-size: 1.05rem;
    margin-top: 0.4rem;
    font-weight: 500;
}

div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

div[data-testid="stMetric"] label {
    color: #9CA3AF !important;
    font-size: 0.82rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem !important;
    font-weight: 700;
    color: #F3F4F6;
}

div[data-testid="column"]:nth-of-type(1) div[data-testid="stMetric"] { border-top: 4px solid #3B82F6; }
div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetric"] { border-top: 4px solid #EF4444; }
div[data-testid="column"]:nth-of-type(3) div[data-testid="stMetric"] { border-top: 4px solid #F59E0B; }
div[data-testid="column"]:nth-of-type(4) div[data-testid="stMetric"] { border-top: 4px solid #10B981; }

.badge-red {
    background: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}

.badge-yellow {
    background: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.4);
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}

.badge-green {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.4);
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}

.ai-rationale-card {
    background: #111827;
    border-left: 4px solid #8B5CF6;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

.ai-rationale-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #A78BFA;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.ai-rationale-body {
    font-size: 0.98rem;
    color: #E5E7EB;
    line-height: 1.6;
}

section[data-testid="stSidebar"] {
    background-color: #0D1117;
    border-right: 1px solid #1F2937;
}

div[data-testid="stExpander"] {
    background-color: #111827;
    border: 1px solid #1F2937 !important;
    border-radius: 12px;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_summary() -> pd.DataFrame:
    try:
        conn = psycopg2.connect(PG_CONN_STR)
        df = pd.read_sql("SELECT * FROM claims.claim_risk_summary", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


st.markdown("""
<div class="hero-header">
    <div class="hero-title">🛡️ Insurance Claims Risk & Fraud Intelligence</div>
    <div class="hero-subtitle">Real-Time Ingestion • XGBoost & Isolation Forest Risk Triage • Gemini GenAI Adjuster Rationales</div>
</div>
""", unsafe_allow_html=True)

df = load_summary()

if df.empty:
    st.info("💡 Run the scoring pipeline to populate live claim triage metrics.")
    st.code("python models/score_all.py", language="bash")
    st.stop()

total_claims = len(df)
red_cnt = int((df.triage_flag == "RED").sum())
yellow_cnt = int((df.triage_flag == "YELLOW").sum())
green_cnt = int((df.triage_flag == "GREEN").sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Claims Processed", f"{total_claims:,}")
col2.metric("High Risk (RED)", f"{red_cnt:,}", f"{(red_cnt/total_claims):.1%}")
col3.metric("Medium Risk (YELLOW)", f"{yellow_cnt:,}", f"{(yellow_cnt/total_claims):.1%}")
col4.metric("Low Risk (GREEN)", f"{green_cnt:,}", f"{(green_cnt/total_claims):.1%}")

st.write("")

with st.sidebar:
    st.markdown("### 🎛️ Triage Controls")
    
    selected_flags = st.multiselect(
        "Filter by Triage Status",
        options=["RED", "YELLOW", "GREEN"],
        default=["RED", "YELLOW"]
    )
    
    categories = sorted([x for x in df["vehicle_category"].dropna().unique() if x])
    selected_categories = st.multiselect("Vehicle Category", options=categories)
    
    policies = sorted([x for x in df["base_policy"].dropna().unique() if x])
    selected_policies = st.multiselect("Policy Type", options=policies)
    
    search_query = st.text_input("🔍 Search Claim ID", placeholder="e.g. 1042")

filtered_df = df.copy()
if selected_flags:
    filtered_df = filtered_df[filtered_df["triage_flag"].isin(selected_flags)]
if selected_categories:
    filtered_df = filtered_df[filtered_df["vehicle_category"].isin(selected_categories)]
if selected_policies:
    filtered_df = filtered_df[filtered_df["base_policy"].isin(selected_policies)]
if search_query:
    filtered_df = filtered_df[filtered_df["claim_id"].astype(str).str.contains(search_query, case=False)]

filtered_df = filtered_df.sort_values("fraud_probability", ascending=False)

tab1, tab2, tab3 = st.tabs(["🚨 Triage Queue & AI Explanations", "📊 Risk Analytics & Heatmaps", "📋 Claim Explorer"])

with tab1:
    st.markdown(f"### Triage Queue (`{len(filtered_df)}` Claims Match Filters)")
    
    badge_map = {"RED": "badge-red", "YELLOW": "badge-yellow", "GREEN": "badge-green"}
    icon_map = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}
    
    for _, row in filtered_df.head(50).iterrows():
        status = row.get("triage_flag", "GREEN")
        badge_style = badge_map.get(status, "badge-green")
        icon = icon_map.get(status, "🟢")
        
        fraud_pct = f"{row['fraud_probability']:.1%}" if pd.notna(row['fraud_probability']) else "N/A"
        anomaly_pct = f"{row['anomaly_score']:.1%}" if pd.notna(row['anomaly_score']) else "N/A"
        
        header_text = f"{icon} Claim #{row['claim_id']}  │  Vehicle: {row.get('vehicle_category', 'N/A')}  │  Fraud Risk: {fraud_pct}  │  Anomaly: {anomaly_pct}"
        
        with st.expander(header_text):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fraud Probability (XGBoost)", fraud_pct)
            m2.metric("Anomaly Score (Isolation Forest)", anomaly_pct)
            m3.metric("Customer Risk Index", f"{row['customer_risk_index']:.1f}" if pd.notna(row['customer_risk_index']) else "N/A")
            m4.markdown(f"<div style='margin-top: 1rem;'><span class='{badge_style}'>{status} TRIAGE</span></div>", unsafe_allow_html=True)
            
            st.divider()
            
            d1, d2, d3, d4 = st.columns(4)
            d1.write(f"**Policyholder Age:** {row.get('age_of_policyholder', 'N/A')}")
            d2.write(f"**Fault:** {row.get('fault', 'N/A')}")
            d3.write(f"**Base Policy:** {row.get('base_policy', 'N/A')}")
            d4.write(f"**Past Claims:** {row.get('past_number_of_claims', 'N/A')}")
            
            rationale = row.get("rationale_text")
            if pd.notna(rationale) and rationale:
                st.markdown(f"""
                <div class="ai-rationale-card">
                    <div class="ai-rationale-title">🤖 Gemini GenAI Adjuster Rationale</div>
                    <div class="ai-rationale-body">{rationale}</div>
                </div>
                """, unsafe_allow_html=True)
                
                reasons = row.get("flagged_reasons")
                if reasons and isinstance(reasons, list):
                    st.markdown("**Key Risk Factors Identified:**")
                    for r in reasons:
                        st.markdown(f"- ⚠️ {r}")
            else:
                st.caption("ℹ️ *No GenAI rationale generated yet for this claim (pending batch run or below threshold).*")

with tab2:
    st.markdown("### 📈 Risk Analytics & Distribution Patterns")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### 🎯 Claims by Triage Priority")
        flag_counts = df["triage_flag"].value_counts().reset_index()
        flag_counts.columns = ["Triage Flag", "Count"]
        st.bar_chart(flag_counts.set_index("Triage Flag"), color="#3B82F6")
        
    with chart_col2:
        st.markdown("#### 🚗 Fraud Risk by Vehicle Category")
        if "vehicle_category" in df.columns:
            cat_risk = df.groupby("vehicle_category")["fraud_probability"].mean().reset_index()
            cat_risk.columns = ["Vehicle Category", "Avg Fraud Probability"]
            st.bar_chart(cat_risk.set_index("Vehicle Category"), color="#8B5CF6")
            
    st.divider()
    
    st.markdown("#### ⚡ Fraud Risk vs. Anomaly Score Scatter Matrix")
    if "fraud_probability" in df.columns and "anomaly_score" in df.columns:
        st.scatter_chart(
            df[["fraud_probability", "anomaly_score"]],
            x="fraud_probability",
            y="anomaly_score",
            color="#EF4444"
        )

with tab3:
    st.markdown("### 📋 Full Scored Claims Database")
    
    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud Probability",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "anomaly_score": st.column_config.ProgressColumn(
                "Anomaly Score",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
        }
    )
    
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Scored Claims CSV",
        data=csv_data,
        file_name="scored_claims_report.csv",
        mime="text/csv"
    )

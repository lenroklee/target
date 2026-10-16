import os
import duckdb
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_FILE = "data/eurosystem_infrastructure.duckdb"

st.set_page_config(
    page_title="ECB DG-MIP: TARGET Infrastructure & Liquidity Risk Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_db_connection():
    if not os.path.exists(DB_FILE):
        st.error(f"Database '{DB_FILE}' not found. Please run Phase 1 first.")
        st.stop()
    return duckdb.connect(DB_FILE, read_only=True)

conn = get_db_connection()

# Header Section
st.title("Eurosystem Market Infrastructure Analytics")
st.caption("Directorate General Market Infrastructure and Payments (DG-MIP) — TARGET Services & Liquidity Stress Monitor")
st.markdown("---")

# Sidebar Controls
st.sidebar.header("Filter & Analysis Parameters")
selected_service = st.sidebar.multiselect(
    "TARGET Platforms:",
    options=['T2_CASH', 'T2S_DVP', 'ECMS_COLLATERAL', 'TIPS_INSTANT', 'PONTES_DLT'],
    default=['T2_CASH', 'T2S_DVP', 'ECMS_COLLATERAL', 'TIPS_INSTANT', 'PONTES_DLT']
)

# Key Performance Indicators (KPIs)
st.subheader("1. Executive Summary & Liquidity Overview")

kpi_query = """
    SELECT 
        COUNT(*) AS total_tx,
        SUM(amount_eur) AS total_vol_eur,
        AVG(amount_eur) AS avg_tx_eur,
        SUM(CASE WHEN status = 'QUEUED' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS queue_pct
    FROM target_transactions
    WHERE service_type IN ({});
""".format(", ".join([f"'{s}'" for s in selected_service]))

kpi_res = conn.execute(kpi_query).fetchone()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{kpi_res[0]:,}")
col2.metric("Total Volume Settled", f"€{kpi_res[1] / 1e9:,.2f} B")
col3.metric("Average Tx Size", f"€{kpi_res[2]:,.2f}")
col4.metric("Liquidity Gridlock Rate", f"{kpi_res[3]:.2f}%")

st.markdown("---")

# Time Series Trends
st.subheader("2. Platform Settlement Dynamics Over Time")

ts_query = """
    SELECT 
        CAST(timestamp_utc AS DATE) AS date_id,
        service_type,
        SUM(amount_eur) / 1e9 AS volume_billions
    FROM target_transactions
    WHERE service_type IN ({})
    GROUP BY CAST(timestamp_utc AS DATE), service_type
    ORDER BY date_id;
""".format(", ".join([f"'{s}'" for s in selected_service]))

df_ts = conn.execute(ts_query).df()

fig_ts = px.line(
    df_ts, 
    x="date_id", 
    y="volume_billions", 
    color="service_type",
    title="Daily Settled Volume per Eurosystem Service (€ Billions)",
    labels={"volume_billions": "Volume (€ Billions)", "date_id": "Date", "service_type": "Service Platform"}
)
st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("---")

# Network Topology & Systemic Importance
st.subheader("3. Interbank Interconnectedness & Network Topology")

net_query = """
    SELECT sender_bic, receiver_bic, SUM(amount_eur) AS total_flow
    FROM target_transactions
    WHERE service_type IN ('T2_CASH', 'T2S_DVP') AND status = 'SETTLED'
    GROUP BY sender_bic, receiver_bic;
"""
df_net = conn.execute(net_query).df()

G = nx.DiGraph()
for _, row in df_net.iterrows():
    G.add_edge(row['sender_bic'], row['receiver_bic'], weight=row['total_flow'])

pagerank = nx.pagerank(G, weight='weight')
betweenness = nx.betweenness_centrality(G, weight='weight')

df_centrality = pd.DataFrame({
    'BIC': list(G.nodes()),
    'PageRank (Systemic Score)': [pagerank[node] for node in G.nodes()],
    'Betweenness Centrality': [betweenness[node] for node in G.nodes()]
}).sort_values(by='PageRank (Systemic Score)', ascending=False).reset_index(drop=True)

col_net1, col_net2 = st.columns([1, 1])

with col_net1:
    st.write("**Top 10 Systemically Important Institutions**")
    st.dataframe(df_centrality.head(10), use_container_width=True)

with col_net2:
    fig_scatter = px.scatter(
        df_centrality, 
        x="Betweenness Centrality", 
        y="PageRank (Systemic Score)",
        text="BIC",
        title="Institutional Systemic Risk Matrix",
        size="PageRank (Systemic Score)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# Econometric & Shock Transmission View
st.subheader("4. Econometric Macro Transmission (€STR Rate Shock)")

col_irf1, col_irf2 = st.columns([1, 1])

with col_irf1:
    st.markdown("""
    **Transmission Channel Analysis:**
    * **TWFE Panel Estimate:** Evaluates liquidity hoarding ($\text{Settlement Delay Ratio}$) as a function of monetary policy tightening ($\text{€STR}$).
    * **VAR Impulse Response:** Traces the structural response of intraday payment flows to an unexpected 25 bps rate shock over a 10-day horizon.
    """)
    if os.path.exists("var_irf_transmission.png"):
        st.image("var_irf_transmission.png", caption="Impulse Response Function (VAR Model Output)", use_container_width=True)
    else:
        st.warning("IRF image not found. Execute Phase 3 script to generate plot.")

with col_irf2:
    st.write("**Operational Policy Recommendations**")
    st.info("""
    1. **Intraday Credit Lines:** Increase auto-collateralization thresholds on T2S during peak policy-rate volatility windows.
    2. **Gridlock Resolution:** Trigger automated liquidity recycling algorithms when total queued volume exceeds 1.5% of total daily exposure.
    3. **Pontes DLT Monitoring:** Establish atomic settlement bridges between wholesale DLT nodes and T2 cash accounts.
    """)
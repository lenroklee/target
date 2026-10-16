# Eurosystem TARGET Infrastructure & Liquidity Risk Analytics


##  Project Overview
This repository contains an end-to-end quantitative and data engineering pipeline simulating the **European Central Bank's (ECB) TARGET Services** ecosystem. It is designed to evaluate systemic liquidity risk, interbank interconnectedness, and the transmission of monetary policy shocks (€STR) into intraday settlement delays.

The project mimics the analytical workflows of the **Directorate General Market Infrastructure and Payments (DG-MIP)**, processing a synthetic micro-level dataset of 250,000 transactions across 50 commercial banks and 5 core Eurosystem platforms:
* **T2**: Wholesale Real-Time Gross Settlement (RTGS)
* **T2S**: Securities Settlement (Delivery-versus-Payment)
* **ECMS**: Central Bank Collateral Management
* **TIPS**: Instant Retail Payments
* **Pontes**: DLT / Tokenized Asset Settlement

## Key Analytical Phases

### 1. Data Engineering & ETL (DuckDB)
* Synthesized 250,000 highly realistic transaction logs incorporating log-normal financial volumes and timestamp distributions.
* Built a high-performance analytical database using **DuckDB** to aggregate intraday liquidity flows, calculate net positions, and flag queued transactions (liquidity gridlocks).

### 2. Network Topology & Systemic Risk (NetworkX)
* Constructed a directed, weighted multigraph of interbank payment exposures.
* Calculated systemic importance using **PageRank** and **Betweenness Centrality**.
* **Stress Testing:** Simulated an operational outage on the most critical node (Too-Connected-to-Fail), measuring the immediate contagion effect on frozen liquidity (€23.7B) and structural network resilience.

### 3. Econometric Modeling (PanelOLS & VAR)
* **Two-Way Fixed Effects (TWFE) Panel Regression:** Modeled the phenomenon of *liquidity hoarding*, controlling for unobserved bank heterogeneity and daily macroeconomic shocks.
* **Vector Autoregression (VAR):** Estimated a system-level macro model to trace the Impulse Response Function (IRF) of an unexpected 25 bps hike in the €STR rate on intraday settlement delays.

### 4. Interactive Dashboard & Policy Reporting (Streamlit)
* Developed an executive BI dashboard in **Streamlit** integrating Plotly visualizations for transaction volumes, network centrality scatter plots, and econometric shock transmission charts.
* Includes an institutional-grade **Policy Briefing** detailing operational recommendations for automated liquidity recycling and auto-collateralization.

## Tech Stack
* **Data Engineering:** `duckdb`, `pandas`, `numpy`, `SQL`
* **Graph Theory & Risk:** `networkx`
* **Econometrics:** `statsmodels`, `linearmodels`
* **Data Visualization:** `streamlit`, `plotly`, `matplotlib`
* **Environment Management:** `uv`

## Installation & Usage

This project uses `uv` for lightning-fast dependency management and environment isolation.

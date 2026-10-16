import os
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Econometric modeling libraries
from linearmodels.panel import PanelOLS
from statsmodels.tsa.api import VAR

DB_FILE = "data/eurosystem_infrastructure.duckdb"

def extract_and_build_panel_data(db_path: str) -> pd.DataFrame:
    """
    Queries raw transaction logs and compiles a balanced panel dataset (N banks x T days)
    with aggregated liquidity and operational metrics.
    """
    conn = duckdb.connect(db_path)
    
    # SQL aggregation: calculate daily bank-level metrics
    query = """
        WITH daily_bank_stats AS (
            SELECT 
                CAST(timestamp_utc AS DATE) AS date_id,
                sender_bic AS bic,
                COUNT(*) AS total_tx_sent,
                SUM(amount_eur) AS total_outflow_eur,
                SUM(CASE WHEN status = 'QUEUED' THEN 1 ELSE 0 END) AS queued_tx_count,
                SUM(CASE WHEN status = 'QUEUED' THEN amount_eur ELSE 0 END) AS queued_outflow_eur
            FROM target_transactions
            WHERE service_type IN ('T2_CASH', 'T2S_DVP')
            GROUP BY CAST(timestamp_utc AS DATE), sender_bic
        ),
        daily_inflows AS (
            SELECT 
                CAST(timestamp_utc AS DATE) AS date_id,
                receiver_bic AS bic,
                SUM(amount_eur) AS total_inflow_eur
            FROM target_transactions
            WHERE service_type IN ('T2_CASH', 'T2S_DVP') AND status = 'SETTLED'
            GROUP BY CAST(timestamp_utc AS DATE), receiver_bic
        )
        SELECT 
            s.date_id,
            s.bic,
            s.total_tx_sent,
            s.total_outflow_eur,
            COALESCE(i.total_inflow_eur, 0) AS total_inflow_eur,
            (COALESCE(i.total_inflow_eur, 0) - s.total_outflow_eur) AS net_liquidity_eur,
            CAST(s.queued_tx_count AS DOUBLE) / NULLIF(s.total_tx_sent, 0) AS settlement_delay_ratio
        FROM daily_bank_stats s
        LEFT JOIN daily_inflows i ON s.date_id = i.date_id AND s.bic = i.bic
        ORDER BY s.date_id, s.bic;
    """
    df_panel = conn.execute(query).df()
    conn.close()
    
    # Simulate realistic daily monetary policy / rate environment (€STR shock series)
    unique_dates = df_panel['date_id'].unique()
    np.random.seed(42)
    
    # Baseline €STR rate around 3.25% with random daily fluctuations and a mid-month rate hike
    estr_shocks = np.random.normal(loc=0.0, scale=0.005, size=len(unique_dates))
    estr_rates = 3.25 + np.cumsum(estr_shocks)
    
    # Inject exogenous ECB policy shock on day 15
    if len(estr_rates) >= 15:
        estr_rates[14:] += 0.25
        
    date_rate_map = pd.DataFrame({'date_id': unique_dates, 'estr_rate': estr_rates})
    df_panel = df_panel.merge(date_rate_map, on='date_id')
    
    # Normalize Net Liquidity to Millions for cleaner regression scaling
    df_panel['net_liquidity_m_eur'] = df_panel['net_liquidity_eur'] / 1e6
    
    return df_panel


def run_panel_fixed_effects_model(df_panel: pd.DataFrame):
    """
    Estimates a Two-Way Fixed Effects (TWFE) Panel Regression.
    Dependent Variable: settlement_delay_ratio (Liquidity Hoarding proxy)
    Independent Variables: estr_rate, net_liquidity_m_eur
    Includes Bank (Entity) and Date (Time) Fixed Effects with clustered SEs.
    """
    print("\n" + "="*80)
    print("1. ESTIMATING TWO-WAY FIXED EFFECTS (TWFE) PANEL REGRESSION MODEL")
    print("="*80)
    
    # Set multi-index for panel structure (Entity, Time)
    df_reg = df_panel.set_index(['bic', 'date_id'])
    
    dependent = df_reg['settlement_delay_ratio']
    exog_vars = df_reg[['estr_rate', 'net_liquidity_m_eur']]
    
    # Two-Way Fixed Effects model
    mod = PanelOLS(
        dependent=dependent,
        exog=exog_vars,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    
    # Fit with clustered standard errors by Entity (Bank)
    results = mod.fit(cov_type='clustered', cluster_entity=True)
    print(results.summary)
    
    return results


def run_macro_var_model_and_irf(df_panel: pd.DataFrame):
    """
    Aggregates panel to system-level time series and estimates a Vector Autoregressions (VAR) model.
    Traces Impulse Response Functions (IRFs) of system liquidity delays to €STR shocks.
    """
    print("\n" + "="*80)
    print("2. ESTIMATING MACRO SYSTEM-LEVEL VECTOR AUTOREGRESSION (VAR)")
    print("="*80)
    
    # Aggregate daily system totals
    df_macro = df_panel.groupby('date_id').agg({
        'estr_rate': 'mean',
        'settlement_delay_ratio': 'mean',
        'total_outflow_eur': 'sum'
    }).reset_index()
    
    df_macro['total_outflow_b_eur'] = df_macro['total_outflow_eur'] / 1e9
    
    # Prepare stationary time-series data
    var_data = df_macro[['estr_rate', 'settlement_delay_ratio', 'total_outflow_b_eur']].copy()
    
    # Fit VAR model (Lag order p=2)
    var_model = VAR(var_data)
    var_results = var_model.fit(maxlags=2)
    print(var_results.summary())
    
    # Calculate Cholesky-orthogonalized Impulse Response Functions (10 days horizon)
    irf = var_results.irf(10)
    
    # Plot IRF: Response of Settlement Delay Ratio to an €STR Rate Shock
    fig = irf.plot(orth=True, response='settlement_delay_ratio')
    plt.suptitle("Eurosystem Macro Transmission: IRF of Settlement Delay to €STR Shock", y=1.02)
    
    output_img = "var_irf_transmission.png"
    plt.savefig(output_img, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\nImpulse Response Function plot saved successfully as '{output_img}'.")


if __name__ == "__main__":
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Database file '{DB_FILE}' not found. Run Phase 1 first.")
        
    print("Extracting panel data from DuckDB...")
    df_panel = extract_and_build_panel_data(DB_FILE)
    print(f"Panel initialized: {len(df_panel)} total observations ({df_panel['bic'].nunique()} Banks x {df_panel['date_id'].nunique()} Days).")
    
    # Step 1: Panel Fixed Effects Regression
    fe_results = run_panel_fixed_effects_model(df_panel)
    
    # Step 2: VAR Macro Modeling and IRF Plotting
    run_macro_var_model_and_irf(df_panel)
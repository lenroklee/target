import pandas as pd
import numpy as np
import duckdb
from datetime import datetime, timedelta
import uuid

DB_FILE = "eurosystem_infrastructure.duckdb"
NUM_BANKS = 50
NUM_TRANSACTIONS = 250000

def generate_synthetic_target_data(n_transactions: int, n_banks: int) -> pd.DataFrame:
    np.random.seed(42)
    
    bank_bics = [f"BANK{str(i).zfill(3)}XXX" for i in range(n_banks)]
    
    services = ['T2_CASH', 'TIPS_INSTANT', 'T2S_DVP', 'ECMS_COLLATERAL', 'PONTES_DLT']
    service_probs = [0.40, 0.35, 0.15, 0.05, 0.05]
    
    transaction_types = np.random.choice(services, size=n_transactions, p=service_probs)
    
    senders = np.random.choice(bank_bics, size=n_transactions)
    receivers = np.random.choice(bank_bics, size=n_transactions)
    
    # Force sender and receiver to be different
    same_bank_mask = senders == receivers
    receivers[same_bank_mask] = np.roll(receivers, 1)[same_bank_mask]
    
    # Generate realistic timestamps over a 30-day period
    start_date = datetime(2026, 10, 1)
    random_seconds = np.random.randint(0, 30 * 24 * 3600, size=n_transactions)
    timestamps = [start_date + timedelta(seconds=int(s)) for s in random_seconds]
    
    # Assign monetary values based on service type
    amounts = np.zeros(n_transactions)
    amounts[transaction_types == 'T2_CASH'] = np.random.lognormal(mean=14, sigma=2, size=sum(transaction_types == 'T2_CASH'))
    amounts[transaction_types == 'TIPS_INSTANT'] = np.random.lognormal(mean=5, sigma=1.5, size=sum(transaction_types == 'TIPS_INSTANT'))
    amounts[transaction_types == 'T2S_DVP'] = np.random.lognormal(mean=12, sigma=1.5, size=sum(transaction_types == 'T2S_DVP'))
    amounts[transaction_types == 'ECMS_COLLATERAL'] = np.random.lognormal(mean=15, sigma=1, size=sum(transaction_types == 'ECMS_COLLATERAL'))
    amounts[transaction_types == 'PONTES_DLT'] = np.random.lognormal(mean=13, sigma=2, size=sum(transaction_types == 'PONTES_DLT'))
    
    # Simulate liquidity gridlocks (e.g., 2% of T2 transactions are queued due to lack of funds)
    statuses = np.full(n_transactions, 'SETTLED')
    queued_mask = (transaction_types == 'T2_CASH') & (np.random.rand(n_transactions) < 0.02)
    statuses[queued_mask] = 'QUEUED'
    
    df = pd.DataFrame({
        "transaction_id": [str(uuid.uuid4()) for _ in range(n_transactions)],
        "timestamp_utc": timestamps,
        "service_type": transaction_types,
        "sender_bic": senders,
        "receiver_bic": receivers,
        "amount_eur": np.round(amounts, 2),
        "status": statuses
    })
    
    return df.sort_values("timestamp_utc").reset_index(drop=True)

def init_and_load_duckdb(df: pd.DataFrame, db_path: str):
    conn = duckdb.connect(db_path)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS target_transactions (
            transaction_id VARCHAR PRIMARY KEY,
            timestamp_utc TIMESTAMP NOT NULL,
            service_type VARCHAR NOT NULL,
            sender_bic VARCHAR NOT NULL,
            receiver_bic VARCHAR NOT NULL,
            amount_eur DOUBLE NOT NULL,
            status VARCHAR NOT NULL
        );
    """)
    
    conn.register("staging_df", df)
    
    conn.execute("""
        INSERT OR REPLACE INTO target_transactions 
        SELECT * FROM staging_df;
    """)
    
    conn.close()

if __name__ == "__main__":
    print("Initializing synthetic data generation...")
    df_transactions = generate_synthetic_target_data(NUM_TRANSACTIONS, NUM_BANKS)
    
    print(f"Data generated. Loading {len(df_transactions)} records into DuckDB...")
    init_and_load_duckdb(df_transactions, DB_FILE)
    
    print(f"Phase 1 complete. Database saved as '{DB_FILE}'.")
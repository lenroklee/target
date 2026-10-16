import duckdb

conn = duckdb.connect("data/eurosystem_infrastructure.duckdb")

# 1. Verifica conteggio totale delle transazioni
total_count = conn.execute("SELECT COUNT(*) FROM target_transactions;").fetchone()[0]
print(f"Total transactions in DB: {total_count:,}")

# 2. Ripartizione dei volumi per ciascun servizio dell'Eurosistema
breakdown = conn.execute("""
    SELECT 
        service_type, 
        COUNT(*) AS transaction_count,
        ROUND(SUM(amount_eur) / 1e9, 2) AS total_volume_billions_eur
    FROM target_transactions
    GROUP BY service_type
    ORDER BY total_volume_billions_eur DESC;
""").df()

print("\n--- Eurosystem Services Breakdown ---")
print(breakdown)

conn.close()
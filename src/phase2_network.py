import duckdb
import pandas as pd
import networkx as nx

DB_FILE = "data/eurosystem_infrastructure.duckdb"

def load_network_data(db_path: str) -> pd.DataFrame:
    conn = duckdb.connect(db_path)
    
    # Extract aggregated payment exposures between banks for wholesale services
    query = """
        SELECT 
            sender_bic,
            receiver_bic,
            COUNT(*) AS transaction_count,
            SUM(amount_eur) AS total_volume_eur
        FROM target_transactions
        WHERE service_type IN ('T2_CASH', 'T2S_DVP')
          AND status = 'SETTLED'
        GROUP BY sender_bic, receiver_bic;
    """
    df_edges = conn.execute(query).df()
    conn.close()
    return df_edges

def build_eurosystem_graph(df_edges: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    
    for _, row in df_edges.iterrows():
        G.add_edge(
            row['sender_bic'], 
            row['receiver_bic'], 
            weight=row['total_volume_eur'],
            count=row['transaction_count']
        )
    return G

def calculate_systemic_metrics(G: nx.DiGraph) -> pd.DataFrame:
    degree_centrality = nx.degree_centrality(G)
    in_degree = nx.in_degree_centrality(G)
    out_degree = nx.out_degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight='weight')
    pagerank = nx.pagerank(G, weight='weight')
    
    metrics_df = pd.DataFrame({
        'bic': list(G.nodes()),
        'total_degree': [degree_centrality[node] for node in G.nodes()],
        'in_degree': [in_degree[node] for node in G.nodes()],
        'out_degree': [out_degree[node] for node in G.nodes()],
        'betweenness_centrality': [betweenness[node] for node in G.nodes()],
        'pagerank_score': [pagerank[node] for node in G.nodes()]
    }).sort_values(by='pagerank_score', ascending=False).reset_index(drop=True)
    
    return metrics_df

def simulate_operational_outage(G: nx.DiGraph, target_node: str):
    print(f"\n--- Simulating Operational Outage on Target Node: {target_node} ---")
    
    if not G.has_node(target_node):
        raise ValueError(f"Node {target_node} not found in the network.")
        
    in_edges = list(G.in_edges(target_node, data=True))
    out_edges = list(G.out_edges(target_node, data=True))
    
    blocked_inflow = sum(data['weight'] for _, _, data in in_edges)
    blocked_outflow = sum(data['weight'] for _, _, data in out_edges)
    
    # Create a copy and remove node to measure residual structural connectivity
    G_stressed = G.copy()
    G_stressed.remove_node(target_node)
    
    is_connected = nx.is_strongly_connected(G_stressed)
    num_components = nx.number_strongly_connected_components(G_stressed)
    
    print(f"Blocked Inbound Volume (Frozen Liquidity): €{blocked_inflow / 1e9:.2f} Billion")
    print(f"Blocked Outbound Volume (Settlement Failure): €{blocked_outflow / 1e9:.2f} Billion")
    print(f"Network Strongly Connected Post-Outage: {is_connected}")
    print(f"Number of Isolated Components: {num_components}")

if __name__ == "__main__":
    print("Loading transaction exposure matrix from DuckDB...")
    df_edges = load_network_data(DB_FILE)
    
    print("Building directed payment graph...")
    G_target = build_eurosystem_graph(df_edges)
    print(f"Graph initialized: {G_target.number_of_nodes()} Nodes (Banks), {G_target.number_of_edges()} Edges (Flows)")
    
    print("\nCalculating systemic importance metrics...")
    df_systemic = calculate_systemic_metrics(G_target)
    print("\nTop 5 Systemically Important Banks (by PageRank):")
    print(df_systemic.head())
    
    # Run shock simulation on the most critical node
    top_node = df_systemic.iloc[0]['bic']
    simulate_operational_outage(G_target, top_node)
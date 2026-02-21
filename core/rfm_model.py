
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

def calculate_rfm(df):
    """
    Computes Recency, Frequency, and Monetary values for each customer.
    """
    last_date = df["invoicedate"].max()
    
    rfm = df.groupby('customerid').agg({
        'invoicedate': lambda x: (last_date - x.max()).days,
        'invoiceno': 'nunique',
        'total_price': 'sum'
    }).reset_index()
    
    rfm.columns = ['customerid', 'recency', 'frequency', 'monetary']
    rfm["monetary"] = rfm["monetary"].astype(int)
    return rfm

def train_kmeans(rfm, n_clusters=5):
    """
    Trains KMeans model on RFM data and assigns segments.
    Returns:
        rfm_with_clusters (pd.DataFrame)
        kmeans_model (KMeans)
        scaler (StandardScaler)
    """
    rfm_data = rfm[['recency', 'frequency', 'monetary']]
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Auto-Labeling Logic
    # We want to map cluster IDs to meaningful names based on centroids.
    # High Monetary + High Frequency + Low Recency = Champions
    
    cluster_means = rfm.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()
    
    # We can rank clusters. 
    # A simple scoring: Rank clusters by Monetary (desc), Frequency (desc), Recency (asc)
    # This is a bit complex to generalize perfectly, but let's try a heuristic.
    # We will assign labels based on sorted Monetary value for simplicity, 
    # assuming higher spenders are better segments.
    
    sorted_clusters = cluster_means.sort_values(by="monetary", ascending=False).index.tolist()
    
    # Map ranked clusters to labels
    # Rank 0 (Highest Money) -> Champions
    # Rank 1 -> Loyal Customers
    # Rank 2 -> Big Spenders (At Risk) - (Naming from notebook, though "Big Spender" usually implies loyal)
    # Rank 3 -> Low Value
    # Rank 4 -> Lost
    
    # Wait, notebook labels:
    # 2: 'Champions', 3: 'Loyal', 4: 'Big Spenders', 0: 'Low Value', 1: 'Lost'
    # Use mapping based on sorted order to be consistent roughly with "Value"
    
    labels_map = {
        sorted_clusters[0]: "Champions",
        sorted_clusters[1]: "Loyal Customers",
        sorted_clusters[2]: "Potential Loyalists", # Changed "Big Spenders (At Risk)" to standard term if slightly lower
        sorted_clusters[3]: "At Risk",
        sorted_clusters[4]: "Lost"
    }
    
    # Override with notebook specific names if needed, but dynamic is safer.
    # Let's stick to notebook names but apply them dynamically.
    notebook_names = ["Champions", "Loyal Customers", "Big Spenders", "Low Value Customers", "Lost Customers"]
    
    # Re-map: 
    # To do this safely, we might want to standardize. 
    # For now, let's just map rank 0->4 to the list above.
    
    final_mapping = {}
    for rank, cluster_id in enumerate(sorted_clusters):
        if rank < len(notebook_names):
            final_mapping[cluster_id] = notebook_names[rank]
        else:
            final_mapping[cluster_id] = f"Segment {rank}"
            
    rfm['segment'] = rfm['cluster'].map(final_mapping)
    
    return rfm, kmeans, scaler, final_mapping

def save_model(model, filename, save_dir="artifacts"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    joblib.dump(model, os.path.join(save_dir, filename))

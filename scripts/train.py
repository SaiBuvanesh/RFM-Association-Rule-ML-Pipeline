import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_processing import load_and_clean_data
from core.rfm_model import calculate_rfm, train_kmeans, save_model
from core.recommendation import generate_rules, save_rules
import os
import joblib

def main():
    # Define relative paths from script location
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, "data", "data.csv")
    
    print(f"Loading data from {data_path}...")
    df = load_and_clean_data(data_path)
    print(f"Data loaded. Shape: {df.shape}")
    
    print("Running RFM analysis...")
    rfm = calculate_rfm(df)
    print("Training KMeans...")
    rfm_labeled, kmeans, scaler, segment_map = train_kmeans(rfm)
    
    print("Saving RFM artifacts...")
    save_model(kmeans, "kmeans_model.pkl")
    save_model(scaler, "scaler.pkl")
    joblib.dump(segment_map, os.path.join(base_dir, "artifacts", "segment_map.pkl"))
    
    rfm_segments_path = os.path.join(base_dir, "artifacts", "rfm_segments.csv")
    rfm_labeled.to_csv(rfm_segments_path, index=False)
    print(f"RFM segments saved to {rfm_segments_path}.")

    all_products = sorted(df['description'].unique().astype(str))
    products_pkl_path = os.path.join(base_dir, "artifacts", "unique_products.pkl")
    joblib.dump(all_products, products_pkl_path)
    print(f"Unique products saved to {products_pkl_path}: {len(all_products)}")

    print("Generating association rules...")
    rules = generate_rules(df, min_support=0.005, min_threshold=0.2) 
    
    print(f"Rules generated: {len(rules)}")
    save_rules(rules, "association_rules.pkl")
    
    print("Saving product price mapping...")
    product_prices = df.groupby('description')['unitprice'].mean().to_dict()
    joblib.dump(product_prices, os.path.join("artifacts", "product_prices.pkl"))
    
    print("Optimization complete.")

if __name__ == "__main__":
    main()

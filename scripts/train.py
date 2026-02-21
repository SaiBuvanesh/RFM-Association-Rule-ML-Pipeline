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
    print("Loading data...")
    df = load_and_clean_data(r"d:/data/data/data.csv")
    print(f"Data loaded. Shape: {df.shape}")
    
    print("Running RFM analysis...")
    rfm = calculate_rfm(df)
    print("Training KMeans...")
    rfm_labeled, kmeans, scaler, segment_map = train_kmeans(rfm)
    
    print("Saving RFM artifacts...")
    save_model(kmeans, "kmeans_model.pkl")
    save_model(scaler, "scaler.pkl")
    
    rfm_labeled.to_csv("d:/data/artifacts/rfm_segments.csv", index=False)
    print("RFM segments saved.")

    all_products = sorted(df['description'].unique().astype(str))
    joblib.dump(all_products, "d:/data/artifacts/unique_products.pkl")
    print(f"Unique products saved: {len(all_products)}")

    print("Generating association rules...")
    rules = generate_rules(df, min_support=0.005, min_threshold=0.2) 
    
    print(f"Rules generated: {len(rules)}")
    save_rules(rules, "association_rules.pkl")
    print("Optimization complete.")

if __name__ == "__main__":
    main()


import pandas as pd
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from core.recommendation import recommend_for_basket

def debug_flow():
    rules_path = r"d:/data/artifacts/association_rules.pkl"
    rules = joblib.load(rules_path)
    
    basket = ["REGENCY CAKESTAND 3 TIER"]
    normalized_basket = {str(item).upper().strip() for item in basket}
    print(f"Normalized Basket: {normalized_basket}")
    
    # Check if any rule consequent matches the normalized basket
    total_consequents = []
    for _, row in rules.iterrows():
        for item in row['consequents']:
            clean = str(item).upper().strip()
            if clean in normalized_basket:
                print(f"MATCH FOUND in rules: Original='{item}', Clean='{clean}'")
    
    recs = recommend_for_basket(basket, rules, top_n=4)
    print(f"Final Recs: {recs}")

if __name__ == "__main__":
    debug_flow()

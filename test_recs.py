
import pandas as pd
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from core.recommendation import recommend_for_basket

def test_recommendations():
    rules_path = r"d:/data/artifacts/association_rules.pkl"
    if not os.path.exists(rules_path):
        print("Rules not found")
        return
    
    rules = joblib.load(rules_path)
    
    test_baskets = [
        ["WHITE HANGING HEART T-LIGHT HOLDER"],
        ["REGENCY CAKESTAND 3 TIER"],
        ["JUMBO BAG RED RETROSPOT"],
        ["PARTY BUNTING"],
        ["ASSORTED COLOUR BIRD ORNAMENT"]
    ]
    
    for basket in test_baskets:
        recs = recommend_for_basket(basket, rules, top_n=4)
        print(f"Basket: {basket}")
        print(f"Recs  : {recs}\n")

if __name__ == "__main__":
    test_recommendations()

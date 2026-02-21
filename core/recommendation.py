
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import joblib
import os

def generate_rules(df, min_support=0.01, min_threshold=0.3):
    """
    Generates association rules from transaction data.
    """
    invoice_grouped = df.groupby('invoiceno')['description'].apply(list).reset_index()
    transactions = invoice_grouped['description'].tolist()
    
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_array, columns=te.columns_)
    
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
    rules = rules.sort_values(['lift', 'confidence'], ascending=False)
    
    return rules

def recommend_for_basket(basket_items, rules, top_n=5):
    """
    Returns recommendations with a 2-stage fallback strategy.
    1. Strict Match: Rules where Antecedents are SUBSET of Basket.
    2. Partial Match: Rules where Antecedents share AT LEAST ONE item with Basket.
    """
    basket_items = [item.upper() for item in basket_items]
    basket_set = set(basket_items)
    
    recommendations = []
    
    # --- Strategy 1: Strict Match ---
    # (Existing logic: if Cart has A, B and Rule is A->C, it matches. If Rule is A,B->C, it matches)
    strict_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(basket_set))]
    strict_rules = strict_rules.sort_values(['confidence', 'lift'], ascending=False)
    
    for _, row in strict_rules.iterrows():
        for item in row['consequents']:
            if item not in basket_items:
                recommendations.append(item)
    
    # Remove duplicates
    recommendations = list(dict.fromkeys(recommendations))
    
    if len(recommendations) >= top_n:
        return recommendations[:top_n]
        
    # --- Strategy 2: Partial Match (Relaxed) ---
    # If we don't have enough recommendations, look for rules where ANY valid item is in antecedent
    # e.g. Rule A,B -> C. Cart has A. This is a partial match.
    
    def is_partial_match(antecedents):
        # Return True if intersection is not empty
        return not antecedents.isdisjoint(basket_set)
        
    partial_rules = rules[rules['antecedents'].apply(is_partial_match)]
    partial_rules = partial_rules.sort_values(['lift', 'confidence'], ascending=False) # Sort by Lift for partials
    
    for _, row in partial_rules.iterrows():
        for item in row['consequents']:
            if item not in basket_items and item not in recommendations:
                recommendations.append(item)
                
    recommendations = list(dict.fromkeys(recommendations))
    
    if len(recommendations) >= top_n:
        return recommendations[:top_n]
        
    # --- Strategy 3: Global Popularity (Fallback) ---
    # If still not enough, return top consequents by support (global popularity)
    popular_rules = rules.sort_values('support', ascending=False)
    for _, row in popular_rules.iterrows():
        for item in row['consequents']:
            if item not in basket_items and item not in recommendations:
                recommendations.append(item)
                if len(recommendations) >= top_n:
                    return recommendations
                    
    return recommendations[:top_n]

def save_rules(rules, filename="rules.pkl", save_dir="artifacts"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    joblib.dump(rules, os.path.join(save_dir, filename))

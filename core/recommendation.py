
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import joblib
import os

def generate_rules(df, min_support=0.01, min_threshold=0.3):
    """
    Generates association rules from transaction data using FPGrowth for better performance.
    """
    # Ensure all descriptions are strings and handle potential nulls
    df['description'] = df['description'].astype(str)
    
    invoice_grouped = df.groupby('invoiceno')['description'].apply(list).reset_index()
    transactions = invoice_grouped['description'].tolist()
    
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_array, columns=te.columns_).astype(bool) # Force boolean
    
    # Using FPGrowth instead of Apriori for memory efficiency
    frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        return pd.DataFrame()
        
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
    rules = rules.sort_values(['lift', 'confidence'], ascending=False)
    
    return rules

def recommend_for_basket(basket_items, rules, top_n=5):
    """
    Returns recommendations with a 2-stage fallback strategy.
    1. Strict Match: Rules where Antecedents are SUBSET of Basket.
    2. Partial Match: Rules where Antecedents share AT LEAST ONE item with Basket.
    """
    # Normalize basket items for comparison (Upper and Strip)
    normalized_basket = {str(item).upper().strip() for item in basket_items}
    # print(f"DEBUG: normalized_basket={normalized_basket}")
    
    recommendations = []
    
    def get_clean_consequents(rules_df):
        res = []
        for _, row in rules_df.iterrows():
            for item in row['consequents']:
                clean_item = str(item).upper().strip()
                if clean_item not in normalized_basket:
                    res.append(item)
        return list(dict.fromkeys(res))

    # --- Strategy 1: Strict Match ---
    # We use the normalized_basket for matching since it's the most robust
    strict_rules = rules[rules['antecedents'].apply(lambda x: {str(a).upper().strip() for a in x}.issubset(normalized_basket))]
    strict_rules = strict_rules.sort_values(['confidence', 'lift'], ascending=False)
    
    recommendations.extend(get_clean_consequents(strict_rules))
    recommendations = list(dict.fromkeys(recommendations))
    
    if len(recommendations) >= top_n + 4: # If we have plenty, we can shuffle a bit
        import random
        pool = recommendations[:top_n + 4]
        random.shuffle(pool)
        return pool[:top_n]
    
    if len(recommendations) >= top_n:
        return recommendations[:top_n]
        
    # --- Strategy 2: Partial Match ---
    def is_partial_match(antecedents):
        clean_ants = {str(a).upper().strip() for a in antecedents}
        return not clean_ants.isdisjoint(normalized_basket)
        
    partial_rules = rules[rules['antecedents'].apply(is_partial_match)]
    partial_rules = partial_rules.sort_values(['lift', 'confidence'], ascending=False)
    
    new_recs = get_clean_consequents(partial_rules)
    for r in new_recs:
        if r not in recommendations:
            recommendations.append(r)
                
    if len(recommendations) >= top_n:
        # Shuffle the newly added ones for diversity if we have enough
        return recommendations[:top_n]
        
    # --- Strategy 3: Global Diversity (Improved Fallback) ---
    top_global = rules.sort_values('support', ascending=False).head(30)
    global_recs = get_clean_consequents(top_global)
    
    import random
    random.shuffle(global_recs)
    
    for r in global_recs:
        if r not in recommendations:
            recommendations.append(r)
            if len(recommendations) >= top_n:
                break
                    
    return recommendations[:top_n]

def save_rules(rules, filename="rules.pkl", save_dir="artifacts"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    joblib.dump(rules, os.path.join(save_dir, filename))

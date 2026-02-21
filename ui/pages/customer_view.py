
import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Add parent directory to path to import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import importlib
import core.recommendation
importlib.reload(core.recommendation)
from core.recommendation import recommend_for_basket

def load_resources():
    rules_path = r"d:/data/artifacts/association_rules.pkl"
    if not os.path.exists(rules_path):
        st.error("Resource files not found. Please ensure training is complete.")
        return None, None
    
    rules = joblib.load(rules_path)
    
    products_path = r"d:/data/artifacts/unique_products.pkl"
    if os.path.exists(products_path):
        items = joblib.load(products_path)
    else:
        items = sorted(list(set(rules['antecedents'].explode()) | set(rules['consequents'].explode())))
        
    return rules, items

st.title("Shopping Assistant")
st.caption("Personalized product recommendations powered by association rule mining.")

rules, product_list = load_resources()

if rules is not None:
    # Sidebar Selection
    with st.sidebar:
        st.header("Your Basket")
        selected_items = st.multiselect("Search Products", product_list, placeholder="Type to search items...")
        
        st.divider()
        if selected_items:
            st.success(f"{len(selected_items)} items selected")
            st.button("Complete Purchase", type="primary")
        else:
            st.info("Your basket is empty.")

    # Main Interface
    if selected_items:
        st.subheader("Current Selection")
        cols = st.columns(4)
        for i, item in enumerate(selected_items):
            with cols[i % 4]:
                st.container(border=True).markdown(f"**{item}**\n\nQty: 1")
    
        st.divider()
        st.subheader("Recommended Additions")
        
        # Strip items to ensure matching logic handles potential trailing spaces from data
        stripped_selection = [item.strip() for item in selected_items]
        recommendations = recommend_for_basket(stripped_selection, rules, top_n=4)
        
        if recommendations:
            rec_cols = st.columns(4)
            for i, rec in enumerate(recommendations):
                with rec_cols[i % 4]:
                    with st.container(border=True):
                        st.markdown(f"**{rec}**")
                        st.caption("Frequently purchased together")
                        if st.button(f"Add to Basket", key=f"add_{i}"):
                            st.toast(f"Added {rec} to basket")
        else:
            st.info("No specific recommendations found for the current selection.")
            
    else:
        st.markdown("""
        ### Build your basket
        Select products in the sidebar to receive AI-generated recommendations based on historical transaction patterns.
        
        #### Popular Collections
        - **Decor**: WHITE HANGING HEART T-LIGHT HOLDER
        - **Kitchen**: REGENCY CAKESTAND 3 TIER
        - **Storage**: JUMBO BAG RED RETROSPOT
        """)

    with st.expander("Technical Details (Association Rules)"):
        # Formatting for readability
        display_rules = rules.copy()
        for col in ['antecedents', 'consequents']:
            display_rules[col] = display_rules[col].apply(lambda x: list(x))
        st.dataframe(display_rules.head(10), use_container_width=True)

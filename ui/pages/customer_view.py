
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
    # Define base path relative to the current file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    rules_path = os.path.join(base_dir, "artifacts", "association_rules.pkl")
    if not os.path.exists(rules_path):
        st.error(f"Resource files not found at {rules_path}. Please ensure training is complete.")
        return None, None
    
    rules = joblib.load(rules_path)
    
    products_path = os.path.join(base_dir, "artifacts", "unique_products.pkl")
    if os.path.exists(products_path):
        items = joblib.load(products_path)
    else:
        items = sorted(list(set(rules['antecedents'].explode()) | set(rules['consequents'].explode())))
        
    return rules, items

st.title("Shopping Assistant")
st.caption("Personalized product recommendations powered by association rule mining.")

# Initialize session state for basket
if 'basket' not in st.session_state:
    st.session_state.basket = []

rules, product_list = load_resources()

if rules is not None:
    # Sidebar Selection
    with st.sidebar:
        st.header("Your Basket")
        selected_items = st.multiselect(
            "Search Products", 
            product_list, 
            key="basket",
            placeholder="Type to search items...",
            help="Search and select products to add them to your shopping basket. Recommendations will update automatically."
        )
        
        st.divider()
        if selected_items:
            st.success(f"{len(selected_items)} items selected")
            if st.button("Complete Purchase", type="primary", help="Proceed to checkout with your current selection."):
                st.balloons()
                st.success("Purchase logic would be integrated here!")
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
                        
                        # Add to Basket Button logic
                        if st.button(f"Add to Basket", key=f"add_{i}", help=f"Add {rec} to your current selection."):
                            if rec not in st.session_state.basket:
                                st.session_state.basket.append(rec)
                                st.toast(f"Added {rec} to basket")
                                st.rerun() # Refresh to update the multiselect and view
                            else:
                                st.toast(f"{rec} is already in your basket!")
        else:
            st.info("No specific recommendations found for the current selection.")
            
    else:
        st.markdown("""
        ### Build your basket
        Select products in the sidebar to receive AI-generated recommendations based on historical transaction patterns.
        
        #### Popular Collections
        - Decor: WHITE HANGING HEART T-LIGHT HOLDER
        - Kitchen: REGENCY CAKESTAND 3 TIER
        - Storage: JUMBO BAG RED RETROSPOT
        """)

    with st.expander("Technical Details (Association Rules)"):
        # Formatting for readability
        display_rules = rules.copy()
        for col in ['antecedents', 'consequents']:
            display_rules[col] = display_rules[col].apply(lambda x: list(x))
        st.dataframe(display_rules.head(10), use_container_width=True)

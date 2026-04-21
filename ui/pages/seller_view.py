
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
import joblib
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def load_data_and_models():
    # Define base path relative to the current file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    artifact_dir = os.path.join(base_dir, "artifacts")
    
    rfm_path = os.path.join(artifact_dir, "rfm_segments.csv")
    if not os.path.exists(rfm_path):
        return None, None, None, None, None
    
    df = pd.read_csv(rfm_path)
    df['customerid'] = df['customerid'].astype(int)
    
    # Load Models for simulation
    try:
        scaler = joblib.load(os.path.join(artifact_dir, "scaler.pkl"))
        kmeans = joblib.load(os.path.join(artifact_dir, "kmeans_model.pkl"))
        prices = joblib.load(os.path.join(artifact_dir, "product_prices.pkl"))
        seg_map = joblib.load(os.path.join(artifact_dir, "segment_map.pkl"))
    except Exception as e:
        st.warning(f"Some analytical components are missing: {e}")
        return df, None, None, None, None
        
    return df, scaler, kmeans, prices, seg_map

st.title("Business Intelligence")
st.caption("Strategic analysis of customer segments and purchasing behaviors.")

rfm_df, scaler, kmeans, prices, seg_map = load_data_and_models()

if rfm_df is not None:
    # Use tabs for a cleaner interface
    tab_overview, tab_session, tab_lookup = st.tabs(["Market Overview", "Active Session Analytics", "Customer Deep Dive"])

    with tab_overview:
        # Key Performance Indicators
        st.markdown("#### Performance Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Customer Base", f"{len(rfm_df):,}", help="Total number of unique customers identified in the dataset.")
        k2.metric("Average CLV", f"£{rfm_df['monetary'].mean():.0f}", help="Average Customer Lifetime Value (Total Spend) across all segments.")
        k3.metric("Purchase Frequency", f"{rfm_df['frequency'].mean():.1f}", help="Average number of distinct transactions per customer.")
        k4.metric("Market Segments", f"{rfm_df['segment'].nunique()}", help="Number of distinct behavioral groups identified by AI clustering.")
        
        st.divider()
        
        # Visualizations
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown("#### Customer Segmentation (RFM)", help="Relationship between Recency and Monetary values. Use the legend to filter cohorts.")
            fig = px.scatter(
                rfm_df, x='recency', y='monetary', color='segment',
                size='monetary', hover_name='customerid',
                log_y=True, template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Prism,
                labels={"recency": "Days Since Last Purchase", "monetary": "Total Spend (£)"},
                height=500 
            )
            fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### Segment Distribution")
            seg_counts = rfm_df['segment'].value_counts().reset_index()
            seg_counts.columns = ['segment', 'count']
            fig_pie = px.pie(seg_counts, values='count', names='segment', hole=0.5, 
                             color_discrete_sequence=px.colors.qualitative.Prism,
                             height=400)
            fig_pie.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        with st.expander("Understanding the Segments"):
            st.markdown("""
            - **Champions**: High frequency, high monetary, and low recency. These are your most valuable and active customers.
            - **Loyal Customers**: Moderate to high frequency and monetary. They buy regularly.
            - **Big Spenders**: High monetary values but may have higher recency (at risk of churn).
            - **Low Value**: Lower spending and frequency across the board.
            - **Lost Customers**: Highest recency and lowest frequency. Customers who have likely moved to a competitor.
            """)

    with tab_session:
        st.markdown("#### Real-Time Behavior Simulation")
        st.markdown("This module analyzes the active shopping session to predict which customer cohort the current user behavior aligns with.")
        
        # Get active basket from session state
        active_basket = st.session_state.get('basket', [])
        
        if not active_basket:
            st.info("No items currently in the active session basket. Visit the Shopping Assistant to build a session for analysis.")
        elif scaler is None or kmeans is None or prices is None:
            st.error("Analytical models required for session simulation are not available.")
        else:
            # Simulation Logic
            st.markdown(f"**Analyzing Session Basket ({len(active_basket)} items)**")
            
            # 1. Calculate Simulated Monetary (Sum of average prices)
            total_value = sum([prices.get(item, 10.0) for item in active_basket]) # Fallback to 10.0 if unknown
            
            # 2. Simulated RFM
            # Recency = 0 (Just shopped), Frequency = 1 (Current basket), Monetary = total_value
            sim_rfm = np.array([[0, 1, total_value]])
            
            # 3. Scale and Predict
            sim_scaled = scaler.transform(sim_rfm)
            cluster_id = kmeans.predict(sim_scaled)[0]
            
            # 4. Map to Segment
            predicted_segment = seg_map.get(cluster_id, f"Cluster {cluster_id}")
            
            # UI Display
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Simulated Basket Value", f"£{total_value:.2f}")
                st.metric("Predicted Segment Persona", predicted_segment)
            
            with c2:
                # Show items being analyzed
                st.write("Basket Composition:")
                for item in active_basket[:10]: # Cap at 10 for UI
                    st.caption(f"- {item}")
                if len(active_basket) > 10:
                    st.caption(f"... and {len(active_basket)-10} more")
            
            st.divider()
            st.subheader("Market Potential Analysis")
            st.write(f"Based on the current selection, this session profile aligns most closely with the **{predicted_segment}** cohort.")
            
            # Engagement strategy for this simulated profile
            strategies = {
                "Champions": "High immediate conversion probability. Recommend premium upsells.",
                "Loyal Customers": "Strong repeat potential. Highlight membership benefits.",
                "Big Spenders": "Value-driven profile. Offer bundle discounts to increase basket size.",
                "Low Value": "Transactional profile. Focus on low-friction checkout.",
                "Lost Customers": "Reactivation simulation. Show significant first-purchase incentives."
            }
            st.info(f"**Recommended Approach:** {strategies.get(predicted_segment, 'Monitor session progression for clearer behavior patterns.')}")

    with tab_lookup:
        st.markdown("#### Customer Intelligence Search")
        
        col_search, col_res = st.columns([1, 2])
        
        with col_search:
            st.markdown("Analyze individual customer profiles for tailored engagement strategies.")
            search_id = st.text_input(
                "Enter Customer ID", 
                placeholder="e.g., 17850",
                help="Enter a numeric Customer ID to retrieve historical RFM performance and cohort alignment."
            )
            
            if not rfm_df.empty:
                sample_ids = rfm_df['customerid'].sample(min(3, len(rfm_df))).tolist()
                st.caption(f"Reference IDs: {', '.join(map(str, sample_ids))}")

        with col_res:
            if search_id:
                try:
                    cust_int = int(float(search_id))
                    cust_data = rfm_df[rfm_df['customerid'] == cust_int]
                    
                    if not cust_data.empty:
                        c = cust_data.iloc[0]
                        with st.container(border=True):
                            st.markdown(f"**Customer Profile: #{cust_int}**")
                            st.markdown(f"Segment: **{c['segment']}**")
                            
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Recency", f"{c['recency']}d")
                            m2.metric("Frequency", f"{c['frequency']}")
                            m3.metric("Value", f"£{c['monetary']}")
                            
                            st.divider()
                            
                            # Strategies
                            strategies = {
                                "Champions": "Maintain engagement with exclusive previews and loyalty rewards.",
                                "Loyal Customers": "Focus on cross-selling and requesting referrals.",
                                "Potential Loyalists": "Offer bundle discounts to increase purchase frequency.",
                                "At Risk": "Implement win-back campaigns with time-limited offers.",
                                "Lost": "Conduct exit surveys and offer significant re-entry incentives."
                            }
                            
                            strategy = strategies.get(c['segment'], "Monitor behavior for shifts in cohort.")
                            st.success(f"**Engagement Strategy:** {strategy}")
                            
                    else:
                        st.error(f"Customer ID {cust_int} not found.")
                except ValueError:
                    st.error("Invalid ID format. Please enter a numeric value.")
            else:
                st.info("Enter a Customer ID to view specific behavioral insights.")

else:
    st.warning("Analysis required. Please run the training pipeline to generate artifacts.")

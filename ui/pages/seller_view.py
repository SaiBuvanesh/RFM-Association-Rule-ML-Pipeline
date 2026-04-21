
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def load_data():
    # Define base path relative to the current file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    rfm_path = os.path.join(base_dir, "artifacts", "rfm_segments.csv")
    if not os.path.exists(rfm_path):
        return None
    df = pd.read_csv(rfm_path)
    df['customerid'] = df['customerid'].astype(int)
    return df

st.title("Business Intelligence")
st.caption("Strategic analysis of customer segments and purchasing behaviors.")

rfm_df = load_data()

if rfm_df is not None:
    # Use tabs for a cleaner interface
    tab_overview, tab_lookup = st.tabs(["📊 Market Overview", "🔍 Customer Deep Dive"])

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
            st.markdown("#### Customer Segmentation (RFM)", help="This scatter plot shows the relationship between Recency (how recently they shopped) and Monetary (how much they spent). Use the legend to toggle segments.")
            fig = px.scatter(
                rfm_df, x='recency', y='monetary', color='segment',
                size='monetary', hover_name='customerid',
                log_y=True, template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Prism,
                labels={"recency": "Days Since Last Purchase", "monetary": "Total Spend (£)"},
                height=500 # Explicit height for better visibility
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

        with st.expander("📝 Understanding the Segments"):
            st.markdown("""
            - **Champions**: Your best customers. They bought recently, buy often, and spend the most.
            - **Loyal Customers**: Frequent buyers who respond well to loyalty programs.
            - **Big Spenders**: Customers who have high total spend but haven't bought in a while (At Risk).
            - **Low Value**: Customers who spend little and buy infrequently.
            - **Lost Customers**: Lowest recency and frequency scores. Require significant win-back effort.
            """)

    with tab_lookup:
        st.markdown("#### Strategic Insights")
        
        col_search, col_res = st.columns([1, 2])
        
        with col_search:
            st.markdown("Analyze individual customer profiles for tailored engagement strategies.")
            search_id = st.text_input(
                "Enter Customer ID", 
                placeholder="e.g., 17850",
                help="Type a specific Customer ID to see their personal RFM metrics and a recommended business strategy for them."
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
                            
                            strategy = strategies.get(c['segment'], "Monitor purchasing behavior for shifts in cohort.")
                            st.success(f"**Engagement Strategy:** {strategy}")
                            
                    else:
                        st.error(f"Customer ID {cust_int} not found.")
                except ValueError:
                    st.error("Invalid ID format. Please enter a numeric value.")
            else:
                st.info("Query a customer ID in the search box to view specific segment insights.")

else:
    st.warning("Analysis required. Please execute the training scripts.")

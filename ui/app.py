
import streamlit as st
import os

# Set page config
st.set_page_config(
    page_title="Customer Segmentation & Basket Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Classic" feel
st.markdown("""
<style>
    .main {
        background-color: #FFFFFF;
        color: #1E293B;
    }
    .stAppHeader {
        background-color: transparent !important;
    }
    h1, h2, h3, h4 {
        color: #1E3A8A;
        font-family: 'Georgia', serif;
        font-weight: 500;
    }
    .stMetric {
        background-color: #F8FAFC !important;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    [data-testid="stMetricLabel"] > div {
        color: #64748B !important;
        font-family: 'Segoe UI', sans-serif !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stMetricValue"] > div {
        color: #1E3A8A !important;
        font-family: 'Georgia', serif !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        background-color: #1E3A8A;
        color: white;
        border: none;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("""
        <div style="background-color: #1E3A8A; padding: 3rem; border-radius: 10px; text-align: center; margin-bottom: 2rem;">
            <h1 style="color: white; margin-bottom: 0.5rem; font-family: 'Georgia', serif;">Customer Segmentation & Basket Intelligence System</h1>
            <p style="color: #E2E8F0; font-size: 1.1rem; font-family: 'Segoe UI', sans-serif;">AI-powered segmentation and product recommendation system for modern retail.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Customer Experience")
        st.info("Interactive shopping assistant that provides personalized product suggestions using association rule mining.")
    
    with col2:
        st.markdown("#### Business Analytics")
        st.success("Strategic insights for retailers, featuring customer segmentation through RFM analysis and behavioral cohorts.")

    st.markdown("---")
    st.markdown("#### System Capabilities")
    blocks = st.columns(4)
    blocks[0].metric("Segmentation", "K-Means", "Clustering")
    blocks[1].metric("Intelligence", "Apriori", "Association")
    blocks[2].metric("Analysis", "RFM", "Behavioral")
    blocks[3].metric("Interface", "Streamlit", "Interactive")

if __name__ == "__main__":
    main()

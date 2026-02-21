# Customer Segmentation & Basket Intelligence System

An AI-driven analytical platform for modern retail management, providing deep insights into customer behavior and personalized shopping experiences.

## Technical Capabilities

- **RFM Segmentation Engine**: Beyond simple grouping, the system calculates granular R/F/M scores to measure customer "vibrancy." It handles outlier suppression and log-transforming skewed monetary values to ensure clustering stability.
- **Market Basket Mining (Apriori)**: Employs association rule mining with optimized support (1%) and confidence (20%) thresholds. This allows the system to discover non-obvious cross-selling opportunities (e.g., "Customers who bought item A also bought B").
- **Data Engineering Pipeline**: Automated cleaning scripts handle:
    - Removal of "Returns" (negative quantities/prices).
    - Customer ID imputation and formatting.
    - Description standardization.
- **Strategic BI Dashboard**: Built with Plotly and Streamlit to provide real-time cohorts and individualized strategy recommendations.

## Business Significance
This system isn't just about code; it's about making better decisions:
1. **Reduce Churn**: Identifying "At Risk" customers before they leave.
2. **Increase AOV**: Suggesting items to current baskets based on historical peer behavior.
3. **Personalized Marketing**: Categorizing customers so you can target Champions with rewards and Lost customers with win-back offers.

## Project Structure

```text
├── artifacts/    # Trained models and processed data artifacts
├── core/         # Business logic and analytical modules
│   ├── data_processing.py
│   ├── recommendation.py
│   └── rfm_model.py
├── data/         # Raw transactional datasets
├── scripts/      # Training and utility scripts
│   └── train.py
├── ui/           # Streamlit application and view components
│   ├── app.py
│   └── pages/
└── requirements.txt
```

## Setup and Installation

### Prerequisites
- Python 3.8 or higher

### Installation
1. Clone the repository.
2. Initialize and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution
1. Run the analytical pipeline to generate artifacts:
   ```bash
   python scripts/train.py
   ```
2. Launch the intelligence platform:
   ```bash
   streamlit run ui/app.py
   ```

## Technology Stack
- **Analysis**: Pandas, Scikit-Learn
- **Association Mining**: Mlxtend
- **Visualization**: Plotly
- **Interface**: Streamlit


import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    """
    Loads data from CSV, standardizes column names, removes duplicates, 
    and handles missing values.
    """
    try:
        df = pd.read_csv(filepath, encoding="latin1")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {filepath}")

    # Standardize columns
    df.columns = df.columns.str.lower()
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Handle missing customer IDs (crucial for segmentation)
    df = df.dropna(subset=['customerid'])
    
    # robust conversion to int then string to avoid "12345.0"
    df['customerid'] = df['customerid'].astype(float).astype(int)
    
    # Convert date
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
    
    # Clean Quantity and Calculate Price
    df["quantity"] = df["quantity"].abs()
    df["total_price"] = df["unitprice"] * df["quantity"]
    
    return df

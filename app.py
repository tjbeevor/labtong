import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="LPBF AlSi10Mg Analysis", layout="wide")

def clean_column_names(df):
    """Clean and structure the LPBF dataset"""
    # Read the first row to get actual column names
    actual_headers = pd.read_csv(uploaded_file, nrows=0).columns
    
    # Display the actual column headers for debugging
    st.write("First row headers:", actual_headers.tolist())
    
    # Create a new DataFrame with proper headers
    df_cleaned = pd.read_csv(uploaded_file, header=1)
    
    # Display first few rows for verification
    st.write("First few rows of data:", df_cleaned.head())
    
    return df_cleaned

def main():
    st.title("LPBF AlSi10Mg Process-Property Analysis")
    st.write("""
    Upload your LPBF AlSi10Mg dataset to analyze:
    - Process parameters influence on mechanical properties
    - Heat treatment effects
    - Microstructural relationships
    - Property optimization
    """)
    
    uploaded_file = st.file_uploader("Upload LPBF data (CSV)", type="csv")
    
    if uploaded_file:
        # First, let's look at the raw data structure
        raw_df = pd.read_csv(uploaded_file)
        
        # Display the first few rows of raw data to understand structure
        st.write("### Data Structure Analysis")
        st.write("First few rows of raw data:")
        st.dataframe(raw_df.head())
        
        # Let user confirm if they want to proceed with data processing
        if st.button("Process Data"):
            cleaned_df = clean_column_names(uploaded_file)
            st.write("### Processed Data Preview")
            st.dataframe(cleaned_df.head())

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="LPBF AlSi10Mg Analysis", layout="wide")

def clean_numeric_data(df, column):
    """Clean and convert numeric data, handling errors"""
    try:
        return pd.to_numeric(df[column], errors='coerce')
    except:
        return pd.Series([np.nan] * len(df))

def process_lpbf_data(df):
    """Process LPBF data with robust error handling"""
    processed_df = df.copy()
    
    # Display initial data info for debugging
    st.write("Data shape:", processed_df.shape)
    st.write("Columns:", processed_df.columns.tolist())
    
    # Clean numeric columns
    numeric_columns = {
        'power': 'power',
        'speed': 'speed',
        'UTS': 'UTS',
        'YS': 'YS',
        'Elongation': 'Elongation',
        'solution temp': 'solution temp',
        'hardness': 'hardness'
    }
    
    # Convert each column safely
    for col in numeric_columns:
        if col in processed_df.columns:
            processed_df[col] = clean_numeric_data(processed_df, col)
    
    return processed_df

def main():
    st.title("LPBF AlSi10Mg Process-Property Analysis")
    
    uploaded_file = st.file_uploader("Upload LPBF data (CSV)", type="csv")
    
    if uploaded_file:
        # Read and process data
        df = pd.read_csv(uploaded_file)
        processed_df = process_lpbf_data(df)
        
        # Display data preview
        st.write("### Data Preview")
        st.dataframe(processed_df.head())
        
        # Basic statistical summary
        st.write("### Statistical Summary")
        numeric_df = processed_df.select_dtypes(include=[np.number])
        st.dataframe(numeric_df.describe())
        
        # Create analysis sections with error handling
        try:
            # Process Parameters Analysis
            st.header("Process Parameters Analysis")
            
            # Check if we have valid process parameter data
            if 'power' in processed_df.columns and 'speed' in processed_df.columns:
                valid_data = processed_df.dropna(subset=['power', 'speed'])
                
                if not valid_data.empty:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=valid_data['speed'],
                        y=valid_data['power'],
                        mode='markers',
                        marker=dict(size=8),
                        name='Process Points'
                    ))
                    
                    fig.update_layout(
                        title="Process Window Map",
                        xaxis_title="Scan Speed (mm/s)",
                        yaxis_title="Laser Power (W)"
                    )
                    
                    st.plotly_chart(fig)
            
            # Mechanical Properties Analysis
            st.header("Mechanical Properties Analysis")
            
            if all(col in processed_df.columns for col in ['UTS', 'Elongation']):
                valid_properties = processed_df.dropna(subset=['UTS', 'Elongation'])
                
                if not valid_properties.empty:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=valid_properties['UTS'],
                        y=valid_properties['Elongation'],
                        mode='markers',
                        marker=dict(size=8),
                        name='Properties'
                    ))
                    
                    fig.update_layout(
                        title="Strength-Ductility Relationship",
                        xaxis_title="Ultimate Tensile Strength (MPa)",
                        yaxis_title="Elongation (%)"
                    )
                    
                    st.plotly_chart(fig)
            
            # Heat Treatment Analysis
            st.header("Heat Treatment Analysis")
            
            if 'solution temp' in processed_df.columns and 'UTS' in processed_df.columns:
                valid_ht = processed_df.dropna(subset=['solution temp', 'UTS'])
                
                if not valid_ht.empty:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=valid_ht['solution temp'],
                        y=valid_ht['UTS'],
                        mode='markers',
                        marker=dict(size=8),
                        name='Heat Treatment Effect'
                    ))
                    
                    fig.update_layout(
                        title="Heat Treatment Effect on Strength",
                        xaxis_title="Solution Temperature (°C)",
                        yaxis_title="Ultimate Tensile Strength (MPa)"
                    )
                    
                    st.plotly_chart(fig)
            
            # Property Optimization
            st.header("Property Optimization")
            
            if 'UTS' in processed_df.columns:
                top_properties = processed_df.nlargest(5, 'UTS')
                
                if not top_properties.empty:
                    st.write("### Top 5 Property Combinations")
                    display_columns = ['UTS', 'YS', 'Elongation', 'power', 'speed', 'solution temp']
                    display_columns = [col for col in display_columns if col in top_properties.columns]
                    st.dataframe(top_properties[display_columns])
        
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            st.write("Please check your data format and try again.")

if __name__ == "__main__":
    main()

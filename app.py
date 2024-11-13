import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="LPBF AlSi10Mg Analysis", layout="wide")

def process_lpbf_data(df):
    """Process LPBF data and calculate key parameters"""
    df = df.copy()
    
    # Display column names for debugging
    st.write("Available columns in your data:", df.columns.tolist())
    
    return df

def find_column_containing(df, search_term):
    """Find column names containing the search term"""
    return [col for col in df.columns if search_term.lower() in col.lower()]

def main():
    st.title("LPBF AlSi10Mg Process-Property Explorer")
    
    uploaded_file = st.file_uploader("Upload LPBF data (CSV)", type="csv")
    
    if uploaded_file:
        # Load the data
        data = pd.read_csv(uploaded_file)
        df = process_lpbf_data(data)
        
        # Find relevant columns
        power_cols = find_column_containing(df, 'power')
        speed_cols = find_column_containing(df, 'speed')
        
        if not power_cols or not speed_cols:
            st.error("Could not find power or speed columns in the data. Please verify your column names.")
            return
            
        # Select columns for analysis
        st.subheader("Column Selection")
        power_col = st.selectbox("Select power column:", power_cols)
        speed_col = st.selectbox("Select speed column:", speed_cols)
        
        # Convert selected columns to numeric
        df[power_col] = pd.to_numeric(df[power_col], errors='coerce')
        df[speed_col] = pd.to_numeric(df[speed_col], errors='coerce')
        
        # Sidebar for interactive parameter exploration
        st.sidebar.header("Process Parameter Exploration")
        
        # Power and Speed Ranges with error handling
        try:
            power_min = float(df[power_col].min())
            power_max = float(df[power_col].max())
            speed_min = float(df[speed_col].min())
            speed_max = float(df[speed_col].max())
            
            power_range = st.sidebar.slider(
                "Laser Power Range (W)", 
                min_value=power_min,
                max_value=power_max,
                value=(power_min, power_max)
            )
            
            speed_range = st.sidebar.slider(
                "Scan Speed Range (mm/s)",
                min_value=speed_min,
                max_value=speed_max,
                value=(speed_min, speed_max)
            )
            
            # Filter data based on selections
            filtered_df = df[
                (df[power_col].between(*power_range)) &
                (df[speed_col].between(*speed_range))
            ]
            
            # Main content area
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.header("Process Window Analysis")
                
                # Create process window plot
                fig = go.Figure()
                
                # Basic scatter plot
                fig.add_trace(go.Scatter(
                    x=filtered_df[speed_col],
                    y=filtered_df[power_col],
                    mode='markers',
                    name='Process Points'
                ))
                
                # Update layout
                fig.update_layout(
                    title="Process Window Map",
                    xaxis_title="Scan Speed (mm/s)",
                    yaxis_title="Laser Power (W)",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.header("Data Analysis")
                
                # Display summary statistics
                st.write("### Summary Statistics")
                summary_df = filtered_df[[power_col, speed_col]].describe()
                st.dataframe(summary_df)
            
            # Additional Analysis Sections
            st.header("Additional Properties")
            
            # Find other numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Let user select columns to analyze
            property_col = st.selectbox(
                "Select property to analyze:",
                [col for col in numeric_cols if col not in [power_col, speed_col]]
            )
            
            if property_col:
                col3, col4 = st.columns(2)
                
                with col3:
                    # Property vs Power
                    fig = px.scatter(
                        filtered_df,
                        x=power_col,
                        y=property_col,
                        title=f"{property_col} vs Power",
                        trendline="ols"
                    )
                    st.plotly_chart(fig)
                
                with col4:
                    # Property vs Speed
                    fig = px.scatter(
                        filtered_df,
                        x=speed_col,
                        y=property_col,
                        title=f"{property_col} vs Speed",
                        trendline="ols"
                    )
                    st.plotly_chart(fig)
            
            # Export processed data
            st.download_button(
                label="Download Processed Data",
                data=filtered_df.to_csv(index=False),
                file_name="processed_lpbf_data.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error in data processing: {str(e)}")
            st.write("Please check your data format and column selections.")

if __name__ == "__main__":
    main()

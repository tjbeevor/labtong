import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Set page configuration
st.set_page_config(page_title="LPBF Analysis", layout="wide")

# Cache the data processing
@st.cache_data
def process_data(df):
    """Process and clean the data."""
    numeric_cols = ['speed', 'power', 'UTS', 'YS', 'Elongation', 'Hatch', 'thickness']
    processed_df = df.copy()
    
    for col in numeric_cols:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Remove rows with missing values in key columns
    key_cols = ['speed', 'power', 'UTS', 'YS', 'Elongation']
    key_cols = [col for col in key_cols if col in processed_df.columns]
    return processed_df.dropna(subset=key_cols)

def create_scatter_plot(data, x_col, y_col):
    """Create scatter plot with trend line."""
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        trendline="ols",
        title=f"{y_col} vs {x_col}"
    )
    return fig

def calculate_statistics(data, x_col, y_col):
    """Calculate regression statistics."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        data[x_col],
        data[y_col]
    )
    return {
        'R²': r_value**2,
        'p-value': p_value,
        'slope': slope,
        'intercept': intercept
    }

def get_optimal_parameters(data):
    """Find optimal process parameters based on strength properties."""
    # Get top 10% of samples based on UTS
    if 'UTS' in data.columns:
        top_samples = data.nlargest(int(len(data) * 0.1), 'UTS')
        params = {}
        for col in ['speed', 'power']:
            if col in top_samples.columns:
                params[col] = {
                    'min': top_samples[col].min(),
                    'max': top_samples[col].max(),
                    'mean': top_samples[col].mean()
                }
        return params
    return None

def main():
    st.title("LPBF AlSi10Mg Process-Property Analysis")
    st.write("Upload your LPBF AlSi10Mg data file for analysis of process parameters and mechanical properties.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Load and process data
            data = pd.read_csv(uploaded_file)
            processed_data = process_data(data)

            if processed_data.empty:
                st.error("No valid data found after processing. Please check your input file.")
                return

            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "Data Overview",
                "Process Parameters",
                "Property Analysis",
                "Optimization"
            ])

            with tab1:
                st.header("Data Overview")
                st.subheader("Data Summary")
                st.dataframe(processed_data.describe())
                
                st.subheader("Raw Data Preview")
                st.dataframe(processed_data.head())

            with tab2:
                st.header("Process Parameter Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_param = st.selectbox(
                        "X-axis parameter:",
                        ['power', 'speed']
                    )
                with col2:
                    y_param = st.selectbox(
                        "Y-axis parameter:",
                        ['UTS', 'YS', 'Elongation']
                    )

                if x_param in processed_data.columns and y_param in processed_data.columns:
                    # Create scatter plot
                    fig = create_scatter_plot(processed_data, x_param, y_param)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and display statistics
                    stats_results = calculate_statistics(processed_data, x_param, y_param)
                    st.write("### Statistical Analysis")
                    for stat, value in stats_results.items():
                        st.write(f"**{stat}:** {value:.4f}")

            with tab3:
                st.header("Property Analysis")
                
                # Correlation heatmap
                st.subheader("Correlation Matrix")
                corr_matrix = processed_data[['speed', 'power', 'UTS', 'YS', 'Elongation']].corr()
                fig = px.imshow(
                    corr_matrix,
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Property distributions
                st.subheader("Property Distributions")
                property_to_plot = st.selectbox(
                    "Select property:",
                    ['UTS', 'YS', 'Elongation']
                )
                
                fig = px.box(
                    processed_data,
                    y=property_to_plot,
                    title=f"{property_to_plot} Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab4:
                st.header("Process Optimization")
                
                # Display optimal parameters
                optimal_params = get_optimal_parameters(processed_data)
                if optimal_params:
                    st.subheader("Optimal Process Windows")
                    for param, values in optimal_params.items():
                        st.write(f"""
                        **{param.capitalize()}**
                        - Range: {values['min']:.2f} - {values['max']:.2f}
                        - Optimal (mean): {values['mean']:.2f}
                        """)
                    
                    # Create scatter matrix for top performing samples
                    st.subheader("Parameter Relationships in Top Performing Samples")
                    top_samples = processed_data.nlargest(int(len(processed_data) * 0.1), 'UTS')
                    fig = px.scatter_matrix(
                        top_samples,
                        dimensions=['power', 'speed', 'UTS', 'YS', 'Elongation'],
                        title="Parameter Relationships in Top Performing Samples"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your file contains the required columns and valid data.")

if __name__ == "__main__":
    main()

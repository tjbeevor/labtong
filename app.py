import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Set page configuration
st.set_page_config(page_title="LPBF Analysis", layout="wide")

@st.cache_data
def process_data(df):
    """Process and clean the data."""
    processed_df = df.copy()
    
    # Print column names for debugging
    print("Available columns:", processed_df.columns.tolist())
    
    # Map expected columns to actual columns
    column_mapping = {
        'speed': 'speed',  # mm/s
        'power': 'power',  # W
        'UTS': 'UTS',     # MPa
        'YS': 'YS',       # MPa
        'Elongation': 'Elongation',  # %
        'Hatch': 'Hatch',           # mm
        'thickness': 'thickness'     # mm
    }
    
    # Convert numeric columns, handling both original and mapped names
    for col in processed_df.columns:
        try:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        except:
            continue
    
    return processed_df

def create_scatter_plot(data, x_col, y_col):
    """Create scatter plot with trend line."""
    valid_data = data[[x_col, y_col]].dropna()
    if len(valid_data) > 1:  # Check if we have enough data points
        fig = px.scatter(
            valid_data,
            x=x_col,
            y=y_col,
            trendline="ols",
            title=f"{y_col} vs {x_col}"
        )
        return fig
    return None

def calculate_statistics(data, x_col, y_col):
    """Calculate regression statistics."""
    valid_data = data[[x_col, y_col]].dropna()
    if len(valid_data) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_data[x_col],
            valid_data[y_col]
        )
        return {
            'R²': r_value**2,
            'p-value': p_value,
            'slope': slope,
            'intercept': intercept
        }
    return None

def get_optimal_parameters(data):
    """Find optimal process parameters based on strength properties."""
    if 'UTS' in data.columns:
        # Remove rows with NaN values
        valid_data = data.dropna(subset=['UTS'])
        if not valid_data.empty:
            top_samples = valid_data.nlargest(max(1, int(len(valid_data) * 0.1)), 'UTS')
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
            
            # Display available columns
            st.write("Available columns in your data:", processed_data.columns.tolist())

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
                
                # Get numeric columns for selection
                numeric_columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    x_param = st.selectbox(
                        "X-axis parameter:",
                        numeric_columns
                    )
                with col2:
                    y_param = st.selectbox(
                        "Y-axis parameter:",
                        numeric_columns
                    )

                if x_param and y_param:
                    # Create scatter plot
                    fig = create_scatter_plot(processed_data, x_param, y_param)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate and display statistics
                        stats_results = calculate_statistics(processed_data, x_param, y_param)
                        if stats_results:
                            st.write("### Statistical Analysis")
                            for stat, value in stats_results.items():
                                st.write(f"**{stat}:** {value:.4f}")

            with tab3:
                st.header("Property Analysis")
                
                # Get numeric columns for correlation
                numeric_data = processed_data.select_dtypes(include=[np.number])
                
                # Correlation heatmap
                st.subheader("Correlation Matrix")
                corr_matrix = numeric_data.corr()
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
                    numeric_columns
                )
                
                if property_to_plot:
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
                    if len(numeric_columns) > 1:
                        st.subheader("Parameter Relationships in Top Performing Samples")
                        top_samples = processed_data.nlargest(int(len(processed_data) * 0.1), 'UTS')
                        fig = px.scatter_matrix(
                            top_samples,
                            dimensions=numeric_columns[:5],  # Limit to first 5 numeric columns
                            title="Parameter Relationships in Top Performing Samples"
                        )
                        st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your file contains the required columns and valid data.")

if __name__ == "__main__":
    main()

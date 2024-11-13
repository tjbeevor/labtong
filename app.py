import streamlit as st
import pandas as pd
import numpy as np
from plotly import express as px, graph_objects as go
from scipy import stats
import base64

# Cache the data loading
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Cache the data processing
@st.cache_data
def process_data(df):
    # Your data processing code here
    return processed_df

# Initialize session state at startup
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

class LPBFAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.clean_data = None
        self.process_data()
    
    def process_data(self):
        """Process and clean the data."""
        # Convert numeric columns
        numeric_cols = ['speed', 'power', 'UTS', 'YS', 'Elongation', 'Hatch', 'thickness']
        self.clean_data = self.data.copy()
        
        for col in numeric_cols:
            if col in self.clean_data.columns:
                self.clean_data[col] = pd.to_numeric(self.clean_data[col], errors='coerce')
        
        # Remove rows with missing values in key columns
        key_cols = ['speed', 'power', 'UTS', 'YS', 'Elongation']
        key_cols = [col for col in key_cols if col in self.clean_data.columns]
        self.clean_data = self.clean_data.dropna(subset=key_cols)
    
    def get_data_summary(self):
        """Get summary statistics of key parameters."""
        if self.clean_data is None or self.clean_data.empty:
            return None
        
        summary_cols = ['speed', 'power', 'UTS', 'YS', 'Elongation']
        summary_cols = [col for col in summary_cols if col in self.clean_data.columns]
        
        return self.clean_data[summary_cols].describe()
    
    def calculate_correlations(self):
        """Calculate correlations between process parameters and properties."""
        if self.clean_data is None or self.clean_data.empty:
            return None
        
        process_params = ['speed', 'power']
        properties = ['UTS', 'YS', 'Elongation']
        
        available_cols = [col for col in process_params + properties 
                         if col in self.clean_data.columns]
        
        return self.clean_data[available_cols].corr()
    
    def get_optimal_parameters(self):
        """Find optimal process parameters based on strength properties."""
        if self.clean_data is None or self.clean_data.empty:
            return None
        
        # Get top 10% of samples based on UTS
        if 'UTS' in self.clean_data.columns:
            top_samples = self.clean_data.nlargest(int(len(self.clean_data) * 0.1), 'UTS')
            
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

def create_download_link(df):
    """Create a download link for the processed data."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed Data</a>'

def main():
    st.set_page_config(page_title="LPBF AlSi10Mg Analysis", layout="wide")
    
    st.title("LPBF AlSi10Mg Process-Property Analysis")
    st.write("""
    Upload your LPBF AlSi10Mg data file to analyze process parameters and mechanical properties.
    The file should contain columns for speed, power, UTS, YS, and Elongation.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            analyzer = LPBFAnalyzer(data)
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs([
                "Data Overview", 
                "Process Parameters", 
                "Property Relationships",
                "Optimization"
            ])
            
            with tab1:
                st.header("Data Overview")
                
                # Display data summary
                st.subheader("Summary Statistics")
                summary = analyzer.get_data_summary()
                if summary is not None:
                    st.dataframe(summary)
                
                # Display processed data
                st.subheader("Processed Data Preview")
                st.dataframe(analyzer.clean_data.head())
                
                # Download link for processed data
                st.markdown(create_download_link(analyzer.clean_data), unsafe_allow_html=True)
            
            with tab2:
                st.header("Process Parameter Analysis")
                
                # Scatter plot with trend line
                st.subheader("Process Parameter Relationships")
                
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
                
                if x_param in analyzer.clean_data.columns and y_param in analyzer.clean_data.columns:
                    fig = px.scatter(
                        analyzer.clean_data,
                        x=x_param,
                        y=y_param,
                        trendline="ols",
                        title=f"{y_param} vs {x_param}"
                    )
                    st.plotly_chart(fig)
                    
                    # Calculate and display regression statistics
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        analyzer.clean_data[x_param],
                        analyzer.clean_data[y_param]
                    )
                    
                    st.write(f"""
                    **Regression Statistics:**
                    - R² value: {r_value**2:.3f}
                    - p-value: {p_value:.3e}
                    - Slope: {slope:.3f}
                    - Intercept: {intercept:.3f}
                    """)
            
            with tab3:
                st.header("Property Relationships")
                
                # Correlation matrix
                st.subheader("Correlation Matrix")
                corr_matrix = analyzer.calculate_correlations()
                if corr_matrix is not None:
                    fig = px.imshow(
                        corr_matrix,
                        color_continuous_scale='RdBu',
                        aspect='auto'
                    )
                    st.plotly_chart(fig)
                
                # Box plots
                st.subheader("Property Distributions")
                property_to_plot = st.selectbox(
                    "Select property to analyze:",
                    ['UTS', 'YS', 'Elongation']
                )
                
                if property_to_plot in analyzer.clean_data.columns:
                    fig = px.box(
                        analyzer.clean_data,
                        y=property_to_plot,
                        title=f"{property_to_plot} Distribution"
                    )
                    st.plotly_chart(fig)
            
            with tab4:
                st.header("Process Optimization")
                
                # Display optimal parameters
                st.subheader("Optimal Process Windows")
                optimal_params = analyzer.get_optimal_parameters()
                if optimal_params:
                    for param, values in optimal_params.items():
                        st.write(f"""
                        **{param.capitalize()}**
                        - Range: {values['min']:.2f} - {values['max']:.2f}
                        - Optimal (mean): {values['mean']:.2f}
                        """)
                
                # Scatter plot matrix for optimal samples
                if 'UTS' in analyzer.clean_data.columns:
                    top_samples = analyzer.clean_data.nlargest(
                        int(len(analyzer.clean_data) * 0.1),
                        'UTS'
                    )
                    
                    fig = px.scatter_matrix(
                        top_samples,
                        dimensions=['power', 'speed', 'UTS', 'YS', 'Elongation'],
                        title="Parameter Relationships in Top Performing Samples"
                    )
                    st.plotly_chart(fig)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your file contains the required columns and valid data.")

if __name__ == "__main__":
    main()

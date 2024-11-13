import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="LPBF AlSi10Mg Analysis", layout="wide")

def process_lpbf_data(df):
    """Process LPBF data with materials science focus"""
    # Convert important columns to numeric
    numeric_columns = [
        'power', 'speed', 'UTS', 'YS', 'Elongation', 
        'solution temp', 'ageing temp', 'hardness',
        'Hatch', 'thickness', 'Density'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate Energy Density (J/mm³)
    if all(x in df.columns for x in ['power', 'speed', 'Hatch', 'thickness']):
        df['Energy_Density'] = df['power'] / (df['speed'] * df['Hatch'] * df['thickness'])
    
    return df

def main():
    st.title("LPBF AlSi10Mg Process-Property Analyzer")
    
    uploaded_file = st.file_uploader("Upload LPBF data (CSV)", type="csv")
    
    if uploaded_file:
        # Read data
        df = pd.read_csv(uploaded_file)
        processed_df = process_lpbf_data(df)
        
        # Create main analysis sections
        tab1, tab2, tab3 = st.tabs([
            "Process Parameters", 
            "Heat Treatment Analysis",
            "Property Optimization"
        ])
        
        with tab1:
            st.header("Process Parameter Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Process window plot
                fig = px.scatter(
                    processed_df,
                    x='speed',
                    y='power',
                    color='UTS',
                    size='UTS',
                    hover_data=['Density', 'Energy_Density'],
                    title="Process Window Map (Color indicates UTS)",
                    labels={
                        'speed': 'Scan Speed (mm/s)',
                        'power': 'Laser Power (W)',
                        'UTS': 'Ultimate Tensile Strength (MPa)'
                    }
                )
                st.plotly_chart(fig)
            
            with col2:
                # Key process parameters
                st.write("### Process Parameter Ranges")
                metrics = {
                    'Power (W)': processed_df['power'],
                    'Speed (mm/s)': processed_df['speed'],
                    'Energy Density (J/mm³)': processed_df['Energy_Density']
                }
                
                for name, values in metrics.items():
                    st.metric(
                        name,
                        f"{values.mean():.1f}",
                        f"±{values.std():.1f}"
                    )
        
        with tab2:
            st.header("Heat Treatment Analysis")
            
            # Heat treatment effect visualization
            col3, col4 = st.columns(2)
            
            with col3:
                fig = px.scatter(
                    processed_df,
                    x='solution temp',
                    y='UTS',
                    color='ageing temp',
                    size='Elongation',
                    title="Heat Treatment Effects on Properties",
                    labels={
                        'solution temp': 'Solution Temperature (°C)',
                        'UTS': 'Ultimate Tensile Strength (MPa)',
                        'ageing temp': 'Aging Temperature (°C)'
                    }
                )
                st.plotly_chart(fig)
            
            with col4:
                st.write("### Heat Treatment Optimization")
                
                # Find optimal heat treatment conditions
                best_strength = processed_df.nlargest(5, 'UTS')
                st.write("Top 5 Heat Treatment Conditions:")
                st.dataframe(
                    best_strength[[
                        'solution temp', 'ageing temp', 
                        'UTS', 'YS', 'Elongation'
                    ]]
                )
        
        with tab3:
            st.header("Property Optimization")
            
            # Property relationships
            col5, col6 = st.columns(2)
            
            with col5:
                # Strength-ductility trade-off
                fig = px.scatter(
                    processed_df,
                    x='UTS',
                    y='Elongation',
                    color='Energy_Density',
                    title="Strength-Ductility Trade-off",
                    labels={
                        'UTS': 'Ultimate Tensile Strength (MPa)',
                        'Elongation': 'Elongation (%)',
                        'Energy_Density': 'Energy Density (J/mm³)'
                    }
                )
                st.plotly_chart(fig)
            
            with col6:
                # Quality index analysis
                if 'Quality' in processed_df.columns:
                    fig = px.scatter(
                        processed_df,
                        x='Energy_Density',
                        y='Quality',
                        color='Density',
                        title="Process Quality Analysis",
                        labels={
                            'Energy_Density': 'Energy Density (J/mm³)',
                            'Quality': 'Quality Index',
                            'Density': 'Relative Density (%)'
                        }
                    )
                    st.plotly_chart(fig)
            
            # Optimization guidelines
            st.write("### Process Optimization Guidelines")
            
            # Find optimal process windows
            optimal_conditions = processed_df[
                (processed_df['UTS'] > processed_df['UTS'].quantile(0.75)) &
                (processed_df['Elongation'] > processed_df['Elongation'].quantile(0.75))
            ]
            
            if not optimal_conditions.empty:
                st.write("Recommended Process Windows:")
                st.write(f"- Power: {optimal_conditions['power'].mean():.0f} ± {optimal_conditions['power'].std():.0f} W")
                st.write(f"- Speed: {optimal_conditions['speed'].mean():.0f} ± {optimal_conditions['speed'].std():.0f} mm/s")
                if 'Energy_Density' in optimal_conditions:
                    st.write(f"- Energy Density: {optimal_conditions['Energy_Density'].mean():.2f} ± {optimal_conditions['Energy_Density'].std():.2f} J/mm³")

if __name__ == "__main__":
    main()

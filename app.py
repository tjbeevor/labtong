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
    
    # Convert numeric columns
    numeric_cols = ['power', 'speed', 'Hatch', 'thickness', 'UTS', 'YS', 'Elongation']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate energy density (J/mm³)
    if all(col in df.columns for col in ['power', 'speed', 'Hatch', 'thickness']):
        df['Energy_Density'] = df['power'] / (df['speed'] * df['Hatch'] * df['thickness'])
    
    return df

def main():
    st.title("LPBF AlSi10Mg Process-Property Explorer")
    
    uploaded_file = st.file_uploader("Upload LPBF data (CSV)", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        df = process_lpbf_data(data)
        
        # Sidebar for interactive parameter exploration
        st.sidebar.header("Process Parameter Exploration")
        
        # Power and Speed Ranges
        power_range = st.sidebar.slider(
            "Laser Power Range (W)", 
            min_value=int(df['power'].min()),
            max_value=int(df['power'].max()),
            value=(int(df['power'].min()), int(df['power'].max()))
        )
        
        speed_range = st.sidebar.slider(
            "Scan Speed Range (mm/s)",
            min_value=int(df['speed'].min()),
            max_value=int(df['speed'].max()),
            value=(int(df['speed'].min()), int(df['speed'].max()))
        )
        
        # Filter data based on selections
        filtered_df = df[
            (df['power'].between(*power_range)) &
            (df['speed'].between(*speed_range))
        ]
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Process Window Analysis")
            
            # Create process window plot
            fig = go.Figure()
            
            # Add scatter plot of all data points
            fig.add_trace(go.Scatter(
                x=filtered_df['speed'],
                y=filtered_df['power'],
                mode='markers',
                marker=dict(
                    size=filtered_df['UTS']/10,
                    color=filtered_df['UTS'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="UTS (MPa)")
                ),
                text=filtered_df['UTS'].round(1),
                name='Process Points'
            ))
            
            # Update layout
            fig.update_layout(
                title="Process Window Map (Bubble size and color indicate UTS)",
                xaxis_title="Scan Speed (mm/s)",
                yaxis_title="Laser Power (W)",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.header("Property Analysis")
            
            # Display key statistics
            mean_uts = filtered_df['UTS'].mean()
            max_uts = filtered_df['UTS'].max()
            
            st.metric("Average UTS (MPa)", f"{mean_uts:.1f}")
            st.metric("Maximum UTS (MPa)", f"{max_uts:.1f}")
            
            # Optimal parameter identification
            optimal_row = filtered_df.loc[filtered_df['UTS'].idxmax()]
            
            st.write("### Optimal Parameters")
            st.write(f"Power: {optimal_row['power']:.0f} W")
            st.write(f"Speed: {optimal_row['speed']:.0f} mm/s")
            if 'Energy_Density' in optimal_row:
                st.write(f"Energy Density: {optimal_row['Energy_Density']:.2f} J/mm³")
        
        # Property Relationships Section
        st.header("Property Relationships")
        col3, col4 = st.columns(2)
        
        with col3:
            # Strength-Ductility Relationship
            fig = px.scatter(
                filtered_df,
                x='UTS',
                y='Elongation',
                color='Energy_Density' if 'Energy_Density' in filtered_df.columns else None,
                title="Strength-Ductility Trade-off",
                labels={
                    'UTS': 'Ultimate Tensile Strength (MPa)',
                    'Elongation': 'Elongation (%)',
                    'Energy_Density': 'Energy Density (J/mm³)'
                }
            )
            st.plotly_chart(fig)
        
        with col4:
            # YS/UTS Ratio Analysis
            filtered_df['YS_UTS_Ratio'] = filtered_df['YS'] / filtered_df['UTS']
            
            fig = px.histogram(
                filtered_df,
                x='YS_UTS_Ratio',
                title="YS/UTS Ratio Distribution",
                labels={'YS_UTS_Ratio': 'Yield Strength / Ultimate Tensile Strength'}
            )
            st.plotly_chart(fig)
        
        # Heat Treatment Effects
        st.header("Heat Treatment Effects")
        if 'solution temp' in filtered_df.columns:
            col5, col6 = st.columns(2)
            
            with col5:
                fig = px.box(
                    filtered_df,
                    x=pd.qcut(filtered_df['solution temp'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']),
                    y='UTS',
                    title="Solution Temperature Effect on Strength",
                    labels={
                        'x': 'Solution Temperature Quartile',
                        'UTS': 'Ultimate Tensile Strength (MPa)'
                    }
                )
                st.plotly_chart(fig)
            
            with col6:
                optimal_temp = filtered_df.loc[filtered_df['UTS'].idxmax(), 'solution temp']
                st.write(f"### Optimal Heat Treatment")
                st.write(f"Solution Temperature: {optimal_temp:.0f}°C")
        
        # Quality Metrics
        st.header("Quality Assessment")
        col7, col8 = st.columns(2)
        
        with col7:
            if 'Density' in filtered_df.columns:
                fig = px.histogram(
                    filtered_df,
                    x='Density',
                    title="Part Density Distribution",
                    labels={'Density': 'Relative Density (%)'}
                )
                st.plotly_chart(fig)
        
        with col8:
            # Process stability assessment
            energy_density_range = filtered_df['Energy_Density'].describe()
            st.write("### Process Stability Metrics")
            st.write(f"Energy Density Range: {energy_density_range['min']:.2f} - {energy_density_range['max']:.2f} J/mm³")
            st.write(f"Coefficient of Variation: {filtered_df['UTS'].std() / filtered_df['UTS'].mean() * 100:.1f}%")
        
        # Export processed data
        st.download_button(
            label="Download Processed Data",
            data=filtered_df.to_csv(index=False),
            file_name="processed_lpbf_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
